// External crates
use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};
use polars::error::PolarsResult;
use polars::prelude::*;
use polars::prelude::PlSmallStr;
use ndarray::Array2;
use rayon::prelude::*;
use serde_json;

// Internal modules
use crate::constants::{SEQUENCE_LENGTH, TECHNICAL_INDICATORS, VALIDATION_SPLIT_RATIO, EXTENDED_INDICATORS, PRICE_DENORM_CLIP_MIN};

/// Splits the DataFrame into training and validation sets
///
/// # Arguments
///
/// * `df` - Input DataFrame to split
/// * `validation_split` - Ratio of data to use for validation (0.0 to 1.0)
///
/// # Returns
///
/// Returns a tuple of (training_data, validation_data)
fn split_data(df: &DataFrame, validation_split: f64) -> Result<(DataFrame, DataFrame)> {
    // Validate input
    if df.height() == 0 {
        return Err(PolarsError::ComputeError("Empty DataFrame".into()).into());
    }
    if !(0.0..=1.0).contains(&validation_split) {
        return Err(PolarsError::ComputeError(
            "Validation split must be between 0.0 and 1.0".into(),
        )
        .into());
    }

    let n_samples = df.height();
    let split_idx = (n_samples as f64 * (1.0 - validation_split)) as usize;

    let train_df = df.slice(0, split_idx);
    let val_df = df.slice(split_idx.try_into().unwrap(), n_samples - split_idx);

    Ok((train_df, val_df))
}

/// Checks for NaN values in the specified DataFrame columns
///
/// # Arguments
///
/// * `df` - DataFrame to check
/// * `columns` - Columns to check for NaN values
///
/// # Returns
///
/// Returns a Result with the count of NaN values found, or an error if columns don't exist
pub fn check_for_nans(df: &DataFrame, columns: &[&str]) -> PolarsResult<usize> {
    let mut nan_count = 0;
    
    for &col in columns {
        if let Ok(series) = df.column(col) {
            if matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
                nan_count += series.null_count();
            }
        } else {
            return Err(PolarsError::ComputeError(
                format!("Column '{}' not found", col).into()
            ));
        }
    }
    
    Ok(nan_count)
}

/// Normalize features using Z-score (standardization) and min-max scaling based on feature type
///
/// # Arguments
///
/// * `df` - Input DataFrame to normalize
/// * `price_columns` - Columns containing price data to use z-score normalization
/// * `use_extended_features` - Whether to use the extended feature set
///
/// # Returns
///
/// Returns a PolarsResult indicating success or failure
pub fn normalize_features(df: &mut DataFrame, price_columns: &[&str], use_extended_features: bool) -> PolarsResult<()> {
    // Choose which feature set to use
    let feature_columns = if use_extended_features {
        &EXTENDED_INDICATORS[..]
    } else {
        &TECHNICAL_INDICATORS[..]
    };
    
    // Check for NaN values before normalization
    let nan_count = check_for_nans(df, feature_columns)?;
    if nan_count > 0 {
        // Handle NaNs by dropping rows with nulls
        *df = df.drop_nulls::<String>(None)?;
        
        if df.height() == 0 {
            return Err(PolarsError::ComputeError(
                "All rows contained NaN values and were dropped".into()
            ));
        }
    }

    // Normalization dictionary: stores min/max/mean/std for later denormalization
    let mut norm_params = std::collections::HashMap::new();

    // Process each column based on its normalization strategy
    for col in feature_columns {
        if let Ok(series) = df.column(col) {
            // Skip if column is not numeric
            if !matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
                continue;
            }

            // Convert to Float64 if needed and get a materialized Series reference
            let series_ref = series.as_materialized_series();
            let series = if series_ref.dtype() == &DataType::Int64 {
                series_ref.cast(&DataType::Float64)?
            } else {
                series_ref.clone()
            };

            // For price columns, use z-score normalization
            if price_columns.contains(col) {
                let (mean, std) = match (series.mean(), series.std(1)) {
                    (Some(mean), Some(std)) => (mean, std),
                    // If mean or std calculation fails, skip normalization
                    _ => continue,
                };
                
                // Save params for denormalization
                norm_params.insert(col.to_string(), (0.0, 0.0, mean, std));
                
                // Handle constant columns (where std == 0)
                if std.abs() < f64::EPSILON {
                    // For constant columns, set values to 0 (z-score of mean is 0)
                    let constant_series = Series::new(PlSmallStr::from(*col), vec![0.0f64; df.height()]);
                    df.replace(col, constant_series)?;
                    continue;
                }
                
                // Apply z-score normalization
                let normalized = (series - mean) / std;
                df.replace(col, normalized)?;
            } 
            // For other columns, use min-max scaling
            else {
                let (min, max) = match (series.min::<f64>()?, series.max::<f64>()?) {
                    (Some(min), Some(max)) => (min, max),
                    // If either min or max is missing, skip normalization
                    _ => continue,
                };
                
                // Save params for denormalization
                norm_params.insert(col.to_string(), (min, max, 0.0, 0.0));
                
                // Handle constant columns (where min == max)
                if (max - min).abs() < f64::EPSILON {
                    // For constant columns, set values to 0.5 (middle of range)
                    let constant_series = Series::new(PlSmallStr::from(*col), vec![0.5f64; df.height()]);
                    df.replace(col, constant_series)?;
                    continue;
                }
                
                // Apply min-max scaling for non-constant columns
                let normalized = (series - min) / (max - min);
                df.replace(col, normalized)?;
            }
        }
    }

    // Save normalization parameters in a new dataframe column as serialized JSON
    let norm_params_json = serde_json::to_string(&norm_params)
        .map_err(|e| PolarsError::ComputeError(format!("Failed to serialize normalization parameters: {}", e).into()))?;
    
    // Add normalization parameters as a metadata column for later use
    let params_series = Series::new(PlSmallStr::from("_norm_params"), vec![norm_params_json; df.height()]);
    let df_with_params = df.hstack(&[params_series.clone().into_column()])?;
    *df = df_with_params;

    Ok(())
}

/// Converts a DataFrame to Burn tensors for LSTM input
///
/// # Arguments
///
/// * `df` - Input DataFrame containing normalized features
/// * `sequence_length` - Number of time steps in each sequence
/// * `forecast_horizon` - Number of time steps to forecast
/// * `device` - The device to create tensors on
/// * `use_extended_features` - Whether to use the extended feature set
///
/// # Returns
///
/// Returns a tuple of (features_tensor, target_tensor)
pub fn dataframe_to_tensors<B: Backend>(
    df: &DataFrame,
    sequence_length: usize,
    forecast_horizon: usize,
    device: &B::Device,
    use_extended_features: bool,
) -> PolarsResult<(Tensor<B, 3>, Tensor<B, 2>)> {
    // Choose which feature set to use
    let feature_columns = if use_extended_features {
        &EXTENDED_INDICATORS[..]
    } else {
        &TECHNICAL_INDICATORS[..]
    };
    
    // Validate input DataFrame
    if df.height() == 0 {
        return Err(PolarsError::ComputeError(
            "Empty DataFrame cannot be converted to tensors".into()
        ));
    }
    
    // Convert feature_columns to Vec<String> for selection
    let columns_vec: Vec<String> = feature_columns.iter().map(|&s| s.to_string()).collect();
    let df = df.select(&columns_vec)?;
    
    // Drop initial rows where technical indicators will have NaN values
    // We need at least 50 points for SMA50, which is our longest-window indicator
    let required_initial_points: usize = 50;
    let df = if df.height() > required_initial_points {
        df.tail(Some(df.height() - required_initial_points))
    } else {
        return Err(PolarsError::ComputeError(
            format!("DataFrame needs at least {} rows for technical indicators", required_initial_points).into()
        ));
    };
    
    // Check for any remaining NaN values
    let nan_count = check_for_nans(&df, feature_columns)?;
    if nan_count > 0 {
        // Print which columns have NaN values for debugging
        for &col in feature_columns {
            if let Ok(series) = df.column(col) {
                let null_count = series.null_count();
                if null_count > 0 {
                    eprintln!("Column '{}' has {} NaN values", col, null_count);
                }
            }
        }
        
        // Drop rows with NaN values
        let df = df.drop_nulls::<String>(None)?;
        
        // Validate we have enough data after dropping NaNs
        if df.height() < sequence_length + forecast_horizon {
            return Err(PolarsError::ComputeError(
                format!(
                    "After dropping {} rows with NaN values, not enough data remains for sequence_length ({}) and forecast_horizon ({})",
                    nan_count, sequence_length, forecast_horizon
                ).into()
            ));
        }
    }
    
    let n_samples = df.height();
    let feature_columns_vec: Vec<&str> = feature_columns.iter().copied().collect();
    let n_features = feature_columns_vec.len();
    
    // Ensure all required feature columns are present and have correct types
    for &col in &feature_columns_vec {
        let series = df.column(col)?;
        if !matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
            return Err(PolarsError::ComputeError(
                format!("Column '{}' must be numeric, found {:?}", col, series.dtype()).into()
            ));
        }
    }
    
    // Calculate number of sequences we can create
    let n_sequences = n_samples - sequence_length - forecast_horizon + 1;
    if n_sequences <= 0 {
        return Err(PolarsError::ComputeError(
            format!(
                "Not enough samples ({}) for sequence_length ({}) and forecast_horizon ({})",
                n_samples, sequence_length, forecast_horizon
            ).into()
        ));
    }
    
    // Collect owned Series for each feature column
    let columns: Vec<Series> = feature_columns_vec
        .iter()
        .map(|&name| df.column(name).unwrap().as_series().unwrap().clone())
        .collect();
    
    let close_idx = feature_columns_vec
        .iter()
        .position(|&c| c == "close")
        .expect("'close' column not found");
    
    // Pre-allocate data buffers and fill in parallel
    let mut features_data = vec![0f32; n_sequences * sequence_length * n_features];
    let mut target_data = vec![0f32; n_sequences * forecast_horizon];

    // Parallel fill features_data: each chunk is one sequence
    features_data
        .par_chunks_mut(sequence_length * n_features)
        .enumerate()
        .for_each(|(i, chunk)| {
            for j in 0..sequence_length {
                for k in 0..n_features {
                    let val = columns[k]
                        .f64()
                        .unwrap()
                        .get(i + j)
                        .unwrap_or(0.0) as f32;
                    chunk[j * n_features + k] = val;
                }
            }
        });

    // Parallel fill target_data: each chunk is one forecast horizon
    target_data
        .par_chunks_mut(forecast_horizon)
        .enumerate()
        .for_each(|(i, chunk)| {
            for fh in 0..forecast_horizon {
                let val = columns[close_idx]
                    .f64()
                    .unwrap()
                    .get(i + sequence_length + fh)
                    .unwrap_or(0.0) as f32;
                chunk[fh] = val;
            }
        });

    let expected_len = n_sequences * sequence_length * n_features;
    if features_data.len() != expected_len {
        panic!(
            "Mismatch: features_data.len() = {}, expected = {} (n_sequences={}, sequence_length={}, n_features={})",
            features_data.len(), expected_len, n_sequences, sequence_length, n_features
        );
    }
    
    let features_shape = Shape::new([n_sequences, sequence_length, n_features]);
    let target_shape = Shape::new([n_sequences, forecast_horizon]);
    
    let features_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(features_data.as_slice(), device)
        .reshape(features_shape);
    let target_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(target_data.as_slice(), device)
        .reshape(target_shape);
    
    Ok((features_tensor, target_tensor))
}

/// Creates tensors of price differences rather than absolute prices,
/// which can make the model more robust to price scaling issues
pub fn dataframe_to_diff_tensors<B: Backend>(
    df: &DataFrame,
    sequence_length: usize,
    forecast_horizon: usize,
    device: &B::Device,
    use_extended_features: bool,
) -> PolarsResult<(Tensor<B, 3>, Tensor<B, 2>)> {
    // Create a new DataFrame with price differences
    let mut diff_df = df.clone();
    
    // Get the close column
    let close = df.column("close")?.f64()?;
    
    // Calculate differences manually
    let len = close.len();
    let mut diff_values: Vec<f64> = Vec::with_capacity(len);
    
    // First value is NaN/null since there's no previous value
    diff_values.push(f64::NAN);
    
    // Calculate differences for the rest
    for i in 1..len {
        let current = close.get(i).unwrap_or(f64::NAN);
        let previous = close.get(i-1).unwrap_or(f64::NAN);
        diff_values.push(current - previous);
    }
    
    // Create a new Series with the differences
    let close_diff = Series::new("close".into(), diff_values);
    
    // Replace the close column with differences
    diff_df.replace("close", close_diff)?;
    
    // Now create tensors from the difference DataFrame
    dataframe_to_tensors::<B>(&diff_df, sequence_length, forecast_horizon, device, use_extended_features)
}

// Wrapper for dataframe_to_tensors for compatibility
pub fn build_burn_lstm_model(
    df: DataFrame,
    forecast_horizon: usize,
) -> anyhow::Result<(
    burn::tensor::Tensor<burn::backend::LibTorch<f32>, 3>,
    burn::tensor::Tensor<burn::backend::LibTorch<f32>, 2>,
)> {
    type BurnBackend = burn::backend::LibTorch<f32>;
    let device = <BurnBackend as burn::tensor::backend::Backend>::Device::default();
    
    // Using z-score normalization for price columns
    let mut df_norm = df.clone();
    normalize_features(&mut df_norm, &["close", "open", "high", "low"], false)?;
    
    // Create tensors using normalized data
    dataframe_to_tensors::<BurnBackend>(&df_norm, crate::constants::SEQUENCE_LENGTH, forecast_horizon, &device, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

// Enhanced wrapper for dataframe_to_tensors using extended features
pub fn build_enhanced_lstm_model(
    df: DataFrame,
    forecast_horizon: usize,
) -> anyhow::Result<(
    burn::tensor::Tensor<burn::backend::LibTorch<f32>, 3>,
    burn::tensor::Tensor<burn::backend::LibTorch<f32>, 2>,
)> {
    type BurnBackend = burn::backend::LibTorch<f32>;
    let device = <BurnBackend as burn::tensor::backend::Backend>::Device::default();
    
    // Using z-score normalization for price columns with extended features
    let mut df_norm = df.clone();
    normalize_features(&mut df_norm, &["close", "open", "high", "low"], true)?;
    
    // Create tensors using normalized data with extended features
    dataframe_to_tensors::<BurnBackend>(&df_norm, crate::constants::SEQUENCE_LENGTH, forecast_horizon, &device, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}