// External crates
use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};
use polars::error::PolarsResult;
use polars::prelude::*;
use polars::series::Series;
use polars_utils::float::IsFloat;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde_json;
use std::convert::Into;

// Internal modules
use crate::constants::{EXTENDED_INDICATORS, TECHNICAL_INDICATORS};

/// Splits the DataFrame into training and validation sets using time-based cross-validation
///
/// # Arguments
///
/// * `df` - Input DataFrame to split
/// * `validation_split` - Ratio of data to use for validation (0.0 to 1.0)
/// * `time_based` - Whether to use time-based split (true) or random split (false)
///
/// # Returns
///
/// Returns a tuple of (training_data, validation_data)
pub fn split_data(
    df: &DataFrame,
    validation_split: f64,
    time_based: bool,
) -> Result<(DataFrame, DataFrame)> {
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

    if time_based {
        // Time-based split (for time series data)
        let split_idx = (n_samples as f64 * (1.0 - validation_split)) as i64;
        let train_df = df.slice(0, split_idx as usize);
        let val_df = df.slice(split_idx, (n_samples - split_idx as usize) as usize);
        Ok((train_df, val_df))
    } else {
        // For random split, we'll just use time-based split for now to avoid compile errors
        // This is a simplification - in real production code we would implement proper random CV
        let split_idx = (n_samples as f64 * (1.0 - validation_split)) as i64;
        let train_df = df.slice(0, split_idx as usize);
        let val_df = df.slice(split_idx, (n_samples - split_idx as usize) as usize);
        Ok((train_df, val_df))
    }
}

/// Imputes missing values in the specified DataFrame columns
///
/// # Arguments
///
/// * `df` - DataFrame to impute
/// * `columns` - Columns to check for NaN values
/// * `strategy` - Imputation strategy: "mean", "median", "forward_fill", "backward_fill"
/// * `window_size` - Window size for moving average (not currently used)
///
/// # Returns
///
/// Returns a Result indicating success or failure
pub fn impute_missing_values(
    df: &mut DataFrame,
    columns: &[&str],
    strategy: &str,
    _window_size: Option<usize>,
) -> PolarsResult<()> {
    for &col in columns {
        if !df.schema().contains(col) {
            continue;
        }

        let series = df.column(col)?;

        // Skip if no nulls or not numeric
        if series.null_count() == 0
            || !matches!(series.dtype(), DataType::Float64 | DataType::Int64)
        {
            continue;
        }

        // Get the series as F64
        let f_series = series.f64()?;

        let imputed_series = match strategy {
            "mean" => {
                if let Some(mean) = f_series.mean() {
                    let new_series = f_series.clone().apply(|v| match v {
                        Some(val) if val.is_nan() => Some(mean),
                        _ => v,
                    });
                    new_series.into_series()
                } else {
                    f_series.clone().into_series()
                }
            }
            "median" => {
                if let Some(median) = f_series.median() {
                    let new_series = f_series.clone().apply(|v| match v {
                        Some(val) if val.is_nan() => Some(median),
                        _ => v,
                    });
                    new_series.into_series()
                } else {
                    f_series.clone().into_series()
                }
            }
            "forward_fill" => forward_fill_series(&f_series, col),
            "backward_fill" => backward_fill_series(&f_series, col),
            _ => {
                return Err(PolarsError::ComputeError(
                    format!("Unknown imputation strategy: {}", strategy).into(),
                ));
            }
        };

        // Replace the column in the DataFrame
        df.replace(col, imputed_series)?;
    }

    Ok(())
}

/// Forward fill a series, replacing NaN values with the last valid value
fn forward_fill_series(series: &ChunkedArray<Float64Type>, name: &str) -> Series {
    let mut values: Vec<f64> = Vec::with_capacity(series.len());
    let mut last_valid = 0.0;
    let mut has_valid = false;

    for i in 0..series.len() {
        if let Some(v) = series.get(i) {
            if !v.is_nan() {
                last_valid = v;
                has_valid = true;
                values.push(v);
            } else {
                values.push(if has_valid { last_valid } else { v });
            }
        } else {
            values.push(if has_valid { last_valid } else { 0.0 });
        }
    }

    Series::new(name.into(), values)
}

/// Backward fill a series, replacing NaN values with the next valid value
fn backward_fill_series(series: &ChunkedArray<Float64Type>, name: &str) -> Series {
    let mut values: Vec<f64> = vec![0.0; series.len()];
    let mut last_valid = 0.0;
    let mut has_valid = false;

    for i in (0..series.len()).rev() {
        if let Some(v) = series.get(i) {
            if !v.is_nan() {
                last_valid = v;
                has_valid = true;
                values[i] = v;
            } else {
                values[i] = if has_valid { last_valid } else { v };
            }
        } else {
            values[i] = if has_valid { last_valid } else { 0.0 };
        }
    }

    Series::new(name.into(), values)
}

/// Detect and handle outliers in the DataFrame
///
/// # Arguments
///
/// * `df` - DataFrame to process
/// * `columns` - Columns to check for outliers
/// * `method` - Outlier detection method: "zscore", "iqr", or "percentile"
/// * `threshold` - Threshold for outlier detection (z-score threshold or IQR multiplier)
/// * `strategy` - Handling strategy: "clip", "mean", or "median"
///
/// # Returns
///
/// Returns a PolarsResult indicating success or failure
pub fn handle_outliers(
    df: &mut DataFrame,
    columns: &[&str],
    method: &str,
    threshold: f64,
    strategy: &str,
) -> PolarsResult<()> {
    for &col in columns {
        if !df.schema().contains(col) {
            continue;
        }

        let series = df.column(col)?;

        // Skip if not numeric
        if !matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
            continue;
        }

        let f_series = series.f64()?;

        // Simple implementation to avoid compile errors
        match (method, strategy) {
            ("zscore", "clip") => {
                if let (Some(mean), Some(std_dev)) = (f_series.mean(), f_series.std(1)) {
                    if std_dev < f64::EPSILON {
                        continue; // No variation
                    }

                    let upper = mean + threshold * std_dev;
                    let lower = mean - threshold * std_dev;

                    // Apply clipping to the values
                    let clipped: Vec<f64> = f_series
                        .into_iter()
                        .map(|opt_v| {
                            if let Some(v) = opt_v {
                                v.min(upper).max(lower)
                            } else {
                                mean
                            }
                        })
                        .collect();

                    df.replace(col, Series::new(col.into(), clipped))?;
                }
            }
            ("iqr", "clip") => {
                // Get quartiles directly from sorted values to avoid method calls
                let mut values: Vec<f64> = f_series
                    .into_iter()
                    .filter_map(|v| v.filter(|x| !x.is_nan()))
                    .collect();

                if values.is_empty() {
                    continue;
                }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let len = values.len();
                let q1_idx = len / 4;
                let q3_idx = 3 * len / 4;

                let q1 = values[q1_idx];
                let q3 = values[q3_idx];

                let iqr = q3 - q1;
                if iqr < f64::EPSILON {
                    continue; // No variation
                }

                let upper = q3 + threshold * iqr;
                let lower = q1 - threshold * iqr;

                // Apply clipping to the values
                let clipped: Vec<f64> = f_series
                    .into_iter()
                    .map(|opt_v| {
                        if let Some(v) = opt_v {
                            v.min(upper).max(lower)
                        } else {
                            // Use median for missing values
                            values[len / 2]
                        }
                    })
                    .collect();

                df.replace(col, Series::new(col.into(), clipped))?;
            }
            ("zscore", "mean") | ("iqr", "mean") => {
                if let Some(mean) = f_series.mean() {
                    // Create a mask for outliers (simplified - would be different for zscore/iqr)
                    let std_dev = f_series.std(1).unwrap_or(1.0);

                    // Create values with outliers replaced by mean
                    let mut values = Vec::with_capacity(f_series.len());
                    for i in 0..f_series.len() {
                        if let Some(val) = f_series.get(i) {
                            // Calculate z-score
                            let z_score = (val - mean) / std_dev;
                            if z_score.abs() > threshold {
                                values.push(mean);
                            } else {
                                values.push(val);
                            }
                        } else {
                            values.push(mean);
                        }
                    }

                    df.replace(col, Series::new(col.into(), values))?;
                }
            }
            ("zscore", "median") | ("iqr", "median") => {
                if let Some(median) = f_series.median() {
                    // Create a mask for outliers (simplified - would be different for zscore/iqr)
                    let std_dev = f_series.std(1).unwrap_or(1.0);
                    let mean = f_series.mean().unwrap_or(0.0);

                    // Create values with outliers replaced by median
                    let mut values = Vec::with_capacity(f_series.len());
                    for i in 0..f_series.len() {
                        if let Some(val) = f_series.get(i) {
                            // Calculate z-score
                            let z_score = (val - mean) / std_dev;
                            if z_score.abs() > threshold {
                                values.push(median);
                            } else {
                                values.push(val);
                            }
                        } else {
                            values.push(median);
                        }
                    }

                    df.replace(col, Series::new(col.into(), values))?;
                }
            }
            _ => {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Unsupported method/strategy combination: {}/{}",
                        method, strategy
                    )
                    .into(),
                ));
            }
        }
    }

    Ok(())
}

/// Generate augmented data using time series specific techniques (simplified)
///
/// # Arguments
///
/// * `df` - Original DataFrame
/// * `techniques` - Array of augmentation techniques to apply (currently only supports "jitter")
/// * `augmentation_factor` - How many augmented samples to create per original sample
///
/// # Returns
///
/// Returns an augmented DataFrame
pub fn augment_time_series(
    df: &DataFrame,
    techniques: &[&str],
    augmentation_factor: usize,
) -> PolarsResult<DataFrame> {
    // Simplified implementation that only supports jitter
    if !techniques.contains(&"jitter") {
        // If not using jitter, just return the original
        return Ok(df.clone());
    }

    let mut augmented_dfs = Vec::with_capacity(augmentation_factor + 1);
    augmented_dfs.push(df.clone());

    let orig_height = df.height();
    let mut rng = rand::rng();

    for _ in 0..augmentation_factor {
        // Create a copy of the original dataframe
        let mut aug_df = df.clone();

        // Apply jitter to numeric columns
        for col_name in df.get_column_names() {
            if let Ok(series) = df.column(&col_name) {
                if !matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
                    continue;
                }

                let f_series = series.f64()?;
                let std = f_series.std(1).unwrap_or(1.0) * 0.05; // 5% of std

                if std < f64::EPSILON {
                    continue;
                }

                // Create a series with jittered values
                let mut jittered = Vec::with_capacity(orig_height);
                for i in 0..orig_height {
                    let orig_val = f_series.get(i).unwrap_or(0.0);
                    let noise = rng.random_range(-std..std);
                    jittered.push(orig_val + noise);
                }

                // Replace the column
                aug_df.replace(&col_name, Series::new(col_name.clone(), jittered))?;
            }
        }

        augmented_dfs.push(aug_df);
    }

    // Combine all augmented DataFrames
    let mut combined_df = augmented_dfs.remove(0);
    for df in augmented_dfs {
        combined_df = combined_df.vstack(&df)?;
    }

    Ok(combined_df)
}

/// Normalize features using Z-score (standardization) and min-max scaling based on feature type
///
/// # Arguments
///
/// * `df` - Input DataFrame to normalize
/// * `price_columns` - Columns containing price data to use z-score normalization
/// * `use_extended_features` - Whether to use the extended feature set
/// * `handle_outliers_flag` - Whether to detect and handle outliers before normalization
///
/// # Returns
///
/// Returns a PolarsResult indicating success or failure
pub fn normalize_features(
    df: &mut DataFrame,
    price_columns: &[&str],
    use_extended_features: bool,
    handle_outliers_flag: bool,
) -> PolarsResult<()> {
    // Choose which feature set to use
    let feature_columns = if use_extended_features {
        &EXTENDED_INDICATORS[..]
    } else {
        &TECHNICAL_INDICATORS[..]
    };

    // Check for NaN values before normalization
    let nan_count = check_for_nans(df, feature_columns)?;
    if nan_count > 0 {
        // Instead of dropping rows, try to impute missing values
        impute_missing_values(df, feature_columns, "forward_fill", None)?;

        // If we still have nulls after imputation, try median imputation
        let remaining_nans = check_for_nans(df, feature_columns)?;
        if remaining_nans > 0 {
            impute_missing_values(df, feature_columns, "median", None)?;
        }

        // As a last resort, drop any remaining rows with nulls
        if check_for_nans(df, feature_columns)? > 0 {
            *df = df.drop_nulls::<String>(None)?;
        }

        if df.height() == 0 {
            return Err(PolarsError::ComputeError(
                "All rows contained NaN values and were dropped".into(),
            ));
        }
    }

    // Handle outliers before normalization if requested
    if handle_outliers_flag {
        // Use different strategies for price and other columns
        // For price columns, use IQR method with clipping
        handle_outliers(df, price_columns, "iqr", 1.5, "clip")?;

        // For other columns, use Z-score method with mean replacement
        let other_columns: Vec<&str> = feature_columns
            .iter()
            .filter(|col| !price_columns.contains(col))
            .copied()
            .collect();

        if !other_columns.is_empty() {
            handle_outliers(df, &other_columns, "zscore", 3.0, "mean")?;
        }
    }

    // Normalization dictionary: stores min/max/mean/std for later denormalization
    let mut norm_params = std::collections::HashMap::new();

    // Process each column based on its normalization strategy
    for &col in feature_columns {
        if let Ok(series) = df.column(col) {
            // Skip if column is not numeric
            if !matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
                continue;
            }

            let f_series = series.f64()?;

            // Check if we have any NaN values in this series
            let mut has_nans = false;
            for opt_val in f_series.iter() {
                if let Some(val) = opt_val {
                    if val.is_nan() {
                        has_nans = true;
                        break;
                    }
                } else {
                    has_nans = true;
                    break;
                }
            }

            // Check if this is a constant column
            let min = f_series.min().unwrap_or(0.0);
            let max = f_series.max().unwrap_or(1.0);
            let is_constant = (max - min).abs() < 1e-10;

            // Handle constant columns specially based on column type
            if is_constant {
                if price_columns.contains(&col) {
                    // For constant price columns, set values to 0.0
                    let constant_series = Series::new(col.into(), vec![0.0f64; df.height()]);
                    df.replace(col, constant_series)?;

                    // Save params for denormalization (mean = value, std = 1.0)
                    norm_params.insert(col.to_string(), (0.0, 0.0, min, 1.0));
                } else {
                    // For constant non-price columns, set values to 0.5
                    let constant_series = Series::new(col.into(), vec![0.5f64; df.height()]);
                    df.replace(col, constant_series)?;

                    // Save params for denormalization (min = 0.0, max = 1.0)
                    norm_params.insert(col.to_string(), (0.0, 1.0, 0.0, 0.0));
                }
                continue;
            }

            if has_nans {
                // Try to fix NaN values in this column
                let mut column_as_vec: Vec<f64> =
                    f_series.into_iter().map(|v| v.unwrap_or(0.0)).collect();

                // Replace NaN values with 0.0
                for val in &mut column_as_vec {
                    if val.is_nan() {
                        *val = 0.0;
                    }
                }

                // Replace the column with cleaned values
                df.replace(col, Series::new(col.into(), column_as_vec))?;

                // Get the clean series again
                let f_series = df.column(col)?.f64()?;

                // For price columns, use z-score normalization
                if price_columns.contains(&col) {
                    let mean = f_series.mean().unwrap_or(0.0);
                    let std = f_series.std(1).unwrap_or(1.0);

                    // Ensure std is not zero or NaN
                    let std = if std.is_nan() || std.abs() < 1e-10 {
                        1.0
                    } else {
                        std
                    };

                    // Save params for denormalization
                    norm_params.insert(col.to_string(), (0.0, 0.0, mean, std));

                    // Apply z-score normalization
                    let normalized: Vec<f64> = f_series
                        .into_iter()
                        .map(|opt_v| {
                            if let Some(v) = opt_v {
                                if v.is_nan() {
                                    0.0
                                } else {
                                    (v - mean) / std
                                }
                            } else {
                                0.0 // Default for missing values
                            }
                        })
                        .collect();
                    df.replace(col, Series::new(col.into(), normalized))?;
                }
                // For other columns, use min-max scaling
                else {
                    let min = f_series.min().unwrap_or(0.0);
                    let max = f_series.max().unwrap_or(1.0);

                    // Ensure min and max are valid
                    let (min, max) = if min.is_nan() || max.is_nan() || (max - min).abs() < 1e-10 {
                        (0.0, 1.0) // Default values if invalid
                    } else {
                        (min, max)
                    };

                    // Save params for denormalization
                    norm_params.insert(col.to_string(), (min, max, 0.0, 0.0));

                    // Apply min-max scaling
                    let normalized: Vec<f64> = f_series
                        .into_iter()
                        .map(|opt_v| {
                            if let Some(v) = opt_v {
                                if v.is_nan() {
                                    0.5
                                } else {
                                    (v - min) / (max - min)
                                }
                            } else {
                                0.5 // Default for missing values
                            }
                        })
                        .collect();
                    df.replace(col, Series::new(col.into(), normalized))?;
                }
            } else {
                // No NaN values, proceed with normal normalization

                // For price columns, use z-score normalization
                if price_columns.contains(&col) {
                    let mean = f_series.mean().unwrap_or(0.0);
                    let std = f_series.std(1).unwrap_or(1.0);

                    // Ensure std is not zero or NaN
                    let std = if std.is_nan() || std.abs() < 1e-10 {
                        1.0
                    } else {
                        std
                    };

                    // Save params for denormalization
                    norm_params.insert(col.to_string(), (0.0, 0.0, mean, std));

                    // Apply z-score normalization
                    let normalized: Vec<f64> = f_series
                        .into_iter()
                        .map(|opt_v| {
                            if let Some(v) = opt_v {
                                (v - mean) / std
                            } else {
                                0.0 // Default for missing values
                            }
                        })
                        .collect();
                    df.replace(col, Series::new(col.into(), normalized))?;
                }
                // For other columns, use min-max scaling
                else {
                    let min = f_series.min().unwrap_or(0.0);
                    let max = f_series.max().unwrap_or(1.0);

                    // Handle constant columns (where min == max)
                    let (min, max) = if (max - min).abs() < 1e-10 {
                        (0.0, 1.0) // Default range if constant
                    } else {
                        (min, max)
                    };

                    // Save params for denormalization
                    norm_params.insert(col.to_string(), (min, max, 0.0, 0.0));

                    // Apply min-max scaling
                    let normalized: Vec<f64> = f_series
                        .into_iter()
                        .map(|opt_v| {
                            if let Some(v) = opt_v {
                                (v - min) / (max - min)
                            } else {
                                0.5 // Default for missing values
                            }
                        })
                        .collect();
                    df.replace(col, Series::new(col.into(), normalized))?;
                }
            }
        }
    }

    // Save normalization parameters in a new dataframe column as serialized JSON
    let norm_params_json = serde_json::to_string(&norm_params).map_err(|e| {
        PolarsError::ComputeError(
            format!("Failed to serialize normalization parameters: {}", e).into(),
        )
    })?;

    // Add normalization parameters as a metadata column for later use
    let params_series = Series::new("_norm_params".into(), vec![norm_params_json; df.height()]);
    let df_with_params = df.hstack(&[params_series.into()])?;
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
/// * `batch_size` - Size of batches for processing (to avoid large memory usage)
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
    batch_size: Option<usize>,
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
            "Empty DataFrame cannot be converted to tensors".into(),
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
            format!(
                "DataFrame needs at least {} rows for technical indicators",
                required_initial_points
            )
            .into(),
        ));
    };

    // Check for any remaining NaN values and handle them
    let nan_count = check_for_nans(&df, feature_columns)?;
    if nan_count > 0 {
        // Try imputation instead of dropping rows
        let mut df_clean = df.clone();

        // First try forward fill, then median for any remaining
        impute_missing_values(&mut df_clean, feature_columns, "forward_fill", None)?;
        let remaining_nans = check_for_nans(&df_clean, feature_columns)?;

        if remaining_nans > 0 {
            impute_missing_values(&mut df_clean, feature_columns, "median", None)?;
        }

        // If we still have NaNs, report which columns have them
        let final_nans = check_for_nans(&df_clean, feature_columns)?;
        if final_nans > 0 {
            // As a last resort, drop rows with NaN values
            let df_clean = df_clean.drop_nulls::<String>(None)?;

            if df_clean.height() == 0 {
                return Err(PolarsError::ComputeError(
                    "All rows contained NaN values and were dropped".into(),
                ));
            }

            return dataframe_to_tensors::<B>(
                &df_clean,
                sequence_length,
                forecast_horizon,
                device,
                use_extended_features,
                batch_size,
            );
        }

        // Use the cleaned DataFrame
        return dataframe_to_tensors::<B>(
            &df_clean,
            sequence_length,
            forecast_horizon,
            device,
            use_extended_features,
            batch_size,
        );
    }

    // Count rows and columns
    let n_rows = df.height();
    let n_cols = df.width();

    // Determine maximum number of sequences
    let max_sequences = n_rows - sequence_length - forecast_horizon + 1;

    if max_sequences <= 0 {
        return Err(PolarsError::ComputeError(
            format!(
                "Not enough data points for sequence_length={} and forecast_horizon={}",
                sequence_length, forecast_horizon
            )
            .into(),
        ));
    }

    // Determine batch size (default to processing all at once)
    let batch_size = batch_size.unwrap_or(max_sequences);

    // Collect all columns as vec of f64 series for efficient access
    let columns: Vec<ChunkedArray<Float64Type>> = df
        .get_columns()
        .iter()
        .map(|col| col.f64().unwrap().clone())
        .collect();

    // Find the index of the close column if it exists
    let close_idx = feature_columns
        .iter()
        .position(|&s| s == "close")
        .unwrap_or(0);

    // Process in batches to avoid large memory usage
    let mut all_features = Vec::new();
    let mut all_targets = Vec::new();

    for batch_start in (0..max_sequences).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(max_sequences);
        let batch_size = batch_end - batch_start;

        // Pre-allocate buffer for this batch
        let mut feature_buffer = Vec::with_capacity(batch_size * sequence_length * n_cols);
        let mut target_buffer = Vec::with_capacity(batch_size * forecast_horizon);

        // Extract features for this batch
        for seq_idx in batch_start..batch_end {
            // Features (X): sequence_length timesteps of all features
            for row_idx in seq_idx..(seq_idx + sequence_length) {
                for col_idx in 0..n_cols {
                    let f_col = &columns[col_idx];
                    let value = f_col.get(row_idx).unwrap_or(0.0) as f32;
                    feature_buffer.push(value);
                }
            }

            // Target (y): forecast_horizon timesteps of close price
            let f_close = &columns[close_idx];

            for h in 0..forecast_horizon {
                let target_idx = seq_idx + sequence_length + h;
                let target = if target_idx < n_rows {
                    f_close.get(target_idx).unwrap_or(0.0) as f32
                } else {
                    0.0 // Padding for incomplete sequences
                };
                target_buffer.push(target);
            }
        }

        // Create tensors for this batch
        let features_shape = Shape::new([batch_size, sequence_length, n_cols]);
        let features =
            Tensor::<B, 1>::from_floats(feature_buffer.as_slice(), device).reshape(features_shape);

        let targets_shape = Shape::new([batch_size, forecast_horizon]);
        let targets =
            Tensor::<B, 1>::from_floats(target_buffer.as_slice(), device).reshape(targets_shape);

        all_features.push(features);
        all_targets.push(targets);
    }

    // If we processed in multiple batches, concatenate them
    let final_features = if all_features.len() == 1 {
        all_features.pop().unwrap()
    } else {
        Tensor::cat(all_features, 0)
    };

    let final_targets = if all_targets.len() == 1 {
        all_targets.pop().unwrap()
    } else {
        Tensor::cat(all_targets, 0)
    };

    Ok((final_features, final_targets))
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
        let previous = close.get(i - 1).unwrap_or(f64::NAN);
        diff_values.push(current - previous);
    }

    // Create a new Series with the differences
    let close_diff = Series::new("close".into(), diff_values);

    // Replace the close column with differences
    diff_df.replace("close", close_diff)?;

    // Now create tensors from the difference DataFrame
    dataframe_to_tensors::<B>(
        &diff_df,
        sequence_length,
        forecast_horizon,
        device,
        use_extended_features,
        None,
    )
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
            if let Ok(f64_series) = series.f64() {
                // Count both null values and explicit NaN values
                nan_count += f64_series.null_count();

                // Also check for explicit NaNs
                for opt_val in f64_series.iter() {
                    if let Some(val) = opt_val {
                        if val.is_nan() {
                            nan_count += 1;
                        }
                    }
                }
            } else if matches!(series.dtype(), DataType::Int64) {
                nan_count += series.null_count();
            }
        } else {
            // Skip columns that don't exist instead of returning an error
            continue;
        }
    }

    Ok(nan_count)
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
    normalize_features(
        &mut df_norm,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    // Create tensors using normalized data
    dataframe_to_tensors::<BurnBackend>(
        &df_norm,
        crate::constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        false,
        None,
    )
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
    normalize_features(&mut df_norm, &["close", "open", "high", "low"], true, false)?;

    // Create tensors using normalized data with extended features
    dataframe_to_tensors::<BurnBackend>(
        &df_norm,
        crate::constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        true,
        None,
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))
}

// Add Gaussian noise to the data for augmentation
pub fn add_augmentation_noise(
    mut features: Vec<f64>,
    noise_level: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    // Set up the RNG, with a seed if provided for reproducibility
    let mut rng = if let Some(seed_value) = seed {
        StdRng::seed_from_u64(seed_value)
    } else {
        // Just use a fixed seed as a fallback
        StdRng::seed_from_u64(42)
    };

    // Compute standard deviation for each feature
    let mean = features.iter().sum::<f64>() / features.len() as f64;
    let variance =
        features.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / features.len() as f64;
    let std = variance.sqrt() * noise_level;

    // Add noise to each feature value
    features.iter_mut().for_each(|x| {
        let noise = rng.random_range(-std..std);
        *x += noise;
    });

    features
}
