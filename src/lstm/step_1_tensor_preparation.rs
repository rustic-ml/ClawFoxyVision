// External crates
use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};
use polars::error::PolarsResult;
use polars::prelude::*;
use ndarray::Array2;
use rayon::prelude::*;

// Internal modules
use crate::constants::{SEQUENCE_LENGTH, TECHNICAL_INDICATORS, VALIDATION_SPLIT_RATIO};

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

/// Normalizes the features in the DataFrame using min-max scaling
///
/// # Arguments
///
/// * `df` - Input DataFrame to normalize
///
/// # Returns
///
/// Returns a PolarsResult indicating success or failure
fn normalize_features(df: &mut DataFrame) -> PolarsResult<()> {
    // Get numeric columns to normalize
    let numeric_columns = TECHNICAL_INDICATORS;

    for col in numeric_columns {
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

            // Calculate min and max for min-max scaling, propagating errors
            let (min, max) = match (series.min::<f64>()?, series.max::<f64>()?) {
                (Some(min), Some(max)) => (min, max),
                // If either min or max is missing, skip normalization for this column
                _ => continue,
            };
            // Avoid division by zero
            let range = if (max - min).abs() < f64::EPSILON {
                1.0
            } else {
                max - min
            };

            // Apply min-max scaling
            let normalized = (series - min) / range;
            df.replace(col, normalized)?;
        }
    }

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
///
/// # Returns
///
/// Returns a tuple of (features_tensor, target_tensor)
pub fn dataframe_to_tensors<B: Backend>(
    df: &DataFrame,
    sequence_length: usize,
    forecast_horizon: usize,
    device: &B::Device,
) -> PolarsResult<(Tensor<B, 3>, Tensor<B, 2>)> {
    // Select only the columns in TECHNICAL_INDICATORS, in the correct order
    let df = df.select(TECHNICAL_INDICATORS)?;
    // Drop all rows with nulls to avoid shape mismatches
    let df = df.drop_nulls::<String>(None)?;
    let n_samples = df.height();
    let feature_columns: Vec<&str> = TECHNICAL_INDICATORS.iter().copied().collect();
    let n_features = feature_columns.len();
    if n_samples < sequence_length + forecast_horizon {
        return Err(PolarsError::ComputeError(
            format!(
                "DataFrame has too few rows ({}) for sequence_length ({}) and forecast_horizon ({})",
                n_samples, sequence_length, forecast_horizon
            )
            .into(),
        ));
    }
    // Ensure all required feature columns are present
    for &col in &feature_columns {
        if !df.get_column_names_str().contains(&col) {
            return Err(PolarsError::ComputeError(format!("Column '{}' not found", col).into()));
        }
    }
    let n_sequences = n_samples - sequence_length - forecast_horizon + 1;
    // Collect owned Series for each feature column
    let columns: Vec<Series> = feature_columns
        .iter()
        .map(|&name| df.column(name).unwrap().as_series().unwrap().clone())
        .collect();
    let close_idx = feature_columns
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
    dataframe_to_tensors::<BurnBackend>(&df, crate::constants::SEQUENCE_LENGTH, forecast_horizon, &device)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}