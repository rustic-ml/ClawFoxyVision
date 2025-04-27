// External crates
use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};
use polars::error::PolarsResult;
use polars::prelude::*;
use ndarray::Array2;

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
/// * `device` - The device to create tensors on
///
/// # Returns
///
/// Returns a tuple of (features_tensor, target_tensor)
pub fn dataframe_to_tensors<B: Backend>(
    df: &DataFrame,
    sequence_length: usize,
    device: &B::Device,
) -> PolarsResult<(Tensor<B, 3>, Tensor<B, 2>)> {
    println!("DEBUG: Called dataframe_to_tensors");
    // Select only the columns in TECHNICAL_INDICATORS, in the correct order
    let df = df.select(TECHNICAL_INDICATORS)?;
    // Drop all rows with nulls to avoid shape mismatches
    let df = df.drop_nulls::<String>(None)?;
    println!(
        "DEBUG: DataFrame shape after select/drop_nulls: {:?}, columns: {:?}",
        df.shape(),
        df.get_column_names()
    );
    let n_samples = df.height();
    let feature_columns: Vec<&str> = TECHNICAL_INDICATORS.iter().copied().collect();
    let n_features = feature_columns.len();
    if n_samples < sequence_length + 1 {
        return Err(PolarsError::ComputeError(
            format!(
                "DataFrame has too few rows ({}) for sequence_length ({})",
                n_samples, sequence_length
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
    let n_sequences = n_samples - sequence_length;
    // Collect owned Series for each feature column
    let columns: Vec<Series> = feature_columns
        .iter()
        .map(|&name| df.column(name).unwrap().as_series().unwrap().clone())
        .collect();
    for (idx, col) in columns.iter().enumerate() {
        println!("DEBUG: Column {} (name: {}) length: {}", idx, feature_columns[idx], col.len());
    }
    println!("DEBUG: columns.len() = {}, columns[0].len() = {}", columns.len(), columns[0].len());
    let close_idx = feature_columns
        .iter()
        .position(|&c| c == "close")
        .expect("'close' column not found");
    let mut features_data = Vec::with_capacity(n_sequences * sequence_length * n_features);
    let mut target_data = Vec::with_capacity(n_sequences);

    for i in 0..n_sequences {
        for j in 0..sequence_length {
            for k in 0..n_features {
                let val = columns[k].f64().unwrap().get(i + j).unwrap_or(0.0);
                if features_data.len() < 20 {
                    println!("DEBUG: features_data[{}] = {} (i={}, j={}, k={})", features_data.len(), val, i, j, k);
                }
                features_data.push(val as f32);
            }
        }
        // Target: next close value
        let target = columns[close_idx].f64().unwrap().get(i + sequence_length).unwrap_or(0.0);
        target_data.push(target as f32);
    }
    let expected_len = n_sequences * sequence_length * n_features;
    println!(
        "DEBUG: n_sequences={}, sequence_length={}, n_features={}",
        n_sequences, sequence_length, n_features
    );
    println!(
        "DEBUG: types: n_sequences={}, sequence_length={}, n_features={}",
        std::any::type_name_of_val(&n_sequences),
        std::any::type_name_of_val(&sequence_length),
        std::any::type_name_of_val(&n_features)
    );
    println!(
        "DEBUG: n_sequences * sequence_length = {}",
        n_sequences * sequence_length
    );
    println!(
        "DEBUG: n_sequences * sequence_length * n_features = {}",
        n_sequences * sequence_length * n_features
    );
    println!(
        "DEBUG: n_sequences={}, sequence_length={}, n_features={}, product={}, features_data.len()={}",
        n_sequences, sequence_length, n_features, n_sequences * sequence_length * n_features, features_data.len()
    );
    if features_data.len() != expected_len {
        panic!(
            "Mismatch: features_data.len() = {}, expected = {} (n_sequences={}, sequence_length={}, n_features={})",
            features_data.len(), expected_len, n_sequences, sequence_length, n_features
        );
    }
    let features_shape = Shape::new([n_sequences, sequence_length, n_features]);
    let target_shape = Shape::new([n_sequences, 1]);
    let features_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(features_data.as_slice(), device)
        .reshape(features_shape);
    let target_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(target_data.as_slice(), device)
        .reshape(target_shape);
    Ok((features_tensor, target_tensor))
}

// Wrapper for dataframe_to_tensors for compatibility
pub fn build_burn_lstm_model(
    df: DataFrame,
) -> anyhow::Result<(
    burn::tensor::Tensor<burn::backend::LibTorch<f32>, 3>,
    burn::tensor::Tensor<burn::backend::LibTorch<f32>, 2>,
)> {
    println!("DEBUG: Called build_burn_lstm_model");
    type BurnBackend = burn::backend::LibTorch<f32>;
    let device = <BurnBackend as burn::tensor::backend::Backend>::Device::default();
    dataframe_to_tensors::<BurnBackend>(&df, crate::constants::SEQUENCE_LENGTH, &device)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}