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
use rustalib::util::file_utils::read_financial_data;
use serde_json;
use std::convert::Into;

// Internal modules
use crate::constants::{EXTENDED_INDICATORS, TECHNICAL_INDICATORS};

/// Constants specific to daily data
pub const DAILY_FEATURES: [&str; 8] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "returns",
    "price_range",
];

/// Load a CSV file with daily data
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file
///
/// # Returns
///
/// Returns a DataFrame with the data
pub fn load_daily_csv(csv_path: &str) -> PolarsResult<DataFrame> {
    // Read csv file
    let (df, _) = read_financial_data(csv_path)?;

    // Add derived features
    add_daily_features(&mut df.clone())?;

    Ok(df)
}

/// Adds engineered features specific to daily data
///
/// # Arguments
///
/// * `df` - DataFrame to add features to
///
/// # Returns
///
/// Returns a Result indicating success or failure
fn add_daily_features(df: &mut DataFrame) -> PolarsResult<()> {
    // Calculate returns (daily percent change)
    let close = df.column("close")?.f64()?;
    let close_shifted = close.clone().shift(1);
    let returns = close_shifted
        .into_iter()
        .zip(close.into_iter())
        .map(|(prev, curr)| match (prev, curr) {
            (Some(p), Some(c)) => Some((c - p) / p),
            _ => None,
        });
    let returns_series = Series::new("returns".into(), returns.collect::<Vec<Option<f64>>>());

    // Calculate price range (high - low) / close
    let high = df.column("high")?.f64()?;
    let low = df.column("low")?.f64()?;
    let close_iter = close.clone();
    let price_range = high
        .into_iter()
        .zip(low.into_iter())
        .zip(close_iter.into_iter())
        .map(|((h, l), c)| match (h, l, c) {
            (Some(high), Some(low), Some(close)) => Some((high - low) / close),
            _ => None,
        });
    let price_range_series = Series::new(
        "price_range".into(),
        price_range.collect::<Vec<Option<f64>>>(),
    );

    // Add both series at once to avoid multiple mutable borrows
    df.with_column(returns_series)?
        .with_column(price_range_series)?;

    Ok(())
}

/// Normalizes features in the DataFrame for model training
///
/// # Arguments
///
/// * `df` - DataFrame to normalize
///
/// # Returns
///
/// Returns a Result indicating success or failure
pub fn normalize_daily_features(df: &mut DataFrame) -> PolarsResult<()> {
    // List of columns to normalize
    let columns = &["open", "high", "low", "close", "volume", "adjusted_close"];

    for &col in columns {
        if !df.schema().contains(col) {
            continue;
        }

        // Get the series
        let series = df.column(col)?.f64()?;

        // Calculate min and max for normalization
        if let (Some(min_val), Some(max_val)) = (series.min(), series.max()) {
            if (max_val - min_val).abs() < 1e-6 {
                // Skip normalization if the range is too small
                continue;
            }

            // Apply min-max normalization
            let normalized = series
                .clone()
                .into_iter()
                .map(|opt_val| opt_val.map(|val| (val - min_val) / (max_val - min_val)))
                .collect::<Vec<_>>();

            // Replace the column with normalized values
            df.replace(col, Series::new(col.into(), normalized))?;
        }
    }

    Ok(())
}

/// Splits the DataFrame into training and validation sets
///
/// # Arguments
///
/// * `df` - DataFrame to split
/// * `split_ratio` - Ratio of data to use for training (e.g., 0.8 for 80%)
///
/// # Returns
///
/// Returns a tuple of training and validation DataFrames
pub fn split_daily_data(df: &DataFrame, split_ratio: f64) -> PolarsResult<(DataFrame, DataFrame)> {
    let n_rows = df.height();
    let split_idx = (n_rows as f64 * split_ratio) as i64;

    let train_df = df.slice(0, split_idx as usize);
    let val_df = df.slice(split_idx, (n_rows as i64 - split_idx) as usize);

    Ok((train_df, val_df))
}

/// Converts a DataFrame to input tensors for the LSTM model
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `sequence_length` - Number of time steps in each sequence
/// * `forecast_horizon` - Number of days to forecast ahead
/// * `device` - Tensor device
///
/// # Returns
///
/// Returns a tuple of input and target tensors
pub fn dataframe_to_tensors<B: Backend>(
    df: &DataFrame,
    sequence_length: usize,
    forecast_horizon: usize,
    device: &B::Device,
) -> PolarsResult<(Tensor<B, 3>, Tensor<B, 2>)> {
    // Ensure we have enough data
    if df.height() < sequence_length + forecast_horizon {
        return Err(PolarsError::ComputeError(
            "Not enough data for the requested sequence length and forecast horizon".into(),
        ));
    }

    // Prepare feature and target columns
    let feature_columns = &DAILY_FEATURES;
    let target_column = "adjusted_close";

    // Create feature matrix
    let mut features_vec = Vec::new();
    for &col in feature_columns {
        if !df.schema().contains(col) {
            return Err(PolarsError::ComputeError(
                format!("Column '{}' not found in DataFrame", col).into(),
            ));
        }

        let series = df.column(col)?.f64()?;
        features_vec.push(
            series
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect::<Vec<f64>>(),
        );
    }

    // Create target vector (forecast horizon steps ahead)
    let target_series = df.column(target_column)?.f64()?;
    let target_vec = target_series
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect::<Vec<f64>>();

    // Build sequences
    let num_features = feature_columns.len();
    let num_samples = df.height() - sequence_length - forecast_horizon + 1;

    // Initialize tensors
    let mut x_data = Vec::with_capacity(num_samples * sequence_length * num_features);
    let mut y_data = Vec::with_capacity(num_samples);

    // Fill tensors with data
    for i in 0..num_samples {
        // Extract sequence
        for t in 0..sequence_length {
            for f in 0..num_features {
                x_data.push(features_vec[f][i + t] as f32);
            }
        }

        // Extract target (forecast horizon steps ahead)
        y_data.push(target_vec[i + sequence_length + forecast_horizon - 1] as f32);
    }

    // Create tensors for the feature matrix and target vector
    let features_tensor = Tensor::<B, 1>::from_data(x_data.as_slice(), device).reshape([
        num_samples,
        sequence_length,
        num_features,
    ]);

    let targets_tensor =
        Tensor::<B, 1>::from_data(y_data.as_slice(), device).reshape([num_samples, 1]);

    Ok((features_tensor, targets_tensor))
}

/// Imputes missing values in the DataFrame using the specified strategy
///
/// # Arguments
///
/// * `df` - DataFrame to impute values in
/// * `strategy` - Imputation strategy: "mean", "median", "forward", or "backward"
///
/// # Returns
///
/// Returns a Result indicating success or failure
pub fn impute_missing_values(df: &mut DataFrame, strategy: &str) -> PolarsResult<()> {
    // Columns to check for missing values
    let columns = &["open", "high", "low", "close", "volume", "adjusted_close"];

    for &col in columns {
        if !df.schema().contains(col) {
            continue;
        }

        let series = df.column(col)?.f64()?;

        // Skip if no nulls
        if series.null_count() == 0 {
            continue;
        }

        let new_series = match strategy {
            "mean" => {
                if let Some(mean) = series.mean() {
                    series
                        .clone()
                        .into_iter()
                        .map(|v| v.or(Some(mean)))
                        .collect::<Vec<_>>()
                } else {
                    continue;
                }
            }
            "median" => {
                if let Some(median) = series.median() {
                    series
                        .clone()
                        .into_iter()
                        .map(|v| v.or(Some(median)))
                        .collect::<Vec<_>>()
                } else {
                    continue;
                }
            }
            "forward" => {
                let mut result = Vec::with_capacity(series.len());
                let mut last_valid = None;

                for val in series.into_iter() {
                    if val.is_some() {
                        last_valid = val;
                    }
                    result.push(last_valid);
                }

                result
            }
            "backward" => {
                let mut result = vec![None; series.len()];
                let mut last_valid = None;

                for (i, val) in series.into_iter().enumerate().rev() {
                    if val.is_some() {
                        last_valid = val;
                    }
                    result[i] = last_valid;
                }

                result
            }
            _ => {
                return Err(PolarsError::ComputeError(
                    format!("Unknown imputation strategy: {}", strategy).into(),
                ));
            }
        };

        // Replace the column with imputed values
        df.replace(col, Series::new(col.into(), new_series))?;
    }

    Ok(())
}
