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
use rustalib::indicators::moving_averages::{calculate_ema, calculate_sma};
use rustalib::indicators::oscillators::{calculate_macd, calculate_rsi};
use rustalib::indicators::volatility::{calculate_atr, calculate_bollinger_bands};
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
    let (mut df, _) = read_financial_data(csv_path)?;
    
    // Handle capitalized column names by standardizing them to lowercase
    let mut rename_columns = Vec::new();
    
    // First identify columns to rename based on lowercase matching
    for column_name in df.get_column_names() {
        let col_lower = column_name.to_lowercase();
        
        // Map each column to a standard name based on case-insensitive matching
        let standard_name = match col_lower.as_str() {
            "open" | "o" | "op" | "openprice" | "open_price" => "open",
            "high" | "h" | "highprice" | "high_price" | "max" => "high",
            "low" | "l" | "lowprice" | "low_price" | "min" => "low",
            "close" | "c" | "cl" | "closeprice" | "close_price" => "close",
            "volume" | "vol" | "v" | "volumes" => "volume",
            "timestamp" | "time" | "date" | "t" | "datetime" | "dt" | "day" => "time",
            "vwap" | "vwavg" | "vw" | "vwprice" | "volumeweightedavgprice" => "vwap",
            "adj close" | "adj_close" | "adjusted close" | "adjusted_close" | "adjclose" | "adj" => "adjusted_close",
            _ => continue,
        };
        
        // If the column needs to be renamed (case is different)
        if column_name != standard_name {
            rename_columns.push((column_name.to_string(), standard_name.to_string()));
        }
    }
    
    println!("Original columns: {:?}", df.get_column_names());
    println!("Columns to rename: {:?}", rename_columns);
    
    // Use DataFrame's lazy API to apply all transformations at once
    let mut lazy_df = df.clone().lazy();
    
    // Apply all column renames
    for (old_name, new_name) in rename_columns {
        lazy_df = lazy_df.with_column(col(&old_name).alias(&new_name));
    }
    
    // Apply all transformations
    df = lazy_df.collect()?;
    
    // Cast volume to Float64 if it exists in the dataframe
    if df.schema().contains("volume") {
        let volume = df.column("volume")?;
        let volume_f64 = volume.cast(&DataType::Float64)?;
        df.with_column(volume_f64)?;
    }
    
    // Add the adjusted_close column if it doesn't exist (using close as a fallback)
    if !df.schema().contains("adjusted_close") && df.schema().contains("close") {
        let close = df.column("close")?.clone();
        df.with_column(close.with_name("adjusted_close".into()))?;
    }
    
    println!("DataFrame columns after renaming: {:?}", df.get_column_names());

    // Add derived features - important fix: directly update df instead of cloning
    add_daily_features(&mut df)?;
    
    println!("DataFrame columns after adding features: {:?}", df.get_column_names());

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
    let df_height = df.height();
    
    // Make sure all numerical columns are cast to Float64
    for col_name in ["open", "high", "low", "close", "volume", "vwap"].iter() {
        if df.schema().contains(col_name) {
            let col = df.column(col_name)?;
            let col_f64 = col.cast(&DataType::Float64)?;
            df.with_column(col_f64)?;
        }
    }
    
    // Helper function to ensure all series have the same length as the DataFrame
    fn ensure_same_length(series: Series, df_height: usize) -> Series {
        if series.len() < df_height {
            // Pad with nulls at the beginning to match DataFrame height
            let missing = df_height - series.len();
            let mut padded = vec![None; missing];
            // Collect the series values
            let values: Vec<Option<f64>> = series.f64().unwrap().into_iter().collect();
            // Append the non-null values
            padded.extend(values);
            Series::new(series.name().to_string().into(), padded)
        } else {
            series
        }
    }
    
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
    let close_iter = df.column("close")?.f64()?;
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
        
    // Add technical indicators using rustalib
    // SMA
    let sma_20 = calculate_sma(df, "close", 20)?;
    let sma_20 = ensure_same_length(sma_20.with_name("sma_20".into()), df_height);
    df.with_column(sma_20)?;
    
    let sma_50 = calculate_sma(df, "close", 50)?;
    let sma_50 = ensure_same_length(sma_50.with_name("sma_50".into()), df_height);
    df.with_column(sma_50)?;
    
    // EMA
    let ema_20 = calculate_ema(df, "close", 20)?;
    let ema_20 = ensure_same_length(ema_20.with_name("ema_20".into()), df_height);
    df.with_column(ema_20)?;
    
    // RSI
    let rsi_14 = calculate_rsi(df, 14, "close")?;
    let rsi_14 = ensure_same_length(rsi_14.with_name("rsi_14".into()), df_height);
    df.with_column(rsi_14)?;
    
    // MACD
    let (macd_series, signal_series) = calculate_macd(df, 12, 26, 9, "close")?;
    let macd_series = ensure_same_length(macd_series.with_name("macd".into()), df_height);
    let signal_series = ensure_same_length(signal_series.with_name("macd_signal".into()), df_height);
    df.with_column(macd_series)?;
    df.with_column(signal_series)?;
    
    // Bollinger Bands
    let (bb_middle, bb_upper, bb_lower) = calculate_bollinger_bands(df, 20, 2.0, "close")?;
    let bb_middle = ensure_same_length(bb_middle.with_name("bb_middle".into()), df_height);
    df.with_column(bb_middle)?;
    
    // ATR
    let atr_14 = calculate_atr(df, 14)?;
    let atr_14 = ensure_same_length(atr_14.with_name("atr_14".into()), df_height);
    df.with_column(atr_14)?;

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
