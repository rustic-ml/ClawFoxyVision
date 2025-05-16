use anyhow::Result;
/// Tests for the GRU (Gated Recurrent Unit) implementation
///
/// This module contains unit tests for the GRU neural network components:
/// - Tests for the GRU cell forward pass
/// - Tests for bidirectional GRU functionality
/// - Tests for the TimeSeriesGru model architecture
/// - Tests for prediction functions with both single and multiple step forecasting
///
/// The module includes helper functions for generating test data with realistic
/// price relationships and computing various technical indicators.
// External imports
use burn::backend::LibTorch;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use chrono::{Duration, NaiveDateTime};
use polars::prelude::*;
use rand::rng;
use rand::Rng;
use rustalib::indicators::moving_averages::{calculate_ema, calculate_sma};
use rustalib::indicators::oscillators::{calculate_macd, calculate_rsi};
use rustalib::indicators::volatility::{calculate_atr, calculate_bollinger_bands};

// Internal imports
use crate::minute::gru::step_2_gru_cell::GRU;
use crate::minute::gru::step_3_gru_model_arch::TimeSeriesGru;
use crate::minute::gru::step_5_prediction::{predict_multiple_steps, predict_next_step};
use crate::minute::lstm::step_1_tensor_preparation::normalize_features;

/// Generates a synthetic financial time series DataFrame for testing
///
/// Creates a DataFrame with realistic price data including:
/// - Time series dates at 1-minute intervals
/// - Symbol column (all set to "AAPL")
/// - OHLC (Open, High, Low, Close) price data with realistic inter-relationships
/// - Volume data
/// - Technical indicators including:
///   - SMA (Simple Moving Average) 20 and 50
///   - EMA (Exponential Moving Average) 20
///   - RSI (Relative Strength Index) 14
///   - MACD (Moving Average Convergence Divergence)
///   - Bollinger Bands middle value
///   - ATR (Average True Range) 14
///   - Price returns
///   - Price range
///
/// # Arguments
///
/// * `num_rows` - The number of data points to generate
///
/// # Returns
///
/// A `Result<DataFrame>` containing the generated data
fn generate_test_dataframe(num_rows: usize) -> Result<DataFrame> {
    let mut rng = rng();

    // Create time series dates
    let base_date =
        NaiveDateTime::parse_from_str("2023-01-01 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let times: Vec<String> = (0..num_rows)
        .map(|i| {
            (base_date + Duration::minutes(i as i64))
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .collect();

    // Generate random price data with realistic relationships
    let mut close_prices = Vec::with_capacity(num_rows);
    let mut open_prices = Vec::with_capacity(num_rows);
    let mut high_prices = Vec::with_capacity(num_rows);
    let mut low_prices = Vec::with_capacity(num_rows);
    let mut volume = Vec::with_capacity(num_rows);

    // Start with a base price around $100 - explicitly typed
    let mut current_price: f64 = 100.0 + (rng.random::<f64>() * 50.0);

    for _ in 0..num_rows {
        // Random price movement between -1% and +1%
        let movement = (rng.random::<f64>() * 2.0 - 1.0) * 0.01;
        current_price = current_price * (1.0 + movement);

        // Generate open, high, low with realistic relationships to close
        let open = current_price * (1.0 + (rng.random::<f64>() * 0.01 - 0.005));
        let high = current_price.max(open) * (1.0 + rng.random::<f64>() * 0.005);
        let low = current_price.min(open) * (1.0 - rng.random::<f64>() * 0.005);

        // Add some random volume
        let vol = rng.random::<u32>() % 100_000 + 10_000;

        // Push to vectors
        close_prices.push(current_price);
        open_prices.push(open);
        high_prices.push(high);
        low_prices.push(low);
        volume.push(vol as f64);
    }

    // Create a symbol column (all the same value)
    let symbol = vec!["AAPL".to_string(); num_rows];

    // Create base DataFrame
    let df = DataFrame::new(vec![
        Series::new("time".into(), times).into(),
        Series::new("symbol".into(), symbol).into(),
        Series::new("close".into(), close_prices.clone()).into(),
        Series::new("open".into(), open_prices.clone()).into(),
        Series::new("high".into(), high_prices.clone()).into(),
        Series::new("low".into(), low_prices.clone()).into(),
        Series::new("volume".into(), volume).into(),
    ])?;

    // Add technical indicators using rustalib
    // Function to ensure consistent length between original DataFrame and indicator Series
    fn ensure_consistent_length(df: &DataFrame, series: Series, name: &str) -> Result<Series> {
        let missing_len = df.height() - series.len();
        if missing_len > 0 {
            // Use 0.0 as default value for most indicators
            let mut fill_values = vec![0.0; missing_len];
            let values: Vec<f64> = series.f64()?.iter().map(|opt| opt.unwrap_or(0.0)).collect();
            fill_values.extend(values);
            Ok(Series::new(name.into(), fill_values))
        } else {
            Ok(series)
        }
    }

    // SMA
    let sma_20 = calculate_sma(&df, "close", 20)?;
    let sma_20 = ensure_consistent_length(&df, sma_20, "sma_20")?;
    
    let sma_50 = calculate_sma(&df, "close", 50)?;
    let sma_50 = ensure_consistent_length(&df, sma_50, "sma_50")?;
    
    // EMA
    let ema_20 = calculate_ema(&df, "close", 20)?;
    let ema_20 = ensure_consistent_length(&df, ema_20, "ema_20")?;
    
    // RSI
    let mut rsi_14 = calculate_rsi(&df, 14, "close")?;
    // Fill the missing value with neutral RSI (50.0)
    let missing_len = df.height() - rsi_14.len();
    if missing_len > 0 {
        let mut fill_values = vec![50.0; missing_len];
        let rsi_values: Vec<f64> = rsi_14.f64()?.iter().map(|opt| opt.unwrap_or(50.0)).collect();
        fill_values.extend(rsi_values);
        rsi_14 = Series::new("rsi_14".into(), fill_values);
    }
    
    // MACD
    let (macd_series, signal_series) = calculate_macd(&df, 12, 26, 9, "close")?;
    let macd_series = ensure_consistent_length(&df, macd_series, "macd")?;
    let signal_series = ensure_consistent_length(&df, signal_series, "macd_signal")?;
    
    // Bollinger Bands
    let (bb_middle, _, _) = calculate_bollinger_bands(&df, 20, 2.0, "close")?;
    let bb_middle = ensure_consistent_length(&df, bb_middle, "bb_middle")?;
    
    // ATR
    let atr_14 = calculate_atr(&df, 14)?;
    let atr_14 = ensure_consistent_length(&df, atr_14, "atr_14")?;
    
    // Calculate simple returns (not included in rustalib)
    let mut returns = Vec::with_capacity(close_prices.len());
    returns.push(0.0); // First point has no return
    
    for i in 1..close_prices.len() {
        let prev = close_prices[i-1];
        let curr = close_prices[i];
        returns.push(if prev != 0.0 { (curr - prev) / prev } else { 0.0 });
    }
    
    // Calculate price range (High - Low) / Close (not included in rustalib)
    let mut price_range = Vec::with_capacity(close_prices.len());
    for i in 0..close_prices.len() {
        let high = high_prices[i];
        let low = low_prices[i];
        let close = close_prices[i];
        price_range.push(if close != 0.0 { (high - low) / close } else { 0.0 });
    }

    // Add technical indicators to the DataFrame
    df.hstack(&[
        sma_20.with_name("sma_20".into()).into(),
        sma_50.with_name("sma_50".into()).into(),
        ema_20.with_name("ema_20".into()).into(),
        rsi_14.with_name("rsi_14".into()).into(),
        macd_series.with_name("macd".into()).into(),
        signal_series.with_name("macd_signal".into()).into(),
        bb_middle.with_name("bb_middle".into()).into(),
        atr_14.with_name("atr_14".into()).into(),
        Series::new("returns".into(), returns).into(),
        Series::new("price_range".into(), price_range).into(),
    ])?;

    Ok(df)
}

/// Tests that the GRU cell correctly performs forward pass with proper dimensions
///
/// Verifies that:
/// - The output dimensions match the expected batch_size, sequence_length, and hidden_size
/// - The output contains valid values (no NaN)
#[test]
fn test_gru_cell_forward_pass() {
    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // Create a small batch of data: batch_size=2, seq_len=3, input_size=4
    let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);

    // Create a GRU cell
    let gru = GRU::<TestBackend>::new(4, 5, 1, false, &device);

    // Perform forward pass
    let output = gru.forward(input);

    // Check dimensions of output tensor
    let dims = output.dims();
    assert_eq!(dims[0], 2, "Batch size should be 2");
    assert_eq!(dims[1], 3, "Sequence length should be 3");
    assert_eq!(dims[2], 5, "Hidden size should be 5");

    // Check that output contains valid values (not NaN)
    let data = output.to_data();
    for val in data.convert::<f32>().as_slice::<f32>().unwrap() {
        assert!(!val.is_nan(), "Output contains NaN values");
    }
}

/// Tests that the bidirectional GRU cell works correctly
///
/// Verifies that:
/// - The output dimensions account for bidirectionality (double the hidden size)
/// - The output contains valid values (no NaN)
#[test]
fn test_gru_bidirectional() {
    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // Create a small batch of data: batch_size=2, seq_len=3, input_size=4
    let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);

    // Create a bidirectional GRU cell
    let gru = GRU::<TestBackend>::new(4, 5, 1, true, &device);

    // Perform forward pass
    let output = gru.forward(input);

    // Check dimensions of output tensor
    // For bidirectional, output should have double the hidden size
    let dims = output.dims();
    assert_eq!(dims[0], 2, "Batch size should be 2");
    assert_eq!(dims[1], 3, "Sequence length should be 3");
    assert_eq!(
        dims[2], 10,
        "Hidden size should be 10 (5*2) for bidirectional"
    );

    // Check that output contains valid values (not NaN)
    let data = output.to_data();
    for val in data.convert::<f32>().as_slice::<f32>().unwrap() {
        assert!(!val.is_nan(), "Output contains NaN values");
    }
}

/// Tests the TimeSeriesGru model architecture
///
/// Verifies that:
/// - The model correctly processes input tensors
/// - The output has the expected dimensions
/// - The output values are clamped to [0, 1] range
#[test]
fn test_timeseries_gru_model() {
    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // For testing purposes, we'll explicitly use 10 feature columns
    // to ensure consistency between our test functions
    let expected_features = 10;

    let model = TimeSeriesGru::<TestBackend>::new(
        expected_features, // input_size - must match exactly what we'll provide
        20,                // hidden_size
        1,                 // output_size (for single target forecasting)
        1,                 // num_layers
        true,              // bidirectional
        0.1,               // dropout
        &device,
    );

    // Create a small batch of data: batch_size=2, seq_len=5, input_size=10
    let input = Tensor::<TestBackend, 3>::ones([2, 5, 10], &device);

    // Perform forward pass
    let output = model.forward(input);

    // Check dimensions of output tensor
    let dims = output.dims();
    assert_eq!(dims[0], 2, "Batch size should be 2");
    assert_eq!(dims[1], 1, "Output size should be 1");

    // Check that output values are in range [0, 1] due to clamping
    let data = output.to_data();
    for val in data.convert::<f32>().as_slice::<f32>().unwrap() {
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "Output should be clamped to [0, 1]"
        );
    }
}

/// Tests the single-step prediction function
///
/// This test verifies that the predict_next_step function:
/// - Properly validates required columns in input data
/// - Returns appropriate errors when columns are missing
#[test]
fn test_predict_next_step() {
    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // Generate test data
    let df = generate_test_dataframe(100).unwrap();
    let mut normalized_df = df.clone();

    // Normalize features
    let feature_columns = vec!["close", "open", "high", "low"];
    normalize_features(&mut normalized_df, &feature_columns, false, false).unwrap();

    // Create a GRU model with a reasonable input size
    let model = TimeSeriesGru::<TestBackend>::new(
        10,   // input_size
        20,   // hidden_size
        1,    // output_size
        1,    // num_layers
        true, // bidirectional
        0.1,  // dropout
        &device,
    );

    // Create a DataFrame with only some of the required columns
    let subset_df = DataFrame::new(vec![
        Series::new("close".into(), vec![0.5f64; 100]).into(),
        Series::new("open".into(), vec![0.5f64; 100]).into(),
        Series::new("high".into(), vec![0.5f64; 100]).into(),
        Series::new("low".into(), vec![0.5f64; 100]).into(),
    ])
    .unwrap();

    // This should fail because we're missing required columns
    let result = predict_next_step(&model, subset_df, &device, false);

    // Verify the function properly validates required columns
    assert!(
        result.is_err(),
        "Function should return an error when required columns are missing"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Missing required column"),
        "Error should indicate missing required column, got: {}",
        err
    );
}

/// Tests the multi-step prediction function
///
/// This test verifies that the predict_multiple_steps function:
/// - Properly validates required columns in input data
/// - Returns appropriate errors when columns are missing
/// - Handles zero forecast horizon correctly
#[test]
fn test_predict_multiple_steps() {
    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // Generate test data
    let df = generate_test_dataframe(100).unwrap();
    let mut normalized_df = df.clone();

    // Normalize features
    let feature_columns = vec!["close", "open", "high", "low"];
    normalize_features(&mut normalized_df, &feature_columns, false, false).unwrap();

    // Create a GRU model with a reasonable input size
    let model = TimeSeriesGru::<TestBackend>::new(
        10,   // input_size
        20,   // hidden_size
        1,    // output_size
        1,    // num_layers
        true, // bidirectional
        0.1,  // dropout
        &device,
    );

    // Create a DataFrame with only some of the required columns
    let subset_df = DataFrame::new(vec![
        Series::new("close".into(), vec![0.5f64; 100]).into(),
        Series::new("open".into(), vec![0.5f64; 100]).into(),
        Series::new("high".into(), vec![0.5f64; 100]).into(),
        Series::new("low".into(), vec![0.5f64; 100]).into(),
    ])
    .unwrap();

    // Define the forecast horizon
    let forecast_horizon = 3;

    // This should fail because we're missing required columns
    let result =
        predict_multiple_steps(&model, subset_df.clone(), forecast_horizon, &device, false);

    // Verify the function properly validates required columns
    assert!(
        result.is_err(),
        "Function should return an error when required columns are missing"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Missing required column"),
        "Error should indicate missing required column, got: {}",
        err
    );

    // Verify that zero horizon works correctly
    let zero_result = predict_multiple_steps(&model, subset_df.clone(), 0, &device, false);
    assert!(zero_result.is_ok(), "Zero horizon should succeed");
    assert_eq!(
        zero_result.unwrap().len(),
        0,
        "Zero horizon should return empty vector"
    );
}

/// Tests the GRU model with realistic data
///
/// This test:
/// - Creates a synthetic dataset with realistic price relationships
/// - Normalizes the data
/// - Constructs feature and target tensors
/// - Creates and runs a GRU model
/// - Verifies output dimensions and value ranges
#[test]
fn test_gru_with_real_data() {
    // Generate test data
    let df = generate_test_dataframe(100).unwrap();

    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // Define the four basic features we'll use for testing
    let feature_columns = vec!["close", "open", "high", "low"];

    // Normalize features - but only normalize price columns
    let mut normalized_df = df.clone();
    normalize_features(&mut normalized_df, &feature_columns, false, false).unwrap();

    // Create a custom function that works with our test data
    // Rather than using the standard dataframe_to_tensors that expects all technical indicators
    let sequence_length = 5;
    let forecast_horizon = 1;

    // Manually create feature and target tensors from our dataframe
    let n_rows = normalized_df.height();
    let n_features = feature_columns.len();

    // Need at least sequence_length + forecast_horizon rows
    assert!(n_rows > sequence_length + forecast_horizon);

    let max_sequences = n_rows - sequence_length - forecast_horizon + 1;
    let mut feature_data = Vec::with_capacity(max_sequences * sequence_length * n_features);
    let mut target_data = Vec::with_capacity(max_sequences * forecast_horizon);

    // Extract features and targets
    for seq_idx in 0..max_sequences {
        // Features: sequence_length timesteps of all features
        for row_idx in seq_idx..(seq_idx + sequence_length) {
            for &col in &feature_columns {
                let value = normalized_df
                    .column(col)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(row_idx)
                    .unwrap_or(0.0) as f32;
                feature_data.push(value);
            }
        }

        // Target: forecast_horizon timesteps of close price
        for h in 0..forecast_horizon {
            let target_idx = seq_idx + sequence_length + h;
            let target = normalized_df
                .column("close")
                .unwrap()
                .f64()
                .unwrap()
                .get(target_idx)
                .unwrap_or(0.0) as f32;
            target_data.push(target);
        }
    }

    // Create tensors using the burn API
    let features = Tensor::<TestBackend, 1>::from_floats(feature_data.as_slice(), &device)
        .reshape([max_sequences, sequence_length, n_features]);

    let targets = Tensor::<TestBackend, 1>::from_floats(target_data.as_slice(), &device)
        .reshape([max_sequences, forecast_horizon]);

    // Update the expected features to match the number of columns we're providing
    let expected_features = n_features; // Use the actual number of features we're using (4)

    // Create a GRU model
    let model = TimeSeriesGru::<TestBackend>::new(
        expected_features, // input_size to match the number of required columns
        20,                // hidden_size
        1,                 // output_size (for single target forecasting)
        1,                 // num_layers
        true,              // bidirectional
        0.1,               // dropout
        &device,
    );

    // Perform forward pass
    let output = model.forward(features);

    // Check dimensions match
    assert_eq!(
        output.dims()[0],
        targets.dims()[0],
        "Batch dimension should match"
    );
    assert_eq!(
        output.dims()[1],
        targets.dims()[1],
        "Output dimension should match"
    );

    // Check that output contains valid values
    let data = output.to_data();
    for val in data.convert::<f32>().as_slice::<f32>().unwrap() {
        assert!(!val.is_nan(), "Output contains NaN values");
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "Output should be clamped to [0, 1]"
        );
    }

    // Test loss computation
}
