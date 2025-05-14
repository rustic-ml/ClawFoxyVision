// External imports
use burn::backend::LibTorch;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use polars::prelude::*;
use std::path::PathBuf;
use anyhow::Result;
use chrono::{NaiveDateTime, Duration};
use rand::rng;
use rand::Rng;

// Internal imports
use crate::minute::gru::step_2_gru_cell::GRU;
use crate::minute::gru::step_3_gru_model_arch::TimeSeriesGru;
use crate::minute::lstm::step_1_tensor_preparation::normalize_features;
use crate::constants::TECHNICAL_INDICATORS;

// Define local test utility function since we can't import from main crate
fn generate_test_dataframe(num_rows: usize) -> Result<DataFrame> {
    let mut rng = rng();
    
    // Create time series dates
    let base_date = NaiveDateTime::parse_from_str("2023-01-01 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let times: Vec<String> = (0..num_rows)
        .map(|i| (base_date + Duration::minutes(i as i64)).format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    
    // Generate random price data with realistic relationships
    let mut close_prices = Vec::with_capacity(num_rows);
    let mut open_prices = Vec::with_capacity(num_rows);
    let mut high_prices = Vec::with_capacity(num_rows);
    let mut low_prices = Vec::with_capacity(num_rows);
    let mut volume = Vec::with_capacity(num_rows);
    
    // Start with a base price around $100 - explicitly typed
    let mut current_price: f64 = 100.0 + (rng.gen::<f64>() * 50.0);
    
    for _ in 0..num_rows {
        // Random price movement between -1% and +1%
        let movement = (rng.gen::<f64>() * 2.0 - 1.0) * 0.01;
        current_price = current_price * (1.0 + movement);
        
        // Generate open, high, low with realistic relationships to close
        let open = current_price * (1.0 + (rng.gen::<f64>() * 0.01 - 0.005));
        let high = current_price.max(open) * (1.0 + rng.gen::<f64>() * 0.005);
        let low = current_price.min(open) * (1.0 - rng.gen::<f64>() * 0.005);
        
        // Add some random volume
        let vol = rng.gen::<u32>() % 100_000 + 10_000;
        
        // Push to vectors
        close_prices.push(current_price);
        open_prices.push(open);
        high_prices.push(high);
        low_prices.push(low);
        volume.push(vol as f64);
    }
    
    // Create a symbol column (all the same value)
    let symbol = vec!["AAPL".to_string(); num_rows];
    
    // Clone vectors before using them in Series
    let high_prices_clone = high_prices.clone();
    let low_prices_clone = low_prices.clone();
    
    // Create base DataFrame
    let mut df = DataFrame::new(vec![
        Series::new("time".into(), times).into(),
        Series::new("symbol".into(), symbol).into(),
        Series::new("close".into(), close_prices.clone()).into(),
        Series::new("open".into(), open_prices).into(),
        Series::new("high".into(), high_prices).into(),
        Series::new("low".into(), low_prices).into(),
        Series::new("volume".into(), volume).into(),
    ])?;
    
    // Add technical indicators
    // SMA 20
    let sma_20 = Series::new("sma_20".into(), compute_sma(&close_prices, 20)).into();
    // SMA 50
    let sma_50 = Series::new("sma_50".into(), compute_sma(&close_prices, 50)).into();
    // EMA 20
    let ema_20 = Series::new("ema_20".into(), compute_ema(&close_prices, 20)).into();
    // RSI 14
    let rsi_14 = Series::new("rsi_14".into(), compute_rsi(&close_prices, 14)).into();
    // MACD
    let (macd, signal) = compute_macd(&close_prices);
    let macd_series = Series::new("macd".into(), macd).into();
    let macd_signal = Series::new("macd_signal".into(), signal).into();
    // Bollinger Band Middle
    let bb_middle = Series::new("bb_middle".into(), compute_sma(&close_prices, 20)).into();
    // ATR 14
    let atr_14 = Series::new("atr_14".into(), compute_atr(&close_prices, &high_prices_clone, &low_prices_clone, 14)).into();
    // Returns
    let returns = Series::new("returns".into(), compute_returns(&close_prices)).into();
    // Price Range
    let price_range = Series::new("price_range".into(), compute_price_range(&high_prices_clone, &low_prices_clone)).into();
    
    // Add technical indicators to DataFrame
    df.hstack(&[
        sma_20, sma_50, ema_20, rsi_14, macd_series, macd_signal, 
        bb_middle, atr_14, returns, price_range
    ])?;
    
    Ok(df)
}

// Simple calculation functions for technical indicators
fn compute_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    
    // Handle edge cases
    if data.is_empty() {
        return result;
    }
    
    if period == 0 || period > data.len() {
        // Invalid period, return the original data
        return data.to_vec();
    }
    
    // For the first (period-1) points, we don't have enough data for SMA
    // So we'll just use the data points themselves
    for i in 0..period-1 {
        result.push(data[i]);
    }
    
    // For the remaining points, calculate SMA
    for i in period-1..data.len() {
        let start_idx = if i >= period { i - period + 1 } else { 0 };
        let sum: f64 = data[start_idx..=i].iter().sum();
        let window_size = i - start_idx + 1;
        result.push(sum / window_size as f64);
    }
    
    result
}

fn compute_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    
    // Handle edge cases
    if data.is_empty() {
        return result;
    }
    
    if period == 0 || period > data.len() {
        // Invalid period, return the original data
        return data.to_vec();
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    
    // Start with SMA for the first period points
    let first_sma = data.iter().take(period).sum::<f64>() / period as f64;
    
    // Fill initial values before we have a full period
    for i in 0..period {
        if i < period - 1 {
            result.push(data[i]); // Use the original data for early points
        } else {
            result.push(first_sma); // Use SMA for the period-th point
        }
    }
    
    // Calculate EMA for the rest
    for i in period..data.len() {
        let ema = alpha * data[i] + (1.0 - alpha) * result[i-1];
        result.push(ema);
    }
    
    result
}

fn compute_rsi(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut gains = Vec::with_capacity(data.len());
    let mut losses = Vec::with_capacity(data.len());
    
    // Calculate daily gains and losses
    gains.push(0.0);
    losses.push(0.0);
    
    for i in 1..data.len() {
        let diff = data[i] - data[i-1];
        if diff > 0.0 {
            gains.push(diff);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-diff);
        }
    }
    
    // Placeholder for early points
    for i in 0..period {
        result.push(50.0); // Neutral RSI
    }
    
    // Calculate RSI
    for i in period..data.len() {
        let avg_gain: f64 = gains[(i-period+1)..=i].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[(i-period+1)..=i].iter().sum::<f64>() / period as f64;
        
        if avg_loss.abs() < 1e-10 {
            result.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);
        }
    }
    
    result
}

fn compute_macd(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Define default periods
    let fast_period = 12;
    let slow_period = 26;
    let signal_period = 9;
    
    // Create result vectors
    let mut macd = vec![0.0; data.len()];
    let mut signal = vec![0.0; data.len()];
    
    // Handle edge cases
    if data.len() <= slow_period {
        // Not enough data for proper calculation, return zeros
        return (macd, signal);
    }
    
    // Calculate EMAs
    let fast_ema = compute_ema(data, fast_period);
    let slow_ema = compute_ema(data, slow_period);
    
    // Calculate MACD line
    for i in 0..data.len() {
        macd[i] = fast_ema[i] - slow_ema[i];
    }
    
    // Calculate signal line (EMA of MACD)
    if data.len() > signal_period {
        signal = compute_ema(&macd, signal_period);
    }
    
    (macd, signal)
}

fn compute_atr(close: &[f64], high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(close.len());
    let mut tr = Vec::with_capacity(close.len());
    
    // Calculate True Range
    tr.push(high[0] - low[0]); // First TR is simply High - Low
    
    for i in 1..close.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i-1]).abs();
        let tr3 = (low[i] - close[i-1]).abs();
        tr.push(tr1.max(tr2).max(tr3));
    }
    
    // Placeholder for early points
    for _i in 0..period {
        result.push(tr[0]);  // Use first TR value as placeholder
    }
    
    // Calculate ATR
    let first_atr = tr.iter().take(period).sum::<f64>() / period as f64;
    result[period-1] = first_atr;
    
    for i in period..close.len() {
        let atr = ((period - 1) as f64 * result[i-1] + tr[i]) / period as f64;
        result.push(atr);
    }
    
    result
}

fn compute_returns(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    
    result.push(0.0); // First point has no previous price
    
    for i in 1..data.len() {
        let ret = (data[i] - data[i-1]) / data[i-1];
        result.push(ret);
    }
    
    result
}

fn compute_price_range(high: &[f64], low: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(high.len());
    
    for i in 0..high.len() {
        result.push((high[i] - low[i]) / low[i]);
    }
    
    result
}

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
    assert_eq!(dims[2], 10, "Hidden size should be 10 (5*2) for bidirectional");
    
    // Check that output contains valid values (not NaN)
    let data = output.to_data();
    for val in data.convert::<f32>().as_slice::<f32>().unwrap() {
        assert!(!val.is_nan(), "Output contains NaN values");
    }
}

#[test]
fn test_timeseries_gru_model() {
    // Setup LibTorch backend for testing
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();
    
    // Create a GRU model
    let model = TimeSeriesGru::<TestBackend>::new(
        10,    // input_size
        20,    // hidden_size
        1,     // output_size (for single target forecasting)
        1,     // num_layers
        true,  // bidirectional
        0.1,   // dropout
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
        assert!(*val >= 0.0 && *val <= 1.0, "Output should be clamped to [0, 1]");
    }
}

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
                let value = normalized_df.column(col).unwrap()
                    .f64().unwrap()
                    .get(row_idx).unwrap_or(0.0) as f32;
                feature_data.push(value);
            }
        }
        
        // Target: forecast_horizon timesteps of close price
        for h in 0..forecast_horizon {
            let target_idx = seq_idx + sequence_length + h;
            let target = normalized_df.column("close").unwrap()
                .f64().unwrap()
                .get(target_idx).unwrap_or(0.0) as f32;
            target_data.push(target);
        }
    }
    
    // Create tensors using the burn API
    let features = Tensor::<TestBackend, 1>::from_floats(feature_data.as_slice(), &device)
        .reshape([max_sequences, sequence_length, n_features]);
    
    let targets = Tensor::<TestBackend, 1>::from_floats(target_data.as_slice(), &device)
        .reshape([max_sequences, forecast_horizon]);
    
    // Create a GRU model
    let model = TimeSeriesGru::<TestBackend>::new(
        features.dims()[2], // input_size from features
        20,                 // hidden_size
        targets.dims()[1],  // output_size from targets
        1,                  // num_layers
        true,               // bidirectional
        0.1,                // dropout
        &device,
    );
    
    // Perform forward pass
    let output = model.forward(features);
    
    // Check dimensions match
    assert_eq!(output.dims()[0], targets.dims()[0], "Batch dimension should match");
    assert_eq!(output.dims()[1], targets.dims()[1], "Output dimension should match");
    
    // Check that output contains valid values
    let data = output.to_data();
    for val in data.convert::<f32>().as_slice::<f32>().unwrap() {
        assert!(!val.is_nan(), "Output contains NaN values");
        assert!(*val >= 0.0 && *val <= 1.0, "Output should be clamped to [0, 1]");
    }
    
    // Test loss computation
    let loss = model.mse_loss(output, targets);
    let data = loss.to_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let loss_value = slice[0] as f64;
    
    // Loss should be a valid finite number
    assert!(loss_value.is_finite(), "Loss should be a valid finite number");
} 