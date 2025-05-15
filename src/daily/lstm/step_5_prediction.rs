// External imports
use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::{backend::Backend, Tensor};
use polars::datatypes::{DataType, TimeUnit};
use polars::prelude::*;
use polars::series::Series;
use std::path::Path;

// Internal imports
use super::step_1_tensor_preparation::{
    impute_missing_values, load_daily_csv, normalize_daily_features, DAILY_FEATURES,
};
use super::step_3_lstm_model_arch::DailyLSTMModel;

/// Make a prediction using the trained LSTM model
///
/// # Arguments
///
/// * `model` - Trained LSTM model
/// * `input_data` - Input tensor of shape [batch_size, sequence_length, input_size]
///
/// # Returns
///
/// Returns a tensor with the predictions
pub fn predict_with_model<B: Backend>(
    model: &DailyLSTMModel<B>,
    input_data: Tensor<B, 3>,
) -> Tensor<B, 2> {
    model.predict(input_data)
}

/// Generate forecasts for future days
///
/// # Arguments
///
/// * `model` - Trained LSTM model
/// * `df` - DataFrame with historical data
/// * `sequence_length` - Number of time steps in each sequence
/// * `forecast_days` - Number of days to forecast
/// * `device` - Device to place tensors on
///
/// # Returns
///
/// Returns a vector with the forecasted prices
pub fn generate_forecast<B: Backend>(
    model: &DailyLSTMModel<B>,
    mut df: DataFrame,
    sequence_length: usize,
    forecast_days: usize,
    device: &B::Device,
) -> Result<Vec<f64>> {
    // Ensure we have enough data
    if df.height() < sequence_length {
        return Err(anyhow::anyhow!(
            "Not enough data for the requested sequence length"
        ));
    }

    // Prepare a copy of the original data for de-normalization
    let original_df = df.clone();

    // Normalize features
    normalize_daily_features(&mut df)?;

    // Extract feature columns
    let feature_columns = &DAILY_FEATURES;
    let num_features = feature_columns.len();

    // Initialize forecasts
    let mut forecasts = Vec::with_capacity(forecast_days);

    // Get original min and max values for denormalization
    let close_series = original_df.column("adjusted_close")?.f64()?;
    let min_price = close_series.min().unwrap_or(0.0);
    let max_price = close_series.max().unwrap_or(1.0);

    // Generate forecasts one day at a time
    for _day in 0..forecast_days {
        // Extract the most recent sequence
        let current_data = df.slice(
            (df.height() as i64) - (sequence_length as i64),
            sequence_length,
        );

        // Convert to tensor
        let mut features_vec: Vec<Vec<f64>> = Vec::new();
        for &col in feature_columns {
            if !current_data.schema().contains(col) {
                return Err(anyhow::anyhow!("Column '{}' not found in DataFrame", col));
            }

            let series = current_data.column(col)?.f64()?;
            features_vec.push(
                series
                    .into_iter()
                    .map(|v| v.unwrap_or(0.0))
                    .collect::<Vec<f64>>(),
            );
        }

        // Create input tensor
        let mut x_data = Vec::with_capacity(sequence_length * num_features);
        for t in 0..sequence_length {
            for f in 0..num_features {
                x_data.push(features_vec[f][t] as f32);
            }
        }

        // Create input tensor [1, sequence_length, num_features]
        let x_tensor = Tensor::<B, 1>::from_data(x_data.as_slice(), device).reshape([
            1,
            sequence_length,
            num_features,
        ]);

        // Make prediction
        let prediction = model.predict(x_tensor);

        // Get the predicted value and convert to f64
        let pred_value = prediction.into_scalar().to_f64();

        // Denormalize the prediction
        let denormalized_pred = pred_value * (max_price - min_price) + min_price;

        // Store the forecast
        forecasts.push(denormalized_pred);

        // Add the prediction to the DataFrame for the next iteration
        let mut new_row = df.clone().slice((df.height() as i64) - 1, 1);

        // Update the datetime (add one day)
        let datetime_series = new_row.column("datetime")?.clone();
        let datetime_int = datetime_series.cast(&DataType::Int64)?;
        let datetime_values: Vec<i64> = datetime_int
            .i64()?
            .into_iter()
            .map(|opt_val| opt_val.unwrap_or(0) + 86400000000)
            .collect();

        let new_datetime = Series::new("datetime".into(), datetime_values)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;

        new_row.replace("datetime", new_datetime)?;

        // Update the price columns with the predicted value
        let normalized_pred_array = vec![pred_value];

        // Replace values in the dataframe
        new_row.replace(
            "adjusted_close",
            Series::new("adjusted_close".into(), normalized_pred_array.clone()),
        )?;
        new_row.replace(
            "close",
            Series::new("close".into(), normalized_pred_array.clone()),
        )?;
        new_row.replace(
            "high",
            Series::new("high".into(), normalized_pred_array.clone()),
        )?;
        new_row.replace(
            "low",
            Series::new("low".into(), normalized_pred_array.clone()),
        )?;
        new_row.replace(
            "open",
            Series::new("open".into(), normalized_pred_array.clone()),
        )?;
        new_row.replace("returns", Series::new("returns".into(), vec![0.0]))?;
        new_row.replace("price_range", Series::new("price_range".into(), vec![0.0]))?;

        // Append to the DataFrame
        df = df.vstack(&new_row)?;
    }

    Ok(forecasts)
}

/// Predict the next day's price using the trained model
///
/// # Arguments
///
/// * `model` - Trained LSTM model
/// * `csv_path` - Path to the CSV file with historical data
/// * `sequence_length` - Number of time steps in each sequence
/// * `device` - Device to place tensors on
///
/// # Returns
///
/// Returns the predicted price for the next day
pub fn predict_next_day<B: Backend>(
    model: &DailyLSTMModel<B>,
    csv_path: &str,
    sequence_length: usize,
    device: &B::Device,
) -> Result<f64> {
    // Load and preprocess data
    let mut df = load_daily_csv(csv_path)?;

    // Handle missing values
    impute_missing_values(&mut df, "forward")?;

    // Generate a forecast for 1 day
    let forecasts = generate_forecast(model, df, sequence_length, 1, device)?;

    // Return the forecast
    Ok(forecasts[0])
}

/// Evaluate model performance on test data
///
/// # Arguments
///
/// * `model` - Trained LSTM model
/// * `test_data` - Input tensor of shape [batch_size, sequence_length, input_size]
/// * `test_targets` - Target tensor of shape [batch_size, 1]
///
/// # Returns
///
/// Returns a tuple with the MSE and MAE
pub fn evaluate_model<B: Backend>(
    model: &DailyLSTMModel<B>,
    test_data: Tensor<B, 3>,
    test_targets: Tensor<B, 2>,
) -> (f64, f64) {
    // Make predictions
    let predictions = model.predict(test_data);

    // Calculate MSE
    let two = Tensor::<B, 2>::ones_like(&predictions);
    let squared_diff = (predictions.clone() - test_targets.clone()).powf(two);
    let mse = squared_diff.mean().into_scalar().to_f64();

    // Calculate MAE
    let abs_diff = (predictions - test_targets).abs();
    let mae = abs_diff.mean().into_scalar().to_f64();

    (mse, mae)
}

/// Make forecasts using a trained LSTM model
///
/// # Arguments
///
/// * `model` - Trained LSTM model
/// * `csv_path` - Path to the CSV file with historical data
/// * `sequence_length` - Length of input sequences
/// * `forecast_days` - Number of days to forecast
/// * `device` - Device to run prediction on
///
/// # Returns
///
/// Returns a vector of forecasted values
pub fn forecast_daily_lstm<B: Backend>(
    model: &DailyLSTMModel<B>,
    csv_path: &str,
    sequence_length: usize,
    forecast_days: usize,
    device: &B::Device,
) -> Result<Vec<f64>> {
    // Load and preprocess data
    let mut df = load_daily_csv(csv_path)?;

    // Handle missing values
    impute_missing_values(&mut df, "forward")?;

    // Generate forecasts
    generate_forecast(model, df, sequence_length, forecast_days, device)
}
