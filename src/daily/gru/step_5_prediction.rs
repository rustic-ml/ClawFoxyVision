// External imports
use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Shape;
use burn::tensor::{backend::Backend, Tensor};
use num_traits::cast::NumCast;
use num_traits::Num;
use polars::datatypes::{DataType, TimeUnit};
use polars::prelude::*;
use polars::series::Series;
use std::ops::Add;
use std::path::Path;

// Internal imports
use super::step_1_tensor_preparation::{
    impute_missing_values, load_daily_csv, normalize_daily_features, DAILY_FEATURES,
};
use super::step_3_gru_model_arch::DailyGRUModel;

/// Make a prediction using the trained GRU model
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `input_data` - Input tensor of shape [batch_size, sequence_length, input_size]
///
/// # Returns
///
/// Returns a tensor with the predictions
pub fn predict_with_model<B: Backend>(
    model: &DailyGRUModel<B>,
    input_data: Tensor<B, 3>,
) -> Tensor<B, 2> {
    model.predict(input_data)
}

/// Generate predictions for future days
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `df` - DataFrame with historical data
/// * `sequence_length` - Number of time steps in each sequence
/// * `forecast_days` - Number of days to forecast
/// * `device` - Device to place tensors on
///
/// # Returns
///
/// Returns a vector with the forecasted prices
pub fn generate_forecast<B: Backend>(
    model: &DailyGRUModel<B>,
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
        let current_data = df.slice((df.height() - sequence_length) as i64, sequence_length);

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
                x_data.push(features_vec[f][t]);
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
        let pred_value: f64 = prediction.into_scalar().to_f64();

        // Denormalize the prediction
        let denormalized_pred = pred_value * (max_price - min_price) + min_price;

        // Store the forecast
        forecasts.push(denormalized_pred);

        // Add the prediction to the DataFrame for the next iteration
        let mut new_row = df.clone().slice((df.height() - 1) as i64, 1);

        // Update the datetime (add one day)
        // Check if we have a datetime or time column
        let datetime_column_name = if new_row.schema().contains("datetime") {
            "datetime"
        } else if new_row.schema().contains("time") {
            "time"
        } else {
            return Err(anyhow::anyhow!("Neither 'datetime' nor 'time' column found in DataFrame"));
        };

        // Get the original data type of the date/time column
        let date_col = new_row.column(datetime_column_name)?;
        let date_dtype = date_col.dtype();

        // Determine how to handle the date based on its type
        match date_dtype {
            // If it's already a datetime, use proper datetime arithmetic
            DataType::Datetime(time_unit, tz) => {
                let datetime_series = new_row.column(datetime_column_name)?.clone();
                let datetime_int = datetime_series.cast(&DataType::Int64)?;
                let datetime_values: Vec<i64> = datetime_int
                    .i64()?
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(0) + 86400000000) // Add one day in microseconds
                    .collect();

                let new_datetime = Series::new(datetime_column_name.into(), datetime_values)
                    .cast(&DataType::Datetime(*time_unit, tz.clone()))?;

                new_row.replace(datetime_column_name, new_datetime)?;
            },
            // If it's a string, increment the date as a string
            DataType::String => {
                // Simple string date addition - would need more sophisticated handling in real application
                // This is just an example using a placeholder
                let new_date = Series::new(datetime_column_name.into(), &["next_day"]);
                new_row.replace(datetime_column_name, new_date)?;
            },
            // For other types, convert to string as fallback
            _ => {
                // Just use a placeholder string value
                let new_date = Series::new(datetime_column_name.into(), &["next_day"]);
                new_row.replace(datetime_column_name, new_date)?;
            }
        }

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
        new_row.replace("open", Series::new("open".into(), normalized_pred_array))?;

        // Update other features
        new_row.replace("returns", Series::new("returns".into(), vec![0.0f64]))?;
        new_row.replace(
            "price_range",
            Series::new("price_range".into(), vec![0.0f64]),
        )?;

        // Append to the DataFrame
        df = df.vstack(&new_row)?;
    }

    Ok(forecasts)
}

/// Predict the next day's price using the trained model
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `csv_path` - Path to the CSV file with historical data
/// * `sequence_length` - Number of time steps in each sequence
/// * `device` - Device to place tensors on
///
/// # Returns
///
/// Returns the predicted price for the next day
pub fn predict_next_day<B: Backend>(
    model: &DailyGRUModel<B>,
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
/// * `model` - Trained GRU model
/// * `test_df` - Input DataFrame
/// * `sequence_length` - Number of time steps in each sequence
/// * `device` - Device to place tensors on
///
/// # Returns
///
/// Returns a tuple with the MSE and MAE
pub fn evaluate_model<B: Backend>(
    model: &DailyGRUModel<B>,
    test_df: &DataFrame,
    sequence_length: usize,
    device: &B::Device,
) -> Result<(f64, f64)> {
    // Prepare test data
    let mut test_df = test_df.clone();

    // Normalize features
    normalize_daily_features(&mut test_df)?;

    // Number of samples
    let num_samples = test_df.height() - sequence_length;

    // Prepare tensors to store predictions and actual values
    let mut predictions_vec = Vec::with_capacity(num_samples);
    let mut targets_vec = Vec::with_capacity(num_samples);

    // Get reference to the target column (adjusted_close)
    let target_col = test_df.column("adjusted_close")?.f64()?;

    // Iterate through samples
    for i in 0..num_samples {
        // Extract sequence
        let sequence = test_df.slice(i as i64, sequence_length);

        // Convert to tensor (similar to forecast function)
        let open_col = sequence.column("open")?.f64()?;
        let high_col = sequence.column("high")?.f64()?;
        let low_col = sequence.column("low")?.f64()?;
        let close_col = sequence.column("close")?.f64()?;
        let volume_col = sequence.column("volume")?.f64()?;
        let adj_close_col = sequence.column("adjusted_close")?.f64()?;

        // Stack all features
        let num_features = 6; // OHLCV + adjusted_close
        let mut x_data = Vec::with_capacity(sequence_length * num_features);

        for j in 0..sequence_length {
            // Add all features for this time step
            x_data.push(open_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(high_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(low_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(close_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(volume_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(adj_close_col.get(j).unwrap_or(0.0) as f32);
        }

        // Create input tensor [1, sequence_length, num_features]
        let x_tensor = Tensor::<B, 1>::from_data(x_data.as_slice(), device).reshape([
            1,
            sequence_length,
            num_features,
        ]);

        // Get target (next day's adjusted close)
        let target = target_col.get(i + sequence_length).unwrap_or(0.0);
        targets_vec.push(target as f32);

        // Make prediction
        let pred = model.predict(x_tensor);
        let pred_value = pred.into_scalar().to_f32();
        predictions_vec.push(pred_value);
    }

    // Convert to tensors for easier calculation
    let predictions = Tensor::<B, 1>::from_data(predictions_vec.as_slice(), device);
    let test_targets = Tensor::<B, 1>::from_data(targets_vec.as_slice(), device);

    // Calculate MSE
    let two = Tensor::<B, 1>::ones_like(&predictions);
    let squared_diff = (predictions.clone() - test_targets.clone()).powf(two);
    let mse = squared_diff.mean().into_scalar().to_f64();

    // Calculate MAE
    let abs_diff = (predictions - test_targets).abs();
    let mae = abs_diff.mean().into_scalar().to_f64();

    Ok((mse, mae))
}

/// Make forecasts using a trained GRU model
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `csv_path` - Path to the CSV file with historical data
/// * `sequence_length` - Length of input sequences
/// * `forecast_days` - Number of days to forecast
/// * `device` - Device to run prediction on
///
/// # Returns
///
/// Returns a vector of forecasted values
pub fn forecast_daily_gru<B: Backend>(
    model: &DailyGRUModel<B>,
    csv_path: &str,
    sequence_length: usize,
    forecast_days: usize,
    device: &B::Device,
) -> Result<Vec<f64>> {
    // Load and preprocess data
    let mut df = load_daily_csv(csv_path)?;

    // Handle missing values
    impute_missing_values(&mut df, "forward")?;

    // Get min/max for denormalization later
    let price_col = df.column("adjusted_close")?;
    let price_arr = price_col.f64()?;
    let min_price = price_arr.min().unwrap_or(0.0);
    let max_price = price_arr.max().unwrap_or(1.0);

    // Normalize features
    normalize_daily_features(&mut df)?;

    // Store forecasts
    let mut forecasts = Vec::with_capacity(forecast_days);

    // Make forecasts iteratively
    for _day in 0..forecast_days {
        // Extract the most recent sequence
        let current_data = df.slice(
            (df.height() as i64) - (sequence_length as i64),
            sequence_length,
        );

        // Convert to tensor directly without an intermediate vector
        // Collect all feature columns
        let open_col = current_data.column("open")?.f64()?;
        let high_col = current_data.column("high")?.f64()?;
        let low_col = current_data.column("low")?.f64()?;
        let close_col = current_data.column("close")?.f64()?;
        let volume_col = current_data.column("volume")?.f64()?;
        let adj_close_col = current_data.column("adjusted_close")?.f64()?;

        // Stack all features
        let num_features = 6; // OHLCV + adjusted_close
        let mut x_data = Vec::with_capacity(sequence_length * num_features);

        for i in 0..sequence_length {
            // Add all features for this time step
            x_data.push(open_col.get(i).unwrap_or(0.0) as f32);
            x_data.push(high_col.get(i).unwrap_or(0.0) as f32);
            x_data.push(low_col.get(i).unwrap_or(0.0) as f32);
            x_data.push(close_col.get(i).unwrap_or(0.0) as f32);
            x_data.push(volume_col.get(i).unwrap_or(0.0) as f32);
            x_data.push(adj_close_col.get(i).unwrap_or(0.0) as f32);
        }

        // Create input tensor [1, sequence_length, num_features]
        let x_tensor = Tensor::<B, 1>::from_data(x_data.as_slice(), device).reshape([
            1,
            sequence_length,
            num_features,
        ]);

        // Make prediction
        let pred = model.predict(x_tensor);
        let pred_value = pred.into_scalar().to_f64();

        // Denormalize the prediction
        let denormalized_pred = pred_value * (max_price - min_price) + min_price;

        // Store the forecast
        forecasts.push(denormalized_pred);

        // Add the prediction to the DataFrame for the next iteration
        let mut new_row = df.clone().slice((df.height() as i64) - 1, 1);

        // Update the datetime (add one day)
        // Check if we have a datetime or time column
        let datetime_column_name = if new_row.schema().contains("datetime") {
            "datetime"
        } else if new_row.schema().contains("time") {
            "time"
        } else {
            return Err(anyhow::anyhow!("Neither 'datetime' nor 'time' column found in DataFrame"));
        };

        // Get the original data type of the date/time column
        let date_col = new_row.column(datetime_column_name)?;
        let date_dtype = date_col.dtype();

        // Determine how to handle the date based on its type
        match date_dtype {
            // If it's already a datetime, use proper datetime arithmetic
            DataType::Datetime(time_unit, tz) => {
                let datetime_series = new_row.column(datetime_column_name)?.clone();
                let datetime_int = datetime_series.cast(&DataType::Int64)?;
                let datetime_values: Vec<i64> = datetime_int
                    .i64()?
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(0) + 86400000000) // Add one day in microseconds
                    .collect();

                let new_datetime = Series::new(datetime_column_name.into(), datetime_values)
                    .cast(&DataType::Datetime(*time_unit, tz.clone()))?;

                new_row.replace(datetime_column_name, new_datetime)?;
            },
            // If it's a string, increment the date as a string
            DataType::String => {
                // Simple string date addition - would need more sophisticated handling in real application
                // This is just an example using a placeholder
                let new_date = Series::new(datetime_column_name.into(), &["next_day"]);
                new_row.replace(datetime_column_name, new_date)?;
            },
            // For other types, convert to string as fallback
            _ => {
                // Just use a placeholder string value
                let new_date = Series::new(datetime_column_name.into(), &["next_day"]);
                new_row.replace(datetime_column_name, new_date)?;
            }
        }

        // Update the price columns with the predicted value
        // Convert to f64 array for Series creation
        let normalized_pred_array = vec![denormalized_pred];

        // Replace values in the dataframe
        new_row.replace(
            "adjusted_close",
            Series::new("adjusted_close".into(), &normalized_pred_array),
        )?;
        new_row.replace("close", Series::new("close".into(), &normalized_pred_array))?;

        // For simplicity, set other prices to the same value
        // In a real application, you might want to estimate high, low, open, etc.
        new_row.replace("high", Series::new("high".into(), &normalized_pred_array))?;
        new_row.replace("low", Series::new("low".into(), &normalized_pred_array))?;
        new_row.replace("open", Series::new("open".into(), &normalized_pred_array))?;

        // Update other features (returns, etc.)
        // In a real application, this would be more sophisticated
        new_row.replace("returns", Series::new("returns".into(), &[0.0f64]))?;
        new_row.replace("price_range", Series::new("price_range".into(), &[0.0f64]))?;

        // Append to the DataFrame
        df = df.vstack(&new_row)?;
    }

    Ok(forecasts)
}

/// Evaluate a trained GRU model on test data
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `test_data` - Test data as DataFrame
/// * `sequence_length` - Length of input sequences
/// * `device` - Device to run evaluation on
///
/// # Returns
///
/// Returns (MSE, MAE) metrics
pub fn evaluate_daily_gru<B: Backend>(
    model: &DailyGRUModel<B>,
    test_data: &DataFrame,
    sequence_length: usize,
    device: &B::Device,
) -> Result<(f64, f64)> {
    // Prepare test data
    let mut test_df = test_data.clone();

    // Normalize features
    normalize_daily_features(&mut test_df)?;

    // Number of samples
    let num_samples = test_df.height() - sequence_length;

    // Prepare tensors to store predictions and actual values
    let mut predictions_vec = Vec::with_capacity(num_samples);
    let mut targets_vec = Vec::with_capacity(num_samples);

    // Get reference to the target column (adjusted_close)
    let target_col = test_df.column("adjusted_close")?.f64()?;

    // Iterate through samples
    for i in 0..num_samples {
        // Extract sequence
        let sequence = test_df.slice(i as i64, sequence_length);

        // Convert to tensor (similar to forecast function)
        let open_col = sequence.column("open")?.f64()?;
        let high_col = sequence.column("high")?.f64()?;
        let low_col = sequence.column("low")?.f64()?;
        let close_col = sequence.column("close")?.f64()?;
        let volume_col = sequence.column("volume")?.f64()?;
        let adj_close_col = sequence.column("adjusted_close")?.f64()?;

        // Stack all features
        let num_features = 6; // OHLCV + adjusted_close
        let mut x_data = Vec::with_capacity(sequence_length * num_features);

        for j in 0..sequence_length {
            // Add all features for this time step
            x_data.push(open_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(high_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(low_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(close_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(volume_col.get(j).unwrap_or(0.0) as f32);
            x_data.push(adj_close_col.get(j).unwrap_or(0.0) as f32);
        }

        // Create input tensor [1, sequence_length, num_features]
        let x_tensor = Tensor::<B, 1>::from_data(x_data.as_slice(), device).reshape([
            1,
            sequence_length,
            num_features,
        ]);

        // Get target (next day's adjusted close)
        let target = target_col.get(i + sequence_length).unwrap_or(0.0);
        targets_vec.push(target as f32);

        // Make prediction
        let pred = model.predict(x_tensor);
        let pred_value = pred.into_scalar().to_f32();
        predictions_vec.push(pred_value);
    }

    // Convert to tensors for easier calculation
    let predictions = Tensor::<B, 1>::from_data(predictions_vec.as_slice(), device);
    let test_targets = Tensor::<B, 1>::from_data(targets_vec.as_slice(), device);

    // Calculate MSE
    let two = Tensor::<B, 1>::ones_like(&predictions);
    let squared_diff = (predictions.clone() - test_targets.clone()).powf(two);
    let mse = squared_diff.mean().into_scalar().to_f64();

    // Calculate MAE
    let abs_diff = (predictions - test_targets).abs();
    let mae = abs_diff.mean().into_scalar().to_f64();

    Ok((mse, mae))
}
