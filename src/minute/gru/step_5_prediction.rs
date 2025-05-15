// External imports
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use polars::prelude::*;

// Internal imports
use super::step_3_gru_model_arch::TimeSeriesGru;
use crate::constants::{EXTENDED_INDICATORS, TECHNICAL_INDICATORS};
use crate::minute::lstm::step_1_tensor_preparation::{dataframe_to_tensors, normalize_features};

/// # Make Single-Step Prediction
///
/// Generates a single-step ahead prediction from a trained GRU model.
/// This function takes the most recent data window and predicts the next value.
///
/// ## Process
///
/// 1. Verifies the input DataFrame contains all required features
/// 2. Creates tensors appropriate for the GRU model
/// 3. Uses the last sequence from the data for prediction
/// 4. Returns the predicted value as a normalized value between 0 and 1
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `df` - DataFrame containing time series features
/// * `device` - Device to run prediction on
/// * `use_extended_features` - Whether to use the extended set of features
///
/// # Returns
///
/// The single-step prediction as a floating-point value (normalized)
pub fn predict_next_step<B: Backend>(
    model: &TimeSeriesGru<B>,
    df: DataFrame,
    device: &B::Device,
    use_extended_features: bool,
) -> Result<f64> {
    // Choose which feature set to use
    let feature_columns = if use_extended_features {
        &EXTENDED_INDICATORS[..]
    } else {
        &TECHNICAL_INDICATORS[..]
    };

    // Validate required columns
    if !df.is_empty() {
        for col in feature_columns {
            if !df.schema().contains(col) {
                return Err(anyhow::anyhow!("Missing required column: {}", col));
            }
        }
    }

    // Clone the DataFrame to avoid modifications affecting the original
    let prediction_df = df.clone();

    // Build sequences tensor with horizon 1
    let (features, _) = dataframe_to_tensors::<B>(
        &prediction_df,
        crate::constants::SEQUENCE_LENGTH,
        1,
        device,
        use_extended_features,
        None,
    )
    .context("Tensor creation failed for prediction")?;

    // Extract the last sequence
    let seq_count = features.dims()[0];
    let seq = features.clone().narrow(0, seq_count - 1, 1);

    // Forward pass for single-step prediction
    let pred_tensor = model.forward(seq);

    // Extract scalar prediction
    let data = pred_tensor.to_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let value = slice[0];
    Ok(value as f64)
}

/// # Generate Multi-Step Forecasts
///
/// Generates predictions for multiple steps into the future using a recursive approach.
/// Each prediction is fed back into the model to generate the next prediction.
///
/// ## Recursive Forecasting
///
/// This method uses the model's own predictions as inputs for future predictions:
/// 1. Make prediction for t+1
/// 2. Add prediction to input data
/// 3. Use updated data to predict t+2
/// 4. Repeat for the desired forecast horizon
///
/// # Arguments
///
/// * `model` - Trained GRU model
/// * `df` - DataFrame containing time series features
/// * `horizon` - Number of future steps to predict
/// * `device` - Device to run prediction on
/// * `use_extended_features` - Whether to use the extended set of features
///
/// # Returns
///
/// A vector of predictions for each requested future time step
pub fn predict_multiple_steps<B: Backend>(
    model: &TimeSeriesGru<B>,
    df: DataFrame,
    horizon: usize,
    device: &B::Device,
    use_extended_features: bool,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Ok(Vec::new());
    }

    // Clone the DataFrame to avoid modifications affecting the original
    let mut prediction_df = df.clone();

    // Normalize the input data for prediction
    normalize_features(
        &mut prediction_df,
        &["close", "open", "high", "low"],
        use_extended_features,
        false,
    )?;

    // Container for predictions
    let mut predictions = Vec::with_capacity(horizon);

    // Make recursive predictions
    for _ in 0..horizon {
        // Make a single prediction
        let next_value =
            predict_next_step(model, prediction_df.clone(), device, use_extended_features)?;
        predictions.push(next_value);

        // Get actual column names from the prediction DataFrame
        let column_names: Vec<String> = prediction_df
            .get_column_names()
            .iter()
            .map(|&s| s.to_string())
            .collect();

        // Add the predicted value as a new row
        let mut next_row_values = Vec::new();

        // Create series for each column in the DataFrame
        for col_name in &column_names {
            let col = prediction_df.column(col_name)?;
            let height = col.len();

            if col_name == "close" {
                next_row_values.push(Series::new(col_name.into(), vec![next_value]).into());
            } else if height > 0 {
                // Use the last value for other columns
                if let Ok(f64_series) = col.f64() {
                    let last_val = f64_series.get(height - 1).unwrap_or(0.0);
                    next_row_values.push(Series::new(col_name.into(), vec![last_val]).into());
                } else {
                    // Fallback for non-float columns - use the actual last value of whatever type it is
                    let last_val = col.get(height - 1);
                    match last_val {
                        Ok(AnyValue::Int32(v)) => {
                            next_row_values.push(Series::new(col_name.into(), vec![v]).into())
                        }
                        Ok(AnyValue::Int64(v)) => {
                            next_row_values.push(Series::new(col_name.into(), vec![v]).into())
                        }
                        Ok(AnyValue::Float32(v)) => {
                            next_row_values.push(Series::new(col_name.into(), vec![v]).into())
                        }
                        Ok(AnyValue::Float64(v)) => {
                            next_row_values.push(Series::new(col_name.into(), vec![v]).into())
                        }
                        Ok(AnyValue::String(v)) => {
                            next_row_values.push(Series::new(col_name.into(), vec![v]).into())
                        }
                        _ => next_row_values.push(Series::new(col_name.into(), vec![0.0]).into()),
                    };
                }
            } else {
                // Column exists but is empty - use default
                next_row_values.push(Series::new(col_name.into(), vec![0.0]).into());
            }
        }

        // Create and append the new row
        if !next_row_values.is_empty() {
            let next_row = DataFrame::new(next_row_values)?;

            // Double-check that the new row has exactly the same schema as the prediction DataFrame
            if next_row.width() != prediction_df.width() {
                return Err(anyhow::anyhow!(
                    "Error: lengths don't match: unable to append to a DataFrame of width {} with a DataFrame of width {}",
                    prediction_df.width(), next_row.width()
                ));
            }

            prediction_df = prediction_df.vstack(&next_row)?;
        }
    }

    Ok(predictions)
}

/// # Compare GRU and LSTM Models
///
/// Compares predictions from GRU and LSTM models on the same dataset.
/// This is useful for evaluating the relative performance of the two model types.
///
/// # Arguments
///
/// * `gru_model` - Trained GRU model
/// * `lstm_model` - Trained LSTM model
/// * `df` - DataFrame containing time series features
/// * `horizon` - Number of future steps to predict
/// * `device` - Device to run prediction on
///
/// # Returns
///
/// A tuple of (GRU predictions, LSTM predictions) with the same horizon
pub fn compare_with_lstm<B: Backend>(
    gru_model: &TimeSeriesGru<B>,
    lstm_model: &crate::minute::lstm::step_3_lstm_model_arch::TimeSeriesLstm<B>,
    df: DataFrame,
    horizon: usize,
    device: &B::Device,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // Normalize data for prediction
    let mut normalized_df = df.clone();
    normalize_features(
        &mut normalized_df,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    // Make GRU predictions
    let gru_predictions =
        predict_multiple_steps(gru_model, normalized_df.clone(), horizon, device, false)?;

    // Make LSTM predictions
    let lstm_predictions =
        crate::minute::lstm::step_5_prediction::generate_forecast_with_correction(
            lstm_model,
            normalized_df,
            horizon,
            device,
            false,
            0.5, // default error correction alpha
        )?;

    Ok((gru_predictions, lstm_predictions))
}
