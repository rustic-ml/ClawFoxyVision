// External imports
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use polars::prelude::*;

// Internal imports
use super::step_3_gru_model_arch::TimeSeriesGru;
use crate::minute::lstm::step_1_tensor_preparation::{
    dataframe_to_tensors, normalize_features
};
use crate::constants::{TECHNICAL_INDICATORS, EXTENDED_INDICATORS};

/// Single-step prediction from the GRU model
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
        None
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

/// Generate multiple future predictions using recursive forecasting
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
    normalize_features(&mut prediction_df, &["close", "open", "high", "low"], use_extended_features, false)?;
    
    // Container for predictions
    let mut predictions = Vec::with_capacity(horizon);
    
    // Make recursive predictions
    for _ in 0..horizon {
        // Make a single prediction
        let next_value = predict_next_step(model, prediction_df.clone(), device, use_extended_features)?;
        predictions.push(next_value);
        
        // Add the predicted value as a new row
        // This is simplified - in a real implementation, you'd create a full row with all required columns
        let mut next_row_values = Vec::new();
        
        // Choose the appropriate set of indicators
        let feature_columns = if use_extended_features {
            &EXTENDED_INDICATORS[..]
        } else {
            &TECHNICAL_INDICATORS[..]
        };
        
        // For a simple implementation, copy the last row and replace the 'close' value
        for col_name in feature_columns {
            let col = prediction_df.column(col_name)?;
            let height = col.len();
            
            if *col_name == "close" {
                next_row_values.push(Series::new((*col_name).into(), vec![next_value]).into());
            } else if height > 0 {
                // Use the last value for other columns - this is a simplification
                let value = col.get(height - 1);
                
                if let Ok(series) = df.column(col_name) {
                    if let Ok(f64_series) = series.f64() {
                        // Try to create a new series of the same type
                        let last_val = f64_series.get(height - 1).unwrap_or(0.0);
                        next_row_values.push(Series::new((*col_name).into(), vec![last_val]).into());
                    } else {
                        // Fallback for non-float columns
                        next_row_values.push(Series::new((*col_name).into(), vec![0.0]).into());
                    }
                } else {
                    // Column not found - use default
                    next_row_values.push(Series::new((*col_name).into(), vec![0.0]).into());
                }
            }
        }
        
        // Create and append the new row
        if !next_row_values.is_empty() {
            let next_row = DataFrame::new(next_row_values)?;
            prediction_df = prediction_df.vstack(&next_row)?;
        }
    }
    
    Ok(predictions)
}

/// Compare GRU model prediction with LSTM
pub fn compare_with_lstm<B: Backend>(
    gru_model: &TimeSeriesGru<B>,
    lstm_model: &crate::minute::lstm::step_3_lstm_model_arch::TimeSeriesLstm<B>,
    df: DataFrame,
    horizon: usize,
    device: &B::Device,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // Normalize data for prediction
    let mut normalized_df = df.clone();
    normalize_features(&mut normalized_df, &["close", "open", "high", "low"], false, false)?;
    
    // Make GRU predictions
    let gru_predictions = predict_multiple_steps(gru_model, normalized_df.clone(), horizon, device, false)?;
    
    // Make LSTM predictions
    let lstm_predictions = crate::minute::lstm::step_5_prediction::generate_forecast_with_correction(
        lstm_model, 
        normalized_df, 
        horizon, 
        device, 
        false,
        0.5 // default error correction alpha
    )?;
    
    Ok((gru_predictions, lstm_predictions))
} 