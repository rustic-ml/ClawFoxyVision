// External imports
use anyhow::Result;
use burn::tensor::backend::Backend;
use polars::prelude::*;

// Internal imports
use super::step_1_tensor_preparation;
use super::step_3_cnn_lstm_model_arch::TimeSeriesCnnLstm;
use crate::constants;

/// Make predictions for the next few time steps using the CNN-LSTM model
/// The forecast horizon is shorter to reduce memory usage
pub fn forecast<B: Backend>(
    model: &TimeSeriesCnnLstm<B>,
    df: &DataFrame,
    device: &B::Device,
    forecast_horizon: usize,
) -> Result<Vec<f64>> {
    // Prepare input sequence from the last sequence_length rows of the dataframe
    let sequence_length = constants::SEQUENCE_LENGTH;
    let input_size = model.input_size();
    
    if df.height() < sequence_length {
        return Err(anyhow::anyhow!("Not enough data points for forecasting. Need at least {} rows.", sequence_length));
    }
    
    // Use a smaller slice of data for reduced memory footprint
    let df_slice = df.slice((df.height() - sequence_length) as i64, sequence_length);
    
    println!("Preparing prediction tensors with reduced forecast horizon of {} steps", forecast_horizon);
    
    // Convert to tensor
    let (features, _) = step_1_tensor_preparation::dataframe_to_tensors::<B>(
        &df_slice,
        sequence_length,
        forecast_horizon,  // Using the reduced forecast horizon
        device,
        true, // indicate we're making a prediction
        None,
    )?;
    
    // Ensure the input has the correct shape: [1, sequence_length, input_size]
    let input_tensor = if features.dims()[0] != 1 {
        features.unsqueeze::<3>().reshape([1, sequence_length, input_size])
    } else {
        features
    };
    
    // Forward pass
    let predictions = model.forward(input_tensor);
    
    // Convert predictions to Vec<f64>
    let predictions_data = predictions.to_data().convert::<f32>();
    let predictions_slice = predictions_data.as_slice::<f32>().unwrap();
    
    // Convert predictions to f64 values
    let result = predictions_slice.iter()
        .map(|&x| x as f64)
        .collect::<Vec<f64>>();
    
    println!("Successfully generated {} prediction steps", result.len());
    
    Ok(result)
}

/// Generate OHLCV predictions using the CNN-LSTM model with reduced memory usage
/// This function produces predictions for a specific horizon and can denormalize the values
pub fn generate_ohlcv_predictions<B: Backend>(
    model: &TimeSeriesCnnLstm<B>,
    df: &DataFrame,
    device: &B::Device,
    forecast_horizon: usize,
    denormalize: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Get normalized predictions
    let predictions = forecast(model, df, device, forecast_horizon)?;
    
    if !denormalize {
        // If no denormalization requested, just repeat the predictions for OHLCV
        // (This is a simplified approach - in reality, we might want to predict each separately)
        return Ok((
            predictions.clone(),  // Open
            predictions.clone(),  // High
            predictions.clone(),  // Low
            predictions.clone(),  // Close
            predictions.clone(),  // Volume
        ));
    }
    
    // For now, we'll just return the normalized predictions
    // In a real implementation, you would need to denormalize these values
    // based on the scaling factors used during training
    println!("Warning: Denormalization is not fully implemented yet. Returning normalized predictions.");
    
    Ok((
        predictions.clone(),  // Open
        predictions.clone(),  // High
        predictions.clone(),  // Low
        predictions.clone(),  // Close
        predictions.clone(),  // Volume
    ))
} 