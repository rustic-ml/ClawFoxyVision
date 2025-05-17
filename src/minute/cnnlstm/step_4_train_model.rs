// External imports
use anyhow::Result;
use burn::optim::AdamConfig;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::tensor::{backend::Backend, Tensor};
use polars::prelude::*;

// Internal imports
use super::step_1_tensor_preparation;
use super::step_3_cnn_lstm_model_arch::TimeSeriesCnnLstm;
use crate::constants;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type BurnBackend = Autodiff<NdArray<f32>>;

/// Configuration for training the CNN-LSTM model
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub test_split: f64,
    pub patience: usize,
    pub min_delta: f64,
    pub dropout: f64,
    pub use_huber_loss: bool,
    pub display_metrics: bool,
    pub display_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            test_split: 0.2,
            patience: 3,          // Early stopping patience
            min_delta: 0.001,     // Minimum improvement threshold
            dropout: 0.3,         // Default higher dropout
            use_huber_loss: true, // Use Huber loss by default
            display_metrics: true,
            display_interval: 1,
        }
    }
}

/// Helper function to safely extract a scalar value from a tensor of any dimension
fn tensor_to_f64<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Result<f64> {
    // Convert tensor to data
    let tensor_data = tensor.to_data();
    
    // Access the first element
    let tensor_slice = tensor_data.as_slice::<f32>()
        .map_err(|_| anyhow::anyhow!("Failed to convert tensor to scalar"))?;
    
    // Make sure we have some data
    if tensor_slice.is_empty() {
        return Err(anyhow::anyhow!("Empty tensor data"));
    }
    
    // Return the first element as f64
    Ok(tensor_slice[0] as f64)
}

/// Train the CNN-LSTM model
pub fn train_model(
    df: DataFrame,
    config: TrainingConfig,
    device: &<BurnBackend as burn::tensor::backend::Backend>::Device,
    ticker: &str,
    _model_type: &str,
    forecast_horizon: usize,
) -> Result<(TimeSeriesCnnLstm<BurnBackend>, Vec<f64>)> {
    println!("Starting CNN-LSTM model training...");

    // Prepare data by splitting into training and validation sets
    // We can reuse the tensor preparation from the LSTM module
    let (features, targets) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
        &df,
        crate::constants::SEQUENCE_LENGTH,
        forecast_horizon,
        device,
        false,
        None,
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!(
        "Data prepared: features shape: {:?}, targets shape: {:?}",
        features.dims(),
        targets.dims()
    );

    // Calculate dataset splits
    let num_samples = features.dims()[0];
    let test_size = (num_samples as f64 * config.test_split).round() as usize;
    let train_size = num_samples - test_size;

    // Get input and output sizes before moving features and targets
    let input_size = features.dims()[2];
    let output_size = forecast_horizon;

    // Split data into train and test sets - using clone to prevent move errors
    let train_features = features.clone().narrow(0, 0, train_size);
    let train_targets = targets.clone().narrow(0, 0, train_size);
    let test_features = features.clone().narrow(0, train_size, test_size);
    let test_targets = targets.clone().narrow(0, train_size, test_size);

    println!(
        "Data split: train samples: {}, test samples: {}",
        train_size, test_size
    );

    // Helper for batching
    fn get_batches<B: Backend, const D: usize>(
        data: &Tensor<B, D>,
        batch_size: usize,
    ) -> Vec<Tensor<B, D>> {
        let num_samples = data.dims()[0];
        let mut batches = Vec::new();
        let mut start = 0;
        while start < num_samples {
            let end = usize::min(start + batch_size, num_samples);
            let batch = data.clone().narrow(0, start, end - start);
            batches.push(batch);
            start = end;
        }
        batches
    }

    // Initialize the model with reduced complexity
    let hidden_size = 32; // Reduced from 64 to use less memory
    let num_layers = 1;   // Reduced from 2 to simplify model
    let bidirectional = false; // Turned off bidirectional to reduce parameters
    let dropout = config.dropout;

    println!("Creating lightweight CNN-LSTM model: hidden_size={}, layers={}, bidirectional={}", 
             hidden_size, num_layers, bidirectional);

    let mut model = TimeSeriesCnnLstm::<BurnBackend>::new(
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
        device,
    );

    // Setup for early stopping and learning rate scheduling
    let mut best_model = model.clone();
    let mut best_val_rmse = f64::INFINITY;
    let mut epochs_no_improve = 0;
    
    // Initialize optimizer
    let mut optimizer = AdamConfig::new().init();

    let mut loss_history = Vec::new();
    let _model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);
    
    println!("Training CNN-LSTM model with {} layers", num_layers);
    
    for epoch in 1..=config.epochs {
        // Update learning rate (linear decay)
        let mut current_lr = config.learning_rate * (1.0 - (epoch as f64) / (config.epochs as f64));
        if current_lr < 1e-8 {
            current_lr = 1e-8;
        }

        let feature_batches = get_batches(&train_features, config.batch_size);
        let target_batches = get_batches(&train_targets, config.batch_size);

        let mut epoch_loss = 0.0;
        for (batch_features, batch_targets) in feature_batches.iter().zip(target_batches.iter()) {
            // Forward pass
            let predictions = model.forward(batch_features.clone());
            
            // Compute loss (MSE or Huber)
            let loss_tensor = if config.use_huber_loss {
                model.huber_loss(predictions.clone(), batch_targets.clone(), 1.0)
            } else {
                model.mse_loss(predictions.clone(), batch_targets.clone())
            };
            
            // Extract loss value
            let loss = match tensor_to_f64(&loss_tensor) {
                Ok(val) => val,
                Err(e) => {
                    println!("Warning: Failed to extract loss value: {}", e);
                    0.0 // Use a fallback value
                }
            };
            
            epoch_loss += loss;

            // Backward pass and optimizer step
            let grads = loss_tensor.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(current_lr, model, grads);
        }
        
        let avg_loss = epoch_loss / feature_batches.len() as f64;
        loss_history.push(avg_loss);
        
        if config.display_metrics && epoch % config.display_interval == 0 {
            println!("Epoch {}/{} - Loss: {:.6}", epoch, config.epochs, avg_loss);
        }

        // Validation pass
        let val_preds = model.forward(test_features.clone());
        let val_diff = val_preds - test_targets.clone();
        
        // Calculate MSE with proper dimension handling
        let val_squared_diff = val_diff.clone() * val_diff;
        let val_mse_tensor = val_squared_diff.mean().reshape([1]);
        
        // Extract MSE value
        let val_mse = match tensor_to_f64(&val_mse_tensor) {
            Ok(val) => val,
            Err(e) => {
                println!("Warning: Failed to extract validation MSE value: {}", e);
                f64::MAX // Use a fallback value that will prevent early stopping
            }
        };
        
        let val_rmse = val_mse.sqrt();
        
        if config.display_metrics && epoch % config.display_interval == 0 {
            println!("Validation RMSE: {:.6}", val_rmse);
        }

        // Early stopping logic
        if best_val_rmse - val_rmse > config.min_delta {
            best_val_rmse = val_rmse;
            best_model = model.clone();
            epochs_no_improve = 0;
            
            if config.display_metrics {
                println!("New best model (RMSE: {:.6})", val_rmse);
            }
        } else {
            epochs_no_improve += 1;
            if epochs_no_improve >= config.patience {
                println!(
                    "Early stopping triggered at epoch {} (best val RMSE = {:.6})",
                    epoch, best_val_rmse
                );
                model = best_model.clone();
                break;
            }
        }
    }

    println!("Training completed. Best validation RMSE: {:.6}", best_val_rmse);

    Ok((model, loss_history))
}

/// Evaluate the CNN-LSTM model on test data
pub fn evaluate_model<B: Backend>(
    model: &TimeSeriesCnnLstm<B>,
    test_df: DataFrame,
    device: &B::Device,
    forecast_horizon: usize,
) -> Result<f64> {
    // Prepare test data
    let (test_features, test_targets) = step_1_tensor_preparation::dataframe_to_tensors::<B>(
        &test_df,
        crate::constants::SEQUENCE_LENGTH,
        forecast_horizon,
        device,
        false,
        None,
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Forward pass
    let predictions = model.forward(test_features);
    
    // Calculate RMSE
    let diff = predictions - test_targets;
    let mse_tensor = (diff.clone() * diff).mean();
    
    // Extract MSE value safely
    let mse = tensor_to_f64(&mse_tensor)?;
    let rmse = mse.sqrt();
    
    Ok(rmse)
} 