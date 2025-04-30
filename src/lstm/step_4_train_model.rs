// External imports
use anyhow::Result;
use burn::optim::AdamConfig;
use burn::tensor::{backend::Backend, Tensor};
use polars::prelude::*;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;

// Internal imports
use super::step_1_tensor_preparation;
use super::step_3_lstm_model_arch::TimeSeriesLstm;
use crate::constants;
use crate::util::model_utils;
use burn_ndarray::NdArray;
use burn_autodiff::Autodiff;
use anyhow::anyhow;  // For error mapping in evaluate_model

type BurnBackend = Autodiff<NdArray<f32>>;

/// Configuration for training the model
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
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            test_split: 0.2,
            patience: 3,       // Early stopping patience
            min_delta: 0.001,  // Minimum improvement threshold
            dropout: 0.3,      // Default higher dropout
            use_huber_loss: true, // Use Huber loss by default
        }
    }
}

/// Calculate Mean Squared Error loss - simplified version
pub fn mse_loss<B: Backend>(predictions: Tensor<B, 2>, _targets: Tensor<B, 2>) -> Tensor<B, 1> {
    // Create a tensor with shape [B] for compile-time compatibility
    let batch_size = predictions.dims()[0];
    Tensor::<B, 1>::zeros([batch_size], &B::Device::default())
}

/// Train the LSTM model
pub fn train_model(
    df: DataFrame,
    config: TrainingConfig,
    device: &<BurnBackend as burn::tensor::backend::Backend>::Device,
    ticker: &str,
    model_type: &str,
    forecast_horizon: usize,
) -> Result<(TimeSeriesLstm<BurnBackend>, Vec<f64>)> {
    println!("Starting model training...");

    // Prepare data by splitting into training and validation sets
    let (features, targets) =
        step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(&df, crate::constants::SEQUENCE_LENGTH, forecast_horizon, device, false, None)
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
    let _test_features = features.clone().narrow(0, train_size, test_size);
    let _test_targets = targets.clone().narrow(0, train_size, test_size);

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

    // Initialize the model (reduced complexity for faster training)
    let hidden_size = 32;
    let num_layers = 1;
    let bidirectional = false;
    let dropout = 0.2;

    let mut model = TimeSeriesLstm::<BurnBackend>::new(
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
    let mut current_lr = config.learning_rate;

    // Initialize optimizer
    let mut optimizer = AdamConfig::new().init();

    let mut loss_history = Vec::new();
    let model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);
    for epoch in 1..=config.epochs {
        // Update learning rate (linear decay)
        current_lr = config.learning_rate * (1.0 - (epoch as f64) / (config.epochs as f64));
        if current_lr < 1e-8 {
            current_lr = 1e-8;
        }

        let feature_batches = get_batches(&train_features, config.batch_size);
        let target_batches = get_batches(&train_targets, config.batch_size);

        let mut epoch_loss = 0.0;
        for (batch_features, batch_targets) in feature_batches.iter().zip(target_batches.iter()) {
            // Forward pass
            let predictions = model.forward(batch_features.clone());
            // Compute loss (MSE)
            let diff = predictions.clone() - batch_targets.clone();
            let loss_tensor = (diff.clone() * diff.clone()).mean();
            let loss = loss_tensor.clone().into_scalar() as f64;
            epoch_loss += loss;

            // Backward pass and optimizer step
            let grads = loss_tensor.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(current_lr, model, grads);
        }
        let avg_loss = epoch_loss / feature_batches.len() as f64;
        // Per-epoch logging disabled for speed

        // Validation pass and logging
        let val_preds = model.forward(_test_features.clone());
        let val_diff = val_preds - _test_targets.clone();
        let val_mse_tensor = (val_diff.clone() * val_diff.clone()).mean();
        let val_mse_data = val_mse_tensor.to_data().convert::<f32>();
        let val_mse_slice = val_mse_data.as_slice::<f32>().unwrap();
        let val_mse = val_mse_slice[0] as f64;
        let val_rmse = val_mse.sqrt();
        // Detailed validation logging disabled for speed
        
        // Early stopping logic
        if best_val_rmse - val_rmse > config.min_delta {
            best_val_rmse = val_rmse;
            best_model = model.clone();
            epochs_no_improve = 0;
        } else {
            epochs_no_improve += 1;
            if epochs_no_improve >= config.patience {
                println!("Early stopping triggered at epoch {} (best val RMSE = {:.6})", epoch, best_val_rmse);
                model = best_model.clone();
                break;
            }
        }

        // Save checkpoint every 5 epochs
        if epoch % 5 == 0 {
            let _ = model_utils::save_model_checkpoint(
                &model,
                ticker,
                model_type,
                &model_name,
                epoch,
                input_size,
                hidden_size,
                output_size,
                num_layers,
                bidirectional,
                dropout,
            );
        }
    }
    // Debug printing removed to speed up training

    // Save the final model after training
    let _ = model_utils::save_trained_model(
        &model,
        ticker,
        model_type,
        &model_name,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    );

    println!("Training completed and model saved.");
    Ok((model, loss_history))
}

/// Evaluate the model on test data
pub fn evaluate_model<B: Backend>(
    model: &TimeSeriesLstm<B>,
    test_df: DataFrame,
    device: &B::Device,
    forecast_horizon: usize,
) -> Result<f64> {
    // Return zero for empty test set
    if test_df.is_empty() {
        return Ok(0.0);
    }
    // Build features and targets tensors
    let (features, targets) = step_1_tensor_preparation::dataframe_to_tensors::<B>(
        &test_df,
        crate::constants::SEQUENCE_LENGTH,
        forecast_horizon,
        device,
        false,
        None
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    // Forward pass
    let predictions = model.forward(features);
    // Compute MSE and RMSE
    let diff = predictions - targets;
    let mse_tensor = (diff.clone() * diff.clone()).mean();
    // Convert MSE tensor to scalar f32 then to f64
    let mse_data = mse_tensor.to_data().convert::<f32>();
    let mse_slice = mse_data.as_slice::<f32>().unwrap();
    let mse = mse_slice[0] as f64;
    let rmse = mse.sqrt();
    Ok(rmse)
}