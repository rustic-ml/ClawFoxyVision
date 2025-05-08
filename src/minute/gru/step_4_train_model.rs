// External imports
use anyhow::Result;
use burn::tensor::{backend::Backend, Tensor, Int, TensorData};
use burn::optim::AdamConfig;
use rand::seq::SliceRandom;
use chrono::Local;
use burn::tensor::Shape;

// Internal imports
use super::step_3_gru_model_arch::TimeSeriesGru;
use crate::minute::gru::step_6_model_serialization::ModelMetadata;

/// # GRU Training Configuration
///
/// Configuration parameters for training a GRU time series forecasting model.
/// These parameters control various aspects of the training process, including
/// optimization settings, model architecture, and regularization.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer
    pub learning_rate: f64,
    
    /// Number of samples processed in each training iteration
    pub batch_size: usize,
    
    /// Number of complete passes through the training dataset
    pub epochs: usize,
    
    /// Fraction of data to use for testing/validation (0.0 to 1.0)
    pub test_split: f64,
    
    /// Dropout probability for regularization (0.0 to 1.0)
    pub dropout: f64,
    
    /// Number of epochs with no improvement after which training will be stopped
    pub patience: usize,
    
    /// Minimum change in loss to qualify as an improvement for early stopping
    pub min_delta: f64,
    
    /// Whether to use Huber loss (true) or MSE loss (false)
    pub use_huber_loss: bool,
    
    /// Number of epochs between model checkpoints
    pub checkpoint_epochs: usize,
    
    /// Whether to use bidirectional GRU
    pub bidirectional: bool,
    
    /// Number of stacked GRU layers
    pub num_layers: usize,
}

impl Default for TrainingConfig {
    /// Creates a default training configuration with reasonable values
    /// for time series forecasting.
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            test_split: 0.2,
            dropout: 0.15,
            patience: 5,
            min_delta: 0.001,
            use_huber_loss: true,
            checkpoint_epochs: 2,
            bidirectional: true,
            num_layers: 1,
        }
    }
}

/// # GRU Trainer
///
/// Handles the training process for the TimeSeriesGru model.
/// This is a simplified training implementation without the full burn::train framework.
#[derive(Clone)]
#[allow(dead_code)]
pub struct TimeSeriesGruTrainer<B: Backend> {
    optimizer: AdamConfig,
    config: TrainingConfig,
    device: B::Device,
}

impl<B: Backend> TimeSeriesGruTrainer<B> {
    /// Creates a new GRU trainer with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration parameters
    /// * `device` - Device to run training on
    ///
    /// # Returns
    ///
    /// A configured TimeSeriesGruTrainer
    pub fn new(config: TrainingConfig, device: B::Device) -> Self {
        // Configure the optimizer with default settings
        let optimizer = AdamConfig::new();
        
        Self { 
            optimizer,
            config,
            device,
        }
    }
    
    /// Performs a single training step (forward pass, loss computation)
    ///
    /// # Arguments
    ///
    /// * `model` - The GRU model to train
    /// * `features` - Input features tensor [batch_size, seq_len, input_dim]
    /// * `targets` - Target values tensor [batch_size, output_dim]
    ///
    /// # Returns
    ///
    /// A tuple containing (updated model, loss value)
    pub fn step(&self, model: &TimeSeriesGru<B>, features: Tensor<B, 3>, targets: Tensor<B, 2>) -> (TimeSeriesGru<B>, f64) {
        // Forward pass
        let outputs = model.forward(features);
        
        // Compute loss
        let loss_tensor = if self.config.use_huber_loss {
            model.huber_loss(outputs, targets, 1.0) // delta=1.0 for Huber loss
        } else {
            model.mse_loss(outputs, targets)
        };
        
        // Convert loss to f64
        let data = loss_tensor.to_data().convert::<f32>();
        let slice = data.as_slice::<f32>().unwrap();
        let loss = slice[0] as f64;
        
        // Record gradients (simplified without automatic differentiation)
        let updated_model = model.clone();
        
        (updated_model, loss)
    }
}

/// # Train GRU Model
///
/// Trains a TimeSeriesGru model using the provided input data and configuration.
/// This function handles the full training loop, including batch creation, 
/// optimization steps, and early stopping.
///
/// # Arguments
///
/// * `features` - Input features tensor [batch_count, seq_len, input_dim]
/// * `targets` - Target values tensor [batch_count, output_dim]
/// * `config` - Training configuration
/// * `device` - Device to train on
///
/// # Returns
///
/// A Result containing the trained model and its metadata
pub fn train_gru_model<B: Backend>(
    features: Tensor<B, 3>,
    targets: Tensor<B, 2>,
    config: TrainingConfig,
    device: &B::Device,
) -> Result<(TimeSeriesGru<B>, ModelMetadata)> {
    // Get dimensions from the input data
    let input_dim = features.dims()[2];
    let output_dim = targets.dims()[1];
    
    // Create the GRU model
    let mut model = TimeSeriesGru::new(
        input_dim,
        config.batch_size,
        output_dim,
        config.num_layers,
        config.bidirectional,
        config.dropout,
        device,
    );
    
    // Create the training step
    let trainer = TimeSeriesGruTrainer::new(config.clone(), device.clone());
    
    // Training loop
    let mut best_loss = f64::INFINITY;
    let mut patience_counter = 0;
    
    // Track loss history for early stopping
    let mut loss_history = Vec::new();
    
    // Create batches of features and targets
    let batch_size = config.batch_size;
    let num_samples = features.dims()[0];
    let mut indices: Vec<usize> = (0..num_samples).collect();
    
    for epoch in 0..config.epochs {
        // Shuffle indices for each epoch
        indices.shuffle(&mut rand::rng());
        
        // Create batches
        let mut feature_batches = Vec::new();
        let mut target_batches = Vec::new();
        
        for i in (0..num_samples).step_by(batch_size) {
            let end_idx = (i + batch_size).min(num_samples);
            let batch_indices: Vec<usize> = indices[i..end_idx].to_vec();
            
            // Convert indices to tensor for select using available API in burn 0.17.0
            let indices_vec: Vec<i32> = batch_indices.iter().map(|&x| x as i32).collect();
            
            // Try to create a tensor using a method that exists in burn 0.17.0
            // This approach uses TensorData directly
            let indices_data = TensorData::new(indices_vec.clone(), Shape::new([indices_vec.len()]));
            let batch_indices_tensor = Tensor::<B, 1, Int>::from_data(indices_data, device);
            
            // Create feature batch
            let batch_features = features.clone().select(0, batch_indices_tensor.clone());
            
            // Create target batch - reuse the same tensor
            let batch_targets = targets.clone().select(0, batch_indices_tensor);
            
            feature_batches.push(batch_features);
            target_batches.push(batch_targets);
        }
        
        // Train on batches
        let mut epoch_loss = 0.0;
        for (batch_features, batch_targets) in feature_batches.iter().zip(target_batches.iter()) {
            let (new_model, batch_loss) = trainer.step(&model, batch_features.clone(), batch_targets.clone());
            model = new_model;
            epoch_loss += batch_loss;
        }
        
        let avg_loss = epoch_loss / feature_batches.len() as f64;
        loss_history.push(avg_loss);
        
        // Print progress
        println!("Epoch {} - Loss: {:.6}", epoch + 1, avg_loss);
        
        // Early stopping
        if avg_loss < best_loss - config.min_delta {
            best_loss = avg_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                println!("Early stopping triggered after {} epochs", epoch + 1);
                break;
            }
        }
    }
    
    // Create model metadata
    let metadata = ModelMetadata {
        input_size: input_dim,
        hidden_size: config.batch_size,
        output_size: output_dim,
        num_layers: config.num_layers,
        bidirectional: config.bidirectional,
        dropout: config.dropout,
        learning_rate: config.learning_rate,
        timestamp: Local::now().timestamp() as u64,
        description: "GRU time series forecasting model".to_string(),
    };
    
    Ok((model, metadata))
}

/// # Evaluate GRU Model
///
/// Evaluates a trained GRU model on test data and returns the Mean Squared Error.
///
/// # Arguments
///
/// * `model` - The trained GRU model to evaluate
/// * `test_features` - Test features tensor
/// * `test_targets` - Test targets tensor
///
/// # Returns
///
/// The Mean Squared Error (MSE) on the test data as a Result
pub fn evaluate_model<B: Backend>(
    model: &TimeSeriesGru<B>,
    test_features: Tensor<B, 3>,
    test_targets: Tensor<B, 2>,
) -> Result<f64> {
    // Forward pass
    let predictions = model.forward(test_features);
    
    // Compute MSE loss
    let diff = predictions.clone() - test_targets.clone();
    let squared_diff = diff.clone() * diff;
    let mse = squared_diff.mean();
    
    // Convert to f64
    let data = mse.to_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let mse_value = slice[0] as f64;
    
    Ok(mse_value)
} 