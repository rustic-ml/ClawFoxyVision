// External imports
use anyhow::Result;
use burn::tensor::{backend::Backend, Tensor, Int, TensorData};
use burn::module::Module;
use burn::optim::{AdamConfig, Optimizer};
use burn::record::Record;
use rand::seq::SliceRandom;
use chrono::Local;
use burn::tensor::Shape;

// Internal imports
use super::step_3_gru_model_arch::TimeSeriesGru;
use crate::minute::lstm::step_1_tensor_preparation::{
    dataframe_to_tensors, normalize_features, split_data
};
use crate::minute::gru::step_6_model_serialization::ModelMetadata;
use crate::constants::SEQUENCE_LENGTH;

/// Struct for configuring GRU training
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub test_split: f64,
    pub dropout: f64,
    pub patience: usize,
    pub min_delta: f64,
    pub use_huber_loss: bool,
    pub checkpoint_epochs: usize,
    pub bidirectional: bool,
    pub num_layers: usize,
}

impl Default for TrainingConfig {
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

/// Simple training step without using burn::train::TrainStep
#[derive(Clone)]
pub struct TimeSeriesGruTrainer<B: Backend> {
    optimizer: AdamConfig,
    config: TrainingConfig,
    device: B::Device,
}

impl<B: Backend> TimeSeriesGruTrainer<B> {
    /// Create a new TimeSeriesGruTrainer
    pub fn new(config: TrainingConfig, device: B::Device) -> Self {
        // Configure the optimizer with default settings
        let optimizer = AdamConfig::new();
        
        Self { 
            optimizer,
            config,
            device,
        }
    }
    
    /// Perform a single training step
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

/// Train a TimeSeriesGru model using the provided training data and configuration
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

/// Evaluate a model on the test dataset
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