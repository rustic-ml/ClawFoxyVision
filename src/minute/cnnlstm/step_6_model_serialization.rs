// External imports
use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use std::path::Path;

// Internal imports
use super::step_3_cnn_lstm_model_arch::TimeSeriesCnnLstm;

/// Save model to disk
pub fn save_model<B: Backend>(
    model: &TimeSeriesCnnLstm<B>,
    path: &Path,
) -> Result<()> {
    // Create directory if it doesn't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Convert path to string
    let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
    
    // Create a recorder with full precision settings
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    
    // Clone the model and save it using module::save_file method
    model.clone().save_file(path_str, &recorder)?;
    
    println!("Model saved to: {}", path_str);
    
    Ok(())
}

/// Load model from disk with the simplified architecture
pub fn load_model<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<TimeSeriesCnnLstm<B>> {
    // Convert path to string
    let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
    
    // Create a recorder with full precision settings
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    
    // Create a new model with the simplified architecture matching our training configuration
    let base_model = TimeSeriesCnnLstm::<B>::new(
        /* input_size */ 7,      // Default input features
        /* hidden_size */ 32,     // Reduced hidden size (was 64)
        /* output_size */ 1,      // Default output size
        /* num_layers */ 1,       // Reduced to 1 layer (was 2)
        /* bidirectional */ false, // No bidirectional (was true)
        /* dropout */ 0.2,        // Default dropout rate
        device,
    );
    
    // Load the model using module::load_file method
    let model = base_model.load_file(path_str, &recorder, device)?;
    
    println!("Lightweight model loaded from: {}", path_str);
    
    Ok(model)
} 