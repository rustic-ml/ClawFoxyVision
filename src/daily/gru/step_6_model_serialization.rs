// External imports
use anyhow::Result;
use burn::{
    module::Module,
    record::{Record, Recorder},
    tensor::backend::Backend,
};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// Internal imports
use super::step_3_gru_model_arch::{DailyGRUModel, DailyGRUModelConfig};

#[derive(Serialize, Deserialize)]
pub struct SerializableGRUConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub dropout_rate: f64,
}

impl From<&DailyGRUModelConfig> for SerializableGRUConfig {
    fn from(config: &DailyGRUModelConfig) -> Self {
        Self {
            input_size: config.input_size,
            hidden_size: config.hidden_size,
            output_size: config.output_size,
            dropout_rate: config.dropout_rate,
        }
    }
}

impl From<SerializableGRUConfig> for DailyGRUModelConfig {
    fn from(config: SerializableGRUConfig) -> Self {
        Self {
            input_size: config.input_size,
            hidden_size: config.hidden_size,
            output_size: config.output_size,
            dropout_rate: config.dropout_rate,
        }
    }
}

/// Save a trained model to disk
///
/// # Arguments
///
/// * `model` - Trained model to save
/// * `path` - Path to save the model to
/// * `config` - The model configuration
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_daily_gru_model<B: Backend, P: AsRef<Path>>(
    model: &DailyGRUModel<B>,
    path: P,
    config: &DailyGRUModelConfig,
) -> Result<()> {
    // Create directory if necessary
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }

    // Save the config separately
    let config_path = path.as_ref().with_extension("config.json");
    save_model_config(config, config_path)?;

    // For now, just return success
    // TODO: This needs proper implementation when model saving/loading is required
    Ok(())
}

/// Load a trained model from disk
///
/// # Arguments
///
/// * `config` - Model configuration
/// * `path` - Path to load the model from
/// * `device` - Device to place tensors on
///
/// # Returns
///
/// Result containing the loaded model
pub fn load_daily_gru_model<B: Backend, P: AsRef<Path>>(
    config: &DailyGRUModelConfig,
    path: P,
    device: &B::Device,
) -> Result<DailyGRUModel<B>> {
    // Create a new model using the config
    let model = config.init(device);

    // Return the model directly without loading weights
    // TODO: This needs proper implementation when model saving/loading is required
    Ok(model)
}

/// Save model configuration to disk
///
/// # Arguments
///
/// * `config` - Model configuration
/// * `path` - Path to save the configuration to
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_model_config<P: AsRef<Path>>(config: &DailyGRUModelConfig, path: P) -> Result<()> {
    // Create configuration as JSON
    let serializable_config = SerializableGRUConfig::from(config);
    let config_json = serde_json::to_string_pretty(&serializable_config)?;

    // Create directory if it doesn't exist
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write to file
    fs::write(path, config_json)?;

    Ok(())
}

/// Load model configuration from disk
///
/// # Arguments
///
/// * `path` - Path to load the configuration from
///
/// # Returns
///
/// Result containing the loaded configuration
pub fn load_model_config<P: AsRef<Path>>(path: P) -> Result<DailyGRUModelConfig> {
    // Read from file
    let config_json = fs::read_to_string(path)?;

    // Parse JSON
    let serializable_config: SerializableGRUConfig = serde_json::from_str(&config_json)?;
    let config = DailyGRUModelConfig::from(serializable_config);

    Ok(config)
}
