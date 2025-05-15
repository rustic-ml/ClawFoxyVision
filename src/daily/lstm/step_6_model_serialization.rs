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
use super::step_3_lstm_model_arch::{DailyLSTMModel, DailyLSTMModelConfig};

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
pub fn save_daily_lstm_model<B: Backend, P: AsRef<Path>>(
    model: &DailyLSTMModel<B>,
    path: P,
    config: &DailyLSTMModelConfig,
) -> Result<()> {
    // Create directory if necessary
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }

    // Save the config separately
    let config_path = path.as_ref().with_extension("config.json");
    save_daily_lstm_config(config, config_path)?;

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
pub fn load_daily_lstm_model<B: Backend, P: AsRef<Path>>(
    config: &DailyLSTMModelConfig,
    path: P,
    device: &B::Device,
) -> Result<DailyLSTMModel<B>> {
    // Create a new model using the config
    let model = config.init(device);

    // Return the model directly without loading weights
    // TODO: This needs proper implementation when model saving/loading is required
    Ok(model)
}

#[derive(Serialize, Deserialize)]
struct SerializableLSTMConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    dropout_rate: f64,
}

impl From<&DailyLSTMModelConfig> for SerializableLSTMConfig {
    fn from(config: &DailyLSTMModelConfig) -> Self {
        Self {
            input_size: config.input_size,
            hidden_size: config.hidden_size,
            output_size: config.output_size,
            dropout_rate: config.dropout_rate,
        }
    }
}

impl From<SerializableLSTMConfig> for DailyLSTMModelConfig {
    fn from(config: SerializableLSTMConfig) -> Self {
        Self {
            input_size: config.input_size,
            hidden_size: config.hidden_size,
            output_size: config.output_size,
            dropout_rate: config.dropout_rate,
        }
    }
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
pub fn save_daily_lstm_config<P: AsRef<Path>>(
    config: &DailyLSTMModelConfig,
    path: P,
) -> Result<()> {
    // Create configuration as JSON
    let serializable_config = SerializableLSTMConfig::from(config);
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
pub fn load_daily_lstm_config<P: AsRef<Path>>(path: P) -> Result<DailyLSTMModelConfig> {
    // Read from file
    let config_json = fs::read_to_string(path)?;

    // Parse JSON
    let serializable_config: SerializableLSTMConfig = serde_json::from_str(&config_json)?;
    let config = DailyLSTMModelConfig::from(serializable_config);

    Ok(config)
}
