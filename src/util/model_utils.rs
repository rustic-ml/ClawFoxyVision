use anyhow::{Context, Result};
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use chrono::{DateTime, Local, Utc};
use serde_json::from_str;
use std::path::Path;
use std::path::PathBuf;

use crate::constants::MODEL_PATH;
use crate::minute::gru::{
    step_3_gru_model_arch::TimeSeriesGru,
    step_6_model_serialization::{load_model as load_gru_model, ModelMetadata as GruMetadata},
};
use crate::minute::lstm::{
    step_3_lstm_model_arch::TimeSeriesLstm,
    step_6_model_serialization::{
        load_model_with_metadata as load_lstm_model, ModelMetadata as LstmMetadata,
    },
};

/// Enum to represent different model types
pub enum ModelType {
    Lstm,
    Gru,
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "lstm" => Ok(ModelType::Lstm),
            "gru" => Ok(ModelType::Gru),
            _ => Err(anyhow::anyhow!("Unsupported model type: {}", s)),
        }
    }
}

/// Get the default path for saving models, overridable via MODEL_PATH env var
pub fn get_model_path(ticker: &str, model_type: &str) -> PathBuf {
    // Get current directory as fallback
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Allow overriding MODEL_PATH via environment variable, fallback to constant or current directory
    let base = if !MODEL_PATH.is_empty() {
        MODEL_PATH.to_string()
    } else {
        std::env::var("MODEL_PATH").unwrap_or_else(|_| {
            // If MODEL_PATH is not set, use "models" in the current directory
            current_dir.join("models").to_string_lossy().to_string()
        })
    };

    let mut path = PathBuf::from(&base);

    // Create directories if they don't exist
    path.push(ticker);
    path.push(model_type);

    if !path.exists() {
        if let Err(e) = std::fs::create_dir_all(&path) {
            eprintln!(
                "Warning: Failed to create model directory at {}: {}",
                path.display(),
                e
            );
            // Continue anyway, the error will be handled when trying to save
        } else {
            println!("Created model directory: {}", path.display());
        }
    }

    path
}

/// Save a trained LSTM model with its configuration to MODEL_PATH
pub fn save_trained_lstm_model<B: Backend>(
    model: &TimeSeriesLstm<B>,
    ticker: &str,
    model_type: &str,
    model_name: &str,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
) -> Result<PathBuf> {
    // Create model directory if it doesn't exist
    let model_dir = get_model_path(ticker, model_type);
    std::fs::create_dir_all(&model_dir).context("Failed to create models directory")?;

    // Create model path
    let model_path = model_dir.join(model_name);

    // Ensure parent directory exists
    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create model parent directory")?;
    }

    // Create metadata
    let metadata = LstmMetadata::new(
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    );

    // Save model with metadata
    save_lstm_model_with_metadata(model, metadata, &model_path).context("Failed to save model")?;

    println!("LSTM model saved successfully to: {}", model_path.display());
    Ok(model_path)
}

/// Save a trained GRU model with its configuration to MODEL_PATH
pub fn save_trained_gru_model<B: Backend>(
    model: &TimeSeriesGru<B>,
    ticker: &str,
    model_type: &str,
    model_name: &str,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
) -> Result<PathBuf> {
    // Create model directory if it doesn't exist
    let model_dir = get_model_path(ticker, model_type);
    std::fs::create_dir_all(&model_dir).context("Failed to create models directory")?;

    // Create model path
    let model_path = model_dir.join(model_name);

    // Ensure parent directory exists
    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create model parent directory")?;
    }

    // Create metadata
    let metadata = GruMetadata {
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
        learning_rate: 0.001, // default value
        timestamp: Local::now().timestamp() as u64,
        description: format!("GRU model for {}", ticker),
    };

    // Save model - using save_model function from GRU module
    crate::minute::gru::step_6_model_serialization::save_model(model, metadata, model_path.clone())
        .context("Failed to save GRU model")?;

    println!("GRU model saved successfully to: {}", model_path.display());
    Ok(model_path)
}

/// Load a trained LSTM model with its configuration from MODEL_PATH
pub fn load_trained_lstm_model<B: Backend>(
    ticker: &str,
    model_type: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(TimeSeriesLstm<B>, LstmMetadata)> {
    let model_path = get_model_path(ticker, model_type).join(model_name);
    println!("Loading LSTM model from: {}", model_path.display());

    // Check if model exists
    if !model_path.exists() && !model_path.with_extension("bin").exists() {
        return Err(anyhow::anyhow!(
            "LSTM model file not found at: {}",
            model_path.display()
        ));
    }

    load_lstm_model(&model_path, device).context("Failed to load LSTM model")
}

/// Load a trained GRU model with its configuration from MODEL_PATH
pub fn load_trained_gru_model<B: Backend>(
    ticker: &str,
    model_type: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(TimeSeriesGru<B>, GruMetadata)> {
    let model_path = get_model_path(ticker, model_type).join(model_name);
    println!("Loading GRU model from: {}", model_path.display());

    // Check if model exists
    if !model_path.exists() && !model_path.with_extension("bin").exists() {
        return Err(anyhow::anyhow!(
            "GRU model file not found at: {}",
            model_path.display()
        ));
    }

    load_gru_model(&model_path, device).context("Failed to load GRU model")
}

/// Save a model checkpoint during training (LSTM)
pub fn save_lstm_model_checkpoint<B: Backend>(
    model: &TimeSeriesLstm<B>,
    ticker: &str,
    model_type: &str,
    model_name: &str,
    epoch: usize,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
) -> Result<PathBuf> {
    let checkpoint_name = format!("{}_epoch_{}", model_name, epoch);
    save_trained_lstm_model(
        model,
        ticker,
        model_type,
        &checkpoint_name,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    )
}

/// Save a model checkpoint during training (GRU)
pub fn save_gru_model_checkpoint<B: Backend>(
    model: &TimeSeriesGru<B>,
    ticker: &str,
    model_type: &str,
    model_name: &str,
    epoch: usize,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
) -> Result<PathBuf> {
    let checkpoint_name = format!("{}_epoch_{}", model_name, epoch);
    save_trained_gru_model(
        model,
        ticker,
        model_type,
        &checkpoint_name,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    )
}

pub fn save_lstm_model_with_metadata<B: Backend>(
    model: &TimeSeriesLstm<B>,
    metadata: LstmMetadata,
    path: impl AsRef<Path>,
) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).context("Failed to create model parent directory")?;
    }
    // Save model
    let model_path = path.as_ref().with_extension("bin");
    model
        .clone()
        .save_file::<BinFileRecorder<FullPrecisionSettings>, _>(&model_path, &Default::default())
        .context("Failed to save model")?;
    // Save metadata
    let metadata_path = path.as_ref().with_extension("meta.json");
    let metadata_json =
        serde_json::to_string_pretty(&metadata).context("Failed to serialize metadata")?;
    std::fs::write(&metadata_path, metadata_json).context("Failed to write metadata file")?;
    Ok(())
}

/// Check if the saved model's version matches the current code version
pub fn is_model_version_current(model_base_path: &Path, current_version: &str) -> bool {
    let metadata_path = model_base_path.with_extension("meta.json");
    if let Ok(metadata_json) = std::fs::read_to_string(&metadata_path) {
        // First try to parse as LSTM metadata
        if let Ok(metadata) = from_str::<LstmMetadata>(&metadata_json) {
            // Check if LSTM metadata has version field and matches current version
            if metadata.version != current_version {
                return false;
            }
            // Check if timestamp is today
            let model_date = DateTime::<Utc>::from_timestamp(metadata.timestamp as i64, 0)
                .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
            let now = Local::now().naive_utc();
            if model_date.date_naive() == now.date() {
                return true;
            }
        }
        // Try as GRU metadata if LSTM parse fails
        else if let Ok(metadata) = from_str::<GruMetadata>(&metadata_json) {
            // GRU metadata doesn't have version field, so just check if it's from today
            let model_date = DateTime::<Utc>::from_timestamp(metadata.timestamp as i64, 0)
                .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
            let now = Local::now().naive_utc();
            if model_date.date_naive() == now.date() {
                return true;
            }
        }
    }
    false
}

// For backward compatibility, original function names
pub fn save_trained_model<B: Backend>(
    model: &TimeSeriesLstm<B>,
    ticker: &str,
    model_type: &str,
    model_name: &str,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
) -> Result<PathBuf> {
    // Forward to new function name
    save_trained_lstm_model(
        model,
        ticker,
        model_type,
        model_name,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    )
}

pub fn save_model_checkpoint<B: Backend>(
    model: &TimeSeriesLstm<B>,
    ticker: &str,
    model_type: &str,
    model_name: &str,
    epoch: usize,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
) -> Result<PathBuf> {
    // Forward to new function name
    save_lstm_model_checkpoint(
        model,
        ticker,
        model_type,
        model_name,
        epoch,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    )
}
