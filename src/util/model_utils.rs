use anyhow::{Context, Result};
use burn::prelude::Backend;
use burn::module::Module;
use std::path::PathBuf;
use std::path::Path;
use std::thread::panicking;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use serde_json::from_str;
use chrono::{NaiveDateTime, Local, Datelike};

use crate::lstm::{
    step_3_lstm_model_arch::TimeSeriesLstm,
    step_6_model_serialization::{ModelMetadata, load_model_with_metadata},
};
use crate::constants::MODEL_PATH;

/// Get the default path for saving models, overridable via MODEL_PATH env var
pub fn get_model_path(ticker: &str, model_type: &str) -> PathBuf {
    // Allow overriding MODEL_PATH via environment variable, fallback to constant
    let base = std::env::var("MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string());
    
    if !Path::new(&base).exists() {
        if let Err(e) = std::fs::create_dir_all(&base) {
            eprintln!("Failed to create models directory: {}", e);
        }
    }
    if base == "" {
        println!("MODEL_PATH is not set, cannot save models");
        panic!("MODEL_PATH is not set, cannot save models");
    }
    
    let mut path = PathBuf::from(base);
    path.push(ticker);
    path.push(model_type);
    path
}

/// Save a trained model with its configuration to MODEL_PATH
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
    // Create model directory if it doesn't exist
    let model_dir = get_model_path(ticker, model_type);
    std::fs::create_dir_all(&model_dir)
        .context("Failed to create models directory")?;

    // Create model path
    let model_path = model_dir.join(model_name);

    // Ensure parent directory exists
    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent)
            .context("Failed to create model parent directory")?;
    }

    // Create metadata
    let metadata = ModelMetadata::new(
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional,
        dropout,
    );

    // Save model with metadata
    save_model_with_metadata(model, metadata, &model_path)
        .context("Failed to save model")?;

    println!("Model saved successfully to: {}", model_path.display());
    Ok(model_path)
}

/// Load a trained model with its configuration from MODEL_PATH
pub fn load_trained_model<B: Backend>(
    ticker: &str,
    model_type: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(TimeSeriesLstm<B>, ModelMetadata)> {
    let model_path = get_model_path(ticker, model_type).join(model_name);
    println!("Loading model from: {}", model_path.display());
    load_model_with_metadata(&model_path, device)
        .context("Failed to load model")
}

/// Save a model checkpoint during training
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
    let checkpoint_name = format!("{}_epoch_{}", model_name, epoch);
    save_trained_model(
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

pub fn save_model_with_metadata<B: Backend>(
    model: &TimeSeriesLstm<B>,
    metadata: ModelMetadata,
    path: impl AsRef<Path>,
) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .context("Failed to create model parent directory")?;
    }
    // Save model
    let model_path = path.as_ref().with_extension("bin");
    model.clone()
        .save_file::<BinFileRecorder<FullPrecisionSettings>, _>(&model_path, &Default::default())
        .context("Failed to save model")?;
    // Save metadata
    let metadata_path = path.as_ref().with_extension("meta.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .context("Failed to serialize metadata")?;
    std::fs::write(&metadata_path, metadata_json)
        .context("Failed to write metadata file")?;
    Ok(())
}

/// Check if the saved model's version matches the current code version
pub fn is_model_version_current(model_base_path: &Path, current_version: &str) -> bool {
    let metadata_path = model_base_path.with_extension("meta.json");
    if let Ok(metadata_json) = std::fs::read_to_string(&metadata_path) {
        if let Ok(metadata) = from_str::<ModelMetadata>(&metadata_json) {
            // Check version
            if metadata.version != current_version {
                return false;
            }
            // Check if timestamp is today
            let model_date = NaiveDateTime::from_timestamp_opt(metadata.timestamp as i64, 0)
                .unwrap_or_else(|| NaiveDateTime::from_timestamp_opt(0, 0).unwrap());
            let now = Local::now().naive_local();
            if model_date.date() == now.date() {
                return true;
            }
        }
    }
    false
}