use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::SystemTime;

use super::step_3_lstm_model_arch::TimeSeriesLstm;

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub version: String,
    pub timestamp: u64,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub dropout: f64,
}

impl ModelMetadata {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f64,
    ) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            input_size,
            hidden_size,
            output_size,
            num_layers,
            bidirectional,
            dropout,
        }
    }
}

/// Save the model with metadata to a file
pub fn save_model_with_metadata<B: Backend>(
    model: &TimeSeriesLstm<B>,
    metadata: ModelMetadata,
    path: impl AsRef<Path>,
) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).context("Failed to create model parent directory")?;
    }
    // Save model artifact
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

/// Load the model and its metadata from a file
pub fn load_model_with_metadata<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(TimeSeriesLstm<B>, ModelMetadata)> {
    // Load metadata first
    let metadata_path = path.as_ref().with_extension("meta.json");
    let metadata_json =
        std::fs::read_to_string(&metadata_path).context("Failed to read metadata file")?;
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_json).context("Failed to parse metadata")?;
    // Now use metadata to construct dummy model
    let model_path = path.as_ref().with_extension("bin");
    let dummy_model = TimeSeriesLstm::new(
        metadata.input_size,
        metadata.hidden_size,
        metadata.output_size,
        metadata.num_layers,
        metadata.bidirectional,
        metadata.dropout,
        device,
    );
    let model = dummy_model
        .load_file::<BinFileRecorder<FullPrecisionSettings>, _>(
            &model_path,
            &Default::default(),
            device,
        )
        .context("Failed to load model")?;
    Ok((model, metadata))
}

/// Save the model to a file (without metadata)
pub fn save_model<B: Backend>(model: &TimeSeriesLstm<B>, path: impl AsRef<Path>) -> Result<()> {
    model
        .clone()
        .save_file::<BinFileRecorder<FullPrecisionSettings>, _>(path.as_ref(), &Default::default())
        .context("Failed to save model")?;
    Ok(())
}

/// Load the model from a file (without metadata)
pub fn load_model<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<TimeSeriesLstm<B>> {
    let dummy_model = TimeSeriesLstm::new(12, 64, 1, 2, false, 0.2, device);
    let model = dummy_model
        .load_file::<BinFileRecorder<FullPrecisionSettings>, _>(
            path.as_ref(),
            &Default::default(),
            device,
        )
        .context("Failed to load model")?;
    Ok(model)
}

/// Check if a model file exists and is valid
pub fn verify_model(path: impl AsRef<Path>) -> Result<bool> {
    let model_path = path.as_ref().with_extension("bin");
    let metadata_path = path.as_ref().with_extension("meta.json");

    // Check if both files exist
    if !model_path.exists() || !metadata_path.exists() {
        return Ok(false);
    }

    // Try to read metadata to verify it's valid
    let metadata_json =
        std::fs::read_to_string(&metadata_path).context("Failed to read metadata file")?;
    let _: ModelMetadata =
        serde_json::from_str(&metadata_json).context("Failed to parse metadata")?;

    Ok(true)
}
