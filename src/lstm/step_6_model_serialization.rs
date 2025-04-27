use anyhow::{Context, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::SystemTime;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use super::step_3_lstm_model_arch::TimeSeriesLstm;

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub version: String,
    pub timestamp: u64,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub dropout: f64,
}

impl ModelMetadata {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
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
        std::fs::create_dir_all(parent)
            .context("Failed to create model parent directory")?;
    }
    // Save model artifact
    let model_path = path.as_ref().with_extension("bin");
    model.clone().save_file::<BinFileRecorder<FullPrecisionSettings>, _>(&model_path, &Default::default())
        .context("Failed to save model")?;
    // Save metadata
    let metadata_path = path.as_ref().with_extension("meta.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .context("Failed to serialize metadata")?;
    std::fs::write(&metadata_path, metadata_json)
        .context("Failed to write metadata file")?;
    Ok(())
}

/// Load the model and its metadata from a file
pub fn load_model_with_metadata<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(TimeSeriesLstm<B>, ModelMetadata)> {
    // Load model
    let model_path = path.as_ref().with_extension("bin");
    let dummy_model = TimeSeriesLstm::new(12, 64, 1, 2, false, 0.2, device);
    let model = dummy_model.load_file::<BinFileRecorder<FullPrecisionSettings>, _>(&model_path, &Default::default(), device)
        .context("Failed to load model")?;
    // Load metadata
    let metadata_path = path.as_ref().with_extension("meta.json");
    let metadata_json = std::fs::read_to_string(&metadata_path)
        .context("Failed to read metadata file")?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
        .context("Failed to parse metadata")?;
    Ok((model, metadata))
}

/// Save the model to a file (without metadata)
pub fn save_model<B: Backend>(
    model: &TimeSeriesLstm<B>,
    path: impl AsRef<Path>,
) -> Result<()> {
    model.clone().save_file::<BinFileRecorder<FullPrecisionSettings>, _>(path.as_ref(), &Default::default())
        .context("Failed to save model")?;
    Ok(())
}

/// Load the model from a file (without metadata)
pub fn load_model<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<TimeSeriesLstm<B>> {
    let dummy_model = TimeSeriesLstm::new(12, 64, 1, 2, false, 0.2, device);
    let model = dummy_model.load_file::<BinFileRecorder<FullPrecisionSettings>, _>(path.as_ref(), &Default::default(), device)
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
    let metadata_json = std::fs::read_to_string(&metadata_path)
        .context("Failed to read metadata file")?;
    let _: ModelMetadata = serde_json::from_str(&metadata_json)
        .context("Failed to parse metadata")?;

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use std::fs;
    use tempfile::tempdir;

    fn create_test_model(device: &NdArrayDevice) -> (TimeSeriesLstm<NdArray>, ModelMetadata) {
        let input_size = 12;
        let hidden_size = 64;
        let num_layers = 2;
        let bidirectional = false;
        let dropout = 0.2;

        let model = TimeSeriesLstm::new(
            input_size,
            hidden_size,
            1,  // output_size
            num_layers,
            bidirectional,
            dropout,
            device,
        );

        let metadata = ModelMetadata::new(
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
        );

        (model, metadata)
    }

    #[test]
    fn test_model_serialization() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("test_model.bin");
        let device = NdArrayDevice::Cpu;
        let model = TimeSeriesLstm::<NdArray>::new(12, 64, 1, 2, false, 0.2, &device);

        save_model(&model, &model_path)?;
        assert!(model_path.exists());

        let _loaded_model: TimeSeriesLstm<NdArray> = load_model(&model_path, &device)?;

        fs::remove_file(&model_path)?;
        temp_dir.close()?;
        Ok(())
    }

    #[test]
    fn test_model_serialization_with_metadata() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_base_path = temp_dir.path().join("test_model");
        let device = NdArrayDevice::Cpu;
        
        let (model, metadata) = create_test_model(&device);

        // Save model with metadata
        save_model_with_metadata(&model, metadata.clone(), &model_base_path)?;

        // Verify files exist
        assert!(model_base_path.with_extension("bin").exists());
        assert!(model_base_path.with_extension("meta.json").exists());

        // Load model and metadata with explicit type annotation
        let (_loaded_model, loaded_metadata): (TimeSeriesLstm<NdArray>, ModelMetadata) =
            load_model_with_metadata(&model_base_path, &device)?;
        // Verify metadata consistency
        assert_eq!(loaded_metadata.input_size, metadata.input_size);
        assert_eq!(loaded_metadata.hidden_size, metadata.hidden_size);
        assert_eq!(loaded_metadata.num_layers, metadata.num_layers);
        assert_eq!(loaded_metadata.bidirectional, metadata.bidirectional);
        assert!((loaded_metadata.dropout - metadata.dropout).abs() < f32::EPSILON.into());

        assert!(verify_model(&model_base_path)?);

        temp_dir.close()?;
        Ok(())
    }
} 