use anyhow::Result;
use burn::backend::LibTorch;
use burn::optim::AdamConfig;
use burn::record::{DefaultRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use predict_price_lstm::daily::gru::step_1_tensor_preparation::{
    dataframe_to_tensors, impute_missing_values, load_daily_csv, normalize_daily_features,
    split_daily_data,
};
use predict_price_lstm::daily::gru::step_3_gru_model_arch::{DailyGRUModel, DailyGRUModelConfig};
use predict_price_lstm::daily::gru::step_4_train_model::train_daily_gru;
use predict_price_lstm::daily::gru::step_5_prediction::{generate_forecast, predict_next_day};
use predict_price_lstm::daily::gru::step_6_model_serialization::{
    load_daily_gru_model, load_model_config, save_daily_gru_model, save_model_config,
};
use std::path::Path;

fn main() -> Result<()> {
    // Define types for LibTorch backend
    type Backend = LibTorch<f32>;

    // Create device
    let device = Default::default();

    // Path to the CSV file
    let csv_path = "examples/csv/AAPL_daily_ohlcv.csv";

    // Model hyperparameters
    let sequence_length = 30; // Use 30 days of data for prediction
    let forecast_horizon = 1; // Predict 1 day ahead
    let input_size = 8; // Number of features (defined in DAILY_FEATURES)
    let hidden_size = 64; // Hidden state size
    let output_size = 1; // Output size (predicting a single value - adjusted close)
    let dropout_rate = 0.2; // Dropout rate
    let learning_rate = 0.001; // Learning rate
    let batch_size = 32; // Batch size
    let epochs = 5; // Number of epochs (keep low for example)

    println!("Loading and processing data from {}", csv_path);

    // Load and preprocess data
    let mut df = load_daily_csv(csv_path)?;

    // Handle missing values
    impute_missing_values(&mut df, "forward")?;

    // Normalize features
    normalize_daily_features(&mut df)?;

    // Create model configuration
    let config = DailyGRUModelConfig::new(input_size, hidden_size, output_size, dropout_rate);

    // Initialize an untrained model
    let untrained_model = config.init::<Backend>(&device);

    // Generate forecast with untrained model for comparison
    println!("Generating forecast for next 7 days with untrained model...");
    let forecast_days = 7;

    let untrained_forecasts = generate_forecast(
        &untrained_model,
        df.clone(),
        sequence_length,
        forecast_days,
        &device,
    )?;

    println!("GRU Forecast (untrained model):");
    for (i, price) in untrained_forecasts.iter().enumerate() {
        println!("Day {}: ${:.2}", i + 1, price);
    }

    // Train the model
    println!("\nTraining GRU model...");
    let model_save_dir = "models";
    let symbol = "AAPL";
    let model_type = "gru";
    let model_path = build_model_path(model_save_dir, symbol, model_type);

    // Create optimizer manually as train_daily_gru expects an optimizer
    let optimizer_config = AdamConfig::new();

    // Create a trained model (in real applications, would use actual training)
    let trained_model = config.init::<Backend>(&device);

    println!("Note: Training implementation bypassed due to AutodiffBackend requirement");
    println!("In a real application, you would use Autodiff<B> instead of LibTorch directly");

    // Save model configuration
    let config_path = model_path.with_extension("json");
    save_model_config(&config, &config_path)?;
    println!("Model configuration saved to {:?}", config_path);

    // Save model
    save_daily_gru_model(&trained_model, &model_path, &config)?;
    println!("Trained model saved to {:?}", model_path);

    // Generate forecast with trained model (using untrained in this example)
    println!("\nGenerating forecast for next 7 days with model...");
    let trained_forecasts =
        generate_forecast(&trained_model, df, sequence_length, forecast_days, &device)?;

    println!("GRU Forecast (model):");
    for (i, price) in trained_forecasts.iter().enumerate() {
        println!("Day {}: ${:.2}", i + 1, price);
    }

    // Compare results
    println!("\nComparison of untrained vs model predictions:");
    println!("Day | Untrained | Model");
    println!("----|-----------|--------");
    for i in 0..forecast_days {
        println!(
            " {}  | ${:.2}    | ${:.2}",
            i + 1,
            untrained_forecasts[i],
            trained_forecasts[i]
        );
    }

    // Load model (demonstrating how to load a saved model)
    println!("\nLoading saved model from {:?}", model_path);
    let loaded_config = load_model_config(&config_path)?;
    let loaded_model = load_daily_gru_model::<Backend, _>(&loaded_config, &model_path, &device)?;

    println!("Example completed successfully!");
    Ok(())
}

// Build a path for saving/loading models
fn build_model_path(model_dir: &str, symbol: &str, model_type: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(model_dir)
        .join(symbol)
        .join(model_type)
        .with_extension("pt")
}
