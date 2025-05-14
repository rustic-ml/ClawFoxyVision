use predict_price_lstm::constants;
use predict_price_lstm::minute::gru::{
    step_4_train_model,
    step_5_prediction,
};
use predict_price_lstm::minute::lstm::step_1_tensor_preparation;
use predict_price_lstm::util::pre_processor;
use predict_price_lstm::util::feature_engineering::add_technical_indicators;
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use polars::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use predict_price_lstm::constants::TECHNICAL_INDICATORS;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GRU Example - Training and prediction with daily MSFT data");
    
    // Setup backend
    type BurnBackend = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();
    println!("Using device: CPU NdArray");
    
    // Load test data from the examples/csv directory
    let csv_path = PathBuf::from("examples/csv/MSFT_daily_ohlcv.csv");
    println!("Loading data from: {}", csv_path.display());
    
    // Load data using our read_csv_to_dataframe function
    let df = pre_processor::read_csv_to_dataframe(
        &csv_path, 
        false, 
        Some(vec!["symbol", "time", "open", "high", "low", "close", "volume", "vwap"])
    )?;
    
    // Check data
    println!("Loaded dataframe with {} rows", df.height());
    println!("Columns: {:?}", df.get_column_names());
    
    // Split into training and testing
    let n_samples = df.height();
    let train_size = (n_samples as f64 * 0.8) as i64;
    let mut train_df = df.slice(0, train_size as usize);
    let mut test_df = df.slice(train_size, (n_samples as i64 - train_size) as usize);
    
    // Add technical indicators
    train_df = add_technical_indicators(&mut train_df)?;
    test_df = add_technical_indicators(&mut test_df)?;
    
    println!("Training dataset size: {} rows", train_df.height());
    println!("Testing dataset size: {} rows", test_df.height());
    
    // Normalize features
    let mut normalized_train = train_df.clone();
    if normalized_train.get_column_names().iter().any(|c| c.as_str() == "_norm_params") {
        normalized_train = normalized_train.drop("_norm_params")?;
    }
    step_1_tensor_preparation::normalize_features(
        &mut normalized_train, 
        &["close", "open", "high", "low"], 
        false,
        false
    )?;
    
    // Prepare data tensors for training
    let forecast_horizon = 5; // Predict 5 days ahead
    
    // Create tensors
    let (features, targets) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
        &normalized_train,
        constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        false,
        None
    )?;
    
    // Configure GRU training
    let training_config = step_4_train_model::TrainingConfig {
        learning_rate: 0.001,
        batch_size: 16,
        epochs: 2,  // Use low number of epochs for example
        test_split: 0.1,
        dropout: constants::DEFAULT_DROPOUT,
        patience: 3,
        min_delta: 0.001,
        use_huber_loss: true,
        checkpoint_epochs: 1,
        bidirectional: true,
        num_layers: 1,
    };
    
    // Train model
    println!("Starting GRU model training...");
    let start_time = Instant::now();
    
    // Train GRU model
    let (trained_gru, _) = step_4_train_model::train_gru_model(
        features, 
        targets, 
        training_config, 
        &device
    )?;
    
    println!("Training completed in {:?}", start_time.elapsed());
    
    // Prepare data for prediction
    println!("Generating predictions for the next {} days...", forecast_horizon);
    
    // Ensure we have exactly the columns needed for prediction
    let mut prediction_input = DataFrame::new(vec![])?;
    
    // Create a DataFrame with only the required columns in the correct order
    for col_name in TECHNICAL_INDICATORS.iter() {
        if test_df.schema().contains(col_name) {
            let series = test_df.column(col_name)?.clone();
            prediction_input.with_column(series)?;
        } else {
            return Err(format!("Required column '{}' not found in test data", col_name).into());
        }
    }
    
    // Verify column count matches expected
    if prediction_input.width() != TECHNICAL_INDICATORS.len() {
        println!("Warning: Column count mismatch. Expected {}, got {}", 
                TECHNICAL_INDICATORS.len(), prediction_input.width());
    }
    
    let predictions = step_5_prediction::predict_multiple_steps(
        &trained_gru,
        prediction_input,
        forecast_horizon,
        &device,
        false
    )?;
    
    println!("Predicted prices for the next {} days:", forecast_horizon);
    for (i, price) in predictions.iter().enumerate() {
        println!("Day {}: ${:.2}", i+1, price);
    }
    
    println!("GRU example completed successfully");
    Ok(())
} 