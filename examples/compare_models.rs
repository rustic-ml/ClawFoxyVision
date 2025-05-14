use predict_price_lstm::constants;
use predict_price_lstm::minute::lstm::{
    step_1_tensor_preparation, 
    step_4_train_model as lstm_train,
    step_5_prediction as lstm_predict,
};
use predict_price_lstm::minute::gru::{
    step_4_train_model as gru_train,
    step_5_prediction as gru_predict,
};
use predict_price_lstm::util::pre_processor;
use predict_price_lstm::util::feature_engineering::add_technical_indicators;

use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Model Comparison Example - LSTM vs GRU with TSLA data");
    
    // Setup backend
    type BurnBackend = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();
    println!("Using device: CPU NdArray");
    
    // Load test data from the examples/csv directory
    let csv_path = PathBuf::from("examples/csv/TSLA_daily_ohlcv.csv");
    println!("Loading data from: {}", csv_path.display());
    
    // Load data using our read_csv_to_dataframe function
    let df = pre_processor::read_csv_to_dataframe(
        &csv_path, 
        false, 
        Some(vec!["symbol", "time", "open", "high", "low", "close", "volume", "vwap"])
    )?;
    
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
    
    // Normalize training data
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
    
    // Create tensors for GRU training
    let forecast_horizon = 5; // Predict 5 days ahead
    let (features, targets) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
        &normalized_train,
        constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        false,
        None
    )?;
    
    // Configure training parameters (shared)
    let training_epochs = 2; // Keep low for example
    let test_percentage = 0.1;
    let batch_size = 16;
    
    // Train LSTM model
    println!("\n=== LSTM Training ===");
    let lstm_start = Instant::now();
    
    let lstm_config = lstm_train::TrainingConfig {
        learning_rate: 0.001,
        batch_size,
        epochs: training_epochs,
        test_split: test_percentage,
        dropout: constants::DEFAULT_DROPOUT,
        patience: 3,
        min_delta: 0.001,
        use_huber_loss: true,
        display_metrics: true,
        display_interval: 1,
    };
    
    let (trained_lstm, _) = lstm_train::train_model(
        normalized_train.clone(), 
        lstm_config, 
        &device, 
        "TSLA", 
        "lstm", 
        forecast_horizon
    )?;
    
    println!("LSTM training completed in {:?}", lstm_start.elapsed());
    
    // Train GRU model
    println!("\n=== GRU Training ===");
    let gru_start = Instant::now();
    
    let gru_config = gru_train::TrainingConfig {
        learning_rate: 0.001,
        batch_size,
        epochs: training_epochs,
        test_split: test_percentage,
        dropout: constants::DEFAULT_DROPOUT,
        patience: 3,
        min_delta: 0.001,
        use_huber_loss: true,
        checkpoint_epochs: 1,
        bidirectional: true,
        num_layers: 1,
    };
    
    let (trained_gru, _) = gru_train::train_gru_model(
        features, 
        targets, 
        gru_config, 
        &device
    )?;
    
    println!("GRU training completed in {:?}", gru_start.elapsed());
    
    // Normalize test data
    let mut normalized_test = test_df.clone();
    if normalized_test.get_column_names().iter().any(|c| c.as_str() == "_norm_params") {
        normalized_test = normalized_test.drop("_norm_params")?;
    }
    step_1_tensor_preparation::normalize_features(
        &mut normalized_test, 
        &["close", "open", "high", "low"], 
        false,
        false
    )?;
    
    // Evaluate both models
    println!("\n=== Model Evaluation ===");
    
    let lstm_rmse = lstm_train::evaluate_model(
        &trained_lstm, 
        normalized_test.clone(), 
        &device, 
        forecast_horizon
    )?;
    
    println!("LSTM Test RMSE: {:.4}", lstm_rmse);
    
    // Create tensors for GRU evaluation
    let (test_features, test_targets) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
        &normalized_test,
        constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        false,
        None
    )?;
    
    let gru_mse = gru_train::evaluate_model(
        &trained_gru, 
        test_features, 
        test_targets
    )?;
    
    println!("GRU Test MSE: {:.4}", gru_mse);
    println!("GRU Test RMSE: {:.4}", gru_mse.sqrt());
    
    // Generate predictions
    println!("\n=== Predictions for the next {} days ===", forecast_horizon);
    
    // Get LSTM predictions
    let lstm_predictions = lstm_predict::ensemble_forecast(
        &trained_lstm, 
        normalized_test.clone(), 
        &device, 
        forecast_horizon
    )?;
    
    // Get GRU predictions
    let gru_predictions = gru_predict::predict_multiple_steps(
        &trained_gru,
        normalized_test.clone(),
        forecast_horizon,
        &device,
        false
    )?;
    
    // Print comparison
    println!("Day | LSTM Prediction | GRU Prediction");
    println!("------------------------------------");
    for i in 0..forecast_horizon {
        println!("{:3} | ${:13.2} | ${:12.2}", 
                i+1, 
                lstm_predictions[i], 
                gru_predictions[i]);
    }
    
    println!("\nModel comparison completed successfully");
    Ok(())
} 