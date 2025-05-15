use predict_price_lstm::constants;
use predict_price_lstm::minute::lstm::{
    step_1_tensor_preparation, step_4_train_model, step_5_prediction,
};
use predict_price_lstm::util::feature_engineering::add_technical_indicators;
use predict_price_lstm::util::pre_processor;

use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LSTM Example - Training and prediction with daily AAPL data");

    // Setup backend
    type BurnBackend = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();
    println!("Using device: CPU NdArray");

    // Load test data from the examples/csv directory
    let csv_path = PathBuf::from("examples/csv/AAPL_daily_ohlcv.csv");
    println!("Loading data from: {}", csv_path.display());

    // Load data using our read_csv_to_dataframe function
    let df = pre_processor::read_csv_to_dataframe(
        &csv_path,
        false,
        Some(vec![
            "symbol", "time", "open", "high", "low", "close", "volume", "vwap",
        ]),
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
    step_1_tensor_preparation::normalize_features(
        &mut normalized_train,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    // Configure LSTM training
    let training_config = step_4_train_model::TrainingConfig {
        learning_rate: 0.001,
        batch_size: 16,
        epochs: 2, // Use low number of epochs for example
        test_split: 0.1,
        dropout: constants::DEFAULT_DROPOUT,
        patience: 3,
        min_delta: 0.001,
        use_huber_loss: true,
        display_metrics: true,
        display_interval: 1,
    };

    // Train model
    println!("Starting LSTM model training...");
    let forecast_horizon = 5; // Predict 5 days ahead
    let start_time = Instant::now();

    // Train the model
    let (trained_lstm, _) = step_4_train_model::train_model(
        normalized_train.clone(),
        training_config,
        &device,
        "AAPL",
        "lstm",
        forecast_horizon,
    )?;

    println!("Training completed in {:?}", start_time.elapsed());

    // Evaluate on test data
    let mut normalized_test = test_df.clone();
    step_1_tensor_preparation::normalize_features(
        &mut normalized_test,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    println!("Evaluating LSTM model on test data...");
    let rmse = step_4_train_model::evaluate_model(
        &trained_lstm,
        normalized_test.clone(),
        &device,
        forecast_horizon,
    )?;

    println!("LSTM Test RMSE: {:.4}", rmse);

    // Generate predictions
    println!(
        "Generating predictions for the next {} days...",
        forecast_horizon
    );
    let predictions = step_5_prediction::ensemble_forecast(
        &trained_lstm,
        normalized_test.clone(),
        &device,
        forecast_horizon,
    )?;

    println!("Predicted prices for the next {} days:", forecast_horizon);
    for (i, price) in predictions.iter().enumerate() {
        println!("Day {}: ${:.2}", i + 1, price);
    }

    println!("LSTM example completed successfully");
    Ok(())
}
