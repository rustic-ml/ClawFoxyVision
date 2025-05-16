use predict_price_lstm::constants;
use predict_price_lstm::minute::gru::{
    step_4_train_model as gru_train,
};
use predict_price_lstm::minute::lstm::{
    step_1_tensor_preparation, step_4_train_model as lstm_train,
};
use predict_price_lstm::util::feature_engineering::add_technical_indicators;
use predict_price_lstm::util::pre_processor;
use predict_price_lstm::util::file_utils;

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
    let mut df = predict_price_lstm::util::file_utils::read_csv_file(&csv_path)?;

    use polars::prelude::*;
    println!("DataFrame columns: {:?}", df.get_column_names());
    println!("DataFrame shape: {:?}", df.shape());
    
    // Create a mapping of columns we need to standardize
    let mut rename_columns = Vec::new();
    
    // First identify columns to rename based on lowercase matching
    for column_name in df.get_column_names() {
        let col_lower = column_name.to_lowercase();
        
        // Map each column to a standard name based on case-insensitive matching
        let standard_name = match col_lower.as_str() {
            "open" | "o" => "open",
            "high" | "h" => "high",
            "low" | "l" => "low",
            "close" | "c" => "close",
            "volume" | "vol" | "v" => "volume",
            "timestamp" | "time" | "date" | "t" => "time",
            "vwap" => "vwap",
            _ => continue,
        };
        
        // If the column needs to be renamed (case is different)
        if column_name != standard_name {
            rename_columns.push((column_name.to_string(), standard_name.to_string()));
        }
    }
    
    println!("Columns to rename: {:?}", rename_columns);
    
    // Check if we need to add a symbol column 
    let has_symbol = df.get_column_names().iter().any(|name| name.to_lowercase() == "symbol");
    
    // Use DataFrame's lazy API to apply all transformations at once
    let mut lazy_df = df.clone().lazy();
    
    // Apply all column renames
    for (old_name, new_name) in rename_columns {
        lazy_df = lazy_df.with_column(col(&old_name).alias(&new_name));
    }
    
    // Add symbol column if needed
    if !has_symbol {
        lazy_df = lazy_df.with_column(lit("TSLA").alias("symbol"));
    }
    
    // Apply all transformations
    df = lazy_df.collect()?;
    
    // Now we need to select only the needed columns, keeping only one instance of each 
    // standardized column (the renamed ones)
    let keep_columns = vec!["open", "high", "low", "close", "volume", "vwap", "time", "symbol"];
    let available_columns = df.get_column_names();
    
    // Convert keep_columns to String to match available_columns
    let available_col_strings: Vec<String> = available_columns.iter()
        .map(|c| c.to_string())
        .collect();
    
    // Find columns that exist in our dataset
    let mut final_columns = Vec::new();
    for &col_name in &keep_columns {
        if available_col_strings.contains(&col_name.to_string()) {
            final_columns.push(col_name);
        }
    }
    
    // Convert to strings for select
    let final_column_strs: Vec<&str> = final_columns;
    df = df.select(final_column_strs)?;
    
    println!("DataFrame columns after renaming: {:?}", df.get_column_names());

    // Split into training and testing
    let n_samples = df.height();
    let train_size = (n_samples as f64 * 0.8) as i64;
    let mut train_df = df.slice(0, train_size as usize);
    let mut test_df = df.slice(train_size, (n_samples as i64 - train_size) as usize);

    println!("Training dataset size: {} rows", train_df.height());
    println!("Testing dataset size: {} rows", test_df.height());

    // Add technical indicators
    train_df = add_technical_indicators(&mut train_df)?;
    test_df = add_technical_indicators(&mut test_df)?;
    
    // The technical indicators function now ensures all columns have the same length,
    // but there may still be null values at the beginning of the timeseries for indicators
    // with longer windows, so we drop all null values to ensure clean data
    train_df = train_df.drop_nulls::<String>(None)?;
    test_df = test_df.drop_nulls::<String>(None)?;

    println!("Training dataset size after adding indicators: {} rows", train_df.height());
    println!("Testing dataset size after adding indicators: {} rows", test_df.height());

    // Normalize training data
    let mut normalized_train = train_df.clone();
    // Check and remove any existing _norm_params column before normalization
    if normalized_train
        .get_column_names()
        .iter()
        .any(|c| c.as_str() == "_norm_params")
    {
        normalized_train = normalized_train.drop("_norm_params")?;
    }
    step_1_tensor_preparation::normalize_features(
        &mut normalized_train,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    // Create tensors for GRU training
    let forecast_horizon = 5; // Predict 5 days ahead
    let (features, targets) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
        &normalized_train,
        constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        false,
        None,
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
        forecast_horizon,
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

    let (trained_gru, _) = gru_train::train_gru_model(features, targets, gru_config, &device)?;

    println!("GRU training completed in {:?}", gru_start.elapsed());

    // Normalize test data
    let mut normalized_test = test_df.clone();
    // Check and remove any existing _norm_params column before normalization
    if normalized_test
        .get_column_names()
        .iter()
        .any(|c| c.as_str() == "_norm_params")
    {
        normalized_test = normalized_test.drop("_norm_params")?;
    }
    step_1_tensor_preparation::normalize_features(
        &mut normalized_test,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    // Evaluate both models
    println!("\n=== Model Evaluation ===");

    let lstm_rmse = lstm_train::evaluate_model(
        &trained_lstm,
        normalized_test.clone(),
        &device,
        forecast_horizon,
    )?;

    println!("LSTM Test RMSE: {:.4}", lstm_rmse);

    // Create tensors for GRU evaluation
    let (test_features, test_targets) =
        step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
            &normalized_test,
            constants::SEQUENCE_LENGTH,
            forecast_horizon,
            &device,
            false,
            None,
        )?;

    let gru_mse = gru_train::evaluate_model(&trained_gru, test_features, test_targets)?;

    println!("GRU Test MSE: {:.4}", gru_mse);
    println!("GRU Test RMSE: {:.4}", gru_mse.sqrt());

    // Generate predictions
    println!(
        "\n=== Predictions for the next {} days ===",
        forecast_horizon
    );

    // Use a simpler direct prediction approach for LSTM to avoid the ensemble_forecast issue
    let mut normalized_test_for_pred = test_df.clone();
    
    // Remove _norm_params if it exists
    if normalized_test_for_pred
        .get_column_names()
        .iter()
        .any(|c| c.as_str() == "_norm_params")
    {
        normalized_test_for_pred = normalized_test_for_pred.drop("_norm_params")?;
    }
    
    // Normalize the test data
    step_1_tensor_preparation::normalize_features(
        &mut normalized_test_for_pred,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;
    
    // Get standard feature columns
    let indicator_names: Vec<String> = constants::TECHNICAL_INDICATORS
        .iter()
        .map(|&s| s.to_string())
        .collect();
    
    // Select only technical indicators and drop nulls
    let df_for_pred = normalized_test_for_pred
        .select(&indicator_names)?
        .drop_nulls::<String>(None)?;
    
    // Prepare data for LSTM prediction
    let sequence_length = constants::SEQUENCE_LENGTH;
    let n_features = constants::TECHNICAL_INDICATORS.len();
    let n_rows = df_for_pred.height();
    
    if n_rows < sequence_length {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Not enough data for prediction (need at least {} rows)", sequence_length),
        )));
    }
    
    // Extract the last sequence_length rows
    let start = n_rows - sequence_length;
    let mut feature_buffer = Vec::with_capacity(sequence_length * n_features);
    
    // Populate buffer
    for row in start..n_rows {
        for &col in constants::TECHNICAL_INDICATORS.iter() {
            let val = df_for_pred.column(col)?.f64()?.get(row).unwrap_or(0.0) as f32;
            feature_buffer.push(val);
        }
    }
    
    // Create tensor with correct shape
    let shape = burn::tensor::Shape::new([1, sequence_length, n_features]);
    let features_tensor = burn::tensor::Tensor::<BurnBackend, 1>::from_floats(
        feature_buffer.as_slice(), 
        &device
    ).reshape(shape);
    
    // Run prediction with LSTM model
    let output = trained_lstm.forward(features_tensor);
    let predictions_data = output.to_data().convert::<f32>();
    let predictions_slice = predictions_data.as_slice::<f32>().unwrap();
    
    // Convert to Vec<f64>
    let lstm_predictions: Vec<f64> = predictions_slice.iter().map(|&v| v as f64).collect();
    
    // Use the same approach for GRU predictions
    let mut normalized_test_for_gru = test_df.clone();
    
    // Remove _norm_params if it exists
    if normalized_test_for_gru
        .get_column_names()
        .iter()
        .any(|c| c.as_str() == "_norm_params")
    {
        normalized_test_for_gru = normalized_test_for_gru.drop("_norm_params")?;
    }
    
    // Normalize the test data for GRU
    step_1_tensor_preparation::normalize_features(
        &mut normalized_test_for_gru,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;
    
    // Prepare tensors for GRU prediction (similar to what we did for training)
    let (gru_features, _) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
        &normalized_test_for_gru,
        constants::SEQUENCE_LENGTH,
        forecast_horizon,
        &device,
        false,
        None,
    )?;
    
    // Get predictions from the GRU model
    let gru_output = trained_gru.forward(gru_features);
    let gru_data = gru_output.to_data().convert::<f32>();
    let gru_slice = gru_data.as_slice::<f32>().unwrap();
    
    // Convert to Vec<f64>
    let gru_predictions: Vec<f64> = gru_slice.iter().map(|&v| v as f64).collect();

    // Print comparison
    println!("Day | LSTM Prediction | GRU Prediction");
    println!("------------------------------------");
    for i in 0..forecast_horizon {
        println!(
            "{:3} | ${:13.2} | ${:12.2}",
            i + 1,
            lstm_predictions[i],
            gru_predictions[i]
        );
    }

    println!("\nModel comparison completed successfully");
    Ok(())
}

