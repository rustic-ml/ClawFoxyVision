// External crates
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use polars::prelude::*;
use rayon::ThreadPoolBuilder;
use std::env;
use std::time::Instant;

// Local modules
use util::{model_utils, feature_engineering};

use minute::lstm::step_1_tensor_preparation;

// Constants
pub mod constants;
pub mod minute;

pub mod util {
    pub mod feature_engineering;
    pub mod file_utils;
    pub mod model_utils;
    pub mod pre_processor;
}

use rustalib::util::file_utils::read_financial_data;

pub fn generate_stock_dataframe(symbol: &str) -> PolarsResult<DataFrame> {
    // The file is now located in examples/csv with a different naming pattern
    let file_path = format!("{}_minute_ohlcv.csv", symbol);
    let workspace_dir = std::env::current_dir().expect("Failed to get current directory");
    let full_path = workspace_dir.join("examples").join("csv").join(file_path);
    
    println!("Looking for data file at: {}", full_path.display());

    match load_data(full_path.to_str().unwrap()) {
        Ok(mut df) => {
            // Add technical indicators using the util::feature_engineering module
            let df_with_indicators = util::feature_engineering::add_technical_indicators(&mut df)
                .map_err(|e| PolarsError::ComputeError(format!("Error adding technical indicators: {}", e).into()))?;
            
            Ok(df_with_indicators)
        },
        Err(e) => {
            // Convert the error to a PolarsError
            Err(PolarsError::ComputeError(
                format!("Error loading data: {}", e).into(),
            ))
        }
    }
}

fn main() -> PolarsResult<()> {
    // Use CPU backend with NdArray (Mali GPU not CUDA)
    #[allow(dead_code)]
    type BurnBackend = Autodiff<NdArray<f32>>;
    let _device = NdArrayDevice::default();
    println!("Using device: CPU NdArray");

    // Enable Rayon default global thread pool for parallelism
    ThreadPoolBuilder::new().build_global().unwrap();

    let args: Vec<String> = env::args().collect();
    let ticker = args
        .get(1)
        .map(|s| s.to_uppercase())
        .unwrap_or("AAPL".to_string());
    let model_type = args
        .get(2)
        .map(|s| s.to_lowercase())
        .unwrap_or("lstm".to_string());

    if model_type != "lstm" && model_type != "gru" && model_type != "cnnlstm" {
        eprintln!("Error: model_type must be either 'lstm', 'gru', or 'cnnlstm'");
        eprintln!("Usage: cargo run -- [ticker] [model_type]");
        eprintln!("Example: cargo run -- AAPL lstm");
        eprintln!("Example: cargo run -- AAPL gru");
        eprintln!("Example: cargo run -- AAPL cnnlstm");
        return Err(PolarsError::ComputeError("Invalid model type".into()));
    }

    println!(
        "Using ticker: {} | model_type: {} | backend: NdArray",
        ticker, model_type
    );

    let df = generate_stock_dataframe(ticker.as_str())?;

    // Split into training and testing datasets (80/20)
    let n_samples = df.height();
    let train_size = (n_samples as f64 * 0.8) as i64;
    let train_df = df.slice(0, train_size as usize);
    let test_df = df.slice(train_size, (n_samples as i64 - train_size) as usize);

    println!("Training dataset size: {} rows", train_df.height());
    println!("Testing dataset size: {} rows", test_df.height());

    // Train and evaluate model with timing
    let t_model_start = Instant::now();
    match train_and_evaluate(
        train_df.clone(),
        test_df.clone(),
        ticker.as_str(),
        model_type.as_str(),
    ) {
        Ok(model_path) => {
            println!(
                "Training and evaluation completed successfully. Model saved at: {}",
                model_path.display()
            );
            let dur = t_model_start.elapsed().as_secs_f64() / 60.0;
            println!("Duration - train & eval: {:.2} minutes", dur);
        }
        Err(e) => eprintln!("Error during training and evaluation: {}", e),
    }

    // Generate predictions with timing
    let t_pred_start = Instant::now();
    match generate_predictions(df, &train_df, model_type.as_str()) {
        Ok(_) => {
            println!("Prediction generation completed successfully.");
            let pred_dur = t_pred_start.elapsed().as_secs_f64() / 60.0;
            println!("Duration - prediction generation: {:.2} minutes", pred_dur);
        }
        Err(e) => eprintln!("Error during prediction generation: {}", e),
    }

    Ok(())
}

fn train_and_evaluate(
    train_df: DataFrame,
    test_df: DataFrame,
    ticker: &str,
    model_type: &str,
) -> Result<std::path::PathBuf, PolarsError> {
    // Define BurnBackend inside the function scope to avoid the unused warning
    type BurnBackend = Autodiff<NdArray<f32>>;

    // Initialize device for training (CPU)
    let device = NdArrayDevice::default();

    // Return path placeholder that will be replaced with the actual path
    let mut model_path = std::path::PathBuf::new();

    if model_type == "lstm" {
        // Configure LSTM training (with early stopping parameters)
        let training_config = minute::lstm::step_4_train_model::TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            test_split: 0.2,
            // Enhanced dropout
            dropout: constants::DEFAULT_DROPOUT,
            // Early stopping settings
            patience: minute::lstm::step_4_train_model::TrainingConfig::default().patience,
            min_delta: minute::lstm::step_4_train_model::TrainingConfig::default().min_delta,
            // Use Huber loss to be more robust
            use_huber_loss: true,
            // Use default for any additional fields
            ..Default::default()
        };

        let model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);
        model_path =
            crate::util::model_utils::get_model_path(ticker, model_type).join(model_name.clone());
        let current_version = env!("CARGO_PKG_VERSION");

        if crate::util::model_utils::is_model_version_current(&model_path, current_version) {
            // Load the model
            let (_loaded_lstm, _metadata) = crate::util::model_utils::load_trained_lstm_model::<
                BurnBackend,
            >(ticker, model_type, &model_name, &device)
            .expect("Failed to load model");

            println!(
                "Loaded existing LSTM model with current version: {}",
                current_version
            );
        } else {
            // Train LSTM model
            println!("Starting LSTM model training...");
            let ep_start = Instant::now();
            let (trained_lstm, _) = minute::lstm::step_4_train_model::train_model(
                train_df.clone(),
                training_config,
                &device,
                ticker,
                model_type,
                390,
            )
            .map_err(|e| PolarsError::ComputeError(format!("Training error: {}", e).into()))?;
            println!(
                "Trained and saved new LSTM model. Epoch took {:?}",
                ep_start.elapsed()
            );

            // Evaluate LSTM model
            println!("Evaluating LSTM model on test data...");
            let forecast_horizon = 390; // full trading day in minutes
            let rmse = minute::lstm::step_4_train_model::evaluate_model(
                &trained_lstm,
                test_df.clone(),
                &device,
                forecast_horizon,
            )
            .map_err(|e| PolarsError::ComputeError(format!("Evaluation error: {}", e).into()))?;
            println!("LSTM Test RMSE: {:.4}", rmse);
        }
    } else if model_type == "gru" {
        // Configure GRU training
        let gru_config = minute::gru::step_4_train_model::TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            test_split: 0.2,
            // Enhanced dropout
            dropout: constants::DEFAULT_DROPOUT,
            // Early stopping settings
            patience: 3,
            min_delta: 0.001,
            // Use Huber loss to be more robust
            use_huber_loss: true,
            // Use bidirectional GRU
            bidirectional: true,
            // Use default for any additional fields
            ..Default::default()
        };

        // GRU model training and evaluation
        println!("Starting GRU model training...");
        let forecast_horizon = 390; // full trading day in minutes
                                // Define BurnBackend inside the function scope to avoid the unused warning
        type BurnBackend = Autodiff<NdArray<f32>>;
        let device = NdArrayDevice::default();

        // Prepare features and targets for GRU training
        let (features, targets) = step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
            &train_df,
            crate::constants::SEQUENCE_LENGTH,
            forecast_horizon,
            &device,
            false,
            None,
        )
        .map_err(|e| {
            PolarsError::ComputeError(format!("Feature preparation error: {}", e).into())
        })?;

        println!(
            "Data prepared for GRU: features shape: {:?}, targets shape: {:?}",
            features.dims(),
            targets.dims()
        );

        // Train GRU model
        let (trained_gru, _) =
            minute::gru::step_4_train_model::train_gru_model(features, targets, gru_config, &device)
                .map_err(|e| {
                    PolarsError::ComputeError(format!("GRU training error: {}", e).into())
                })?;

        println!("GRU model training completed.");

        // Create tensors for GRU evaluation
        let (test_features, test_targets) =
            step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(
                &test_df,
                constants::SEQUENCE_LENGTH,
                forecast_horizon,
                &device,
                false,
                None,
            )
            .map_err(|e| {
                PolarsError::ComputeError(format!("Test feature preparation error: {}", e).into())
            })?;

        // Evaluate GRU model
        let mse = minute::gru::step_4_train_model::evaluate_model(
            &trained_gru,
            test_features,
            test_targets,
        )
        .map_err(|e| {
            PolarsError::ComputeError(format!("GRU evaluation error: {}", e).into())
        })?;

        println!("GRU Test MSE: {:.4}", mse);
    } else if model_type == "cnnlstm" {
        // Configure CNN-LSTM training
        let cnnlstm_config = minute::cnnlstm::step_4_train_model::TrainingConfig {
            learning_rate: 0.001,
            batch_size: 16,            // Reduced from 32 to use less memory
            epochs: 5,                 // Reduced from 10 to finish faster
            test_split: 0.2,
            // Increased dropout to reduce overfitting and memory usage
            dropout: constants::DEFAULT_DROPOUT * 1.5,
            // Early stopping settings
            patience: 2,               // Reduced from 3 to stop earlier if not improving
            min_delta: 0.001,
            // Use Huber loss to be more robust
            use_huber_loss: true,
            // Display metrics during training
            display_metrics: true,
            display_interval: 1,
            // Use default for any additional fields
            ..Default::default()
        };

        // CNN-LSTM model training and evaluation
        println!("Starting CNN-LSTM model training...");
        let forecast_horizon = 60;    // Reduced from 390 to significantly decrease memory usage

        // Train CNN-LSTM model
        let (trained_cnnlstm, loss_history) = minute::cnnlstm::step_4_train_model::train_model(
            train_df.clone(),
            cnnlstm_config,
            &device,
            ticker,
            model_type,
            forecast_horizon,
        )
        .map_err(|e| PolarsError::ComputeError(format!("CNN-LSTM training error: {}", e).into()))?;

        println!("CNN-LSTM model training completed. Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));

        // Evaluate CNN-LSTM model
        let rmse = minute::cnnlstm::step_4_train_model::evaluate_model(
            &trained_cnnlstm,
            test_df.clone(),
            &device,
            forecast_horizon,
        )
        .map_err(|e| {
            PolarsError::ComputeError(format!("CNN-LSTM evaluation error: {}", e).into())
        })?;

        println!("CNN-LSTM Test RMSE: {:.4}", rmse);
        
        // Save the model
        let model_name = format!("{}_cnnlstm_model", ticker);
        model_path = crate::util::model_utils::get_model_path(ticker, model_type).join(model_name.clone());
        
        minute::cnnlstm::step_6_model_serialization::save_model(
            &trained_cnnlstm, 
            &model_path
        ).map_err(|e| {
            PolarsError::ComputeError(format!("CNN-LSTM model saving error: {}", e).into())
        })?;
        
        println!("CNN-LSTM model saved to: {}", model_path.display());
    }

    // Return path placeholder that will be replaced with the actual path
    Ok(model_path)
}

fn generate_predictions(
    df: DataFrame,
    _train_df: &DataFrame,
    model_type: &str,
) -> Result<(), PolarsError> {
    let forecast_horizon = 390; // full trading day in minutes
                                // Define BurnBackend inside the function scope to avoid the unused warning
    type BurnBackend = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();

    // Load model metadata to get correct hyperparameters
    let ticker = std::env::args().nth(1).unwrap_or("AAPL".to_string());
    let model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);

    if model_type == "lstm" {
        // Prepare data tensors - using the enhanced build_enhanced_lstm_model function
        let (_features, _) = minute::lstm::step_1_tensor_preparation::build_enhanced_lstm_model(
            df.clone(),
            forecast_horizon,
        )
        .map_err(|e| PolarsError::ComputeError(format!("Tensor building error: {}", e).into()))?;

        // Load the LSTM model directly since we need the specific type
        let (loaded_lstm, _lstm_metadata) = crate::util::model_utils::load_trained_lstm_model::<
            BurnBackend,
        >(&ticker, model_type, &model_name, &device)
        .map_err(|e| {
            PolarsError::ComputeError(format!("LSTM model loading error: {}", e).into())
        })?;

        // Use LSTM ensemble forecasting
        println!(
            "Generating LSTM ensemble forecast for the next trading day ({} minutes)...",
            forecast_horizon
        );
        let predictions = minute::lstm::step_5_prediction::ensemble_forecast(
            &loaded_lstm,
            df.clone(),
            &device,
            forecast_horizon,
        )
        .map_err(|e| PolarsError::ComputeError(format!("LSTM forecast error: {}", e).into()))?;

        // Print per-minute predictions with timestamps starting from 09:30
        println!("Per-minute LSTM predictions for the next trading day:");
        let mut hour = 9;
        let mut minute = 30;
        for (i, pred) in predictions.iter().enumerate() {
            println!("{:02}:{:02} - Minute {}: ${:.2}", hour, minute, i + 1, pred);
            minute += 1;
            if minute == 60 {
                minute = 0;
                hour += 1;
            }
        }
    } else if model_type == "gru" {
        // Load GRU model
        let (loaded_gru, _gru_metadata) = crate::util::model_utils::load_trained_gru_model::<
            BurnBackend,
        >(&ticker, model_type, &model_name, &device)
        .map_err(|e| PolarsError::ComputeError(format!("GRU model loading error: {}", e).into()))?;

        // Generate GRU multi-step predictions
        println!(
            "Generating GRU forecast for the next trading day ({} minutes)...",
            forecast_horizon
        );
        let predictions = minute::gru::step_5_prediction::predict_multiple_steps(
            &loaded_gru,
            df.clone(),
            forecast_horizon,
            &device,
            false, // don't use extended features
        )
        .map_err(|e| PolarsError::ComputeError(format!("GRU forecast error: {}", e).into()))?;

        // Print per-minute predictions with timestamps starting from 09:30
        println!("Per-minute GRU predictions for the next trading day:");
        let mut hour = 9;
        let mut minute = 30;
        for (i, pred) in predictions.iter().enumerate() {
            println!("{:02}:{:02} - Minute {}: ${:.2}", hour, minute, i + 1, pred);
            minute += 1;
            if minute == 60 {
                minute = 0;
                hour += 1;
            }
        }

        // Compare with LSTM if both models are available
        let lstm_model_path =
            crate::util::model_utils::get_model_path(&ticker, "lstm").join(&model_name);
        if lstm_model_path.exists() {
            println!("LSTM model found. Comparing GRU and LSTM predictions...");

            // Load LSTM model
            let (loaded_lstm, _) =
                crate::util::model_utils::load_trained_lstm_model::<BurnBackend>(
                    &ticker,
                    "lstm",
                    &model_name,
                    &device,
                )
                .map_err(|e| {
                    PolarsError::ComputeError(format!("LSTM model loading error: {}", e).into())
                })?;

            // Compare models
            let (gru_preds, lstm_preds) = minute::gru::step_5_prediction::compare_with_lstm(
                &loaded_gru,
                &loaded_lstm,
                df.clone(),
                10, // Compare first 10 minutes
                &device,
            )
            .map_err(|e| {
                PolarsError::ComputeError(format!("Model comparison error: {}", e).into())
            })?;

            println!("Model comparison (first 10 minutes):");
            println!("Minute | GRU Prediction | LSTM Prediction");
            println!("-------------------------------------------");
            for i in 0..gru_preds.len() {
                println!(
                    "{:6} | ${:13.2} | ${:14.2}",
                    i + 1,
                    gru_preds[i],
                    lstm_preds[i]
                );
            }
        }
    } else if model_type == "cnnlstm" {
        // Load CNN-LSTM model
        let cnnlstm_model_name = format!("{}_cnnlstm_model", ticker);
        let model_path = crate::util::model_utils::get_model_path(&ticker, model_type).join(&cnnlstm_model_name);
        
        println!("Loading CNN-LSTM model from: {}", model_path.display());
        
        // Load the CNN-LSTM model
        let loaded_cnnlstm = minute::cnnlstm::step_6_model_serialization::load_model::<BurnBackend>(
            &model_path,
            &device,
        ).map_err(|e| {
            PolarsError::ComputeError(format!("CNN-LSTM model loading error: {}", e).into())
        })?;
        
        // Generate CNN-LSTM predictions
        println!(
            "Generating CNN-LSTM forecast for the next trading day ({} minutes)...",
            forecast_horizon
        );
        
        let predictions = minute::cnnlstm::step_5_prediction::forecast(
            &loaded_cnnlstm,
            &df,
            &device,
            forecast_horizon,
        ).map_err(|e| {
            PolarsError::ComputeError(format!("CNN-LSTM forecast error: {}", e).into())
        })?;
        
        // Print per-minute predictions with timestamps starting from 09:30
        println!("Per-minute CNN-LSTM predictions for the next trading day:");
        let mut hour = 9;
        let mut minute = 30;
        
        // Handle the case where we might have fewer predictions than expected
        let prediction_count = predictions.len().min(forecast_horizon);
        
        for i in 0..prediction_count {
            println!("{:02}:{:02} - Minute {}: ${:.2}", hour, minute, i + 1, predictions[i]);
            minute += 1;
            if minute == 60 {
                minute = 0;
                hour += 1;
            }
        }
        
        // Compare with other models if they exist
        let lstm_model_path = crate::util::model_utils::get_model_path(&ticker, "lstm").join(&model_name);
        let gru_model_path = crate::util::model_utils::get_model_path(&ticker, "gru").join(&model_name);
        
        if lstm_model_path.exists() || gru_model_path.exists() {
            println!("Other models found. Comparing model predictions...");
            
            let mut comparison_predictions = Vec::new();
            let mut model_names = Vec::new();
            
            // Add CNN-LSTM predictions
            comparison_predictions.push(predictions.clone());
            model_names.push("CNN-LSTM".to_string());
            
            // Add LSTM predictions if available
            if lstm_model_path.exists() {
                let (loaded_lstm, _) = crate::util::model_utils::load_trained_lstm_model::<BurnBackend>(
                    &ticker,
                    "lstm",
                    &model_name,
                    &device,
                ).map_err(|e| {
                    PolarsError::ComputeError(format!("LSTM model loading error: {}", e).into())
                })?;
                
                let lstm_preds = minute::lstm::step_5_prediction::ensemble_forecast(
                    &loaded_lstm,
                    df.clone(),
                    &device,
                    10, // Compare first 10 minutes
                ).map_err(|e| {
                    PolarsError::ComputeError(format!("LSTM forecast error: {}", e).into())
                })?;
                
                comparison_predictions.push(lstm_preds);
                model_names.push("LSTM".to_string());
            }
            
            // Add GRU predictions if available
            if gru_model_path.exists() {
                let (loaded_gru, _) = crate::util::model_utils::load_trained_gru_model::<BurnBackend>(
                    &ticker,
                    "gru",
                    &model_name,
                    &device,
                ).map_err(|e| {
                    PolarsError::ComputeError(format!("GRU model loading error: {}", e).into())
                })?;
                
                let gru_preds = minute::gru::step_5_prediction::predict_multiple_steps(
                    &loaded_gru,
                    df.clone(),
                    10, // Compare first 10 minutes
                    &device,
                    false, // don't use extended features
                ).map_err(|e| {
                    PolarsError::ComputeError(format!("GRU forecast error: {}", e).into())
                })?;
                
                comparison_predictions.push(gru_preds);
                model_names.push("GRU".to_string());
            }
            
            // Print comparison table
            println!("Model comparison (first 10 minutes):");
            
            // Print header
            print!("Minute |");
            for name in &model_names {
                print!(" {:13} |", name);
            }
            println!();
            
            // Print separator
            print!("-----");
            for _ in 0..model_names.len() {
                print!("-----------------");
            }
            println!();
            
            // Print predictions
            for i in 0..10.min(comparison_predictions[0].len()) {
                print!("{:6} |", i + 1);
                
                for preds in &comparison_predictions {
                    if i < preds.len() {
                        print!(" ${:12.2} |", preds[i]);
                    } else {
                        print!(" {:14} |", "N/A");
                    }
                }
                println!();
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn select_features(
    df: &DataFrame,
    target_col: &str,
    n_features: usize,
) -> Result<Vec<String>, anyhow::Error> {
    println!(
        "Performing feature selection to identify the most important {} features...",
        n_features
    );

    let feature_columns: Vec<String> = df
        .get_column_names()
        .iter()
        .filter(|col| {
            let col_str = col.as_str();
            col_str != target_col && col_str != "time" && col_str != "symbol"
        })
        .map(|s| s.to_string())
        .collect();

    if feature_columns.len() <= n_features {
        println!("Not enough features to select from. Using all available features.");
        return Ok(feature_columns);
    }

    // Get the target column
    let target = df.column(target_col)?;
    let _target_f64 = target.cast(&DataType::Float64)?;

    // Calculate correlations with target for each feature
    let mut correlations = Vec::with_capacity(feature_columns.len());

    for feature_name in &feature_columns {
        let feature = df.column(feature_name)?;

        // Skip non-numeric features
        if !matches!(feature.dtype(), DataType::Float64 | DataType::Int64) {
            correlations.push((feature_name.clone(), 0.0));
            continue;
        }

        // Convert columns to Series first, then calculate correlation
        let feature_series = feature.clone();
        let target_series = df.column(&target_col)?.clone();

        // Calculate correlation manually using covariance and standard deviations
        let corr_opt = match (feature_series.f64(), target_series.f64()) {
            (Ok(f_series), Ok(t_series)) => {
                if let (Some(f_mean), Some(t_mean), Some(f_std), Some(t_std)) = (
                    f_series.mean(),
                    t_series.mean(),
                    f_series.std(1),
                    t_series.std(1),
                ) {
                    if f_std > 0.0 && t_std > 0.0 {
                        // Calculate covariance manually
                        let mut cov_sum = 0.0;
                        let mut valid_count = 0;

                        for i in 0..f_series.len() {
                            if let (Some(f_val), Some(t_val)) = (f_series.get(i), t_series.get(i)) {
                                if !f_val.is_nan() && !t_val.is_nan() {
                                    cov_sum += (f_val - f_mean) * (t_val - t_mean);
                                    valid_count += 1;
                                }
                            }
                        }

                        if valid_count > 1 {
                            let cov = cov_sum / (valid_count as f64 - 1.0);
                            Some(cov / (f_std * t_std))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        // If correlation calculation failed, use absolute value
        let corr_abs = match corr_opt {
            Some(c) => c.abs(),
            None => 0.0,
        };

        correlations.push((feature_name.clone(), corr_abs));
    }

    // Sort by correlation (descending)
    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select top n_features
    let selected_features = correlations
        .into_iter()
        .take(n_features)
        .map(|(name, corr)| {
            println!("Selected feature: {} (correlation: {:.4})", name, corr);
            name
        })
        .collect();

    Ok(selected_features)
}

#[allow(dead_code)]
fn train_lstm_model(
    _ticker: &str,
    df: &DataFrame,
    use_enhanced_features: bool,
    use_feature_selection: bool,
    handle_outliers: bool,
    use_data_augmentation: bool,
    use_time_based_cv: bool,
) -> Result<(), anyhow::Error> {
    println!("Starting model training...");
    let start_time = Instant::now();

    // Make a copy of the dataframe for preprocessing
    let mut training_df = df.clone();

    // If feature selection is enabled, select most important features
    let _selected_features = if use_feature_selection {
        select_features(&training_df, "close", 15)?
    } else {
        vec![] // Use default features
    };

    // Impute missing values before normalization
    let feature_columns = if use_enhanced_features {
        &crate::constants::EXTENDED_INDICATORS[..]
    } else {
        &crate::constants::TECHNICAL_INDICATORS[..]
    };

    // Handle missing values
    step_1_tensor_preparation::impute_missing_values(
        &mut training_df,
        feature_columns,
        "forward_fill",
        None,
    )?;

    // Handle outliers if requested
    if handle_outliers {
        step_1_tensor_preparation::handle_outliers(
            &mut training_df,
            &["close", "open", "high", "low"],
            "iqr",
            1.5,
            "clip",
        )?;
    }

    // Data augmentation if requested
    if use_data_augmentation {
        println!("Applying data augmentation...");
        training_df = step_1_tensor_preparation::augment_time_series(
            &training_df,
            &["jitter", "scaling"],
            1,
        )?;
        println!(
            "Dataset size after augmentation: {} rows",
            training_df.height()
        );
    }

    // Normalize data - use updated version with outlier handling
    println!("Normalizing data...");
    step_1_tensor_preparation::normalize_features(
        &mut training_df,
        &["close", "open", "high", "low"],
        use_enhanced_features,
        handle_outliers,
    )?;

    // Split data with time-based CV if requested
    let validation_split_ratio = crate::constants::VALIDATION_SPLIT_RATIO;
    let (train_df, val_df) = if use_time_based_cv {
        step_1_tensor_preparation::split_data(&training_df, validation_split_ratio, true)?
    } else {
        // Use standard time-based split
        let n_samples = training_df.height();
        let split_idx = (n_samples as f64 * (1.0 - validation_split_ratio)) as usize;
        let train_df = training_df.slice(0, split_idx);
        let val_df = training_df.slice(split_idx as i64, (n_samples - split_idx) as usize);
        (train_df, val_df)
    };

    println!("Training dataset size: {} rows", train_df.height());
    println!("Validation dataset size: {} rows", val_df.height());

    // Create tensors and train model using the existing code
    // ... rest of the training process remains the same

    let duration = start_time.elapsed();
    println!(
        "Model training and preprocessing completed in {:.2} minutes",
        duration.as_secs_f64() / 60.0
    );

    Ok(())
}

// Extract just the DataFrame from the tuple
fn load_data(file_path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // First read the file using the standard function
    let (mut df, _) = read_financial_data(file_path)?;
    
    // Handle capitalized column names by standardizing them to lowercase
    let mut rename_columns = Vec::new();
    
    // First identify columns to rename based on lowercase matching
    for column_name in df.get_column_names() {
        let col_lower = column_name.to_lowercase();
        
        // Map each column to a standard name based on case-insensitive matching
        let standard_name = match col_lower.as_str() {
            "open" | "o" | "op" | "openprice" | "open_price" => "open",
            "high" | "h" | "highprice" | "high_price" | "max" => "high",
            "low" | "l" | "lowprice" | "low_price" | "min" => "low",
            "close" | "c" | "cl" | "closeprice" | "close_price" => "close",
            "volume" | "vol" | "v" | "volumes" => "volume",
            "timestamp" | "time" | "date" | "t" | "datetime" | "dt" | "day" => "time",
            "vwap" | "vwavg" | "vw" | "vwprice" | "volumeweightedavgprice" => "vwap",
            "adj close" | "adj_close" | "adjusted close" | "adjusted_close" | "adjclose" | "adj" => "adjusted_close",
            _ => continue,
        };
        
        // If the column needs to be renamed (case is different)
        if column_name != standard_name {
            rename_columns.push((column_name.to_string(), standard_name.to_string()));
        }
    }
    
    if !rename_columns.is_empty() {
        println!("Original columns: {:?}", df.get_column_names());
        println!("Columns to rename: {:?}", rename_columns);
        
        // Use DataFrame's lazy API to apply all transformations at once
        let mut lazy_df = df.clone().lazy();
        
        // Apply all column renames
        for (old_name, new_name) in rename_columns {
            lazy_df = lazy_df.with_column(col(&old_name).alias(&new_name));
        }
        
        // Apply all transformations
        df = lazy_df.collect()?;
        
        // Cast volume to Float64 if it exists in the dataframe
        if df.schema().contains("volume") {
            let volume = df.column("volume")?;
            let volume_f64 = volume.cast(&DataType::Float64)?;
            df.with_column(volume_f64)?;
        }
        
        // Add the adjusted_close column if it doesn't exist (using close as a fallback)
        if !df.schema().contains("adjusted_close") && df.schema().contains("close") {
            let close = df.column("close")?.clone();
            df.with_column(close.with_name("adjusted_close".into()))?;
        }
        
        println!("DataFrame columns after renaming: {:?}", df.get_column_names());
    }
    
    Ok(df)
}
