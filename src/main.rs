// External crates
use burn::backend::LibTorch;
use burn::tensor::backend::Backend as BurnBackendTrait;
use polars::prelude::*;
use std::env;

// Local modules
use util::{feature_engineering, pre_processor};

use lstm::{
    step_1_tensor_preparation, 
    step_3_lstm_model_arch, 
    step_5_prediction,
};

// Constants
pub mod constants;

pub mod util {
    pub mod feature_engineering;
    pub mod pre_processor;
    pub mod model_utils;
}

pub mod lstm {
    pub mod step_1_tensor_preparation;
    pub mod step_2_lstm_cell;
    pub mod step_3_lstm_model_arch;
    pub mod step_4_train_model;
    pub mod step_5_prediction;
    pub mod step_6_model_serialization;
}

pub fn generate_stock_dataframe(symbol: &str) -> PolarsResult<DataFrame> {
    let file_path = format!("{}-ticker_minute_bars.csv", symbol);
    let workspace_dir = std::env::current_dir().expect("Failed to get current directory");
    let full_path = workspace_dir.join(file_path);

    let mut ohlc_df = pre_processor::load_and_preprocess(&full_path)
        .map_err(|e| PolarsError::ComputeError(format!("Preprocessing error: {}", e).into()))?;

    let preprocessed_df = feature_engineering::add_technical_indicators(&mut ohlc_df)?;
    Ok(preprocessed_df)
}

fn main() -> PolarsResult<()> {
    // Accept ticker and model_type as command-line arguments
    let args: Vec<String> = env::args().collect();
    let ticker = args.get(1).map(|s| s.to_uppercase()).unwrap_or("AAPL".to_string());
    let model_type = args.get(2).map(|s| s.to_lowercase()).unwrap_or("lstm".to_string());
    println!("Using ticker: {} | model_type: {}", ticker, model_type);

    let df = generate_stock_dataframe(ticker.as_str())?;

    // Split into training and testing datasets (80/20)
    let n_samples = df.height();
    let train_size = (n_samples as f64 * 0.8) as i64;
    let train_df = df.slice(0, train_size as usize);
    let test_df = df.slice(train_size, (n_samples as i64 - train_size) as usize);

    println!("Training dataset size: {} rows", train_df.height());
    println!("Testing dataset size: {} rows", test_df.height());

    // Train and evaluate model
    match train_and_evaluate(train_df.clone(), test_df.clone(), ticker.as_str(), model_type.as_str()) {
        Ok(model_path) => println!("Training and evaluation completed successfully. Model saved at: {}", model_path.display()),
        Err(e) => eprintln!("Error during training and evaluation: {}", e),
    }

    // Generate predictions
    match generate_predictions(df) {
        Ok(_) => println!("Prediction generation completed successfully."),
        Err(e) => eprintln!("Error during prediction generation: {}", e),
    }

    Ok(())
}

fn train_and_evaluate(train_df: DataFrame, test_df: DataFrame, ticker: &str, model_type: &str) -> Result<std::path::PathBuf, PolarsError> {
    // Initialize device
    type BurnBackend = LibTorch<f32>;
    let device = <BurnBackend as BurnBackendTrait>::Device::default();

    // Create training configuration
    let training_config = lstm::step_4_train_model::TrainingConfig {
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 20,
        test_split: 0.2,
    };

    let model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);
    let model_path = crate::util::model_utils::get_model_path(ticker, model_type).join(model_name);
    let current_version = env!("CARGO_PKG_VERSION");

    if crate::util::model_utils::is_model_version_current(&model_path, current_version) {
        let model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);
        
        // Load the model
        let (trained_model, _metadata) = crate::util::model_utils::load_trained_model::<BurnBackend>(
            ticker,
            model_type,
            &model_name,
            &device,
        )
        .expect("Failed to load model");
        println!("Loaded existing model with current version: {}", current_version);
    } else {
        // Train model
        println!("Starting model training...");
        let (trained_model, _) =
            lstm::step_4_train_model::train_model::<BurnBackend>(train_df.clone(), training_config, &device, ticker, model_type)
                .map_err(|e| PolarsError::ComputeError(format!("Training error: {}", e).into()))?;
        println!("Trained and saved new model.");
    }

    // Evaluate model
    println!("Evaluating model on test data...");
    let model_name = format!("{}{}", ticker, constants::MODEL_FILE_NAME);   
    // Load the trained model for evaluation (ensures `trained_model` is in scope)
    let (trained_model, _metadata) = crate::util::model_utils::load_trained_model::<BurnBackend>(
        ticker,
        model_type,
        &model_name,
        &device
    ).map_err(|e| PolarsError::ComputeError(format!("Model loading error: {}", e).into()))?;
    // Perform evaluation
    let rmse = lstm::step_4_train_model::evaluate_model(&trained_model, test_df.clone(), &device)
        .map_err(|e| PolarsError::ComputeError(format!("Evaluation error: {}", e).into()))?;
    println!("Test RMSE: {:.4}", rmse);

    // Return the path to the saved model
    Ok(model_path)
}

fn generate_predictions(df: DataFrame) -> Result<(), PolarsError> {
    // Initialize device
    type BurnBackend = LibTorch<f32>;
    let device = <BurnBackend as BurnBackendTrait>::Device::default();

    // Prepare data tensors
    let (features, _) = step_1_tensor_preparation::build_burn_lstm_model(df.clone())
        .map_err(|e| PolarsError::ComputeError(format!("Tensor building error: {}", e).into()))?;

    // Initialize model
    let input_dim = features.dims()[2];
    let hidden_dim = 64; // hidden_dim
    let output_dim = 1; // typically 1 for time series prediction
    let num_layers = 2; // number of LSTM layers
    let bidirectional = true; // use bidirectional LSTM
    let dropout = 0.2; // dropout probability

    // Create model with explicit parameters instead of using factory function
    let model: step_3_lstm_model_arch::TimeSeriesLstm<BurnBackend> =
        step_3_lstm_model_arch::TimeSeriesLstm::new(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            bidirectional,
            dropout,
            &device,
        );

    // Generate 5-day forecast
    println!("Generating 5-day forecast...");
    let forecast_horizon = 5;
    let predictions =
        step_5_prediction::generate_forecast(&model, df.clone(), forecast_horizon, &device)
            .map_err(|e| PolarsError::ComputeError(format!("Forecast error: {}", e).into()))?;

    // Denormalize predictions to original scale
    let denormalized = step_5_prediction::denormalize_predictions(predictions, &df, "close")
        .map_err(|e| PolarsError::ComputeError(format!("Denormalization error: {}", e).into()))?;

    // Print predictions
    println!("Predictions for the next {} days:", forecast_horizon);
    for (i, pred) in denormalized.iter().enumerate() {
        println!("Day {}: ${:.2}", i + 1, pred);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_build_burn_lstm_model_with_df_from_csv() {
    //     // Read and preprocess the CSV using step_1 and step_2
    //     let cwd = std::env::current_dir().expect("Failed to get current directory");
    //     let csv_path = cwd.join("AAPL-ticker_minute_bars.csv");

    //     // Load and preprocess
    //     let mut df = match crate::util::pre_processor::load_and_preprocess(&csv_path) {
    //         Ok(df) => df,
    //         Err(e) => {
    //             panic!("Failed to load and preprocess data: {}", e);
    //         }
    //     };

    //     // Add technical indicators
    //     let df = match crate::util::feature_engineering::add_technical_indicators(&mut df) {
    //         Ok(df) => df,
    //         Err(e) => {
    //             panic!("Failed to add technical indicators: {}", e);
    //         }
    //     };

    //     // Verify required indicators exist before tensor conversion
    //     for indicator in constants::TECHNICAL_INDICATORS {
    //         assert!(
    //             df.schema().contains(indicator), 
    //             "Missing required indicator: {}",
    //             indicator
    //         );
    //     }

    //     // Use our custom build function with smaller validation split
    //     let result = build_burn_lstm_model_for_test(df, 0.1);
    //     match &result {
    //         Ok(_) => println!("CSV test succeeded"),
    //         Err(e) => {
    //             println!("CSV test Error: {:?}", e);
    //             panic!("Failed to build LSTM model tensors: {}", e);
    //         }
    //     };

    //     // Verify tensors were created with correct dimensions
    //     let (features, targets) = result.expect("Failed to build model");
        
    //     // Validate the dimensionality and shape of the tensors
    //     assert_eq!(
    //         features.dims().len(),
    //         3,
    //         "Features tensor should have 3 dimensions"
    //     );
        
    //     // Check that the 3rd dimension has the expected number of features
    //     assert_eq!(
    //         features.dims()[2],
    //         constants::TECHNICAL_INDICATORS.len(),
    //         "Features tensor should have {} features",
    //         constants::TECHNICAL_INDICATORS.len()
    //     );
        
    //     // Check that the 2nd dimension has the expected sequence length
    //     assert_eq!(
    //         features.dims()[1],
    //         constants::SEQUENCE_LENGTH,
    //         "Features tensor should have sequence length of {}",
    //         constants::SEQUENCE_LENGTH
    //     );
        
    //     // Validate targets tensor shape
    //     assert_eq!(
    //         targets.dims().len(),
    //         2,
    //         "Targets tensor should have 2 dimensions"
    //     );
        
    //     // Ensure number of samples in features and targets match
    //     assert_eq!(
    //         features.dims()[0],
    //         targets.dims()[0],
    //         "Number of samples in features and targets should match"
    //     );
        
    //     // Verify targets has correct second dimension (should be 1 for single target value)
    //     assert_eq!(
    //         targets.dims()[1],
    //         1,
    //         "Targets tensor second dimension should be 1"
    //     );
    // }
}
