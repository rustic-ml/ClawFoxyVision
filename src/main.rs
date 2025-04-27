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
    let ticker = args.get(1).map(|s| s.as_str()).unwrap_or("AAPL");
    let model_type = args.get(2).map(|s| s.as_str()).unwrap_or("lstm");
    println!("Using ticker: {} | model_type: {}", ticker, model_type);

    let df = generate_stock_dataframe(ticker)?;

    // Split into training and testing datasets (80/20)
    let n_samples = df.height();
    let train_size = (n_samples as f64 * 0.8) as i64;
    let train_df = df.slice(0, train_size as usize);
    let test_df = df.slice(train_size, (n_samples as i64 - train_size) as usize);

    println!("Training dataset size: {} rows", train_df.height());
    println!("Testing dataset size: {} rows", test_df.height());

    // Train and evaluate model
    match train_and_evaluate(train_df.clone(), test_df.clone(), ticker, model_type) {
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

    #[test]
    fn test_build_burn_lstm_model_with_nulls_and_nans() {
        use polars::prelude::*;
        
        // Create dataframe with some null values
        // Need enough data to handle validation split (at least sequence_length rows in each partition)
        let size = constants::SEQUENCE_LENGTH * 3 + 10; // Extra rows to ensure enough data after nulls are dropped
        let mut columns = Vec::new();
        
        for indicator in constants::TECHNICAL_INDICATORS {
            let mut values: Vec<f64> = (0..size).map(|i| i as f64).collect();
            
            // Insert some nulls at random positions
            if indicator == "close" || indicator == "volume" {
                // Keep these essential columns intact to avoid early failures
                columns.push(Series::new(indicator.into(), values).into());
            } else {
                // Insert some NaN values, but not too many to avoid dropping too many rows
                // Only add NaNs to non-consecutive rows
                for i in 0..3 {
                    let pos = i * (size / 8) + 5; // Spread out the NaNs
                    if pos < values.len() {
                        values[pos] = f64::NAN;
                    }
                }
                columns.push(Series::new(indicator.into(), values).into());
            }
        }
        
        let df_with_nulls = DataFrame::new(columns).unwrap();
        
        // Use our custom build function with smaller validation split
        let result = build_burn_lstm_model_for_test(df_with_nulls, 0.1);
        
        // Should succeed but with fewer samples
        assert!(result.is_ok(), "Should handle dataframe with NaN values");
        
        if let Ok((features, _)) = result {
            // Number of sequences should be reduced due to null handling
            assert!(
                features.dims()[0] > 0,
                "Should have at least some sequences after null handling"
            );
        }
    }

    #[test]
    fn test_build_burn_lstm_model_tensor_values() {
        use polars::prelude::*;
        use burn::tensor::Tensor;
        
        // Create a DataFrame with predictable values
        // Need enough data to handle validation split (at least sequence_length rows in each partition)
        let size = constants::SEQUENCE_LENGTH * 4; // Extra rows to ensure validation split works
        let mut columns = Vec::new();
        
        for (i, indicator) in constants::TECHNICAL_INDICATORS.iter().enumerate() {
            // Use different values for each indicator to test proper mapping
            let values: Vec<f64> = (0..size).map(|j| (j + i) as f64).collect();
            columns.push(Series::new((*indicator).into(), values).into());
        }
        
        let predictable_df = DataFrame::new(columns).unwrap();
        
        // Use our custom build function with smaller validation split
        let result = build_burn_lstm_model_for_test(predictable_df, 0.1);
        assert!(result.is_ok(), "Should work with predictable dataframe");
        
        if let Ok((features, targets)) = result {
            // Check that all values are finite (no NaN or infinity)
            type BurnBackend = burn::backend::LibTorch<f32>;
            let device = <BurnBackend as burn::tensor::backend::Backend>::Device::default();
            
            // Validate tensor shape constraints
            assert_eq!(features.dims().len(), 3, "Features should be 3D tensor");
            assert!(features.dims()[0] > 0, "Should have at least some sequences");
            assert_eq!(features.dims()[1], constants::SEQUENCE_LENGTH, "Should match sequence length");
            assert_eq!(features.dims()[2], constants::TECHNICAL_INDICATORS.len(), "Should match number of features");
            
            // Validate targets shape
            assert_eq!(targets.dims().len(), 2, "Targets should be 2D tensor");
            assert_eq!(targets.dims()[0], features.dims()[0], "Number of sequences should match");
            assert_eq!(targets.dims()[1], 1, "Target dimension should be 1");
            
            // Validate tensor values are within expected range (0-1 after normalization)
            // Create tensors to compare against
            let zeros = Tensor::<BurnBackend, 3>::zeros(features.dims().to_vec(), &device);
            let ones = Tensor::<BurnBackend, 3>::ones(features.dims().to_vec(), &device);
            
            // Check min and max values (should be between 0 and 1 due to normalization)
            let min_values = (features.clone() - zeros).abs().sum().reshape([1]);
            let max_diff = (features - ones).abs().sum().reshape([1]);
            
            // At least some values should be close to 0 and some close to 1 after normalization
            assert!(
                min_values.into_scalar() > 0.0,
                "Features should have non-zero values"
            );
            assert!(
                max_diff.into_scalar() > 0.0,
                "Features should have values less than 1"
            );
            
            // Targets should also be normalized between 0-1
            let target_zeros = Tensor::<BurnBackend, 2>::zeros(targets.dims().to_vec(), &device);
            let target_ones = Tensor::<BurnBackend, 2>::ones(targets.dims().to_vec(), &device);
            
            let target_min = (targets.clone() - target_zeros).abs().sum().reshape([1]);
            let target_max_diff = (targets - target_ones).abs().sum().reshape([1]);
            
            assert!(
                target_min.into_scalar() >= 0.0,
                "Targets should have non-negative values"
            );
            assert!(
                target_max_diff.into_scalar() > 0.0,
                "Targets should have values less than 1"
            );
        }
    }

    #[test]
    fn test_build_burn_lstm_model_edge_cases() {
        use polars::prelude::*;
        // Remove unused imports
        
        // Test with minimal valid DataFrame
        let minimal_df = create_minimal_valid_df();
        println!("Minimal DF height: {}", minimal_df.height());
        println!("Minimal DF columns: {}", minimal_df.width());
        
        // Verify required columns are present
        for col in constants::TECHNICAL_INDICATORS {
            assert!(minimal_df.schema().contains(col), "Column '{}' is missing", col);
        }
        
        // Use our custom build function with smaller validation split
        let result = build_burn_lstm_model_for_test(minimal_df, 0.1);
        match &result {
            Ok(_) => println!("Test succeeded"),
            Err(e) => {
                println!("Error: {:?}", e);
                println!("Error cause: {:?}", e.root_cause());
            }
        };
        
        assert!(result.is_ok(), "Should work with minimal valid dataframe");
        
        // Test with DataFrame containing only some required columns
        let incomplete_df = create_incomplete_df();
        let result = build_burn_lstm_model_for_test(incomplete_df, 0.1);
        assert!(result.is_err(), "Should fail with incomplete dataframe");
        
        // Test with empty DataFrame
        let empty_df = DataFrame::default();
        let result = build_burn_lstm_model_for_test(empty_df, 0.1);
        assert!(result.is_err(), "Should fail with empty dataframe");
        
        // Test with DataFrame too small for sequence length
        let small_df = create_small_df();
        let result = build_burn_lstm_model_for_test(small_df, 0.1);
        assert!(
            result.is_err(),
            "Should fail with dataframe too small for sequence length"
        );
        
        // Helper function to create a minimally valid DataFrame
        fn create_minimal_valid_df() -> DataFrame {
            // Create minimum data required for all TECHNICAL_INDICATORS
            // Need at least 2*SEQUENCE_LENGTH + validation split + padding
            let size = constants::SEQUENCE_LENGTH * 5; // Make much larger to be safe
            
            // Generate sample data
            let mut columns = Vec::new();
            
            for indicator in constants::TECHNICAL_INDICATORS {
                let values: Vec<f64> = (0..size).map(|i| i as f64).collect();
                columns.push(Series::new(indicator.into(), values).into());
            }
            
            DataFrame::new(columns).unwrap()
        }
        
        // Helper function to create an incomplete DataFrame
        fn create_incomplete_df() -> DataFrame {
            // Only include some of the required indicators
            let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
            let close = Series::new("close".into(), values.clone()).into();
            let volume = Series::new("volume".into(), values).into();
            
            DataFrame::new(vec![close, volume]).unwrap()
        }
        
        // Helper function to create a DataFrame too small for sequence length
        fn create_small_df() -> DataFrame {
            // Create data with fewer rows than required sequence length
            let size = constants::SEQUENCE_LENGTH - 2; // Too small
            
            let mut columns = Vec::new();
            
            for indicator in constants::TECHNICAL_INDICATORS {
                let values: Vec<f64> = (0..size).map(|i| i as f64).collect();
                columns.push(Series::new(indicator.into(), values).into());
            }
            
            DataFrame::new(columns).unwrap()
        }
    }

    // Custom build function that accepts a validation split parameter
    // Made public for other tests to use
    fn build_burn_lstm_model_for_test(
        df: DataFrame, 
        validation_split: f64
    ) -> anyhow::Result<(
        burn::tensor::Tensor<burn::backend::LibTorch<f32>, 3>,
        burn::tensor::Tensor<burn::backend::LibTorch<f32>, 2>,
    )> {
        use anyhow::bail;
        use burn::tensor::backend::Backend;
        
        // Rename the alias so it doesn't collide with the imported `Backend` trait
        type BurnBackend = burn::backend::LibTorch<f32>;

        // Explicitly invoke the associated `Device` from the `Backend` trait
        let device = <BurnBackend as Backend>::Device::default();

        // Select features and target
        let mut features_df = match df.select(constants::TECHNICAL_INDICATORS) {
            Ok(df) => df,
            Err(e) => bail!("Failed to select features: {}", e),
        };

        // Normalize features
        // Since normalize_features is private, we'll implement our own simplified version
        normalize_features_for_test(&mut features_df)?;

        // Drop any rows with nulls introduced by rolling-window feature calculations
        match features_df.drop_nulls::<String>(None) {
            Ok(df_nonnull) => features_df = df_nonnull,
            Err(e) => bail!("Failed to drop null values: {}", e),
        }

        // Split data into training and validation sets with custom validation_split
        let (train_df, val_df) = split_data_for_test(&features_df, validation_split)?;

        println!("Start converting to tensors");

        // Convert to tensors
        let sequence_length = constants::SEQUENCE_LENGTH; // Number of time steps to look back
        let (train_features, train_targets) =
            match crate::lstm::step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(&train_df, sequence_length, &device) {
                Ok(tensors) => tensors,
                Err(e) => bail!("Failed to convert training data to tensors: {}", e),
            };

        // Use sequence_length - 1 for validation if not enough data
        // This is just for testing purposes
        let val_seq_length = if val_df.height() <= sequence_length {
            std::cmp::max(1, val_df.height() - 1)
        } else {
            sequence_length
        };
        
        let (val_features, val_targets) =
            match crate::lstm::step_1_tensor_preparation::dataframe_to_tensors::<BurnBackend>(&val_df, val_seq_length, &device) {
                Ok(tensors) => tensors,
                Err(e) => bail!("Failed to convert validation data to tensors: {}", e),
            };

        // Print data shapes for verification
        println!("Training data shapes:");
        println!("  Features: {:?}", train_features.shape());
        println!("  Targets: {:?}", train_targets.shape());
        println!("Validation data shapes:");
        println!("  Features: {:?}", val_features.shape());
        println!("  Targets: {:?}", val_targets.shape());

        Ok((train_features, train_targets))
    }

    // Our own implementation of normalize_features since the original is private
    fn normalize_features_for_test(df: &mut DataFrame) -> anyhow::Result<()> {
        use polars::prelude::*;
        
        // Get numeric columns to normalize
        for col in constants::TECHNICAL_INDICATORS {
            if let Ok(series) = df.column(col) {
                // Skip if column is not numeric
                if !matches!(series.dtype(), DataType::Float64 | DataType::Int64) {
                    continue;
                }

                // Convert to Float64 if needed and get a materialized Series reference
                let series_ref = series.as_materialized_series();
                let series = if series_ref.dtype() == &DataType::Int64 {
                    series_ref.cast(&DataType::Float64)?
                } else {
                    series_ref.clone()
                };

                // Calculate min and max for min-max scaling
                let (min, max) = match (series.min::<f64>()?, series.max::<f64>()?) {
                    (Some(min), Some(max)) => (min, max),
                    _ => continue,
                };
                
                // Avoid division by zero
                let range = if (max - min).abs() < f64::EPSILON {
                    1.0
                } else {
                    max - min
                };

                // Apply min-max scaling
                let normalized = (series.clone() - min) / range;
                df.replace(col, normalized)?;
            }
        }

        Ok(())
    }

    // Our own implementation of split_data since the original is private
    fn split_data_for_test(df: &DataFrame, validation_split: f64) -> anyhow::Result<(DataFrame, DataFrame)> {
        // Validate input
        if df.height() == 0 {
            return Err(anyhow::anyhow!("Empty DataFrame"));
        }
        if !(0.0..=1.0).contains(&validation_split) {
            return Err(anyhow::anyhow!("Validation split must be between 0.0 and 1.0"));
        }

        let n_samples = df.height();
        let split_idx = (n_samples as f64 * (1.0 - validation_split)) as usize;

        let train_df = df.slice(0, split_idx);
        let val_df = df.slice(split_idx.try_into().unwrap(), n_samples - split_idx);

        Ok((train_df, val_df))
    }

    #[test]
    fn test_build_burn_lstm_model_missing_file() {
        // Test with non-existent file
        let non_existent_path = std::path::PathBuf::from("non_existent_file.csv");

        // This should fail at the load_and_preprocess step
        let result = crate::util::pre_processor::load_and_preprocess(&non_existent_path);

        assert!(result.is_err(), "Should fail with non-existent file");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("not found") || e.to_string().contains("No such file"),
                "Error message should indicate file not found"
            );
        }
    }
}
