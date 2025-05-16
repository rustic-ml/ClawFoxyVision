use predict_price_lstm::util::file_utils::read_financial_data;
use predict_price_lstm::util::feature_engineering::add_technical_indicators;
use burn::tensor::{backend::Backend, Tensor};
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use polars::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example: Reading Parquet data and processing with LSTM and GRU models");
    
    // Setup backend
    type BurnBackend = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();
    println!("Using device: CPU NdArray");
    
    // Read financial data from Parquet file
    let (df, _) = read_financial_data("examples/csv/AAPL_daily_ohlcv.parquet")?;
    println!("Loaded data with columns: {:?}", df.get_column_names());
    println!("Loaded dataframe with {} rows", df.height());
    
    // Extract features for time series modeling using the project's feature engineering module
    let mut features_df = df.clone();
    features_df = add_technical_indicators(&mut features_df)?;
    println!("After adding technical indicators, columns: {:?}", features_df.get_column_names());
    
    // Select relevant features for modeling
    let selected_features = features_df.select([
        "close", "open", "high", "low", "volume", 
        "sma_20", "ema_20", "returns",
        "rsi_14", "macd", "bb_middle", "atr_14"
    ])?;
    
    println!("Prepared features with columns: {:?}", selected_features.get_column_names());
    
    // Drop rows with null values that resulted from calculations
    let features = selected_features.drop_nulls::<String>(None)?;
    println!("Final feature set has {} rows after removing nulls", features.height());
    
    // Process data for LSTM model
    println!("Processing data for LSTM model...");
    let start_time = Instant::now();
    let lstm_results = process_with_lstm::<BurnBackend>(features.clone(), &device)?;
    println!("LSTM processing completed in {:?}", start_time.elapsed());
    println!("LSTM prediction shape: {:?}", lstm_results.shape());
    
    // Process data for GRU model
    println!("Processing data for GRU model...");
    let start_time = Instant::now();
    let gru_results = process_with_gru::<BurnBackend>(features, &device)?;
    println!("GRU processing completed in {:?}", start_time.elapsed());
    println!("GRU prediction shape: {:?}", gru_results.shape());
    
    println!("Parquet LSTM/GRU example completed successfully");
    Ok(())
}

// Process data with LSTM model
fn process_with_lstm<B: Backend>(
    features: DataFrame,
    device: &B::Device,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    // In a real implementation, you would:
    // 1. Load your trained LSTM model
    // 2. Format the data for LSTM input (sequences)
    // 3. Run prediction
    
    // This is a placeholder for demonstration purposes
    println!("LSTM would process {} rows of data", features.height());
    
    // Return placeholder tensor (in real code, this would be the model output)
    let forecast_horizon = 5;
    Ok(Tensor::<B, 2>::zeros([forecast_horizon, 1], device))
}

// Process data with GRU model
fn process_with_gru<B: Backend>(
    features: DataFrame,
    device: &B::Device,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    // In a real implementation, you would:
    // 1. Load your trained GRU model
    // 2. Format the data for GRU input (sequences)
    // 3. Run prediction
    
    // This is a placeholder for demonstration purposes
    println!("GRU would process {} rows of data", features.height());
    
    // Return placeholder tensor (in real code, this would be the model output)
    let forecast_horizon = 5;
    Ok(Tensor::<B, 2>::zeros([forecast_horizon, 1], device))
} 