// External crates
use polars::error::PolarsError;
use polars::prelude::*;
use std::{error::Error, path::PathBuf};

// Local modules
use crate::feature_engineering::add_technical_indicators;


/// Loads and preprocesses a CSV file into a DataFrame
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file relative to the `src` directory
///
/// # Returns
///
/// Returns a Result containing the preprocessed DataFrame or an error
pub fn load_and_preprocess(full_path: &PathBuf) -> Result<DataFrame, Box<dyn Error>> {
    println!("Loading data from: {}", full_path.display());

    if !full_path.exists() {
        return Err(format!("File not found: {}", full_path.display()).into());
    }

    let file = std::fs::File::open(&full_path)?;
    let mut df = CsvReader::new(file).finish()?;

    // Verify required columns exist
    let required_columns = ["open", "high", "low", "close", "volume"];
    for &col in &required_columns {
        if df.column(col).is_err() {
            return Err(Box::new(PolarsError::ColumnNotFound(
                format!("Required column {} not found", col).into(),
            )) as Box<dyn std::error::Error>);
        }
    }

    // Sort by timestamp if exists
    if df.column("timestamp").is_ok() {
        df = df.sort(vec!["timestamp"], SortMultipleOptions::default())?;
    }

    // Drop any rows with missing values
    df = df.drop_nulls::<String>(None)?;

    Ok(df)
}

/// Prepares data for LSTM model training
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file relative to the `src` directory
/// * `sequence_length` - Length of input sequences for LSTM
///
/// # Returns
///
/// Returns a Result containing the prepared DataFrame with features and sequences
pub fn prepare_lstm_data(
    file_path: &PathBuf,
    _sequence_length: usize,
) -> Result<DataFrame, Box<dyn Error>> {
    // Load and preprocess data
    let mut df = load_and_preprocess(file_path)?;

    // Add technical indicators
    df = add_technical_indicators(&mut df)?;

    // TODO: Add sequence creation for LSTM
    // This will be implemented when we add the LSTM model

    Ok(df)
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_load_and_preprocess() {
//         let workspace_dir = std::env::current_dir().expect("Failed to get current directory");
//         let full_path = workspace_dir.join("AAPL-ticker_minute_bars.csv");

//         let result = load_and_preprocess(&full_path);
//         assert!(result.is_ok());
//         let df = result.unwrap();

//         // Verify required columns exist
//         let required_columns = ["open", "high", "low", "close", "volume"];
//         for column in required_columns {
//             assert!(df.column(column).is_ok(), "Column {} was not found", column);
//         }
//     }

//     #[test]
//     fn test_prepare_lstm_data() {
//         let workspace_dir = std::env::current_dir().expect("Failed to get current directory");
//         let full_path = workspace_dir.join("AAPL-ticker_minute_bars.csv");

//         let result = prepare_lstm_data(&full_path, 10);
//         assert!(result.is_ok());
//         let df = result.unwrap();

//         // Verify technical indicators were added
//         let technical_indicators = [
//             "sma_20",
//             "sma_50",
//             "ema_20",
//             "rsi_14",
//             "macd",
//             "macd_signal",
//             "bb_middle",
//             "bb_upper",
//             "bb_lower",
//             "returns",
//         ];

//         for indicator in technical_indicators {
//             assert!(
//                 df.column(indicator).is_ok(),
//                 "Indicator {} was not found",
//                 indicator
//             );
//         }
//     }
// }
