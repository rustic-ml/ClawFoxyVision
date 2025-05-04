// External crates
use polars::error::PolarsError;
use polars::prelude::*;
use chrono::{Utc, Duration};
use std::{error::Error, path::PathBuf};

// Local modules
use crate::feature_engineering::add_technical_indicators;
use crate::constants::LSTM_TRAINING_DAYS;


/// Loads and preprocesses a CSV file into a DataFrame
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file relative to the `src` directory
/// * `days` - Optional number of days to look back for data (defaults to LSTM_TRAINING_DAYS)
///
/// # Returns
///
/// Returns a Result containing the preprocessed DataFrame or an error
pub fn load_and_preprocess(full_path: &PathBuf, days: Option<i64>) -> Result<DataFrame, Box<dyn Error>> {
    println!("Loading data from: {}", full_path.display());

    if !full_path.exists() {
        return Err(format!("File not found: {}", full_path.display()).into());
    }

    let file = std::fs::File::open(&full_path)?;
    // Compute cutoff from one year ago and filter rows by 'time'
    let training_days = days.unwrap_or(LSTM_TRAINING_DAYS);
    let one_year_ago = Utc::now() - Duration::days(training_days);
    let cutoff_str = one_year_ago.format("%Y-%m-%d %H:%M:%S UTC").to_string();
    use polars::prelude::{col, lit};
    // Read CSV lazily, filter by 'time' > cutoff, then collect
    let mut df = CsvReader::new(file)
        .finish()?
        .lazy()
        .filter(col("time").gt(lit(cutoff_str)))
        .collect()?;

    // Verify required columns exist
    let required_columns = ["open", "high", "low", "close", "volume"];
    for &col in &required_columns {
        if df.column(col).is_err() {
            return Err(Box::new(PolarsError::ColumnNotFound(
                format!("Required column {} not found", col).into(),
            )) as Box<dyn std::error::Error>);
        }
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
    let mut df = load_and_preprocess(file_path, None)?;

    // Add technical indicators
    df = add_technical_indicators(&mut df)?;

    // TODO: Add sequence creation for LSTM
    // This will be implemented when we add the LSTM model

    Ok(df)
}
