// External crates
use chrono::{Duration, Utc};
use polars::error::PolarsError;
use polars::prelude::*;
use std::fs::File;
use std::sync::Arc;
use std::{error::Error, path::Path, path::PathBuf};

// Local modules
use crate::constants::LSTM_TRAINING_DAYS;
use rustalib::util::file_utils::read_financial_data;

/// Read a CSV file into a DataFrame with custom column names
pub fn read_csv_to_dataframe<P: AsRef<Path>>(
    file_path: P,
    _has_header: bool,
    _column_names: Option<Vec<&str>>,
) -> PolarsResult<DataFrame> {
    let (df, _) = read_financial_data(file_path.as_ref().to_str().unwrap())?;
    Ok(df)
}

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
pub fn load_and_preprocess_with_days(
    full_path: &PathBuf,
    days: Option<i64>,
) -> Result<DataFrame, Box<dyn Error>> {
    println!("Loading data from: {}", full_path.display());

    if !full_path.exists() {
        return Err(format!("File not found: {}", full_path.display()).into());
    }

    // Read CSV using standardized function
    let (df, _) = read_financial_data(full_path.to_str().unwrap())?;

    // Compute cutoff from one year ago and filter rows by 'time'
    let training_days = days.unwrap_or(LSTM_TRAINING_DAYS);
    let one_year_ago = Utc::now() - Duration::days(training_days);
    let cutoff_str = one_year_ago.format("%Y-%m-%d %H:%M:%S UTC").to_string();
    use polars::prelude::{col, lit};

    // Filter by 'time' > cutoff
    let df = df
        .lazy()
        .filter(col("time").gt(lit(cutoff_str)))
        .collect()?;

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
    let df = load_and_preprocess_with_days(file_path, None)?;

    // TODO: Add sequence creation for LSTM
    // This will be implemented when we add the LSTM model

    Ok(df)
}
