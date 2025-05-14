// External crates
use polars::error::PolarsError;
use polars::prelude::*;
use chrono::{Utc, Duration};
use std::{error::Error, path::PathBuf, path::Path};
use std::fs::File;
use std::sync::Arc;

// Local modules
use crate::constants::LSTM_TRAINING_DAYS;

/// Read a CSV file into a DataFrame with custom column names
pub fn read_csv_to_dataframe<P: AsRef<Path>>(
    file_path: P,
    has_header: bool,
    column_names: Option<Vec<&str>>,
) -> PolarsResult<DataFrame> {
    let file = File::open(file_path)?;
    
    // Create options
    let mut options = CsvReadOptions::default()
        .with_has_header(has_header);
    
    // If column names were provided, create a schema
    if let Some(names) = column_names {
        let fields: Vec<Field> = names.iter()
            .map(|&name| Field::new(name.to_string().into(), DataType::String))
            .collect();
        
        let schema = Schema::from_iter(fields);
        options = options.with_schema(Some(Arc::new(schema)));
    }
    
    // Create reader and apply options
    let reader = CsvReader::new(file).with_options(options);
    
    // Finish reading
    reader.finish()
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
pub fn load_and_preprocess(full_path: &PathBuf, days: Option<i64>) -> Result<DataFrame, Box<dyn Error>> {
    println!("Loading data from: {}", full_path.display());

    if !full_path.exists() {
        return Err(format!("File not found: {}", full_path.display()).into());
    }

    // Read CSV using our local function instead of ta-lib-in-rust
    let df = read_csv_to_dataframe(full_path, true, None)?;
    
    // Compute cutoff from one year ago and filter rows by 'time'
    let training_days = days.unwrap_or(LSTM_TRAINING_DAYS);
    let one_year_ago = Utc::now() - Duration::days(training_days);
    let cutoff_str = one_year_ago.format("%Y-%m-%d %H:%M:%S UTC").to_string();
    use polars::prelude::{col, lit};
    
    // Filter by 'time' > cutoff
    let mut df = df.lazy()
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
    let df = load_and_preprocess(file_path, None)?;

    // TODO: Add sequence creation for LSTM
    // This will be implemented when we add the LSTM model

    Ok(df)
}
