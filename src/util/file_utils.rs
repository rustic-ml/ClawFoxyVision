// External crates
use polars::prelude::*;
use rustalib::util::file_utils::read_financial_data;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Read a CSV file into a DataFrame with standardized column names
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file
/// * `has_header` - Whether the CSV file has a header row
///
/// # Returns
///
/// Returns a DataFrame with standardized column names
pub fn read_csv_to_dataframe<P: AsRef<Path>>(
    file_path: P,
    _has_header: bool,
) -> PolarsResult<DataFrame> {
    let (df, _) = read_financial_data(file_path.as_ref().to_str().unwrap())?;
    Ok(df)
}

/// Load and preprocess a CSV file into a DataFrame
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file
/// * `has_header` - Whether the CSV file has a header row
///
/// # Returns
///
/// Returns a preprocessed DataFrame
pub fn load_and_preprocess<P: AsRef<Path>>(
    file_path: P,
    has_header: bool,
) -> PolarsResult<DataFrame> {
    read_csv_to_dataframe(file_path, has_header)
}

/// Read a CSV file into a DataFrame
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file
///
/// # Returns
///
/// Returns a DataFrame containing the CSV data
pub fn read_csv_file<P: AsRef<Path>>(file_path: P) -> PolarsResult<DataFrame> {
    let (df, _) = read_financial_data(file_path.as_ref().to_str().unwrap())?;
    Ok(df)
}
