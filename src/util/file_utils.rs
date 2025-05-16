// External crates
use polars::prelude::*;
use rustalib::util::file_utils::read_financial_data as rustalib_read_financial_data;
use std::path::Path;

/// A single entry point for reading financial data from either CSV or Parquet files
///
/// This function automatically detects the file type based on the extension and 
/// uses the appropriate reader from RusTaLib to load the data.
///
/// # Arguments
///
/// * `file_path` - Path to the financial data file (CSV or Parquet)
///
/// # Returns
///
/// Returns a tuple containing the DataFrame with standardized column names and metadata
pub fn read_financial_data<P: AsRef<Path>>(
    file_path: P,
) -> PolarsResult<(DataFrame, rustalib::util::file_utils::FinancialColumns)> {
    // RusTaLib's read_financial_data already handles both CSV and Parquet files
    // by examining the file extension
    let (df, metadata) = rustalib_read_financial_data(file_path.as_ref().to_str().unwrap())?;
    
    // Process the data through our enhanced pipeline for additional standardization
    enhanced_process_dataframe(df, metadata)
}

/// Enhanced financial data processing for standardizing column names
///
/// This function adds support for case-insensitive column names and common abbreviations
///
/// # Arguments
///
/// * `df` - The original DataFrame
/// * `metadata` - Original financial metadata
///
/// # Returns
///
/// Returns a tuple containing the DataFrame with standardized column names and metadata
fn enhanced_process_dataframe(
    mut df: DataFrame,
    metadata: rustalib::util::file_utils::FinancialColumns,
) -> PolarsResult<(DataFrame, rustalib::util::file_utils::FinancialColumns)> {
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
    
    Ok((df, metadata))
}

/// Enhanced CSV reader with case-insensitive column naming
///
/// This function wraps the standard read_financial_data function
/// and adds support for case-insensitive column names and common abbreviations
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file
///
/// # Returns
///
/// Returns a tuple containing the DataFrame with standardized column names and metadata
pub fn enhanced_read_financial_data<P: AsRef<Path>>(
    file_path: P,
) -> PolarsResult<(DataFrame, rustalib::util::file_utils::FinancialColumns)> {
    // First read the file using the standard function
    let (df, metadata) = rustalib_read_financial_data(file_path.as_ref().to_str().unwrap())?;
    
    // Apply our enhanced processing
    enhanced_process_dataframe(df, metadata)
}

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
    let (df, _) = read_financial_data(file_path)?;
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
    let (df, _) = read_financial_data(file_path)?;
    Ok(df)
}
