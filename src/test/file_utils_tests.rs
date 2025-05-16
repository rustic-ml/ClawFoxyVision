#[cfg(test)]
mod tests {
    use crate::util::file_utils::{read_financial_data, read_csv_file};
    use std::path::Path;

    #[test]
    fn test_read_financial_data_csv() {
        // This test verifies that the read_financial_data function works with CSV files
        println!("Testing read_financial_data with CSV files");
        
        let test_file = Path::new("examples/csv/AAPL_daily_ohlcv.csv");
        
        if test_file.exists() {
            let result = read_financial_data(test_file);
            assert!(result.is_ok(), "Should successfully read CSV file");
            
            let (df, _columns) = result.unwrap();
            assert!(df.height() > 0, "DataFrame should have rows");
            assert!(df.width() > 0, "DataFrame should have columns");
            
            // Check that common financial columns were detected
            assert!(df.schema().contains("close"), "DataFrame should contain 'close' column");
        } else {
            println!("Skipping test_read_financial_data_csv: Test file not found");
        }
    }

    #[test]
    fn test_read_financial_data_parquet() {
        // This test verifies that the read_financial_data function works with Parquet files
        println!("Testing read_financial_data with Parquet files");
        
        let test_file = Path::new("examples/csv/AAPL_daily_ohlcv.parquet");
        
        if test_file.exists() {
            let result = read_financial_data(test_file);
            assert!(result.is_ok(), "Should successfully read Parquet file");
            
            let (df, _columns) = result.unwrap();
            assert!(df.height() > 0, "DataFrame should have rows");
            assert!(df.width() > 0, "DataFrame should have columns");
            
            // Check that common financial columns were detected
            assert!(df.schema().contains("close"), "DataFrame should contain 'close' column");
        } else {
            println!("Skipping test_read_financial_data_parquet: Test file not found");
        }
    }

    #[test]
    fn test_read_csv_file() {
        // This test verifies that the read_csv_file function works correctly
        println!("Testing read_csv_file");
        
        let test_file = Path::new("examples/csv/AAPL_daily_ohlcv.csv");
        
        if test_file.exists() {
            let result = read_csv_file(test_file);
            assert!(result.is_ok(), "Should successfully read CSV file");
            
            let df = result.unwrap();
            assert!(df.height() > 0, "DataFrame should have rows");
            assert!(df.width() > 0, "DataFrame should have columns");
            
            // Check that common financial columns were detected
            assert!(df.schema().contains("close"), "DataFrame should contain 'close' column");
        } else {
            println!("Skipping test_read_csv_file: Test file not found");
        }
    }

    #[test]
    fn test_file_format_detection() {
        // This test verifies that the read_financial_data function correctly detects file formats
        println!("Testing file format detection in read_financial_data");
        
        let csv_file = Path::new("examples/csv/AAPL_daily_ohlcv.csv");
        let parquet_file = Path::new("examples/csv/AAPL_daily_ohlcv.parquet");
        
        if csv_file.exists() && parquet_file.exists() {
            // Read both files and verify they have the same key columns
            let csv_result = read_financial_data(csv_file);
            let parquet_result = read_financial_data(parquet_file);
            
            assert!(csv_result.is_ok(), "Should successfully read CSV file");
            assert!(parquet_result.is_ok(), "Should successfully read Parquet file");
            
            let (csv_df, _) = csv_result.unwrap();
            let (parquet_df, _) = parquet_result.unwrap();
            
            // Verify both dataframes have the key financial columns
            for column in ["open", "high", "low", "close", "volume"].iter() {
                let col_name = *column;
                assert!(csv_df.schema().contains(col_name), 
                    "CSV DataFrame should contain expected column");
                assert!(parquet_df.schema().contains(col_name), 
                    "Parquet DataFrame should contain expected column");
            }
        } else {
            println!("Skipping test_file_format_detection: Test files not found");
        }
    }
} 