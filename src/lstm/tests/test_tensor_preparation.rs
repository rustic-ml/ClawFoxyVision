use anyhow::Result;
use polars::prelude::*;
use burn::backend::LibTorch;
use burn::tensor::backend::Backend;

use crate::lstm::step_1_tensor_preparation::{
    self, 
    normalize_features, 
    check_for_nans, 
    dataframe_to_tensors
};
use crate::constants::TECHNICAL_INDICATORS;

// This main function will be compiled in when running tests directly
#[cfg(test)]
pub fn main() {
    println!("Running tensor preparation tests...");
    test_check_for_nans().unwrap();
    test_normalize_features().unwrap();
    test_dataframe_to_tensors().unwrap();
    println!("All tests passed!");
}

// Helper function to create a test DataFrame
fn create_test_dataframe() -> DataFrame {
    // Create dummy data that matches our expected columns
    let mut columns = Vec::new();
    
    for col in TECHNICAL_INDICATORS.iter() {
        let series = Series::new((*col).into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        columns.push(series);
    }
    
    let columns: Vec<Column> = columns.into_iter().map(|s| s.into_column()).collect();
    DataFrame::new(columns).unwrap()
}

// Helper function to create a DataFrame with NaN values
fn create_df_with_nans() -> DataFrame {
    let mut columns = Vec::new();
    
    for (i, &col) in TECHNICAL_INDICATORS.iter().enumerate() {
        let mut values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        
        // Add a NaN to a few columns
        if i % 3 == 0 {
            values[2] = f64::NAN;
        }
        
        let series = Series::new(col.into(), values);
        columns.push(series);
    }
    
    let columns: Vec<Column> = columns.into_iter().map(|s| s.into_column()).collect();
    DataFrame::new(columns).unwrap()
}

// Helper function to create a DataFrame with constant columns
fn create_df_with_constants() -> DataFrame {
    let mut columns = Vec::new();
    
    for (i, &col) in TECHNICAL_INDICATORS.iter().enumerate() {
        let values = if i % 3 == 0 {
            // Make some columns constant
            vec![10.0f64, 10.0, 10.0, 10.0, 10.0]
        } else {
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0]
        };
        
        let series = Series::new(col.into(), values);
        columns.push(series);
    }
    
    let columns: Vec<Column> = columns.into_iter().map(|s| s.into_column()).collect();
    DataFrame::new(columns).unwrap()
}

#[test]
fn test_check_for_nans() -> Result<()> {
    let df = create_test_dataframe();
    let nan_count = check_for_nans(&df, &TECHNICAL_INDICATORS)?;
    assert_eq!(nan_count, 0, "Expected no NaNs in clean DataFrame");
    
    let df_with_nans = create_df_with_nans();
    let nan_count = check_for_nans(&df_with_nans, &TECHNICAL_INDICATORS)?;
    assert!(nan_count > 0, "Expected to find NaNs in DataFrame with NaNs");
    
    // Test with subset of columns
    let subset = &TECHNICAL_INDICATORS[0..2];
    let nan_count = check_for_nans(&df_with_nans, subset)?;
    assert!(nan_count <= 1, "Expected at most 1 NaN in the subset of columns");
    
    Ok(())
}

#[test]
fn test_normalize_features() -> Result<()> {
    // Test with clean data
    let mut df = create_test_dataframe();
    normalize_features(&mut df, &["close", "open", "high", "low"], false)?;
    
    // After normalization, all values should be between 0 and 1
    for col in TECHNICAL_INDICATORS.iter() {
        let series = df.column(*col)?.f64()?;
        let min = series.min().unwrap();
        let max = series.max().unwrap();
        
        assert!(min >= 0.0 && max <= 1.0, 
                "Column {} values should be between 0 and 1, got min={}, max={}", 
                col, min, max);
    }
    
    // Test with constant columns
    let mut df_const = create_df_with_constants();
    normalize_features(&mut df_const, &["close", "open", "high", "low"], false)?;
    
    // Constant columns should be normalized to 0.5
    for (i, &col) in TECHNICAL_INDICATORS.iter().enumerate() {
        let series = df_const.column(col)?.f64()?;
        
        if i % 3 == 0 {
            // This was a constant column, should be normalized to 0.5
            for j in 0..series.len() {
                let val = series.get(j).unwrap();
                assert!((val - 0.5).abs() < 1e-10, 
                       "Constant column {} should be normalized to 0.5, got {}", col, val);
            }
        } else {
            // This was a regular column, should be normalized between 0 and 1
            let min = series.min().unwrap();
            let max = series.max().unwrap();
            assert!(min >= 0.0 && max <= 1.0, 
                   "Column {} values should be between 0 and 1, got min={}, max={}", 
                   col, min, max);
        }
    }
    
    // Test with NaN values
    let mut df_nan = create_df_with_nans();
    let original_height = df_nan.height();
    normalize_features(&mut df_nan, &["close", "open", "high", "low"], false)?;
    
    // NaN rows should be dropped
    assert!(df_nan.height() < original_height, 
            "DataFrame with NaNs should have rows dropped during normalization");
    
    // Remaining values should be normalized
    for col in TECHNICAL_INDICATORS.iter() {
        let series = df_nan.column(*col)?.f64()?;
        let min = series.min().unwrap();
        let max = series.max().unwrap();
        
        assert!(min >= 0.0 && max <= 1.0, 
                "Column {} values should be between 0 and 1 after handling NaNs, got min={}, max={}", 
                col, min, max);
    }
    
    Ok(())
}

#[test]
fn test_dataframe_to_tensors() -> Result<()> {
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();
    
    // Test with clean data
    let mut df = create_test_dataframe();
    normalize_features(&mut df, &["close", "open", "high", "low"], false)?;
    
    // Use small values for testing
    let sequence_length = 2;
    let forecast_horizon = 1;
    
    let (features, targets) = dataframe_to_tensors::<TestBackend>(
        &df, sequence_length, forecast_horizon, &device, false
    )?;
    
    // Verify tensor shapes
    let expected_sequences = df.height() - sequence_length - forecast_horizon + 1;
    assert_eq!(features.dims()[0], expected_sequences, "Wrong number of sequences in features tensor");
    assert_eq!(features.dims()[1], sequence_length, "Wrong sequence length in features tensor");
    assert_eq!(features.dims()[2], TECHNICAL_INDICATORS.len(), "Wrong number of features");
    
    assert_eq!(targets.dims()[0], expected_sequences, "Wrong number of sequences in targets tensor");
    assert_eq!(targets.dims()[1], forecast_horizon, "Wrong forecast horizon in targets tensor");
    
    // Test with NaN values
    let mut df_nan = create_df_with_nans();
    normalize_features(&mut df_nan, &["close", "open", "high", "low"], false)?;
    
    let (features_nan, targets_nan) = dataframe_to_tensors::<TestBackend>(
        &df_nan, sequence_length, forecast_horizon, &device, false
    )?;
    
    // Verify tensor shapes after NaNs are handled
    let expected_sequences_nan = df_nan.height() - sequence_length - forecast_horizon + 1;
    assert_eq!(features_nan.dims()[0], expected_sequences_nan, 
               "Wrong number of sequences in features tensor after NaN handling");
    
    Ok(())
} 