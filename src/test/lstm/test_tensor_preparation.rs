use crate::constants::TECHNICAL_INDICATORS;
use crate::minute::lstm::step_1_tensor_preparation::{
    check_for_nans, dataframe_to_tensors, normalize_features,
};
use anyhow::Result;
use burn::backend::LibTorch;
use burn::tensor::backend::Backend;
use polars::prelude::*;

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

// Helper function to create a test DataFrame with enough rows for sequence processing
fn create_large_test_dataframe() -> DataFrame {
    // Create dummy data that matches our expected columns
    let mut columns = Vec::new();

    // Create 60 rows to exceed the 50-row requirement for technical indicators
    let row_count = 60;

    for &col in TECHNICAL_INDICATORS.iter() {
        let values: Vec<f64> = (1..=row_count).map(|i| i as f64).collect();
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

    // Create a DataFrame with explicit NaN values to ensure they get detected
    let mut df_with_nans = DataFrame::default();

    // Create a series with explicit NaN values
    for (i, &col) in TECHNICAL_INDICATORS.iter().enumerate() {
        let mut values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];

        // Add NaN values to the first technical indicator
        if i == 0 {
            values[2] = f64::NAN;
        }

        let series = Series::new(col.into(), values);
        df_with_nans.with_column(series)?;
    }

    let nan_count = check_for_nans(&df_with_nans, &TECHNICAL_INDICATORS)?;
    assert!(
        nan_count > 0,
        "Expected to find NaNs in DataFrame with NaNs"
    );

    // Test with subset of columns
    let subset = &TECHNICAL_INDICATORS[0..2];
    let nan_count = check_for_nans(&df_with_nans, subset)?;
    assert!(
        nan_count <= 1,
        "Expected at most 1 NaN in the subset of columns"
    );

    Ok(())
}

#[test]
fn test_normalize_features() -> Result<()> {
    // Test with clean data
    let mut df = create_test_dataframe();
    normalize_features(&mut df, &["close", "open", "high", "low"], false, false)?;

    // After normalization, price columns should be z-score normalized,
    // other columns should be min-max normalized (between 0 and 1)
    for &col in TECHNICAL_INDICATORS.iter() {
        let series = df.column(col)?.f64()?;

        // Check if this is a price column
        if ["close", "open", "high", "low"].contains(&col) {
            // Price columns use z-score normalization - not constrained to 0-1 range
            // But should have mean close to 0 and std close to 1
            let mean = series.mean().unwrap();
            let std = series.std(1).unwrap();

            assert!(
                mean.abs() < 1e-10,
                "Z-score normalized column {} should have mean close to 0, got {}",
                col,
                mean
            );

            assert!(
                (std - 1.0).abs() < 1e-10,
                "Z-score normalized column {} should have std close to 1, got {}",
                col,
                std
            );
        } else {
            // Non-price columns use min-max normalization
            let min = series.min().unwrap();
            let max = series.max().unwrap();

            assert!(
                min >= 0.0 && max <= 1.0,
                "Column {} values should be between 0 and 1, got min={}, max={}",
                col,
                min,
                max
            );
        }
    }

    // Test with constant columns
    let mut df_const = create_df_with_constants();
    normalize_features(
        &mut df_const,
        &["close", "open", "high", "low"],
        false,
        false,
    )?;

    // Constant columns should be normalized differently based on type
    for (i, &col) in TECHNICAL_INDICATORS.iter().enumerate() {
        let series = df_const.column(col)?.f64()?;

        if i % 3 == 0 {
            // This was a constant column
            if ["close", "open", "high", "low"].contains(&col) {
                // Constant price columns should be normalized to 0 (z-score of mean is 0)
                for j in 0..series.len() {
                    let val = series.get(j).unwrap();
                    assert!(
                        (val - 0.0).abs() < 1e-10,
                        "Constant price column {} should be normalized to 0.0, got {}",
                        col,
                        val
                    );
                }
            } else {
                // Constant non-price columns should be normalized to 0.5
                for j in 0..series.len() {
                    let val = series.get(j).unwrap();
                    assert!(
                        (val - 0.5).abs() < 1e-10,
                        "Constant column {} should be normalized to 0.5, got {}",
                        col,
                        val
                    );
                }
            }
        } else {
            // This was a regular column with variation
            if ["close", "open", "high", "low"].contains(&col) {
                // Price columns use z-score normalization
                let mean = series.mean().unwrap();
                let std = series.std(1).unwrap();

                assert!(
                    mean.abs() < 1e-10,
                    "Z-score normalized column {} should have mean close to 0, got {}",
                    col,
                    mean
                );

                assert!(
                    (std - 1.0).abs() < 1e-10,
                    "Z-score normalized column {} should have std close to 1, got {}",
                    col,
                    std
                );
            } else {
                // Non-price columns use min-max normalization
                let min = series.min().unwrap();
                let max = series.max().unwrap();

                assert!(
                    min >= 0.0 && max <= 1.0,
                    "Column {} values should be between 0 and 1, got min={}, max={}",
                    col,
                    min,
                    max
                );
            }
        }
    }

    // Test with NaN values
    let mut df_nan = create_df_with_nans();
    let _original_height = df_nan.height();
    normalize_features(&mut df_nan, &["close", "open", "high", "low"], false, false)?;

    // Remaining values should be normalized based on column type
    for &col in TECHNICAL_INDICATORS.iter() {
        let series = df_nan.column(col)?.f64()?;

        // Skip empty series
        if series.len() == 0 {
            continue;
        }

        if ["close", "open", "high", "low"].contains(&col) {
            // Price columns use z-score normalization
            let mean = series.mean().unwrap();
            let std = series.std(1).unwrap();

            assert!(mean.abs() < 1e-2, 
                   "Z-score normalized column {} should have mean close to 0 after handling NaNs, got {}", 
                   col, mean);

            assert!((std - 1.0).abs() < 1e-1, 
                   "Z-score normalized column {} should have std close to 1 after handling NaNs, got {}", 
                   col, std);
        } else {
            // Non-price columns use min-max normalization
            let min = series.min().unwrap();
            let max = series.max().unwrap();

            assert!(min >= 0.0 && max <= 1.0, 
                    "Column {} values should be between 0 and 1 after handling NaNs, got min={}, max={}", 
                    col, min, max);
        }
    }

    Ok(())
}

#[test]
fn test_dataframe_to_tensors() -> Result<()> {
    type TestBackend = LibTorch<f32>;
    let device = <TestBackend as Backend>::Device::default();

    // Test with large dataset that has enough rows for technical indicators
    let mut df = create_large_test_dataframe();
    normalize_features(&mut df, &["close", "open", "high", "low"], false, false)?;

    // Use small values for testing
    let sequence_length = 5;
    let forecast_horizon = 1;

    let (features, targets) = dataframe_to_tensors::<TestBackend>(
        &df,
        sequence_length,
        forecast_horizon,
        &device,
        false,
        None,
    )?;

    // Verify tensor shapes
    let expected_sequences = df.height() - 50 - sequence_length - forecast_horizon + 1;
    assert_eq!(
        features.dims()[0],
        expected_sequences,
        "Wrong number of sequences in features tensor"
    );
    assert_eq!(
        features.dims()[1],
        sequence_length,
        "Wrong sequence length in features tensor"
    );
    assert_eq!(
        features.dims()[2],
        TECHNICAL_INDICATORS.len(),
        "Wrong number of features"
    );

    assert_eq!(
        targets.dims()[0],
        expected_sequences,
        "Wrong number of sequences in targets tensor"
    );
    assert_eq!(
        targets.dims()[1],
        forecast_horizon,
        "Wrong forecast horizon in targets tensor"
    );

    // Test with batch size specified
    let batch_size = 3;
    let (features_batched, _targets_batched) = dataframe_to_tensors::<TestBackend>(
        &df,
        sequence_length,
        forecast_horizon,
        &device,
        false,
        Some(batch_size),
    )?;

    // Verify batched tensor shapes - should still preserve the same total number of sequences
    assert_eq!(
        features_batched.dims()[0],
        expected_sequences,
        "Batch size should not affect the total number of sequences"
    );

    Ok(())
}
