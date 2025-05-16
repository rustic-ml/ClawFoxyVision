pub mod gru;
/// Test modules for the time series prediction package
///
/// This module contains various test suites organized by the neural network architecture they test:
///
/// * `lstm` - Tests for Long Short-Term Memory network implementations
/// * `gru` - Tests for Gated Recurrent Unit network implementations, including prediction functionality
/// * `main_tests` - Tests for overall system functionality
/// * `file_utils_tests` - Tests for file reading utilities, including CSV and Parquet support
///
/// The tests verify model architectures, training procedures, and prediction capabilities,
/// ensuring that the time series forecasting system works correctly and produces valid results.
pub mod lstm;
pub mod main_tests;
pub mod file_utils_tests;
