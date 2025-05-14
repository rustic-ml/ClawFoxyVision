/// Tests for the GRU (Gated Recurrent Unit) implementation
///
/// This module contains tests for the GRU neural network implementation, including:
/// 
/// * Basic GRU cell functionality and forward pass
/// * Bidirectional GRU behavior
/// * TimeSeriesGRU model architecture
/// * Single-step prediction with `predict_next_step`
/// * Multi-step prediction with `predict_multiple_steps`
/// * GRU with real training data
/// 
/// The tests verify that the GRU components work correctly, can process data properly,
/// and produce valid predictions, especially focusing on the fixes we made to the
/// prediction functions to handle column naming and DataFrame manipulation correctly.
pub mod test_gru; 