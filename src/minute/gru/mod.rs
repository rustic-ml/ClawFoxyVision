/// # GRU Implementation Module
/// 
/// This module implements a Gated Recurrent Unit (GRU) neural network for time series forecasting.
/// The GRU is a more efficient alternative to LSTM that maintains competitive performance while
/// using fewer parameters and requiring less computational resources.
///
/// ## Module Structure:
///
/// 1. **step_1_tensor_preparation**: Data preparation utilities (imports from LSTM implementation)
/// 2. **step_2_gru_cell**: Core GRU cell implementation with gates and state management
/// 3. **step_3_gru_model_arch**: Complete GRU architecture with attention mechanism
/// 4. **step_4_train_model**: Training workflow and configuration
/// 5. **step_5_prediction**: Single and multi-step prediction utilities
/// 6. **step_6_model_serialization**: Model saving and loading functionality
///
/// The GRU implementation follows a similar architecture to the LSTM implementation
/// but uses the simpler GRU cell which combines the forget and input gates into a single
/// update gate, and merges the cell state and hidden state.
///
pub mod step_1_tensor_preparation;
pub mod step_2_gru_cell;
pub mod step_3_gru_model_arch;
pub mod step_4_train_model;
pub mod step_5_prediction;
pub mod step_6_model_serialization;

#[cfg(test)]
pub mod tests {
    pub mod test_gru;
} 