// External imports
use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor, activation};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use crate::lstm::step_2_lstm_cell::LSTM;

/// TimeSeriesLstm architecture for forecasting
#[derive(Module, Debug)]
pub struct TimeSeriesLstm<B: Backend> {
    // Using placeholder tensors instead of actual Lstm component
    // since the Burn API has changed significantly
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    attention: Attention<B>,
    dropout: Dropout,
    output: Linear<B>,
    lstm: LSTM<B>,
}

impl<B: Backend> TimeSeriesLstm<B> {
    /// Create a new TimeSeriesLstm model
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        // Configure LSTM
        let lstm_output_size = if bidirectional { 2 * hidden_size } else { hidden_size };
        let attention = Attention::new(lstm_output_size, device);
        let dropout_config = DropoutConfig::new(dropout_prob);
        let dropout = dropout_config.init();
        let output_config = LinearConfig::new(lstm_output_size, output_size);
        let output = output_config.init(device);
        let lstm = LSTM::new(input_size, hidden_size, num_layers, bidirectional, device);
        Self {
            input_size,
            hidden_size,
            output_size,
            attention,
            dropout,
            output,
            lstm,
        }
    }

    /// Getter for input_size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Forward pass of the LSTM model
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // Apply LSTM cell to the sequence
        let lstm_out = self.lstm.forward(x);

        // Apply attention mechanism
        let attended = self.attention.forward(lstm_out);

        // Apply pooling (currently using last step)
        let batch_size = attended.dims()[0];
        let last_step_idx = attended.dims()[1] - 1;
        let lstm_output_size = attended.dims()[2];
        
        // Extract the last step from the sequence and reshape
        let pooled = attended.narrow(1, last_step_idx, 1)
            .reshape([batch_size, lstm_output_size]);
        
        // Apply dropout before the final layer
        let dropped = self.dropout.forward(pooled);
        
        // Apply the output layer - now outputs [batch_size, output_size]
        let output = self.output.forward(dropped);
        // Clamp output to [0.0, 1.0] to match normalized target range
        output.clamp(0.0, 1.0)
    }
}

/// Attention mechanism to weight the importance of different timesteps
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
}

impl<B: Backend> Attention<B> {
    /// Create a new attention module
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        let query_config = LinearConfig::new(hidden_dim, hidden_dim);
        let key_config = LinearConfig::new(hidden_dim, hidden_dim);
        let value_config = LinearConfig::new(hidden_dim, hidden_dim);

        Self {
            query: query_config.init(device),
            key: key_config.init(device),
            value: value_config.init(device),
        }
    }

    /// Forward pass through the attention mechanism
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Get dimensions
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];
        let hidden_dim = x.dims()[2];
        
        // Apply linear transformations to get query, key, and value
        // Reshape to [batch_size * seq_len, hidden_dim] for linear layers
        let x_reshaped = x.clone().reshape([batch_size * seq_len, hidden_dim]);
        
        let q = self.query.forward(x_reshaped.clone())
            .reshape([batch_size, seq_len, hidden_dim]);
        let k = self.key.forward(x_reshaped.clone())
            .reshape([batch_size, seq_len, hidden_dim]);
        let v = self.value.forward(x_reshaped)
            .reshape([batch_size, seq_len, hidden_dim]);

        // Compute attention scores (with scaling)
        let scale = (hidden_dim as f64).sqrt();
        
        // For matrix multiplication, we need to transpose k
        let k_t = k.permute([0, 2, 1]); // [batch, hidden_dim, seq_len]
        
        // Compute scores: [batch, seq_len, seq_len]
        let scores = q.matmul(k_t) / scale;

        // Apply softmax to get attention weights
        let weights = activation::softmax(scores, 2);

        // Apply attention weights to values
        weights.matmul(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::Tensor;

    #[test]
    fn test_attention_creation() {
        let device = NdArrayDevice::default();
        let hidden_dim = 64;
        let attention: Attention<NdArray> = Attention::new(hidden_dim, &device);

        // Check weight dimensions for query, key, and value projections
        // Each should map from hidden_dim to hidden_dim: [out_features, in_features]
        assert_eq!(attention.query.weight.dims(), [hidden_dim, hidden_dim]);
        assert_eq!(attention.key.weight.dims(), [hidden_dim, hidden_dim]);
        assert_eq!(attention.value.weight.dims(), [hidden_dim, hidden_dim]);
    }

    #[test]
    fn test_attention_forward() {
        let device = NdArrayDevice::default();
        let hidden_dim = 64;
        let attention: Attention<NdArray> = Attention::new(hidden_dim, &device);

        // Create input tensor [batch_size, seq_len, hidden_dim]
        let batch_size = 2;
        let seq_len = 5;
        
        let input = Tensor::<NdArray, 3>::ones([batch_size, seq_len, hidden_dim], &device);
        
        // Test forward pass
        let output = attention.forward(input);
        
        // Output should maintain the same dimensions
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_dim]);
    }

    #[test]
    fn test_lstm_model_creation() {
        let device = NdArrayDevice::default();
        let input_size = 10;
        let hidden_size = 64;
        let output_size = 1;
        let num_layers = 2;
        let dropout_prob = 0.1;
        
        let model: TimeSeriesLstm<NdArray> = TimeSeriesLstm::new(
            input_size,
            hidden_size,
            output_size,
            num_layers,
            false,
            dropout_prob,
            &device
        );

        // Check model parameters
        assert_eq!(model.input_size, input_size);
        assert_eq!(model.hidden_size, hidden_size);
        
        // Check output layer dimensions [in_features, out_features]
        assert_eq!(model.output.weight.dims(), [hidden_size, output_size]);
    }

    #[test]
    fn test_lstm_model_forward() {
        let device = NdArrayDevice::default();
        let input_size = 10;
        let hidden_size = 64;
        let output_size = 1;
        let num_layers = 2;
        let dropout_prob = 0.1;
        
        let model: TimeSeriesLstm<NdArray> = TimeSeriesLstm::new(
            input_size,
            hidden_size,
            output_size,
            num_layers,
            false,
            dropout_prob,
            &device
        );

        // Create input tensor [batch_size, seq_len, input_size]
        let batch_size = 2;
        let seq_len = 20;
        
        let input = Tensor::<NdArray, 3>::ones([batch_size, seq_len, input_size], &device);
        
        // Test forward pass
        let output = model.forward(input);
        
        // Check output dimensions [batch_size, output_size]
        assert_eq!(output.dims(), [batch_size, output_size]);
    }
}
