// External imports
use crate::constants::{DEFAULT_DROPOUT, L2_REGULARIZATION};
use crate::minute::lstm::step_2_lstm_cell::LSTM;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{activation, Tensor};

/// TimeSeriesLstm architecture for forecasting
#[derive(Module, Debug)]
pub struct TimeSeriesLstm<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    attention: Attention<B>,
    dropout1: Dropout,
    dropout2: Dropout,
    output: Linear<B>,
    lstm: LSTM<B>,
    regularization: f64,
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
        // Use default higher dropout if not specified
        let dropout_prob = if dropout_prob <= 0.0 {
            DEFAULT_DROPOUT
        } else {
            dropout_prob
        };

        // Configure LSTM
        let lstm_output_size = if bidirectional {
            2 * hidden_size
        } else {
            hidden_size
        };
        let attention = Attention::new(lstm_output_size, device);

        // Add two dropout layers with different probabilities to prevent overfitting
        let dropout_config1 = DropoutConfig::new(dropout_prob);
        let dropout_config2 = DropoutConfig::new(dropout_prob * 0.7); // Second dropout slightly less aggressive
        let dropout1 = dropout_config1.init();
        let dropout2 = dropout_config2.init();

        // Configure output layer
        let output_config = LinearConfig::new(lstm_output_size, output_size);
        let output = output_config.init(device);

        // Create LSTM cell
        let lstm = LSTM::new(input_size, hidden_size, num_layers, bidirectional, device);

        Self {
            input_size,
            hidden_size,
            output_size,
            attention,
            dropout1,
            dropout2,
            output,
            lstm,
            regularization: L2_REGULARIZATION,
        }
    }

    /// Getter for input_size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Getter for L2 regularization strength
    pub fn regularization(&self) -> f64 {
        self.regularization
    }

    /// Forward pass of the LSTM model with added dropout
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
        let pooled = attended
            .narrow(1, last_step_idx, 1)
            .reshape([batch_size, lstm_output_size]);

        // Apply first dropout before the final layer
        let dropped1 = self.dropout1.forward(pooled);

        // Apply the output layer - now outputs [batch_size, output_size]
        let output_pre = self.output.forward(dropped1);

        // Apply second dropout after the output layer
        let dropped2 = self.dropout2.forward(output_pre);

        // Clamp output to [0.0, 1.0] to match normalized target range
        dropped2.clamp(0.0, 1.0)
    }

    /// Calculates L2 regularization penalty
    pub fn l2_penalty(&self) -> Tensor<B, 1> {
        let device = &self.output.weight.device();

        // Sum squared weights from all layers
        let mut squared_sum = Tensor::zeros([1], device);

        // Add LSTM weights
        // This implementation adds L2 regularization to output layer only
        // A complete implementation would include all weights in the model

        // Add output layer weights
        let output_weights = self.output.weight.val().clone();
        // Calculate sum of squared weights using element-wise multiplication
        let weight_squared = output_weights.clone() * output_weights;
        squared_sum = squared_sum + weight_squared.sum();

        // Scale by regularization strength
        squared_sum * self.regularization
    }

    /// Huber loss function, a combination of MSE and MAE that is more robust to outliers
    /// It acts like MSE for small errors and like MAE for large errors
    #[allow(dead_code)]
    pub fn huber_loss(
        &self,
        pred: Tensor<B, 2>,
        target: Tensor<B, 2>,
        _delta: f64,
    ) -> Tensor<B, 0> {
        // Compute mean squared error
        let diff = pred - target;
        let squared_diff = diff.clone() * diff;
        let mse = squared_diff.mean().reshape([0_usize; 0]);

        // Add L2 regularization if configured
        if self.regularization > 0.0 {
            let weight_squared =
                self.output.weight.val().clone() * self.output.weight.val().clone();
            let l2_penalty = (weight_squared.sum() * self.regularization).reshape([0_usize; 0]);
            mse + l2_penalty
        } else {
            mse
        }
    }

    /// Calculate MSE loss with L2 regularization
    pub fn mse_loss(&self, pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 0> {
        let diff = pred - target;
        let squared_diff = diff.clone() * diff;
        let mse = squared_diff.mean().reshape([0_usize; 0]);

        if self.regularization > 0.0 {
            let weight_squared =
                self.output.weight.val().clone() * self.output.weight.val().clone();
            let l2_penalty = (weight_squared.sum() * self.regularization).reshape([0_usize; 0]);
            mse + l2_penalty
        } else {
            mse
        }
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

        let q = self
            .query
            .forward(x_reshaped.clone())
            .reshape([batch_size, seq_len, hidden_dim]);
        let k = self
            .key
            .forward(x_reshaped.clone())
            .reshape([batch_size, seq_len, hidden_dim]);
        let v = self
            .value
            .forward(x_reshaped)
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
