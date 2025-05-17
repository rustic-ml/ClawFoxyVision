// External imports
use crate::constants::{DEFAULT_DROPOUT, L2_REGULARIZATION};
use crate::minute::cnnlstm::step_2_cnn_lstm_cell::CNNLSTM;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{activation, Tensor};

/// TimeSeriesCnnLstm architecture for forecasting
/// This combines CNN for feature extraction with LSTM for temporal modeling
#[derive(Module, Debug)]
pub struct TimeSeriesCnnLstm<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    // Removed attention to save memory
    dropout: Dropout,
    output: Linear<B>,
    cnn_lstm: CNNLSTM<B>,
    regularization: f64,
}

impl<B: Backend> TimeSeriesCnnLstm<B> {
    /// Create a new TimeSeriesCnnLstm model
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

        // Configure CNN-LSTM
        let lstm_output_size = if bidirectional {
            2 * hidden_size
        } else {
            hidden_size
        };

        // Single dropout layer to reduce memory
        let dropout_config = DropoutConfig::new(dropout_prob);
        let dropout = dropout_config.init();

        // Configure output layer
        let output_config = LinearConfig::new(lstm_output_size, output_size);
        let output = output_config.init(device);

        // Create CNN-LSTM cell
        let cnn_lstm = CNNLSTM::new(input_size, hidden_size, num_layers, bidirectional, device);

        Self {
            input_size,
            hidden_size,
            output_size,
            dropout,
            output,
            cnn_lstm,
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

    /// Forward pass of the CNN-LSTM model with added dropout
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // Apply CNN-LSTM cell to the sequence
        let cnn_lstm_out = self.cnn_lstm.forward(x);

        // Apply pooling (using last step instead of attention)
        let batch_size = cnn_lstm_out.dims()[0];
        let last_step_idx = cnn_lstm_out.dims()[1] - 1;
        let lstm_output_size = cnn_lstm_out.dims()[2];

        // Extract the last step from the sequence and reshape
        let pooled = cnn_lstm_out
            .narrow(1, last_step_idx, 1)
            .reshape([batch_size, lstm_output_size]);

        // Apply dropout before the final layer
        let dropped = self.dropout.forward(pooled);

        // Apply the output layer
        let output = self.output.forward(dropped);

        // Clamp output to [0.0, 1.0] to match normalized target range
        output.clamp(0.0, 1.0)
    }

    /// Calculates L2 regularization penalty
    pub fn l2_penalty(&self) -> Tensor<B, 1> {
        let device = &self.output.weight.device();

        // Sum squared weights from all layers
        let mut squared_sum = Tensor::zeros([1], device);

        // Add output layer weights
        let output_weights = self.output.weight.val().clone();
        // Calculate sum of squared weights using element-wise multiplication
        let weight_squared = output_weights.clone() * output_weights;
        squared_sum = squared_sum + weight_squared.sum();

        // Scale by regularization strength
        squared_sum * self.regularization
    }

    /// Huber loss function, a combination of MSE and MAE that is more robust to outliers
    pub fn huber_loss(
        &self,
        pred: Tensor<B, 2>,
        target: Tensor<B, 2>,
        _delta: f64,
    ) -> Tensor<B, 1> {
        // Compute mean squared error
        let diff = pred - target;
        let squared_diff = diff.clone() * diff;
        
        // Fix: Ensure mean() results in a Tensor<B, 1>
        let mse = squared_diff.mean().reshape([1]);

        // Add L2 regularization if configured
        if self.regularization > 0.0 {
            let weight_squared =
                self.output.weight.val().clone() * self.output.weight.val().clone();
            let l2_penalty = (weight_squared.sum() * self.regularization).reshape([1]);
            mse + l2_penalty
        } else {
            mse
        }
    }

    /// Calculate MSE loss with L2 regularization
    pub fn mse_loss(&self, pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
        let diff = pred - target;
        let squared_diff = diff.clone() * diff;
        
        // Fix: Ensure mean() results in a Tensor<B, 1>
        let mse = squared_diff.mean().reshape([1]);

        if self.regularization > 0.0 {
            let weight_squared =
                self.output.weight.val().clone() * self.output.weight.val().clone();
            let l2_penalty = (weight_squared.sum() * self.regularization).reshape([1]);
            mse + l2_penalty
        } else {
            mse
        }
    }
} 