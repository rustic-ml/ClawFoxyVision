// External imports
use crate::constants::{DEFAULT_DROPOUT, L2_REGULARIZATION};
use crate::minute::gru::step_2_gru_cell::GRU;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{activation, Tensor};

/// # TimeSeriesGru Architecture
///
/// A GRU-based architecture specialized for time series forecasting. It combines a GRU layer
/// with attention mechanisms and multiple dropout layers to improve prediction accuracy.
///
/// ## Architecture Overview
///
/// 1. **Input Layer**: Accepts time series features in shape [batch_size, seq_len, input_size]
/// 2. **GRU Layer**: Processes sequential data, with optional bidirectional processing
/// 3. **Attention Layer**: Learns to focus on the most important time steps
/// 4. **Dropout Layers**: Two dropout layers to prevent overfitting
/// 5. **Output Layer**: Projects to the desired forecast horizon dimension
///
/// ## Key Features
///
/// - **Bidirectional Processing**: Can process the time series in both directions
/// - **Attention Mechanism**: Helps the model focus on relevant time steps
/// - **Regularization**: L2 regularization and dropout to prevent overfitting
/// - **Multiple Loss Functions**: Supports both MSE and Huber loss
///
#[derive(Module, Debug)]
pub struct TimeSeriesGru<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    attention: Attention<B>,
    dropout1: Dropout,
    dropout2: Dropout,
    output: Linear<B>,
    gru: GRU<B>,
    regularization: f64,
}

impl<B: Backend> TimeSeriesGru<B> {
    /// Creates a new TimeSeriesGru model configured for time series forecasting
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features per time step
    /// * `hidden_size` - Dimension of the GRU hidden state
    /// * `output_size` - Number of output features (forecast horizon)
    /// * `num_layers` - Number of stacked GRU layers (note: currently only single layer is implemented)
    /// * `bidirectional` - Whether to use bidirectional GRU
    /// * `dropout_prob` - Dropout probability for regularization
    /// * `device` - Device to allocate tensors on
    ///
    /// # Returns
    ///
    /// A configured TimeSeriesGru model ready for training or inference
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

        // Configure GRU
        let gru_output_size = if bidirectional {
            2 * hidden_size
        } else {
            hidden_size
        };
        let attention = Attention::new(gru_output_size, device);

        // Add two dropout layers with different probabilities to prevent overfitting
        let dropout_config1 = DropoutConfig::new(dropout_prob);
        let dropout_config2 = DropoutConfig::new(dropout_prob * 0.7); // Second dropout slightly less aggressive
        let dropout1 = dropout_config1.init();
        let dropout2 = dropout_config2.init();

        // Configure output layer
        let output_config = LinearConfig::new(gru_output_size, output_size);
        let output = output_config.init(device);

        // Create GRU cell
        let gru = GRU::new(input_size, hidden_size, num_layers, bidirectional, device);

        Self {
            input_size,
            hidden_size,
            output_size,
            attention,
            dropout1,
            dropout2,
            output,
            gru,
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

    /// Performs forward pass through the GRU model to generate predictions
    ///
    /// # Process Flow
    ///
    /// 1. Pass input through the GRU layer
    /// 2. Apply attention mechanism to focus on important time steps
    /// 3. Extract the relevant time step (currently the last one)
    /// 4. Apply dropout for regularization
    /// 5. Project to output dimension
    /// 6. Apply second dropout
    /// 7. Clamp values to [0,1] range to match normalized target
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_size]
    ///
    /// # Returns
    ///
    /// Predictions tensor of shape [batch_size, output_size]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // Apply GRU cell to the sequence
        let gru_out = self.gru.forward(x);

        // Apply attention mechanism
        let attended = self.attention.forward(gru_out);

        // Apply pooling (currently using last step)
        let batch_size = attended.dims()[0];
        let last_step_idx = attended.dims()[1] - 1;
        let gru_output_size = attended.dims()[2];

        // Extract the last step from the sequence and reshape
        let pooled = attended
            .narrow(1, last_step_idx, 1)
            .reshape([batch_size, gru_output_size]);

        // Apply first dropout before the final layer
        let dropped1 = self.dropout1.forward(pooled);

        // Apply the output layer - now outputs [batch_size, output_size]
        let output_pre = self.output.forward(dropped1);

        // Apply second dropout after the output layer
        let dropped2 = self.dropout2.forward(output_pre);

        // Clamp output to [0.0, 1.0] to match normalized target range
        dropped2.clamp(0.0, 1.0)
    }

    /// Calculates L2 regularization penalty for the model weights
    ///
    /// This helps prevent overfitting by penalizing large weight values
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

    /// Huber loss function for more robust regression against outliers
    ///
    /// Combines MSE for small errors and MAE (Mean Absolute Error) for large errors,
    /// making it less sensitive to outliers than pure MSE.
    pub fn huber_loss(
        &self,
        pred: Tensor<B, 2>,
        target: Tensor<B, 2>,
        _delta: f64,
    ) -> Tensor<B, 1> {
        // Compute mean squared error first with traditional method to avoid dimension issues
        let diff = pred - target;
        let squared_diff = diff.clone() * diff.clone();

        // Use tensor operations to get a scalar result
        let total = squared_diff.sum();
        let count = diff.dims().iter().product::<usize>() as f64;
        let mse = total / count;

        // For L2 regularization
        if self.regularization > 0.0 {
            // Get the sum of squared weights
            let weight_squared =
                self.output.weight.val().clone() * self.output.weight.val().clone();
            let l2_sum = weight_squared.sum();

            // Add scaled regularization to MSE
            mse + (l2_sum * self.regularization)
        } else {
            mse
        }
    }

    /// Calculate MSE loss with L2 regularization
    ///
    /// The standard loss function for regression problems, with added
    /// L2 regularization to prevent overfitting.
    pub fn mse_loss(&self, pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
        // Compute MSE using tensor operations to avoid dimension issues
        let diff = pred - target;
        let squared_diff = diff.clone() * diff.clone();

        // Use tensor operations to get a scalar result
        let total = squared_diff.sum();
        let count = diff.dims().iter().product::<usize>() as f64;
        let mse = total / count;

        // For L2 regularization
        if self.regularization > 0.0 {
            // Get the sum of squared weights
            let weight_squared =
                self.output.weight.val().clone() * self.output.weight.val().clone();
            let l2_sum = weight_squared.sum();

            // Add scaled regularization to MSE
            mse + (l2_sum * self.regularization)
        } else {
            mse
        }
    }
}

/// # Attention Mechanism
///
/// Implements a self-attention mechanism for time series that helps the model
/// focus on the most relevant time steps when making predictions.
///
/// ## How Attention Works
///
/// 1. Projects the input into Query, Key, and Value spaces
/// 2. Computes similarity between Query and Key to get attention weights
/// 3. Scales and applies softmax to get probability distribution
/// 4. Uses weights to create a weighted sum of Value vectors
///
/// This implementation follows the general self-attention pattern used in
/// transformer architectures, but adapted for time series data.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
}

impl<B: Backend> Attention<B> {
    /// Creates a new attention module
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Dimension of the hidden states to attend to
    /// * `device` - Device to allocate tensors on
    ///
    /// # Returns
    ///
    /// An initialized Attention module
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

    /// Applies the attention mechanism to the input sequence
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, hidden_dim]
    ///
    /// # Returns
    ///
    /// Attention-weighted sequence of shape [batch_size, seq_len, hidden_dim]
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
