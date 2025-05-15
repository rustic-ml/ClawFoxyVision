// External imports
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

// Internal imports
use super::step_2_gru_cell::DailyGRU;

/// GRU model architecture for daily stock price prediction
#[derive(Module, Debug)]
pub struct DailyGRUModel<B: Backend> {
    // Model hyperparameters
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    // Model layers
    gru: DailyGRU<B>,
    dropout: Dropout,
    output_layer: Linear<B>,
}

impl<B: Backend> DailyGRUModel<B> {
    /// Create a new GRU model for daily stock prediction
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Size of hidden state
    /// * `output_size` - Size of output (usually 1 for regression)
    /// * `dropout_rate` - Dropout rate for regularization
    /// * `device` - Device to place tensors on
    ///
    /// # Returns
    ///
    /// Returns a new GRU model
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        dropout_rate: f64,
        device: &B::Device,
    ) -> Self {
        // Create GRU layer
        let gru = DailyGRU::new(input_size, hidden_size, dropout_rate, device);

        // Create dropout layer
        let dropout_config = DropoutConfig::new(dropout_rate);
        let dropout = dropout_config.init();

        // Create output layer
        let output_config = LinearConfig::new(hidden_size, output_size);
        let output_layer = output_config.init(device);

        Self {
            input_size,
            hidden_size,
            output_size,
            gru,
            dropout,
            output_layer,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, sequence_length, input_size]
    /// * `is_training` - Whether the model is in training mode (affects dropout)
    ///
    /// # Returns
    ///
    /// Returns the output tensor of shape [batch_size, output_size]
    pub fn forward(&self, x: Tensor<B, 3>, is_training: bool) -> Tensor<B, 2> {
        let batch_size = x.dims()[0];
        let sequence_length = x.dims()[1];

        // Pass input through GRU layer
        let gru_out = self.gru.forward(x);

        // We only need the last output from the sequence
        let last_output = gru_out
            .narrow(1, sequence_length - 1, 1)
            .reshape([batch_size, self.hidden_size]);

        // Apply dropout if in training mode
        let dropped = if is_training {
            self.dropout.forward(last_output)
        } else {
            last_output
        };

        // Pass through output layer
        self.output_layer.forward(dropped)
    }

    /// Predict using the model (convenience wrapper around forward)
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, sequence_length, input_size]
    ///
    /// # Returns
    ///
    /// Returns the prediction tensor of shape [batch_size, output_size]
    pub fn predict(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x, false)
    }
}

/// Configuration for the DailyGRUModel
#[derive(Debug, Clone)]
pub struct DailyGRUModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub dropout_rate: f64,
}

impl DailyGRUModelConfig {
    /// Create a new configuration for the DailyGRUModel
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Size of hidden state
    /// * `output_size` - Size of output (usually 1 for regression)
    /// * `dropout_rate` - Dropout rate for regularization
    ///
    /// # Returns
    ///
    /// Returns a new configuration
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        dropout_rate: f64,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            dropout_rate,
        }
    }

    /// Initialize a model from this configuration
    ///
    /// # Arguments
    ///
    /// * `device` - Device to place tensors on
    ///
    /// # Returns
    ///
    /// Returns a new GRU model
    pub fn init<B: Backend>(&self, device: &B::Device) -> DailyGRUModel<B> {
        DailyGRUModel::new(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.dropout_rate,
            device,
        )
    }
}
