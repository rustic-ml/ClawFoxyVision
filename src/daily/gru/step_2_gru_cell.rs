// External imports
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

/// GRU Cell implementation for daily stock prediction
#[derive(Module, Debug)]
pub struct DailyGRU<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    dropout_rate: f64,

    // GRU components
    update_gate_input: Linear<B>,
    update_gate_hidden: Linear<B>,
    reset_gate_input: Linear<B>,
    reset_gate_hidden: Linear<B>,
    output_gate_input: Linear<B>,
    output_gate_hidden: Linear<B>,
}

impl<B: Backend> DailyGRU<B> {
    /// Create a new GRU cell for daily prediction
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Size of hidden state
    /// * `dropout_rate` - Dropout rate for regularization
    /// * `device` - Device to place tensors on
    ///
    /// # Returns
    ///
    /// Returns a new GRU cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        dropout_rate: f64,
        device: &B::Device,
    ) -> Self {
        // Initialize the gate weights
        let update_gate_input_config = LinearConfig::new(input_size, hidden_size);
        let update_gate_hidden_config = LinearConfig::new(hidden_size, hidden_size);
        let reset_gate_input_config = LinearConfig::new(input_size, hidden_size);
        let reset_gate_hidden_config = LinearConfig::new(hidden_size, hidden_size);
        let output_gate_input_config = LinearConfig::new(input_size, hidden_size);
        let output_gate_hidden_config = LinearConfig::new(hidden_size, hidden_size);

        // Create the linear layers
        let update_gate_input = update_gate_input_config.init(device);
        let update_gate_hidden = update_gate_hidden_config.init(device);
        let reset_gate_input = reset_gate_input_config.init(device);
        let reset_gate_hidden = reset_gate_hidden_config.init(device);
        let output_gate_input = output_gate_input_config.init(device);
        let output_gate_hidden = output_gate_hidden_config.init(device);

        Self {
            input_size,
            hidden_size,
            dropout_rate,
            update_gate_input,
            update_gate_hidden,
            reset_gate_input,
            reset_gate_hidden,
            output_gate_input,
            output_gate_hidden,
        }
    }

    /// Forward pass through the GRU cell
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, sequence_length, input_size]
    ///
    /// # Returns
    ///
    /// Returns the output tensor of shape [batch_size, sequence_length, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let batch_size = x.dims()[0];
        let sequence_length = x.dims()[1];

        // Initialize hidden state
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &device);

        // Initialize output tensor
        let mut outputs = Tensor::zeros([batch_size, sequence_length, self.hidden_size], &device);

        // Process each time step
        for t in 0..sequence_length {
            // Get the input at the current time step [batch_size, input_size]
            let x_t = x
                .clone()
                .narrow(1, t, 1)
                .reshape([batch_size, self.input_size]);

            // Calculate update gate
            let z_t = activation::sigmoid(
                self.update_gate_input.forward(x_t.clone())
                    + self.update_gate_hidden.forward(h.clone()),
            );

            // Calculate reset gate
            let r_t = activation::sigmoid(
                self.reset_gate_input.forward(x_t.clone())
                    + self.reset_gate_hidden.forward(h.clone()),
            );

            // Calculate candidate hidden state
            let h_tilde = activation::tanh(
                self.output_gate_input.forward(x_t)
                    + self.output_gate_hidden.forward(r_t * h.clone()),
            );

            // Update hidden state
            h = (Tensor::ones_like(&z_t) - z_t.clone()) * h_tilde + z_t * h;

            // Store the output for this time step
            let h_reshaped = h.clone().reshape([batch_size, 1, self.hidden_size]);
            outputs =
                outputs.slice_assign([0..batch_size, t..t + 1, 0..self.hidden_size], h_reshaped);
        }

        outputs
    }

    /// Get the hidden size of the GRU
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
