// External imports
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

/// LSTM Cell implementation for daily stock prediction
#[derive(Module, Debug)]
pub struct DailyLSTM<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    dropout_rate: f64,

    // LSTM components
    input_gate: Linear<B>,
    forget_gate: Linear<B>,
    cell_gate: Linear<B>,
    output_gate: Linear<B>,

    // Recurrent connections
    input_recurrent: Linear<B>,
    forget_recurrent: Linear<B>,
    cell_recurrent: Linear<B>,
    output_recurrent: Linear<B>,
}

impl<B: Backend> DailyLSTM<B> {
    /// Create a new LSTM cell for daily prediction
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
    /// Returns a new LSTM cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        dropout_rate: f64,
        device: &B::Device,
    ) -> Self {
        // Initialize the gate weights
        let input_gate_config = LinearConfig::new(input_size, hidden_size);
        let forget_gate_config = LinearConfig::new(input_size, hidden_size);
        let cell_gate_config = LinearConfig::new(input_size, hidden_size);
        let output_gate_config = LinearConfig::new(input_size, hidden_size);

        // Initialize the recurrent weights
        let input_recurrent_config = LinearConfig::new(hidden_size, hidden_size);
        let forget_recurrent_config = LinearConfig::new(hidden_size, hidden_size);
        let cell_recurrent_config = LinearConfig::new(hidden_size, hidden_size);
        let output_recurrent_config = LinearConfig::new(hidden_size, hidden_size);

        // Create the linear layers
        let input_gate = input_gate_config.init(device);
        let forget_gate = forget_gate_config.init(device);
        let cell_gate = cell_gate_config.init(device);
        let output_gate = output_gate_config.init(device);

        let input_recurrent = input_recurrent_config.init(device);
        let forget_recurrent = forget_recurrent_config.init(device);
        let cell_recurrent = cell_recurrent_config.init(device);
        let output_recurrent = output_recurrent_config.init(device);

        Self {
            input_size,
            hidden_size,
            dropout_rate,
            input_gate,
            forget_gate,
            cell_gate,
            output_gate,
            input_recurrent,
            forget_recurrent,
            cell_recurrent,
            output_recurrent,
        }
    }

    /// Forward pass through the LSTM cell
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

        // Initialize hidden and cell states
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &device);
        let mut c = Tensor::zeros([batch_size, self.hidden_size], &device);

        // Initialize output tensor
        let mut outputs = Tensor::zeros([batch_size, sequence_length, self.hidden_size], &device);

        // Process each time step
        for t in 0..sequence_length {
            // Get the input at the current time step [batch_size, input_size]
            let x_t = x
                .clone()
                .narrow(1, t, 1)
                .reshape([batch_size, self.input_size]);

            // Calculate gate values
            let i_t = activation::sigmoid(
                self.input_gate.forward(x_t.clone()) + self.input_recurrent.forward(h.clone()),
            );

            let f_t = activation::sigmoid(
                self.forget_gate.forward(x_t.clone()) + self.forget_recurrent.forward(h.clone()),
            );

            let g_t = activation::tanh(
                self.cell_gate.forward(x_t.clone()) + self.cell_recurrent.forward(h.clone()),
            );

            let o_t = activation::sigmoid(
                self.output_gate.forward(x_t) + self.output_recurrent.forward(h.clone()),
            );

            // Update cell state
            c = f_t * c + i_t * g_t;

            // Update hidden state
            h = o_t * activation::tanh(c.clone());

            // Store the output for this time step
            let h_reshaped = h.clone().reshape([batch_size, 1, self.hidden_size]);
            outputs =
                outputs.slice_assign([0..batch_size, t..t + 1, 0..self.hidden_size], h_reshaped);
        }

        outputs
    }

    /// Get the hidden size of the LSTM
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
