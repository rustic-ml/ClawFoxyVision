// External imports
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

/// # GRU Cell Implementation
///
/// The Gated Recurrent Unit (GRU) is a type of recurrent neural network that is more
/// efficient than LSTM while offering similar performance for many tasks. It was introduced
/// by Cho et al. in 2014.
///
/// ## Architecture
///
/// A GRU has two gates:
/// - **Update Gate (z)**: Controls how much of the previous hidden state should be kept
/// - **Reset Gate (r)**: Controls how much of the previous hidden state should influence the new candidate state
///
/// Unlike LSTM which has separate cell and hidden states, GRU combines them into a single hidden state.
///
/// ## Mathematical Representation
///
/// For input x_t at time t and previous hidden state h_(t-1):
///
/// 1. Update gate: z_t = σ(W_z · x_t + U_z · h_(t-1))
/// 2. Reset gate: r_t = σ(W_r · x_t + U_r · h_(t-1))
/// 3. Candidate state: n_t = tanh(W_n · x_t + r_t ∘ (U_n · h_(t-1)))
/// 4. New hidden state: h_t = (1 - z_t) ∘ n_t + z_t ∘ h_(t-1)
///
/// Where:
/// - σ is the sigmoid function
/// - ∘ denotes element-wise multiplication
///
#[derive(Module, Debug)]
pub struct GRU<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,

    // Single-layer GRU components
    input_weights: Linear<B>,
    hidden_weights: Linear<B>,

    // Optional bidirectional components
    reverse_input_weights: Option<Linear<B>>,
    reverse_hidden_weights: Option<Linear<B>>,
}

impl<B: Backend> GRU<B> {
    /// Create a new GRU cell with specified dimensions
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of expected features in the input
    /// * `hidden_size` - Number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (currently only single layer is fully implemented)
    /// * `bidirectional` - If true, becomes a bidirectional GRU
    /// * `device` - The device to allocate tensors on
    ///
    /// # Returns
    ///
    /// A new GRU cell instance with the specified configuration
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        device: &B::Device,
    ) -> Self {
        // Initialize weights
        // For GRU we need 3 gates (reset, update, new) combined
        let gate_size = 3 * hidden_size;

        // Create linear layers with correct dimensions [in_features, out_features]
        let input_weights_config = LinearConfig::new(input_size, gate_size);
        let hidden_weights_config = LinearConfig::new(hidden_size, gate_size);

        let input_weights = input_weights_config.init(device);
        let hidden_weights = hidden_weights_config.init(device);

        // Initialize bidirectional components
        let (reverse_input_weights, reverse_hidden_weights) = if bidirectional {
            let rev_input_weights_config = LinearConfig::new(input_size, gate_size);
            let rev_hidden_weights_config = LinearConfig::new(hidden_size, gate_size);

            let rev_input_weights = rev_input_weights_config.init(device);
            let rev_hidden_weights = rev_hidden_weights_config.init(device);

            (Some(rev_input_weights), Some(rev_hidden_weights))
        } else {
            (None, None)
        };

        Self {
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            input_weights,
            hidden_weights,
            reverse_input_weights,
            reverse_hidden_weights,
        }
    }

    /// Process a sequence in a single direction (forward or backward)
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_size]
    /// * `reverse` - If true, process the sequence backwards
    /// * `device` - Device to allocate tensors on
    ///
    /// # Returns
    ///
    /// Output tensor containing hidden states for each time step with shape [batch_size, seq_len, hidden_size]
    fn process_direction(
        &self,
        x: Tensor<B, 3>,
        reverse: bool,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        // Select the appropriate weights based on direction
        let (input_weights, hidden_weights) = if reverse && self.bidirectional {
            (
                self.reverse_input_weights.as_ref().unwrap(),
                self.reverse_hidden_weights.as_ref().unwrap(),
            )
        } else {
            (&self.input_weights, &self.hidden_weights)
        };

        // Initial hidden state (zeros)
        let mut h = Tensor::zeros([batch_size, self.hidden_size], device);

        // Initialize a tensor to store the sequence of hidden states
        let mut output_sequence = Tensor::zeros([batch_size, seq_len, self.hidden_size], device);

        // Process the sequence
        for t in 0..seq_len {
            // Get the input at the current time step
            // If processing in reverse, flip the sequence order
            let time_idx = if reverse { seq_len - 1 - t } else { t };
            let x_t = x
                .clone()
                .narrow(1, time_idx, 1)
                .reshape([batch_size, self.input_size]);

            // Calculate gates with correct matrix multiplication
            let input_projection = input_weights.forward(x_t);
            let hidden_projection = hidden_weights.forward(h.clone());

            // Split into individual gates
            // For GRU, we have 3 gates (update, reset, new/candidate)

            // Input projections
            let input_gates = input_projection.reshape([batch_size, 3, self.hidden_size]);
            let z_input = input_gates
                .clone()
                .narrow(1, 0, 1)
                .reshape([batch_size, self.hidden_size]); // update gate
            let r_input = input_gates
                .clone()
                .narrow(1, 1, 1)
                .reshape([batch_size, self.hidden_size]); // reset gate
            let n_input = input_gates
                .narrow(1, 2, 1)
                .reshape([batch_size, self.hidden_size]); // new gate

            // Hidden projections
            let hidden_gates = hidden_projection.reshape([batch_size, 3, self.hidden_size]);
            let z_hidden = hidden_gates
                .clone()
                .narrow(1, 0, 1)
                .reshape([batch_size, self.hidden_size]); // update gate
            let r_hidden = hidden_gates
                .clone()
                .narrow(1, 1, 1)
                .reshape([batch_size, self.hidden_size]); // reset gate
            let n_hidden = hidden_gates
                .narrow(1, 2, 1)
                .reshape([batch_size, self.hidden_size]); // new gate

            // Apply activations for gates
            let z = activation::sigmoid(z_input + z_hidden); // update gate
            let r = activation::sigmoid(r_input + r_hidden); // reset gate

            // Calculate candidate hidden state using the reset gate
            let n = activation::tanh(n_input + (r * n_hidden)); // candidate hidden state

            // Update hidden state using update gate
            // h = (1-z) * n + z * h - this is the key GRU equation
            h = (Tensor::ones_like(&z) - z.clone()) * n + z * h;

            // Store the hidden state in the output sequence
            output_sequence = output_sequence.slice_assign(
                [0..batch_size, t..t + 1, 0..self.hidden_size],
                h.clone().reshape([batch_size, 1, self.hidden_size]),
            );
        }

        output_sequence
    }

    /// Forward pass through the GRU for the entire input sequence
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_size]
    ///
    /// # Returns
    ///
    /// Output tensor containing hidden states for each time step.
    /// If bidirectional, the output shape is [batch_size, seq_len, 2*hidden_size],
    /// otherwise it's [batch_size, seq_len, hidden_size].
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        // Process forward direction
        let forward_output = self.process_direction(x.clone(), false, &device);

        if self.bidirectional {
            // Process reverse direction
            let reverse_output = self.process_direction(x, true, &device);

            // Concatenate forward and reverse outputs along the last dimension
            let mut combined_output =
                Tensor::zeros([batch_size, seq_len, 2 * self.hidden_size], &device);

            for t in 0..seq_len {
                let forward_h = forward_output.clone().narrow(1, t, 1);
                let reverse_h = reverse_output.clone().narrow(1, t, 1);

                // Concatenate the two hidden states
                let combined_h = Tensor::cat(vec![forward_h, reverse_h], 2);

                // Assign to the combined output
                combined_output = combined_output.slice_assign(
                    [0..batch_size, t..t + 1, 0..2 * self.hidden_size],
                    combined_h.reshape([batch_size, 1, 2 * self.hidden_size]),
                );
            }

            combined_output
        } else {
            forward_output
        }
    }
}
