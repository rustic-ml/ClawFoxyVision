// External imports
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

/// LSTM Cell implementation
#[derive(Module, Debug)]
pub struct LSTM<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,

    // Single-layer LSTM components
    input_weights: Linear<B>,
    hidden_weights: Linear<B>,

    // Optional bidirectional components
    reverse_input_weights: Option<Linear<B>>,
    reverse_hidden_weights: Option<Linear<B>>,
}

impl<B: Backend> LSTM<B> {
    /// Create a new LSTM cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        device: &B::Device,
    ) -> Self {
        // Initialize weights
        // For simplicity, we're using a single linear layer for all gates combined
        let gate_size = 4 * hidden_size; // input, forget, cell, output gates combined

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

    /// Process a single direction of the LSTM
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

        // Initial hidden and cell states (zeros)
        let mut h = Tensor::zeros([batch_size, self.hidden_size], device);
        let mut c = Tensor::zeros([batch_size, self.hidden_size], device);

        // Initialize a tensor to store the sequence of hidden states
        let mut output_sequence = Tensor::zeros([batch_size, seq_len, self.hidden_size], device);

        // Process the sequence
        for t in 0..seq_len {
            // Get the input at the current time step
            let time_idx = if reverse { seq_len - 1 - t } else { t };
            let x_t = x
                .clone()
                .narrow(1, time_idx, 1)
                .reshape([batch_size, self.input_size]);

            // Calculate gates with correct matrix multiplication
            let input_projection = input_weights.forward(x_t);
            let hidden_projection = hidden_weights.forward(h);

            // Combine projections and reshape for gate operations
            let gates = input_projection + hidden_projection;

            // Split into individual gates - each gate should have shape [batch_size, hidden_size]
            let gates = gates.reshape([batch_size, 4, self.hidden_size]);
            let i_gate = gates
                .clone()
                .narrow(1, 0, 1)
                .reshape([batch_size, self.hidden_size]);
            let f_gate = gates
                .clone()
                .narrow(1, 1, 1)
                .reshape([batch_size, self.hidden_size]);
            let g_gate = gates
                .clone()
                .narrow(1, 2, 1)
                .reshape([batch_size, self.hidden_size]);
            let o_gate = gates
                .narrow(1, 3, 1)
                .reshape([batch_size, self.hidden_size]);

            // Apply activations
            let i = activation::sigmoid(i_gate);
            let f = activation::sigmoid(f_gate);
            let g = activation::tanh(g_gate);
            let o = activation::sigmoid(o_gate);

            // Update cell state
            c = f * c + i * g;

            // Update hidden state
            h = o * activation::tanh(c.clone());

            // Store the hidden state in the output sequence
            output_sequence = output_sequence.slice_assign(
                [0..batch_size, t..t + 1, 0..self.hidden_size],
                h.clone()
                    .unsqueeze::<3>()
                    .reshape([batch_size, 1, self.hidden_size]),
            );
        }

        output_sequence
    }

    /// Forward pass through the LSTM
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
