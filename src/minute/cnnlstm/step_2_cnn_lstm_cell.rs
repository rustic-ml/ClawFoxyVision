// External imports
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::{activation, backend::Backend, Tensor};
use crate::minute::lstm::step_2_lstm_cell::LSTM;

/// CNN-LSTM Cell implementation
/// This combines convolutional layers for feature extraction with an LSTM for sequential processing
#[derive(Module, Debug)]
pub struct CNNLSTM<B: Backend> {
    // CNN components
    conv1: Conv1d<B>,
    
    // LSTM component (reusing existing LSTM implementation)
    lstm: LSTM<B>,
    
    // Model dimensions
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    cnn_output_size: usize,
}

impl<B: Backend> CNNLSTM<B> {
    /// Create a new CNN-LSTM cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        device: &B::Device,
    ) -> Self {
        // CNN configuration - simplified to a single layer with fewer features
        // [batch_size, sequence_length, input_size] -> [batch_size, sequence_length, cnn_output_size]
        let conv1_config = Conv1dConfig::new(input_size, 16, 3) // input channels, output channels, kernel size
            .with_padding(PaddingConfig1d::Same)
            .with_stride(1); 
            
        let conv1 = conv1_config.init(device);
        
        // The CNN output size which becomes the LSTM input
        let cnn_output_size = 16; // Reduced from 64 to 16
        
        // Create LSTM layer
        let lstm = LSTM::new(cnn_output_size, hidden_size, num_layers, bidirectional, device);

        Self {
            conv1,
            lstm,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            cnn_output_size,
        }
    }

    /// Forward pass through the CNN-LSTM cell
    /// Input shape: [batch_size, seq_len, input_size]
    /// Output shape: [batch_size, seq_len, hidden_size] (or [batch_size, seq_len, 2*hidden_size] if bidirectional)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // We need to permute from [batch, seq_len, features] to [batch, features, seq_len] for Conv1d
        let x_permuted = x.permute([0, 2, 1]);
        
        // First conv layer with ReLU
        let conv1_out = self.conv1.forward(x_permuted);
        let conv1_out = activation::relu(conv1_out);
        
        // Permute back to [batch, seq_len, features] for LSTM
        let cnn_features = conv1_out.permute([0, 2, 1]);
        
        // Pass through LSTM
        let lstm_out = self.lstm.forward(cnn_features);
        
        lstm_out
    }
} 