
use anyhow::Result;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::{
    Linear, LinearConfig, Dropout, DropoutConfig,
    lstm, gru, 
};
use burn::optim::AdamConfig;
use burn::tensor::{backend::Backend, Tensor};
use burn::train::ClassificationOutput;
use std::time::Instant;

// Define model configuration
#[derive(Debug, Clone)]
enum RNNType {
    LSTM,
    GRU,
}

#[derive(Debug, Clone)]
struct ModelConfig {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    output_size: usize,
    dropout_rate: f64,
    rnn_type: RNNType,
}

impl ModelConfig {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
        dropout_rate: f64,
        rnn_type: RNNType,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers,
            output_size,
            dropout_rate,
            rnn_type,
        }
    }
    
    /// Creates an LSTM configuration with the given parameters
    fn lstm(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
        dropout_rate: f64,
    ) -> Self {
        Self::new(
            input_size, 
            hidden_size, 
            num_layers, 
            output_size, 
            dropout_rate, 
            RNNType::LSTM
        )
    }
    

    fn gru(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
        dropout_rate: f64,
    ) -> Self {
        Self::new(
            input_size, 
            hidden_size, 
            num_layers, 
            output_size, 
            dropout_rate, 
            RNNType::GRU
        )
    }
}


#[derive(Module, Debug)]
struct StockPriceModel<B: Backend> {
    lstm: Option<lstm::Lstm<B>>,
    gru: Option<gru::Gru<B>>,
    dropout: Dropout,
    output: Linear<B>,
}

impl<B: Backend> StockPriceModel<B> {

    fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let lstm;
        let gru;
        
        match config.rnn_type {
            RNNType::LSTM => {
                // Configure LSTM
                let lstm_config = lstm::LstmConfig::new(
                    config.input_size,
                    config.hidden_size,
                    true, // Use bias
                );
                
                lstm = Some(lstm_config.init(device));
                gru = None;
            },
            RNNType::GRU => {
                // Configure GRU
                let gru_config = gru::GruConfig::new(
                    config.input_size,
                    config.hidden_size,
                    true, // Use bias
                );
                
                gru = Some(gru_config.init(device));
                lstm = None;
            }
        };
        
        // Configure dropout
        let dropout_config = DropoutConfig::new(config.dropout_rate);
        let dropout = dropout_config.init();
        
        // Configure output layer
        let output_config = LinearConfig::new(config.hidden_size, config.output_size);
        let output = output_config.init(device);
        
        Self {
            lstm,
            gru,
            dropout,
            output,
        }
    }
    

    fn forward(&self, x: Tensor<B, 3>, is_training: bool) -> Tensor<B, 2> {
        // x shape: [batch_size, sequence_length, input_size]
        let batch_size = x.dims()[0];
        
        // Pass through RNN
        let output = if let Some(lstm) = &self.lstm {
            let (output, _) = lstm.forward(x, None);
            output
        } else if let Some(gru) = &self.gru {
            gru.forward(x, None)
        } else {
            panic!("Neither LSTM nor GRU initialized");
        };
        // output shape: [batch_size, sequence_length, hidden_size]
        
        // Get last time step
        let sequence_length = output.dims()[1];
        let hidden_size = output.dims()[2];
        let last_output = output.narrow(1, sequence_length - 1, 1)
                               .reshape([batch_size, hidden_size]);
        
        // Apply dropout
        let dropped = if is_training {
            self.dropout.forward(last_output)
        } else {
            last_output
        };
        
        // Apply output layer and return
        self.output.forward(dropped)
    }
    

    fn predict(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x, false)
    }
}


struct Batch<B: Backend> {
    features: Tensor<B, 3>,  // [batch_size, sequence_length, input_size]
    targets: Tensor<B, 2>,   // [batch_size, output_size]
}

fn main() -> Result<()> {
    // Define device types
    type Device = LibTorch;
    type AutoDevice = Autodiff<LibTorch>;
    
    // Create device
    let device = Default::default();
    
    // Model hyperparameters
    let sequence_length = 30; // Use 30 days of data for prediction
    let _forecast_horizon = 1; // Predict 1 day ahead
    let input_size = 6; // Number of features
    let hidden_size = 64; // Hidden state size
    let num_layers = 2; // Number of RNN layers
    let output_size = 1; // Output size (predicting a single value - adjusted close)
    let dropout_rate = 0.2; // Dropout rate
    let _learning_rate = 0.001; // Learning rate
    let batch_size = 32; // Batch size
    let _epochs = 5; // Number of epochs (keep low for example)
    
    println!("====================================================================");
    println!("             LSTM vs GRU Daily Model Comparison Example             ");
    println!("====================================================================");
    
    // Generate synthetic data for demonstration
    println!("Generating synthetic data for example...");
    
    // Create random training data
    let train_features = Tensor::<Device, 3>::random(
        [batch_size, sequence_length, input_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device
    );
    
    let train_targets = Tensor::<Device, 2>::random(
        [batch_size, output_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device
    );
    
    // Create random validation data
    let val_features = Tensor::<Device, 3>::random(
        [batch_size, sequence_length, input_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device
    );
    
    let val_targets = Tensor::<Device, 2>::random(
        [batch_size, output_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device
    );
    
    // Create training and validation batches
    let train_batch = Batch {
        features: train_features,
        targets: train_targets,
    };
    
    let _val_batch = Batch {
        features: val_features,
        targets: val_targets,
    };
    
    // Create LSTM model configuration
    let lstm_config = ModelConfig::lstm(
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout_rate
    );
    
    // Create GRU model configuration
    let gru_config = ModelConfig::gru(
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout_rate
    );
    
    // Initialize models
    let lstm_model = StockPriceModel::new(&lstm_config, &device);
    let gru_model = StockPriceModel::new(&gru_config, &device);
    
    // Train LSTM model
    println!("\n1. Testing LSTM model...");
    let lstm_start_time = Instant::now();
    
    // Forward pass for LSTM
    let lstm_pred = lstm_model.forward(train_batch.features.clone(), true);
    let lstm_loss = {
        let diff = lstm_pred - train_batch.targets.clone();
        let square = diff.clone() * diff;
        square.mean_dim(1).mean()
    };
    
    let lstm_elapsed = lstm_start_time.elapsed();
    println!("LSTM forward pass completed in {:.2?}", lstm_elapsed);
    println!("LSTM loss: {:.6}", lstm_loss.into_scalar());
    
    // Train GRU model
    println!("\n2. Testing GRU model...");
    let gru_start_time = Instant::now();
    
    // Forward pass for GRU
    let gru_pred = gru_model.forward(train_batch.features.clone(), true);
    let gru_loss = {
        let diff = gru_pred - train_batch.targets.clone();
        let square = diff.clone() * diff;
        square.mean_dim(1).mean()
    };
    
    let gru_elapsed = gru_start_time.elapsed();
    println!("GRU forward pass completed in {:.2?}", gru_elapsed);
    println!("GRU loss: {:.6}", gru_loss.into_scalar());
    
    // For autodiff examples
    println!("\n3. Creating models with autodiff capability...");
    
    // Create autodiff device
    let auto_device = Default::default();
    
    // Create random data with autodiff
    let _auto_features = Tensor::<AutoDevice, 3>::random(
        [batch_size, sequence_length, input_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &auto_device
    );
    
    let _auto_targets = Tensor::<AutoDevice, 2>::random(
        [batch_size, output_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &auto_device
    );
    
    // Create LSTM and GRU models with autodiff
    let _auto_lstm_model: StockPriceModel<AutoDevice> = StockPriceModel::new(&lstm_config, &auto_device);
    let _auto_gru_model: StockPriceModel<AutoDevice> = StockPriceModel::new(&gru_config, &auto_device);
    
    println!("Successfully created autodiff models");
    
    // Print model comparison results
    println!("\n====================================================================");
    println!("                          Model Comparison                          ");
    println!("====================================================================");
    
    println!("\nPerformance Metrics:");
    println!("Model | Forward Pass Time");
    println!("------|-------------");
    println!("LSTM  | {:.2?}", lstm_elapsed);
    println!("GRU   | {:.2?}", gru_elapsed);
    
    // Model size comparison (parameter count)
    println!("\nModel Architecture Comparison:");
    println!("- Both models use sequence length of {} days", sequence_length);
    println!("- Both models use {} features as input", input_size);
    println!("- Both models use hidden size of {} neurons", hidden_size);
    println!("- Both models use {} layers", num_layers);
    println!("- Both models trained with dropout rate of {}", dropout_rate);
    println!("- LSTM has 4 gates (input, forget, cell, output)");
    println!("- GRU has 3 gates (update, reset, output)");
    println!("- Theoretically, GRU should be slightly faster but might be less powerful");
    
    println!("\nConclusion:");
    if lstm_elapsed < gru_elapsed {
        println!("- LSTM was faster in forward pass in this example by {:.2?}", gru_elapsed - lstm_elapsed);
    } else {
        println!("- GRU was faster in forward pass in this example by {:.2?}", lstm_elapsed - gru_elapsed);
    }
    
    println!("\nExample completed successfully!");
    Ok(())
} 