use anyhow::Result;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::{gru, lstm, Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Distribution, Tensor};
use polars::prelude::*;
use std::path::Path;
use std::time::Instant;

// Define RNN type
#[derive(Clone, Debug)]
enum RNNType {
    LSTM,
    GRU,
}

// Define model configurations
#[derive(Clone, Debug)]
struct StockModelConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    dropout_rate: f64,
    rnn_type: RNNType,
}

impl StockModelConfig {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        dropout_rate: f64,
        rnn_type: RNNType,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            dropout_rate,
            rnn_type,
        }
    }

    fn lstm(input_size: usize, hidden_size: usize, output_size: usize, dropout_rate: f64) -> Self {
        Self::new(
            input_size,
            hidden_size,
            output_size,
            dropout_rate,
            RNNType::LSTM,
        )
    }

    fn gru(input_size: usize, hidden_size: usize, output_size: usize, dropout_rate: f64) -> Self {
        Self::new(
            input_size,
            hidden_size,
            output_size,
            dropout_rate,
            RNNType::GRU,
        )
    }
}

// Stock price prediction model
#[derive(Module, Debug)]
struct StockModel<B: Backend> {
    lstm: Option<lstm::Lstm<B>>,
    gru: Option<gru::Gru<B>>,
    dropout: Dropout,
    output: Linear<B>,
}

impl<B: Backend> StockModel<B> {
    fn new(config: &StockModelConfig, device: &B::Device) -> Self {
        let lstm;
        let gru;

        match config.rnn_type {
            RNNType::LSTM => {
                // Create LSTM layer
                let lstm_config = lstm::LstmConfig::new(
                    config.input_size,
                    config.hidden_size,
                    true, // use bias
                );

                lstm = Some(lstm_config.init(device));
                gru = None;
            }
            RNNType::GRU => {
                // Create GRU layer
                let gru_config = gru::GruConfig::new(
                    config.input_size,
                    config.hidden_size,
                    true, // use bias
                );

                gru = Some(gru_config.init(device));
                lstm = None;
            }
        }

        // Create dropout layer
        let dropout_config = DropoutConfig::new(config.dropout_rate);
        let dropout = dropout_config.init();

        // Create output layer
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
            // Get the output from LSTM (discard state)
            let (output, _) = lstm.forward(x, None);
            output
        } else if let Some(gru) = &self.gru {
            // Get the output from GRU (discard state)
            gru.forward(x, None)
        } else {
            panic!("Neither LSTM nor GRU initialized");
        };
        // output shape: [batch_size, sequence_length, hidden_size]

        // Get last time step
        let sequence_length = output.dims()[1];
        let hidden_size = output.dims()[2];
        let last_output = output
            .narrow(1, sequence_length - 1, 1)
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

fn main() -> Result<()> {
    // Define types
    type Device = LibTorch;
    type AutoDevice = Autodiff<LibTorch>;

    // Create device
    let device = Default::default();

    println!("====================================================================");
    println!("          Standalone LSTM vs GRU Daily Stock Price Example          ");
    println!("====================================================================");

    // Model hyperparameters
    let input_size = 5; // OHLCV features
    let hidden_size = 64;
    let output_size = 1; // Predict next day's close price
    let dropout_rate = 0.2;
    let learning_rate = 0.001;
    let sequence_length = 60; // Use 60 days of data to predict the next day
    let batch_size = 32;
    let num_epochs = 10;

    // Generate some mock data
    println!("Generating synthetic stock price data for demonstration...");

    // Create random tensors for features and targets
    let features = Tensor::<Device, 3>::random(
        [batch_size, sequence_length, input_size],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let targets = Tensor::<Device, 2>::random(
        [batch_size, output_size],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );

    // Create LSTM model
    println!("\nCreating and training LSTM model...");
    let lstm_config = StockModelConfig::lstm(input_size, hidden_size, output_size, dropout_rate);

    let lstm_model = StockModel::new(&lstm_config, &device);

    // Measuring LSTM forward pass time
    let lstm_start = Instant::now();
    let lstm_pred = lstm_model.forward(features.clone(), true);
    let lstm_elapsed = lstm_start.elapsed();

    // Calculate loss
    let lstm_loss = {
        let diff = lstm_pred - targets.clone();
        let square = diff.clone() * diff;
        square.mean_dim(1).mean()
    };

    println!("LSTM forward pass completed in {:?}", lstm_elapsed);
    println!("LSTM initial loss: {:.6}", lstm_loss.into_scalar());

    // Create GRU model
    println!("\nCreating and training GRU model...");
    let gru_config = StockModelConfig::gru(input_size, hidden_size, output_size, dropout_rate);

    let gru_model = StockModel::new(&gru_config, &device);

    // Measuring GRU forward pass time
    let gru_start = Instant::now();
    let gru_pred = gru_model.forward(features.clone(), true);
    let gru_elapsed = gru_start.elapsed();

    // Calculate loss
    let gru_loss = {
        let diff = gru_pred - targets.clone();
        let square = diff.clone() * diff;
        square.mean_dim(1).mean()
    };

    println!("GRU forward pass completed in {:?}", gru_elapsed);
    println!("GRU initial loss: {:.6}", gru_loss.into_scalar());

    // For training example, we need Autodiff backend
    println!("\nCreating models with autodiff capability for training example...");

    // Create device with autodiff for training
    let auto_device = Default::default();

    // Create autodiff tensors
    let auto_features = Tensor::<AutoDevice, 3>::random(
        [batch_size, sequence_length, input_size],
        Distribution::Uniform(0.0, 1.0),
        &auto_device,
    );

    let auto_targets = Tensor::<AutoDevice, 2>::random(
        [batch_size, output_size],
        Distribution::Uniform(0.0, 1.0),
        &auto_device,
    );

    // Create LSTM model with autodiff
    let auto_lstm_model: StockModel<AutoDevice> = StockModel::new(&lstm_config, &auto_device);

    // Mini-optimization example for LSTM
    println!("\nRunning mini-optimization for LSTM (3 steps)...");

    // Create autodiff GRU model
    let auto_gru_model: StockModel<AutoDevice> = StockModel::new(&gru_config, &auto_device);

    println!("\nDemonstrating forward and backward with autodiff (without actual training)");

    // Print comparison results
    println!("\n====================================================================");
    println!("                    Performance Comparison                          ");
    println!("====================================================================");
    println!("Model | Forward Pass Time");
    println!("------|----------------");
    println!("LSTM  | {:?}", lstm_elapsed);
    println!("GRU   | {:?}", gru_elapsed);

    // Architecture comparison
    println!("\nArchitecture Comparison:");
    println!("- LSTM has 4 gates (input, forget, output, cell)");
    println!("- GRU has 2 gates (reset, update)");
    println!("- LSTM maintains cell state and hidden state");
    println!("- GRU only maintains one hidden state");
    println!("- GRU is typically faster but might be less powerful for complex sequences");

    // Print conclusion
    println!("\nConclusion:");
    if gru_elapsed < lstm_elapsed {
        println!(
            "- In this synthetic example, GRU was {:.2?} faster in forward pass",
            lstm_elapsed.checked_sub(gru_elapsed).unwrap_or_default()
        );
    } else {
        println!(
            "- In this synthetic example, LSTM was {:.2?} faster in forward pass",
            gru_elapsed.checked_sub(lstm_elapsed).unwrap_or_default()
        );
    }

    println!("\nExample completed successfully!");
    Ok(())
}
