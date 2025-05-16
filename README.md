# ClawFoxyVision 🔮

![ClawFoxyVision Logo](ClawFoxyVision_250px.png)

**ClawFoxyVision: Your Sharper View into Financial Fortunes.**

Clawy and Foxy, our visionary duo, power this library to help you navigate the complexities of financial time series data. Clawy's razor-sharp analytical abilities dissect intricate market patterns, while Foxy's cunning intelligence detects subtle movements often missed by others.

ClawFoxyVision empowers traders and analysts with enhanced foresight into market trends. By transforming raw data into actionable insights, our advanced vision algorithms cut through market noise, revealing the true signals that can inform tomorrow's movements. Trust ClawFoxyVision to illuminate your path through the often murky waters of financial forecasting.

**Built with [Burn](https://github.com/tracel-ai/burn), a deep learning framework in Rust.**

## ✨ Features

* **Advanced Recurrent Neural Networks:** Implements LSTM and GRU models tailored for time series forecasting.
* **Flexible Configuration:**
    * Configurable sequence length and forecast horizon.
    * Bidirectional processing capabilities.
    * Integrated attention mechanisms.
    * Adjustable hyperparameters.
    * L2 regularization and dropout for robust training.
* **Streamlined Data Handling:**
    * Built-in data normalization and preprocessing.
    * Supports OHLC (Open, High, Low, Close) data from CSV files.
* **Persistent Models:** Save trained models and load them for later use.
* **Comparative Analysis:** Directly compare prediction performance between LSTM and GRU models for the same ticker.

## 🚀 Getting Started

### Prerequisites

* Rust 1.65 or higher installed on your system.
* Your stock data should be in CSV files, with columns for Open, High, Low, and Close values. For example: `AAPL-ticker_minute_bars.csv`.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd ClawFoxyVision
    ```
2.  The project uses Cargo, Rust's package manager. Dependencies will be handled automatically.

### Running the Models

You can train and run forecasting models using either the provided shell script or directly with Cargo.

**Using the shell script:**

```bash
./run_model.sh [ticker] [model_type]

## Usage

### Running the Models

You can run either the LSTM or GRU model using the provided shell script:

```bash
./run_model.sh [ticker] [model_type]
```

Examples:
```bash
# Run with LSTM model
./run_model.sh AAPL lstm

# Run with GRU model
./run_model.sh AAPL gru
```

Alternatively, you can run directly with Cargo:

```bash
cargo run --release -- [ticker] [model_type]
```

### Model Types

The application supports two types of recurrent neural networks:

1. **LSTM (Long Short-Term Memory)** - Default model with gates to control information flow and mitigate vanishing gradients
2. **GRU (Gated Recurrent Unit)** - More efficient model with fewer parameters

Both models support:
- Bidirectional processing
- Attention mechanisms
- Configurable hyperparameters
- L2 regularization and dropout

### Input Data

The application expects stock data in CSV format with OHLC (Open, High, Low, Close) values.
Example input file: `AAPL_minute_ohlcv.csv`

## Implementation Details

### Project Structure

- `src/minute/lstm/` - LSTM implementation modules
- `src/minute/gru/` - GRU implementation modules
- `src/constants.rs` - Common configuration constants
- `src/main.rs` - Entry point and execution logic

### Module Organization

Each model implementation follows the same module pattern:

1. `step_1_tensor_preparation.rs` - Data preparation utilities
2. `step_2_*_cell.rs` - Core cell implementation
3. `step_3_*_model_arch.rs` - Complete architecture
4. `step_4_train_model.rs` - Training workflow
5. `step_5_prediction.rs` - Prediction utilities
6. `step_6_model_serialization.rs` - Model saving/loading

## Comparing Models

When a GRU model is run with an existing LSTM model for the same ticker, the application will automatically compare predictions from both models. This helps in evaluating which model performs better for your specific dataset.

## Requirements

- Rust 1.65 or higher
- CSV input data files in the expected format

## License

MIT License

## Project Structure

```
ClawFoxyVision/
├── src/                    # Main source code
│   ├── util/              # Utility functions
│   ├── daily/             # Daily data processing
│   ├── minute/            # Minute data processing
│   ├── constants/         # Project constants
│   └── test/             # Test files and utilities
├── examples/              # Example code
│   └── csv/              # Sample data files
└── .cursor/              # Cursor IDE configuration
    └── baseline.json     # Project baseline
```

## Coding Standards

### Rust Style Guide

- **Formatting**:
  - Max line length: 100 characters
  - Use spaces (4) for indentation
  - No tabs

- **Naming Conventions**:
  - Modules: `snake_case`
  - Functions: `snake_case`
  - Structs: `PascalCase`
  - Traits: `PascalCase`
  - Constants: `SCREAMING_SNAKE_CASE`

- **Documentation**:
  - All public items must be documented
  - Use rustdoc style
  - Include sections for Arguments, Returns, and Examples

### Data Processing

#### File Handling
- Use `read_financial_data` as a single entry point for all financial data operations
- Automatically detects file type based on extension
- Supported file formats:
  - CSV
  - Parquet
- Required columns:
  - symbol
  - datetime
  - open
  - high
  - low
  - close
  - volume
- Optional columns:
  - adjusted_close

Example usage:
```rust
use predict_price_lstm::util::file_utils::read_financial_data;

// Read financial data from either CSV or Parquet with the same function
let (df, metadata) = read_financial_data("path/to/data.csv")?;
// OR
let (df, metadata) = read_financial_data("path/to/data.parquet")?;
```

### Model Guidelines

#### LSTM Default Parameters
- sequence_length: 30
- hidden_size: 64
- num_layers: 2
- dropout_rate: 0.2

#### GRU Default Parameters
- sequence_length: 30
- hidden_size: 64
- num_layers: 2
- dropout_rate: 0.2

## Dependencies

### Required
- polars (>=0.47.1): Data manipulation
- burn (>=0.17.0): Deep learning framework
- chrono (>=0.4.41): Date and time handling
- anyhow (>=1.0.98): Error handling
- rustalib (>=1.0.0): Financial data processing

### Optional
- rayon (>=1.10.0): Parallel processing
- serde (>=1.0.219): Serialization

### Version Policy
- Never downgrade dependencies
- Use caret (^) versioning to allow compatible updates
- Pin versions only when necessary for stability
- Regularly update dependencies to latest compatible versions
- Test thoroughly after dependency updates

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   cargo build
   ```
3. Run examples:
   ```bash
   cargo run --example lstm_example
   cargo run --example gru_example
   cargo run --example parquet_lstm_gru_example  # Example using Parquet files
   ```

## Development Guidelines

1. **Module Organization**:
   - One file per module
   - Tests adjacent to source files
   - Examples in the examples directory

2. **Error Handling**:
   - Use `anyhow::Result` for error propagation
   - Provide meaningful error messages
   - Handle all potential error cases

3. **Testing**:
   - Write unit tests for all public functions
   - Include integration tests for complex features
   - Maintain test coverage above 80%

4. **Documentation**:
   - Keep documentation up to date
   - Include examples in doc comments
   - Document all public APIs

## Contributing

1. Follow the coding standards
2. Add tests for new features
3. Update documentation
4. Submit pull requests with clear descriptions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Examples

The repository includes several example implementations to demonstrate the usage of LSTM and GRU models:

### Standalone Examples

- **standalone_lstm_gru_daily.rs**: A self-contained example that demonstrates both LSTM and GRU models for daily stock price prediction. It includes:
  - Forward pass timing comparisons
  - Loss calculation
  - Implementation with both regular tensors and autodiff tensors
  - Architecture comparison

- **simplified_lstm_gru_comparison.rs**: A comprehensive comparison between LSTM and GRU models focusing on:
  - Performance metrics (speed and accuracy)
  - Architecture differences
  - Forward pass implementation
  - Layer-by-layer configuration

- **parquet_lstm_gru_example.rs**: A complete example demonstrating how to use Parquet files with LSTM and GRU models:
  - Reading financial data from Parquet files using the unified read_financial_data function
  - Adding technical indicators through feature engineering
  - Processing data with both LSTM and GRU models
  - Performance timing for both models
  - Suitable as a starting point for new projects using Parquet files

### Other Examples

- **daily_lstm_example.rs**: LSTM implementation for daily data
- **daily_gru_example.rs**: GRU implementation for daily data
- **daily_model_comparison.rs**: Comparison between daily LSTM and GRU models
- **lstm_example.rs**: Basic LSTM implementation
- **gru_example.rs**: Basic GRU implementation
- **compare_models.rs**: Utility for detailed model comparison

## Burn 0.17.0 API Notes

Our examples use the Burn 0.17.0 neural network API. There are some important implementation details to be aware of:

### LSTM Implementation

The LSTM forward method returns a tuple containing the output tensor and a state:

```rust
// LSTM forward signature
fn forward(&self, x: Tensor<B, 3>, state: Option<LstmState<B, 2>>) -> (Tensor<B, 3>, LstmState<B, 2>)

// Usage example
let (output, _) = lstm.forward(x, None);
```

### GRU Implementation

The GRU forward method returns just the output tensor:

```rust
// GRU forward signature
fn forward(&self, x: Tensor<B, 3>, state: Option<Tensor<B, 2>>) -> Tensor<B, 3>

// Usage example
let output = gru.forward(x, None);
```

### Tensor Operations

When working with tensors, be careful with moved values. It's often necessary to:
- Store dimension values before using tensor operations
- Clone tensors that will be used multiple times

```rust
// Get dimensions before using tensor
let sequence_length = output.dims()[1];
let hidden_size = output.dims()[2];

// Shape transformation
let last_output = output.narrow(1, sequence_length - 1, 1)
                       .reshape([batch_size, hidden_size]);
```

### Generic Backend Type

When using autodiff backends, make sure to specify the type explicitly:

```rust
let auto_lstm_model: StockModel<AutoDevice> = StockModel::new(&lstm_config, &auto_device);
```
