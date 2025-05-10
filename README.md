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
Example input file: `AAPL-ticker_minute_bars.csv`

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
