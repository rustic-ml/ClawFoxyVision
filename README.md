![ClawFoxyVision](ClawFoxyVision_100px.png)

## ClawFoxyVision

Clawy and Foxy, with their ClawFoxyVision library here to give you a better view to a fortune lies ahead. These two visionary companions combine their unique talents to unravel the mysteries of financial time series data. Clawy, with razor-sharp analytical claws, dissects complex market patterns, while Foxy brings cunning intelligence to detect subtle market movements others might miss. Together, they empower traders and analysts with unprecedented foresight into market trends, turning raw data into actionable insights. Their advanced vision algorithms cut through market noise to reveal the true signals that predict tomorrow's movements. Trust in ClawFoxyVision to illuminate your path through the often murky waters of financial forecasting.

## Under the hood

A time series forecasting library built with [Burn](https://github.com/tracel-ai/burn), a deep learning framework in Rust.

## Features

- LSTM and GRU model implementations for time series forecasting
- Configurable sequence length and forecast horizon
- Data normalization and preprocessing
- Model saving and loading
- Compare predictions between LSTM and GRU models

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