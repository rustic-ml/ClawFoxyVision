#!/bin/bash

# Script to run either LSTM or GRU models for time series forecasting
# Usage: ./run_model.sh [ticker] [model_type]
# Example: ./run_model.sh AAPL lstm
# Example: ./run_model.sh AAPL gru

# Default values
TICKER=${1:-AAPL}
MODEL_TYPE=${2:-lstm}

# Convert model_type to lowercase
MODEL_TYPE=$(echo "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')

# Validate model type
if [[ "$MODEL_TYPE" != "lstm" && "$MODEL_TYPE" != "gru" && "$MODEL_TYPE" != "cnnlstm" ]]; then
    echo "Error: model_type must be either 'lstm', 'gru', or 'cnnlstm'"
    echo "Usage: ./run_model.sh [ticker] [model_type]"
    echo "Example: ./run_model.sh AAPL lstm"
    echo "Example: ./run_model.sh AAPL gru"
    echo "Example: ./run_model.sh AAPL cnnlstm"
    exit 1
fi

echo "Running with ticker: $TICKER, model_type: $MODEL_TYPE"
cargo run --release -- "$TICKER" "$MODEL_TYPE" 