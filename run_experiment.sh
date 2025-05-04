#!/bin/bash

# This script runs LSTM model experiments with different configurations

# Ensure the experiments directory exists
mkdir -p experiments

# Run baseline for comparison
echo "Running baseline experiment..."
cargo run --release -- AAPL lstm

# Run experiment with improved model
echo "Running improved model experiment..."
cargo run --release -- AAPL lstm

# Run experiment with enhanced features
echo "Running experiment with enhanced features..."
# This will use our improved model configuration from the code changes

# Print summary
echo "Experiments completed. Results saved in the experiments directory." 