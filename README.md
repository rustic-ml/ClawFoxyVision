# Rust ML Stock Prediction Project

This project implements a machine learning pipeline for stock price prediction using Rust, Polars, and Burn (PyTorch backend).

---

## Project Structure

```
├── src/
│   ├── main.rs                # Entry point, orchestrates the pipeline
│   ├── constants.rs           # Technical indicators, model constants
│   ├── util/
│   │   ├── feature_engineering.rs   # Adds technical indicators
│   │   ├── pre_processor.rs        # Loads and preprocesses data
│   │   ├── model_utils.rs          # Model saving/loading utilities
│   │   └── mod.rs
│   └── lstm/
│       ├── step_1_tensor_preparation.rs   # Data to tensor conversion
│       ├── step_2_lstm_cell.rs           # LSTM cell implementation
│       ├── step_3_lstm_model_arch.rs     # LSTM model architecture
│       ├── step_4_train_model.rs         # Training and evaluation
│       ├── step_5_prediction.rs          # Prediction utilities
│       ├── step_6_model_serialization.rs # Model (de)serialization
│       └── mod.rs
├── build.rs
├── Cargo.toml
├── .gitignore
├── README.md
├── LICENSE
└── [large CSV files, e.g., AAPL-ticker_minute_bars.csv] (ignored by git)
```

---

## Main Pipeline

1. **Data Preprocessing** (`util/pre_processor.rs`)
   - Loads and preprocesses stock market data from CSV
   - Handles missing values and sorts by timestamp
2. **Feature Engineering** (`util/feature_engineering.rs`)
   - Adds technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR, returns, etc.)
3. **Tensor Preparation** (`lstm/step_1_tensor_preparation.rs`)
   - Converts DataFrame to tensors for model input
4. **LSTM Model** (`lstm/step_3_lstm_model_arch.rs`, `lstm/step_2_lstm_cell.rs`)
   - Implements an LSTM neural network using Burn
5. **Training** (`lstm/step_4_train_model.rs`)
   - Trains the model and evaluates on a test split
6. **Prediction** (`lstm/step_5_prediction.rs`)
   - Generates forecasts and denormalizes predictions
7. **Model Serialization** (`lstm/step_6_model_serialization.rs`, `util/model_utils.rs`)
   - Saves and loads model checkpoints with metadata

---

## Dependencies

- Rust (edition 2021)
- polars = { version = "0.46.0", features = ["lazy", "strings", "temporal", "rolling_window"] }
- burn = { version = "0.17.0", features = ["tch", "ndarray"] }
- burn-core, burn-autodiff, burn-train, burn-tch, burn-ndarray = "0.17.0"
- bincode = "2.0.1"
- chrono = "0.4.34"
- thiserror = "2.0.11"
- anyhow = "1.0.75"
- serde = { version = "1.0", features = ["derive"] }
- serde_json = "1.0"
- tempfile = "3.0"
- ndarray = "0.16.1"

---

## Data Format

The input data should be in CSV format with the following fields:
- `symbol`: Stock symbol (e.g., AAPL, MSFT)
- `time`: Timestamp of the data point
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume
- `vwap`: Volume-weighted average price

> **Note:** Large CSV files (such as `AAPL-ticker_minute_bars.csv`) are ignored by git. Place your data files in the project root as needed.

---

## Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd burn-timeseries
```
2. Build the project:
```bash
cargo build
```
3. Run the project:
```bash
cargo run -- [TICKER] [MODEL_TYPE]
# Example: cargo run -- AAPL lstm
```

---

## Features

- Efficient data processing with Polars
- Rich technical indicator calculation
- LSTM-based price prediction using Burn
- Model versioning and checkpointing
- ARM64 and CPU-only PyTorch (LibTorch) support

---

## License

[Your chosen license]

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Predict Price with LSTM

This project demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM) network in Rust. It focuses on minute-by-minute stock trading data.

## Dependencies

*   `polars = { version = "0.46.0", features = ["csv", "lazy", "serde"] }`
*   `burn = { version = "0.16.1", features = ["ndarray", "train"] }`
*   `chrono = "0.4.40"`

## Data Format

The input data should be in CSV format with the following fields:

*   `symbol`: Stock symbol (e.g., AAPL, MSFT).
*   `time`: Timestamp of the data point.
*   `open`: Opening price.
*   `high`: Highest price during the minute.
*   `low`: Lowest price during the minute.
*   `close`: Closing price.
*   `volume`: Trading volume.
*   `vwap`: Volume-weighted average price.

## Steps to Predict Price

1.  **Data Preparation:** Load the CSV data using the polars library. Clean and preprocess the data as needed.
2.  **Feature Engineering:** Extract relevant features from the data. For example, calculate moving averages, price differences, etc.
3.  **LSTM Model:** Build an LSTM model using the Burn machine learning library.
4.  **Training:** Train the LSTM model on historical data.
5.  **Prediction:** Use the trained model to predict future stock prices.
6.  **Evaluation:** Evaluate the model's performance using appropriate metrics.


Building PyTorch (LibTorch) for ARM64
This guide provides step-by-step instructions to build PyTorch (LibTorch) from source for ARM64 (aarch64-unknown-linux-gnu) on a Linux system. The resulting LibTorch installation is suitable for use with Rust projects (e.g., those using burn-tch or torch-sys) or C++ applications. The build is configured for CPU-only usage (no CUDA).
Prerequisites

System: ARM64-based Linux (e.g., Ubuntu 20.04 or later).
Disk Space: At least 20 GB free (for source, dependencies, and build artifacts).
Memory: Minimum 8 GB RAM (16 GB recommended for faster compilation).
Compiler: GCC/G++ 9 or later.
Python: Python 3.8 or later.

Step 1: Install Dependencies
Install required system packages and Python dependencies.
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip ninja-build libopenblas-dev liblapack-dev
pip3 install pyyaml


build-essential: Provides GCC/G++, make, etc.
cmake: Required for building PyTorch.
ninja-build: Speeds up the build process.
libopenblas-dev, liblapack-dev: Linear algebra libraries for CPU performance.
pyyaml: Required for PyTorch's build scripts.

Step 2: Clone PyTorch Source
Clone the PyTorch repository and check out the desired version (e.g., 2.3.0 for compatibility with burn-tch v0.16.1).
cd ~/workspace
rm -rf pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.3.0
git submodule sync
git submodule update --init --recursive


--recursive: Ensures submodules (e.g., third_party) are cloned.
v2.3.0: Matches LibTorch version for torch-sys v0.15.0. Adjust if needed (e.g., git checkout v2.4.0 for newer versions).
submodule update: Fetches dependencies like sleef, gloo, etc.

Step 3: Set Environment Variables
Configure the build for CPU-only usage and optimize the process.
export USE_CUDA=0
export BUILD_TEST=0
export USE_DISTRIBUTED=0
export MAX_JOBS=4


USE_CUDA=0: Disables GPU support (required for CPU-only ARM64 build).
BUILD_TEST=0: Skips building tests to reduce build time.
USE_DISTRIBUTED=0: Disables distributed training features.
MAX_JOBS=4: Limits parallel jobs to avoid memory issues (adjust based on CPU cores, e.g., nproc).

Step 4: Build PyTorch (LibTorch)
Run the build process.
python3 setup.py build


This builds both the Python PyTorch package and LibTorch (C++ library).
The process may take 1–4 hours depending on your system (e.g., 4-core ARM64 with 16 GB RAM).
Monitor for errors. If the build fails, check the last 20 lines of output or build/log.txt.

Step 5: Install LibTorch
Copy the built LibTorch artifacts to a designated directory (e.g., $HOME/libtorch).
rm -rf $HOME/libtorch
mkdir -p $HOME/libtorch
cp -r build/lib $HOME/libtorch/lib
cp -r torch/include $HOME/libtorch/include
cp -r torch/share $HOME/libtorch/share


lib/: Contains libtorch.so, libtorch_cpu.so, libc10.so, etc.
include/: Contains headers like torch/torch.h, torch/csrc/autograd/engine.h.
share/: Contains CMake configuration for linking.

Step 6: Verify the Build
Confirm the build is correct for ARM64 and functional.
6.1 Check Library Architecture
ls -l $HOME/libtorch/lib
file $HOME/libtorch/lib/libtorch.so
file $HOME/libtorch/lib/libtorch_cpu.so
file $HOME/libtorch/lib/libc10.so


Expected output:$HOME/libtorch/lib/libtorch.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...
$HOME/libtorch/lib/libtorch_cpu.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...
$HOME/libtorch/lib/libc10.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...


If the architecture shows x86-64 or libraries are missing, the build failed.

6.2 Check Headers
Verify the presence of required headers.
ls $HOME/libtorch/include/torch
ls $HOME/libtorch/include/torch/csrc/autograd


Expect torch.h in include/torch/ and engine.h in include/torch/csrc/autograd/.

6.3 Test with C++ Program
Create a simple C++ program to test LibTorch functionality.
// test_torch.cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    return 0;
}

Compile and run:
g++ -o test_torch test_torch.cpp -I$HOME/libtorch/include -L$HOME/libtorch/lib -ltorch -ltorch_cpu -lc10 -std=c++17
LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH ./test_torch


Expected output: A random 2x3 tensor (e.g., tensor([[0.1234, 0.5678, 0.9012], ...]])).
If it fails (e.g., "cannot find torch/torch.h" or linker errors), check include/lib paths.

6.4 Verify Build Configuration
Check the CMake configuration used during the build.
cat ~/workspace/pytorch/build/CMakeCache.txt | grep -E "TORCH|CMAKE_SYSTEM_PROCESSOR"


Expect:
CMAKE_SYSTEM_PROCESSOR:STRING=aarch64
No CUDA-related flags (e.g., TORCH_CUDA=OFF).



Step 7: Configure Environment
