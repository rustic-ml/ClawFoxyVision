# Rust ML Stock Prediction Project

This project implements a machine learning pipeline for stock price prediction using Rust, Polars, and Burn (PyTorch backend).

## Project Structure

The project is organized into three main steps:

1. **Data Preprocessing** (`step_1_preprocessor.rs`)
   - Loads and preprocesses stock market data
   - Handles data cleaning and initial transformations

2. **Feature Engineering** (`step_2_feature_engineering.rs`)
   - Adds technical indicators (SMA, RSI, MACD)
   - Prepares features for model training

3. **LSTM Model** (`step_3_burn_lstm_model.rs`)
   - Implements an LSTM neural network using Burn
   - Trains on the prepared features to predict stock prices

## Dependencies

- Rust (latest stable version)
- Polars
- Burn (with PyTorch backend)
- PyTorch (LibTorch) for ARM64

## Building PyTorch (LibTorch) for ARM64

To build PyTorch for ARM64 architecture:

```bash
# Clone PyTorch repository
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Configure build for ARM64
export USE_CUDA=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_XNNPACK=0

# Build PyTorch
python setup.py build

# Install LibTorch
python setup.py install
```

## Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd rust-ml
```

2. Build the project:
```bash
cargo build
```

3. Run the project:
```bash
cargo run
```

## Features

- Efficient data processing with Polars
- Technical indicator calculation
- LSTM-based price prediction
- GPU acceleration support (when available)

## License

[Your chosen license]

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
Copy the built LibTorch artifacts to a designated directory (e.g., /home/orange/libtorch).
rm -rf /home/orange/libtorch
mkdir -p /home/orange/libtorch
cp -r build/lib /home/orange/libtorch/lib
cp -r torch/include /home/orange/libtorch/include
cp -r torch/share /home/orange/libtorch/share


lib/: Contains libtorch.so, libtorch_cpu.so, libc10.so, etc.
include/: Contains headers like torch/torch.h, torch/csrc/autograd/engine.h.
share/: Contains CMake configuration for linking.

Step 6: Verify the Build
Confirm the build is correct for ARM64 and functional.
6.1 Check Library Architecture
ls -l /home/orange/libtorch/lib
file /home/orange/libtorch/lib/libtorch.so
file /home/orange/libtorch/lib/libtorch_cpu.so
file /home/orange/libtorch/lib/libc10.so


Expected output:/home/orange/libtorch/lib/libtorch.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...
/home/orange/libtorch/lib/libtorch_cpu.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...
/home/orange/libtorch/lib/libc10.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...


If the architecture shows x86-64 or libraries are missing, the build failed.

6.2 Check Headers
Verify the presence of required headers.
ls /home/orange/libtorch/include/torch
ls /home/orange/libtorch/include/torch/csrc/autograd


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
g++ -o test_torch test_torch.cpp -I/home/orange/libtorch/include -L/home/orange/libtorch/lib -ltorch -ltorch_cpu -lc10 -std=c++17
LD_LIBRARY_PATH=/home/orange/libtorch/lib:$LD_LIBRARY_PATH ./test_torch


Expected output: A random 2x3 tensor (e.g., tensor([[0.1234, 0.5678, 0.9012], ...]])).
If it fails (e.g., "cannot find torch/torch.h" or linker errors), check include/lib paths.

6.4 Verify Build Configuration
Check the CMake configuration used during the build.
cat ~/workspace/pytorch/build/CMakeCache.txt | grep -E "TORCH|CMAKE_SYSTEM_PROCESSOR"


Expect:
CMAKE_SYSTEM_PROCESSOR:STRING=aarch64
No CUDA-related flags (e.g., TORCH_CUDA=OFF).



Step 7: Configure Environment
