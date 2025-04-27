// Technical indicators and feature names
pub const TECHNICAL_INDICATORS: [&str; 12] = [
    "close",
    "volume",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_middle",
    "atr_14",
    "returns",
    "price_range",
];

// Model parameters
pub const SEQUENCE_LENGTH: usize = 10; // Number of time steps to look back

// Data preprocessing
pub const VALIDATION_SPLIT_RATIO: f64 = 0.2; // 20% of data for validation

// Model paths
pub const MODEL_PATH: &str = "/mnt/nfs_mount/barData/stock/analysis/result/model";
pub const MODEL_FILE_NAME: &str = "_lstm_model";

