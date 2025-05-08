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

// Extended technical indicators including time-based features and lag features
pub const EXTENDED_INDICATORS: [&str; 23] = [
    // Original indicators
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
    // New enhanced indicators
    "bb_b",
    "gk_volatility",
    // Time-based features
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    // Lag features
    "close_lag_5",
    "close_lag_15",
    "close_lag_30",
    "returns_5min",
    "volatility_15min",
];

// Model parameters
pub const SEQUENCE_LENGTH: usize = 10; // Number of time steps to look back

// Data preprocessing
pub const LSTM_TRAINING_DAYS: i64 = 200; // Reduced from 400 to prevent memory issues
pub const VALIDATION_SPLIT_RATIO: f64 = 0.2; // 20% of data for validation
pub const DEFAULT_DROPOUT: f64 = 0.15;  // Reduced from 0.3 to prevent overfitting while allowing more learning
pub const PRICE_DENORM_CLIP_MIN: f64 = 0.0; // Prevent negative price predictions
pub const L2_REGULARIZATION: f64 = 0.01; // L2 regularization strength

// Model paths
pub const MODEL_PATH: &str = "models"; // Use "models" directory in the current workspace
pub const MODEL_FILE_NAME: &str = "_lstm_model";

