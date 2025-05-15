// Re-export tensor preparation functionality from LSTM implementation
// since GRU uses the same tensor format and preparation methods
pub use crate::minute::lstm::step_1_tensor_preparation::*;
