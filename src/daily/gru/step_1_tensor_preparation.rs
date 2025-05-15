// Re-export the tensor preparation functions from the LSTM module
// This avoids code duplication as the tensor preparation is identical
pub use crate::daily::lstm::step_1_tensor_preparation::*;
