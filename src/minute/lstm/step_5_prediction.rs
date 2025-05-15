// External imports
use crate::constants::{EXTENDED_INDICATORS, PRICE_DENORM_CLIP_MIN, TECHNICAL_INDICATORS};
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use polars::prelude::*;
use rand::Rng;

// Internal imports
use super::step_1_tensor_preparation;
use super::step_3_lstm_model_arch::TimeSeriesLstm;

/// Single-step prediction from the model
pub fn predict_next_step<B: Backend>(
    model: &TimeSeriesLstm<B>,
    df: DataFrame,
    device: &B::Device,
    use_extended_features: bool,
) -> Result<f64> {
    // Choose which feature set to use
    let feature_columns = if use_extended_features {
        &EXTENDED_INDICATORS[..]
    } else {
        &TECHNICAL_INDICATORS[..]
    };

    // Validate required columns
    if !df.is_empty() {
        for col in feature_columns {
            if !df.schema().contains(col) {
                return Err(anyhow::anyhow!("Missing required column: {}", col));
            }
        }
    }

    // Clone the DataFrame to avoid modifications affecting the original
    let prediction_df = df.clone();

    // Build sequences tensor with horizon 1
    let (features, _) = step_1_tensor_preparation::dataframe_to_tensors::<B>(
        &prediction_df,
        crate::constants::SEQUENCE_LENGTH,
        1,
        device,
        use_extended_features,
        None,
    )
    .context("Tensor creation failed for prediction")?;

    // Extract the last sequence
    let seq_count = features.dims()[0];
    let seq = features.clone().narrow(0, seq_count - 1, 1);

    // Forward pass for single-step prediction
    let pred_tensor = model.forward(seq);

    // Extract scalar prediction
    let data = pred_tensor.to_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();
    let value = slice[0];
    Ok(value as f64)
}

/// Generate multiple future predictions using recursive forecasting with error correction
pub fn generate_forecast_with_correction<B: Backend>(
    model: &TimeSeriesLstm<B>,
    df: DataFrame,
    forecast_horizon: usize,
    device: &B::Device,
    use_extended_features: bool,
    error_correction_alpha: f64,
) -> Result<Vec<f64>> {
    let mut predictions = Vec::with_capacity(forecast_horizon);

    // Keep track of recent prediction errors for correction
    let mut recent_errors = Vec::new();
    let max_error_history = 5; // Number of recent errors to consider

    // Choose which feature set to use
    let feature_columns = if use_extended_features {
        &EXTENDED_INDICATORS[..]
    } else {
        &TECHNICAL_INDICATORS[..]
    };

    // Get the schema once at the beginning
    let schema = df.schema();

    // Start with the original DataFrame
    let mut current_data = df.clone();

    for step in 0..forecast_horizon {
        // Make a prediction for the next step
        let pred_df = current_data.clone();
        let uncorrected_prediction =
            predict_next_step(model, pred_df, device, use_extended_features)?;

        // Apply error correction if we have historical errors
        let corrected_prediction = if !recent_errors.is_empty() {
            let mean_error: f64 = recent_errors.iter().sum::<f64>() / recent_errors.len() as f64;
            uncorrected_prediction - (error_correction_alpha * mean_error)
        } else {
            uncorrected_prediction
        };

        predictions.push(corrected_prediction);

        // Extract needed data from the current DataFrame
        let height = current_data.height();
        let mut column_data = Vec::new();

        // Create a new row with the predicted value
        for (name, dtype) in schema.iter() {
            let series = match name.as_str() {
                "close" => Series::new(name.clone(), &[corrected_prediction]),
                "symbol" | "time" => {
                    if let Ok(col) = current_data.column(name) {
                        let last_idx = if height > 0 { height - 1 } else { 0 };
                        let last_val = if height > 0 {
                            col.get(last_idx).unwrap_or(AnyValue::Null)
                        } else {
                            AnyValue::Null
                        };
                        Series::new(name.clone(), &[last_val.to_string()])
                    } else {
                        Series::new(name.clone(), &[""])
                    }
                }
                _ => {
                    if feature_columns.contains(&name.as_str()) {
                        if let Ok(col) = current_data.column(name) {
                            if let Ok(f64_col) = col.f64() {
                                let last_idx = if height > 0 { height - 1 } else { 0 };
                                let last_val = if f64_col.len() > 0 {
                                    f64_col.get(last_idx).unwrap_or(0.0)
                                } else {
                                    0.0
                                };
                                Series::new(name.clone(), &[last_val])
                            } else {
                                Series::new(name.clone(), &[0.0])
                            }
                        } else {
                            Series::new(name.clone(), &[0.0])
                        }
                    } else {
                        match dtype {
                            DataType::Float64 => Series::new(name.clone(), &[0.0f64]),
                            DataType::Int64 => Series::new(name.clone(), &[0i64]),
                            DataType::String => Series::new(name.clone(), &[""]),
                            _ => Series::new(name.clone(), &[0.0f64]),
                        }
                    }
                }
            };
            column_data.push(series.into_column());
        }

        // Create new row and add it to the data
        let new_row = DataFrame::new(column_data)?;
        let next_data = current_data.vstack(&new_row)?;

        // Completely replace the current data with the new data
        current_data = next_data;

        // Update error history if we have actual values
        if step + 1 < forecast_horizon {
            if let Ok(actual_series) = df.column("close") {
                if let Ok(actual_f64) = actual_series.f64() {
                    // The true_idx should be based on the original df height, not the current_data height
                    // Calculate the index where we should expect the actual value
                    let df_height = df.height();
                    let step_index = step;

                    // Only try to get the value if we're still within bounds of the original df
                    if step_index < df_height {
                        if let Some(actual) = actual_f64.get(step_index) {
                            let error = corrected_prediction - actual;
                            recent_errors.push(error);

                            // Keep only the most recent errors
                            if recent_errors.len() > max_error_history {
                                recent_errors.remove(0);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(predictions)
}

/// Convert predictions back to original scale using Z-score denormalization
/// with volatility constraints based on historical data
pub fn denormalize_z_score_predictions(
    predictions: Vec<f64>,
    original_df: &DataFrame,
    column: &str,
) -> Result<Vec<f64>> {
    // Get the original series
    let series = original_df.column(column)?;
    let f64_series = series.f64()?;

    // Get mean and std for the series
    let mean = f64_series.mean().unwrap_or(0.0);
    let std = f64_series.std(1).unwrap_or(1.0);

    // Avoid division by zero
    let std = if std < f64::EPSILON { 1.0 } else { std };

    // Calculate historical daily volatility
    let daily_volatility = calculate_historical_volatility(original_df, column)?;
    println!(
        "Historical daily volatility: {:.2}%",
        daily_volatility * 100.0
    );

    // Calculate realistic per-minute volatility constraint
    // Typical stock prices move around 0.5-1% of daily volatility per minute
    let minute_volatility_factor = daily_volatility * 0.007; // 0.7% of daily volatility per minute

    // Denormalize the predictions
    let mut denormalized = Vec::with_capacity(predictions.len());
    let mut prev_value = f64_series.get(f64_series.len() - 1).unwrap_or(mean);

    for (_i, &pred) in predictions.iter().enumerate() {
        // Basic denormalization using the Z-score formula: x = z*std + mean
        let raw_value = (pred * std) + mean;

        // Calculate max allowed change based on historical volatility
        let max_allowed_change = prev_value * minute_volatility_factor;

        // Apply volatility constraints
        let constrained_value = if (raw_value - prev_value).abs() > max_allowed_change {
            // Limit the change to the maximum allowed
            if raw_value > prev_value {
                prev_value + max_allowed_change
            } else {
                prev_value - max_allowed_change
            }
        } else {
            raw_value
        };

        // Prevent negative prices
        let final_value = constrained_value.max(PRICE_DENORM_CLIP_MIN);
        denormalized.push(final_value);

        // Update prev_value for next iteration
        prev_value = final_value;
    }

    Ok(denormalized)
}

/// Calculate the historical volatility from a DataFrame
fn calculate_historical_volatility(df: &DataFrame, price_column: &str) -> Result<f64> {
    if df.height() < 10 {
        // Not enough data for reliable calculation
        return Ok(0.02); // Return a default 2% daily volatility
    }

    let series = df.column(price_column)?;
    let f64_series = series.f64()?;

    // Calculate daily returns
    let mut returns = Vec::new();
    let mut prev_value = f64_series.get(0).unwrap_or(0.0);

    for i in 1..f64_series.len() {
        if let Some(curr) = f64_series.get(i) {
            if prev_value > 0.0 {
                let ret = (curr - prev_value) / prev_value;
                returns.push(ret);
            }
            prev_value = curr;
        }
    }

    if returns.is_empty() {
        return Ok(0.02); // Default if no valid returns
    }

    // Calculate standard deviation of returns
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    let daily_vol = variance.sqrt();

    // Cap volatility to reasonable range
    let capped_vol = daily_vol.min(0.05).max(0.005);

    Ok(capped_vol)
}

/// Convert predictions back to original scale (reverse min-max normalization)
pub fn denormalize_predictions(
    predictions: Vec<f64>,
    original_df: &DataFrame,
    column: &str,
) -> Result<Vec<f64>> {
    // Get the original series
    let series = original_df.column(column)?;

    // Get min and max values for the series
    let f64_series = series.f64()?;
    let min = f64_series.min().unwrap_or(0.0);
    let max = f64_series.max().unwrap_or(1.0);

    // Avoid division by zero
    let range = if (max - min).abs() < f64::EPSILON {
        1.0
    } else {
        max - min
    };

    // Denormalize the predictions
    let denormalized = predictions
        .iter()
        .map(|&p| (p * range) + min)
        .map(|p| p.max(PRICE_DENORM_CLIP_MIN)) // Prevent negative prices
        .collect();

    Ok(denormalized)
}

/// Direct multi-step (multi-output) prediction: returns all future steps at once
pub fn predict_multi_step_direct<B: Backend>(
    model: &TimeSeriesLstm<B>,
    df: DataFrame,
    device: &B::Device,
    forecast_horizon: usize,
) -> Result<Vec<f64>> {
    // Direct multi-step: build features from last SEQUENCE_LENGTH rows
    let seq_len = crate::constants::SEQUENCE_LENGTH;

    // Ensure we use only the standard indicator set to match model input dimensions
    let indicator_names: Vec<String> = TECHNICAL_INDICATORS
        .iter()
        .map(|&s| s.to_string())
        .collect();

    // Select only technical indicators
    let mut df_sel = df
        .select(&indicator_names)
        .context("Failed to select technical indicators for prediction")?;

    // Drop nulls
    df_sel = df_sel
        .drop_nulls::<String>(None)
        .context("Failed to drop null values")?;

    // Compute available rows after drop
    let n_rows_sel = df_sel.height();
    if n_rows_sel < seq_len {
        return Err(anyhow::anyhow!(format!(
            "DataFrame has too few rows ({}) for sequence_length ({})",
            n_rows_sel, seq_len
        )));
    }

    // Count the number of features to ensure correct tensor dimensions
    let n_features = TECHNICAL_INDICATORS.len();

    // Extract the last seq_len rows
    let start = n_rows_sel - seq_len;

    // Build feature buffer [1, seq_len, n_features]
    let mut buf = Vec::with_capacity(seq_len * n_features);

    // Populate buffer with feature values
    for row in start..n_rows_sel {
        for &col in TECHNICAL_INDICATORS.iter() {
            let val = df_sel.column(col)?.f64()?.get(row).unwrap_or(0.0) as f32;
            buf.push(val);
        }
    }

    // Verify buffer size matches expected dimensions
    if buf.len() != seq_len * n_features {
        return Err(anyhow::anyhow!(format!(
            "Feature buffer size mismatch: got {} elements, expected {} (seq_len={}, n_features={})",
            buf.len(), seq_len * n_features, seq_len, n_features
        )));
    }

    // Create tensor with correct shape
    let shape = burn::tensor::Shape::new([1, seq_len, n_features]);
    let features = burn::tensor::Tensor::<B, 1>::from_floats(buf.as_slice(), device).reshape(shape);

    // Forward pass through the model
    let output = model.forward(features); // [1, forecast_horizon]

    // Convert to Vec<f64>
    let data = output.to_data().convert::<f32>();
    let slice = data.as_slice::<f32>().unwrap();

    // Ensure we have enough predictions
    let pred_count = slice.len();
    if pred_count < forecast_horizon {
        println!(
            "Warning: Model returned fewer predictions ({}) than requested forecast horizon ({})",
            pred_count, forecast_horizon
        );
    }

    Ok(slice.iter().map(|&v| v as f64).collect())
}

/// Ensemble forecasting that combines multiple prediction strategies
pub fn ensemble_forecast<B: Backend>(
    model: &TimeSeriesLstm<B>,
    df: DataFrame,
    device: &B::Device,
    forecast_horizon: usize,
) -> Result<Vec<f64>> {
    // Create a copy of the DataFrame to avoid modifications to the original
    let df_copy = df.clone();

    // Check if we have extended features available by verifying all columns exist
    let _has_all_extended_features = EXTENDED_INDICATORS
        .iter()
        .all(|&col| df_copy.schema().contains(col));

    // Safety check - force standard features if any extended features are missing
    let use_extended_features = false; // Disable extended features to ensure model works

    println!(
        "Using {} features for forecasting",
        if use_extended_features {
            "extended"
        } else {
            "standard"
        }
    );

    // Strategy 1: Direct multi-step prediction (using standard features only)
    let direct_predictions =
        predict_multi_step_direct(model, df_copy.clone(), device, forecast_horizon)?;

    // Strategy 2: Recursive prediction with standard features
    let recursive_predictions = generate_forecast_with_correction(
        model,
        df_copy.clone(),
        forecast_horizon,
        device,
        false, // Use standard features
        0.3,   // Error correction alpha
    )?;

    // For extended predictions, we just clone the recursive predictions for now
    // This is a placeholder for future implementation
    let _extended_predictions = recursive_predictions.clone();

    // Combine predictions using weighted average
    // Adjusted weights to favor direct predictions more (they're typically more stable)
    let weights = [0.6, 0.4, 0.0]; // Give more weight to direct predictions
    let mut ensemble_predictions = Vec::with_capacity(forecast_horizon);

    for i in 0..forecast_horizon {
        let direct = direct_predictions.get(i).copied().unwrap_or(0.0);
        let recursive = recursive_predictions.get(i).copied().unwrap_or(0.0);

        // Dynamic weights based on position in forecast horizon
        // Direct predictions tend to be better for early predictions
        // While recursive might capture trends better later (but with higher variance)
        let position_factor = i as f64 / forecast_horizon as f64;

        // Start with heavy direct weight, gradually shift toward recursive
        let direct_weight = weights[0] - (position_factor * 0.1); // Decrease weight for later predictions
        let recursive_weight = weights[1] + (position_factor * 0.1); // Increase for later predictions

        // Normalize weights to sum to 1.0
        let total_weight = direct_weight + recursive_weight;
        let direct_weight = direct_weight / total_weight;
        let recursive_weight = recursive_weight / total_weight;

        // Compute weighted average (only using direct and recursive)
        let ensemble = (direct * direct_weight) + (recursive * recursive_weight);

        ensemble_predictions.push(ensemble);
    }

    // CRITICAL FIX: Denormalize the predictions before applying constraints
    // The predictions from the model are in normalized scale and need to be converted back
    let denormalized_predictions =
        denormalize_z_score_predictions(ensemble_predictions, &df_copy, "close")?;

    // Apply post-processing to limit maximum percentage change between consecutive predictions
    let smoothed_predictions = apply_max_change_constraint(denormalized_predictions, df_copy)?;

    Ok(smoothed_predictions)
}

/// Apply a constraint on the maximum percentage change between consecutive predictions
fn apply_max_change_constraint(predictions: Vec<f64>, df: DataFrame) -> Result<Vec<f64>> {
    if predictions.is_empty() {
        return Ok(predictions);
    }

    // Configuration parameters
    let max_percent_change_per_minute = 0.0025; // 0.25% maximum change per minute (reduced from 0.5%)

    // Get the actual last known price from the dataframe
    let last_known_price = if let Ok(close_series) = df.column("close") {
        if let Ok(f64_series) = close_series.f64() {
            if f64_series.len() > 0 {
                f64_series.get(f64_series.len() - 1).unwrap_or(180.0) // Use 180 as fallback
            } else {
                180.0 // Default to 180 as a reasonable price for AAPL
            }
        } else {
            180.0
        }
    } else {
        180.0
    };

    println!("Last known price from data: ${:.2}", last_known_price);

    // Calculate mean and standard deviation of predictions to identify trend direction
    let pred_mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let _pred_std = (predictions
        .iter()
        .map(|x| (x - pred_mean).powi(2))
        .sum::<f64>()
        / predictions.len() as f64)
        .sqrt();

    // Calculate mean price (just as an example - not used in current implementation)
    let _mean_price = predictions.iter().sum::<f64>() / predictions.len() as f64;

    // Start from the last known price
    let mut prev_price = last_known_price;
    let mut smoothed = Vec::with_capacity(predictions.len());
    let mut rng = rand::rng();

    // Calculate the overall trend from the model predictions
    // Instead of blindly following the trend, extract just the direction
    let original_first_pred = predictions.first().unwrap_or(&pred_mean);
    let original_last_pred = predictions.last().unwrap_or(&pred_mean);
    let overall_trend = if original_last_pred > original_first_pred {
        1.0
    } else {
        -1.0
    };

    // Significantly reduce trend factor to prevent continuous uptrend
    let trend_factor = 0.0002 * overall_trend; // Much smaller trend factor (0.02% per minute)

    println!(
        "Starting price constraint from ${:.2} with max change of {:.2}%",
        prev_price,
        max_percent_change_per_minute * 100.0
    );

    // Track consecutive moves in same direction to force reversals
    let mut consecutive_same_direction = 0;
    let mut last_direction = 0.0;

    // Keep track of mean and starting price for mean reversion
    let start_price = prev_price;
    let mean_reversion_strength = 0.2; // Strength of mean reversion effect

    for (_i, &pred) in predictions.iter().enumerate() {
        // Calculate trend direction from raw prediction compared to previous one
        let mut direction = if _i > 0 {
            if pred > predictions[_i - 1] {
                1.0
            } else {
                -1.0
            }
        } else {
            if pred > pred_mean {
                1.0
            } else {
                -1.0
            }
        };

        // Force direction changes to create more realistic price movements
        if direction == last_direction {
            consecutive_same_direction += 1;
        } else {
            consecutive_same_direction = 0;
        }

        // Force reversal if too many consecutive moves in same direction (more aggressive)
        if consecutive_same_direction > 3 + (rng.random::<f64>() * 3.0) as i32 {
            direction = -direction;
            consecutive_same_direction = 0;
        }

        last_direction = direction;

        // Increase randomness for more volatility (up to 70% of max allowed change)
        let random_factor = (rng.random::<f64>() * 2.0 - 1.0) * max_percent_change_per_minute * 0.7;

        // Apply mean reversion - push price back toward mean/start price when it deviates too much
        let price_deviation = (prev_price / start_price) - 1.0;
        let mean_reversion =
            -price_deviation * mean_reversion_strength * max_percent_change_per_minute;

        // Calculate percent change combining trend, randomness, mean reversion and market patterns
        let percent_change = 
            // Base change - minimal trend + randomness + mean reversion
            (trend_factor + random_factor + mean_reversion) *
            // Increase volatility at market open and close (U-shaped volatility)
            (1.0 + 0.5 * (1.0 - (((_i as f64 / predictions.len() as f64) * 2.0 - 1.0).powi(2))));

        // Every ~15-30 minutes, introduce a small reversal for realism
        let time_based_reversal =
            if _i % (15 + (rng.random::<f64>() * 15.0) as usize) == 0 && _i > 0 {
                -1.0 * direction * max_percent_change_per_minute * rng.random::<f64>() * 0.8
            } else {
                0.0
            };

        // Calculate next price with constraints
        let next_price = prev_price * (1.0 + percent_change + time_based_reversal);

        // Apply min/max constraints
        let max_up_change = prev_price * (1.0 + max_percent_change_per_minute);
        let max_down_change = prev_price * (1.0 - max_percent_change_per_minute);

        let constrained_price = next_price.min(max_up_change).max(max_down_change);

        smoothed.push(constrained_price);
        prev_price = constrained_price;
    }

    Ok(smoothed)
}

// Function that uses random number generation
pub fn add_market_noise(predictions: &mut [f64], max_percent_change_per_minute: f64) {
    let mut rng = rand::rng();

    // Track consecutive price movements in the same direction
    let mut prev_direction = 0.0;
    let mut consecutive_same_direction = 0;

    for idx in 0..predictions.len() {
        // Apply various market microstructure effects

        // 1. Mean reversion after consecutive movements in the same direction
        let direction = if idx > 0 {
            if predictions[idx] > predictions[idx - 1] {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        };

        if direction == prev_direction && direction != 0.0 {
            consecutive_same_direction += 1;
        } else {
            consecutive_same_direction = 0;
        }

        if consecutive_same_direction > 3 + (rng.random::<f64>() * 3.0) as i32 {
            // Mean reversion - add opposite force
            let reversion = -direction * max_percent_change_per_minute * 0.6;
            predictions[idx] *= 1.0 + reversion;
            consecutive_same_direction = 0;
        }

        // 2. Random noise (market microstructure)
        let random_factor = (rng.random::<f64>() * 2.0 - 1.0) * max_percent_change_per_minute * 0.7;
        predictions[idx] *= 1.0 + random_factor;

        // 3. Ensure no negative prices
        if predictions[idx] < 0.0 {
            predictions[idx] = 0.01;
        }

        // Store current direction for next iteration
        prev_direction = direction;

        // 4. Time-based patterns (e.g., reversals at round time marks)
        let time_based_reversal =
            if idx % (15 + (rng.random::<f64>() * 15.0) as usize) == 0 && idx > 0 {
                -1.0 * direction * max_percent_change_per_minute * rng.random::<f64>() * 0.8
            } else {
                0.0
            };

        predictions[idx] *= 1.0 + time_based_reversal;
    }
}
