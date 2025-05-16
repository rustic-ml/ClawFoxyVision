// External crates
use polars::prelude::*;

// Use rustalib for technical indicators
use rustalib::indicators::moving_averages::{calculate_ema, calculate_sma};
use rustalib::indicators::oscillators::{calculate_macd, calculate_rsi};
use rustalib::indicators::volatility::{calculate_atr, calculate_bollinger_bands};

/// Calculates lagged features for a given column
pub fn calculate_lagged_features(
    df: &DataFrame,
    column: &str,
    lags: &[usize],
) -> PolarsResult<Vec<Series>> {
    let series = df.column(column)?.as_materialized_series().clone();
    let mut result = Vec::with_capacity(lags.len());

    for &lag in lags {
        let lagged = series.shift(lag as i64);
        let name = format!("{}_lag_{}", column, lag);
        result.push(lagged.with_name(name.into()));
    }

    Ok(result)
}

/// Calculates returns over different periods
pub fn calculate_period_returns(df: &DataFrame, periods: &[usize]) -> PolarsResult<Vec<Series>> {
    let close = df.column("close")?.f64()?.clone().into_series();
    let mut result = Vec::with_capacity(periods.len());

    for &period in periods {
        let shifted = close.shift(period as i64);
        // Manual calculation of return
        let mut returns = Vec::with_capacity(df.height());

        let close_values: Vec<_> = close.f64()?.into_iter().collect();
        let shifted_values: Vec<_> = shifted.f64()?.into_iter().collect();

        for i in 0..close_values.len() {
            if let (Some(curr), Some(prev)) = (close_values[i], shifted_values[i]) {
                if prev != 0.0 {
                    returns.push(Some((curr - prev) / prev));
                } else {
                    returns.push(Some(0.0));
                }
            } else {
                returns.push(None);
            }
        }

        let name = format!("returns_{}min", period);
        result.push(Series::new(name.into(), returns));
    }
    Ok(result)
}

/// Adds all technical indicators to the DataFrame
pub fn add_technical_indicators(df: &mut DataFrame) -> PolarsResult<DataFrame> {
    // Clone the DataFrame to avoid borrowing issues
    let mut result_df = df.clone();
    let df_height = result_df.height();

    // Ensure all price columns are Float64
    for col_name in ["open", "high", "low", "close", "volume"].iter() {
        if result_df.column(col_name).is_ok() {
            // Convert to Float64 if needed
            let series = result_df.column(col_name)?;
            let f64_series = series.cast(&DataType::Float64)?;
            result_df.with_column(f64_series)?;
        }
    }

    // Helper function to ensure all series have the same length as the DataFrame
    fn ensure_same_length(series: Series, df_height: usize) -> Series {
        if series.len() < df_height {
            // Pad with nulls at the beginning to match DataFrame height
            let missing = df_height - series.len();
            let mut padded = vec![None; missing];
            // Collect the series values
            let values: Vec<Option<f64>> = series.f64().unwrap().into_iter().collect();
            // Append the non-null values
            padded.extend(values);
            Series::new(series.name().to_string().into(), padded)
        } else {
            series
        }
    }

    // Add SMA using rustalib
    let sma_20 = calculate_sma(&result_df, "close", 20)?;
    let sma_20 = ensure_same_length(sma_20.with_name("sma_20".into()), df_height);
    result_df.with_column(sma_20)?;
    
    let sma_50 = calculate_sma(&result_df, "close", 50)?;
    let sma_50 = ensure_same_length(sma_50.with_name("sma_50".into()), df_height);
    result_df.with_column(sma_50)?;

    // Add EMA using rustalib
    let ema_20 = calculate_ema(&result_df, "close", 20)?;
    let ema_20 = ensure_same_length(ema_20.with_name("ema_20".into()), df_height);
    result_df.with_column(ema_20)?;

    // Calculate returns
    let close_series = result_df.column("close")?.clone();
    let close_vals: Vec<Option<f64>> = close_series.f64()?.into_iter().collect();
    let mut returns = Vec::with_capacity(close_vals.len());

    // Manual calculation to avoid operator issues
    for i in 1..close_vals.len() {
        if let (Some(curr), Some(prev)) = (close_vals[i], close_vals[i - 1]) {
            if prev != 0.0 {
                returns.push(Some((curr - prev) / prev));
            } else {
                returns.push(Some(0.0));
            }
        } else {
            returns.push(None);
        }
    }

    // Add leading value to match length
    returns.insert(0, None);
    result_df.with_column(Series::new("returns".into(), returns))?;

    // Calculate price range (High - Low) / Close using manual calculation
    let high_vals: Vec<Option<f64>> = result_df.column("high")?.f64()?.into_iter().collect();
    let low_vals: Vec<Option<f64>> = result_df.column("low")?.f64()?.into_iter().collect();
    let close_vals: Vec<Option<f64>> = result_df.column("close")?.f64()?.into_iter().collect();

    let mut price_range = Vec::with_capacity(close_vals.len());
    for i in 0..close_vals.len() {
        if let (Some(h), Some(l), Some(c)) = (high_vals[i], low_vals[i], close_vals[i]) {
            if c != 0.0 {
                price_range.push(Some((h - l) / c));
            } else {
                price_range.push(Some(0.0));
            }
        } else {
            price_range.push(None);
        }
    }

    result_df.with_column(Series::new("price_range".into(), price_range))?;

    // Add RSI using rustalib
    let rsi_14 = calculate_rsi(&result_df, 14, "close")?;
    let rsi_14 = ensure_same_length(rsi_14.with_name("rsi_14".into()), df_height);
    result_df.with_column(rsi_14)?;

    // Add MACD using rustalib
    let (macd_series, signal_series) = calculate_macd(&result_df, 12, 26, 9, "close")?;
    let macd_series = ensure_same_length(macd_series.with_name("macd".into()), df_height);
    let signal_series = ensure_same_length(signal_series.with_name("macd_signal".into()), df_height);
    result_df.with_column(macd_series)?;
    result_df.with_column(signal_series)?;

    // Add Bollinger Bands using rustalib
    let (bb_middle, bb_upper, bb_lower) = calculate_bollinger_bands(&result_df, 20, 2.0, "close")?;
    let bb_middle = ensure_same_length(bb_middle.with_name("bb_middle".into()), df_height);
    let bb_upper = ensure_same_length(bb_upper.with_name("bb_upper".into()), df_height);
    let bb_lower = ensure_same_length(bb_lower.with_name("bb_lower".into()), df_height);
    result_df.with_column(bb_middle)?;
    result_df.with_column(bb_upper)?;
    result_df.with_column(bb_lower)?;

    // Add ATR using rustalib
    let atr_14 = calculate_atr(&result_df, 14)?;
    let atr_14 = ensure_same_length(atr_14.with_name("atr_14".into()), df_height);
    result_df.with_column(atr_14)?;

    Ok(result_df)
}
