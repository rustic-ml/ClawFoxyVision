// External crates
use polars::frame::column::Column;
use polars::prelude::*;
use chrono::{NaiveDateTime, Timelike, Datelike};
use std::f64::consts::PI;

/// Calculates Simple Moving Average (SMA) using rolling_map
pub fn calculate_sma(df: &DataFrame, column: &str, window: usize) -> PolarsResult<Series> {
    let series = df.column(column)?.f64()?.clone().into_series();
    
    if series.len() < window {
        return Err(PolarsError::ComputeError(
            format!("Not enough data points ({}) for SMA window ({})", series.len(), window).into()
        ));
    }
    
    series.rolling_mean(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })
}

/// Calculates Exponential Moving Average (EMA)
pub fn calculate_ema(df: &DataFrame, column: &str, window: usize) -> PolarsResult<Series> {
    let series = df.column(column)?.f64()?.clone().into_series();
    
    if series.len() < window {
        return Err(PolarsError::ComputeError(
            format!("Not enough data points ({}) for EMA window ({})", series.len(), window).into()
        ));
    }
    
    let alpha = 2.0 / (window as f64 + 1.0);
    series.rolling_mean(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })
}

/// Calculates Relative Strength Index (RSI)
pub fn calculate_rsi(df: &DataFrame, window: usize) -> PolarsResult<Series> {
    let close = df.column("close")?.f64()?.clone().into_series();
    let prev_close = close.shift(1);
    
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    // Handle first value
    gains.push(0.0);
    losses.push(0.0);
    
    for i in 1..close.len() {
        let curr = close.f64()?.get(i).unwrap_or(0.0);
        let prev = prev_close.f64()?.get(i).unwrap_or(0.0);
        let change = curr - prev;
        
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    let gains_series = Series::new("gains".into(), gains);
    let losses_series = Series::new("losses".into(), losses);
    
    let avg_gain = gains_series.rolling_mean(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })?;
    let avg_loss = losses_series.rolling_mean(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })?;
    
    let mut rsi = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let g = avg_gain.f64()?.get(i).unwrap_or(0.0);
        let l = avg_loss.f64()?.get(i).unwrap_or(0.0);
        
        let rsi_val = if l == 0.0 {
            100.0
        } else {
            let rs = g / l;
            100.0 - (100.0 / (1.0 + rs))
        };
        rsi.push(rsi_val);
    }
    
    Ok(Series::new("RSI".into(), rsi))
}

/// Calculates Moving Average Convergence Divergence (MACD)
pub fn calculate_macd(df: &DataFrame) -> PolarsResult<(Series, Series)> {
    // MACD requires at least 26 points for the longer EMA
    if df.height() < 26 {
        return Err(PolarsError::ComputeError(
            "Not enough data points for MACD calculation (need at least 26)".into()
        ));
    }
    
    let ema12 = calculate_ema(df, "close", 12)?;
    let ema26 = calculate_ema(df, "close", 26)?;
    
    let macd = (&ema12 - &ema26)?.with_name("macd".into());
    
    // Create a temporary DataFrame for signal line calculation
    let mut signal_df = DataFrame::new(vec![macd.clone().into_column()])?;
    let signal = calculate_ema(&signal_df, "macd", 9)?.with_name("macd_signal".into());
    
    Ok((macd, signal))
}

/// Calculates price returns
pub fn calculate_returns(df: &DataFrame) -> PolarsResult<Series> {
    let close: Series = df.column("close")?.as_materialized_series().clone();
    let shifted: Series = close.shift(1);
    let diff: Series = (close.clone() - shifted.clone())?;
    let returns: Series = (&diff / &shifted)?;
    Ok(returns)
}

/// Calculates lagged features for a given column
pub fn calculate_lagged_features(df: &DataFrame, column: &str, lags: &[usize]) -> PolarsResult<Vec<Series>> {
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
        let diff = (&close - &shifted)?;
        // Convert Series to ChunkedArray<Float64Type> before division
        let shifted_f64 = shifted.f64()?;
        let diff_f64 = diff.f64()?;
        // Perform division on ChunkedArrays
        let period_return = (diff_f64 / shifted_f64).into_series();
        let name = format!("returns_{}min", period).into();
        result.push(period_return.with_name(name));
    }
    Ok(result)
}

/// Calculates volatility over different periods
pub fn calculate_volatility(df: &DataFrame, windows: &[usize]) -> PolarsResult<Vec<Series>> {
    let returns = calculate_returns(df)?;
    let mut result = Vec::with_capacity(windows.len());
    
    for &window in windows {
        let volatility = returns.rolling_std(RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            center: false,
            weights: None,
            fn_params: None,
        })?;
        let name = format!("volatility_{}min", window).into();
        result.push(volatility.with_name(name));
    }
    
    Ok(result)
}

/// Create time-based features from the time column
pub fn calculate_time_features(df: &DataFrame) -> PolarsResult<Vec<Series>> {
    // Check if there's a time column
    if !df.schema().contains("time") {
        return Err(PolarsError::ComputeError("Time column not found".into()));
    }
    
    let time_col = df.column("time")?.str()?;
    let n_rows = df.height();
    
    // Create vectors for hour and day of week features
    let mut hour_sin = Vec::with_capacity(n_rows);
    let mut hour_cos = Vec::with_capacity(n_rows);
    let mut day_sin = Vec::with_capacity(n_rows);
    let mut day_cos = Vec::with_capacity(n_rows);
    
    for i in 0..n_rows {
        let time_str = time_col.get(i).unwrap_or("");
        let datetime = match NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d %H:%M:%S UTC") {
            Ok(dt) => dt,
            Err(_) => {
                // Default values if parsing fails
                hour_sin.push(0.0);
                hour_cos.push(1.0);
                day_sin.push(0.0);
                day_cos.push(1.0);
                continue;
            }
        };
        
        // Extract hour (0-23) and day of week (0-6)
        let hour = datetime.hour() as f64;
        let day = datetime.weekday().num_days_from_monday() as f64;
        
        // Encode using sine and cosine to capture cyclical patterns
        hour_sin.push((2.0 * PI * hour / 24.0).sin());
        hour_cos.push((2.0 * PI * hour / 24.0).cos());
        day_sin.push((2.0 * PI * day / 7.0).sin());
        day_cos.push((2.0 * PI * day / 7.0).cos());
    }
    
    // Create series
    let result = vec![
        Series::new("hour_sin".into(), hour_sin),
        Series::new("hour_cos".into(), hour_cos),
        Series::new("day_of_week_sin".into(), day_sin),
        Series::new("day_of_week_cos".into(), day_cos),
    ];
    
    Ok(result)
}

/// Calculates Bollinger Bands
pub fn calculate_bollinger_bands(
    df: &DataFrame,
    window: usize,
    num_std: f64,
) -> PolarsResult<(Series, Series, Series)> {
    let close = df.column("close")?.f64()?.clone().into_series();
    
    if close.len() < window {
        return Err(PolarsError::ComputeError(
            format!("Not enough data points ({}) for Bollinger Bands window ({})", close.len(), window).into()
        ));
    }
    
    let sma = close.rolling_mean(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })?;
    let std = close.rolling_std(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })?;
    
    let mut upper_band = Vec::with_capacity(close.len());
    let mut lower_band = Vec::with_capacity(close.len());
    
    for i in 0..close.len() {
        let ma = sma.f64()?.get(i).unwrap_or(0.0);
        let std_val = std.f64()?.get(i).unwrap_or(0.0);
        
        upper_band.push(ma + num_std * std_val);
        lower_band.push(ma - num_std * std_val);
    }
    
    Ok((
        sma,
        Series::new("upper_band".into(), upper_band),
        Series::new("lower_band".into(), lower_band)
    ))
}

/// Calculates Average True Range (ATR)
pub fn calculate_atr(df: &DataFrame, window: usize) -> PolarsResult<Series> {
    let high = df.column("high")?.f64()?.clone().into_series();
    let low = df.column("low")?.f64()?.clone().into_series();
    let close = df.column("close")?.f64()?.clone().into_series();
    
    if df.height() < window + 1 {
        return Err(PolarsError::ComputeError(
            format!("Not enough data points ({}) for ATR calculation (need {})", df.height(), window + 1).into()
        ));
    }
    
    let prev_close = close.shift(1);
    let mut tr_values = Vec::with_capacity(df.height());
    
    let first_tr = {
        let h = high.f64()?.get(0).unwrap_or(0.0);
        let l = low.f64()?.get(0).unwrap_or(0.0);
        h - l
    };
    tr_values.push(first_tr);
    
    for i in 1..df.height() {
        let h = high.f64()?.get(i).unwrap_or(0.0);
        let l = low.f64()?.get(i).unwrap_or(0.0);
        let pc = prev_close.f64()?.get(i).unwrap_or(0.0);
        
        let tr = if pc == 0.0 {
            h - l
        } else {
            (h - l).max((h - pc).abs()).max((l - pc).abs())
        };
        tr_values.push(tr);
    }
    
    let tr_series = Series::new("TR".into(), tr_values);
    tr_series.rolling_mean(RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        center: false,
        weights: None,
        fn_params: None,
    })
}

/// Adds all technical indicators to the DataFrame
pub fn add_technical_indicators(df: &mut DataFrame) -> PolarsResult<DataFrame> {
    // Convert numeric columns to Float64 by mutating in-place via Column
    let numeric_columns = ["open", "high", "low", "close", "volume"];
    for col_name in numeric_columns {
        // 1) Wrap the existing Series in a Column for in-place mutation
        let s: Series = df.column(col_name)?.as_materialized_series().clone();
        let mut col: Column = s.into_column();

        // 2) Materialize and get a &mut Series to cast in place
        let series_mut: &mut Series = col.into_materialized_series();
        *series_mut = series_mut.cast(&DataType::Float64)?;

        // 3) Convert the Column back into a Series and replace it in the DataFrame
        let series: Series = col.take_materialized_series();
        df.replace(col_name, series)?;
    }

    // Calculate all indicators
    let sma20 = calculate_sma(df, "close", 20)?;
    let sma50 = calculate_sma(df, "close", 50)?;
    let ema20 = calculate_ema(df, "close", 20)?;
    let rsi = calculate_rsi(df, 14)?;
    let (macd, signal) = calculate_macd(df)?;
    let (bb_middle, bb_upper, bb_lower) = calculate_bollinger_bands(df, 20, 2.0)?;
    let returns = calculate_returns(df)?;
    let atr_14 = calculate_atr(df, 14)?;

    // Add volume indicators
    let volume_sma20 = calculate_sma(df, "volume", 20)?;
    let volume_ema20 = calculate_ema(df, "volume", 20)?;

    // Add price range
    let high: Series = df.column("high")?.as_materialized_series().clone();
    let low: Series = df.column("low")?.as_materialized_series().clone();
    let price_range: Series = (&high - &low)?;
    let price_range = price_range.with_name("price_range".into());

    // Calculate lag features
    let lags = [5, 15, 30];
    let lagged_close = calculate_lagged_features(df, "close", &lags)?;
    
    // Calculate returns over different periods
    let periods = [5, 15, 30];
    let period_returns = calculate_period_returns(df, &periods)?;
    
    // Calculate volatility
    let windows = [15, 30];
    let volatility = calculate_volatility(df, &windows)?;
    
    // Calculate time-based features
    let time_features = match calculate_time_features(df) {
        Ok(features) => features,
        Err(_) => {
            // If time column is missing or parsing fails, create empty features
            let empty = vec![0.0; df.height()];
            vec![
                Series::new("hour_sin".into(), empty.clone()),
                Series::new("hour_cos".into(), empty.clone()),
                Series::new("day_of_week_sin".into(), empty.clone()),
                Series::new("day_of_week_cos".into(), empty),
            ]
        }
    };

    // Create a vec of all indicators
    let mut all_indicators = vec![
        sma20.with_name("sma_20".into()).into(),
        sma50.with_name("sma_50".into()).into(),
        ema20.with_name("ema_20".into()).into(),
        rsi.with_name("rsi_14".into()).into(),
        macd.with_name("macd".into()).into(),
        signal.with_name("macd_signal".into()).into(),
        bb_middle.with_name("bb_middle".into()).into(),
        bb_upper.with_name("bb_upper".into()).into(),
        bb_lower.with_name("bb_lower".into()).into(),
        atr_14.with_name("atr_14".into()).into(),
        returns.with_name("returns".into()).into(),
        volume_sma20.with_name("volume_sma_20".into()).into(),
        volume_ema20.with_name("volume_ema_20".into()).into(),
        price_range.into(),
    ];
    
    // Add lag features
    all_indicators.extend(lagged_close);
    
    // Add period returns
    all_indicators.extend(period_returns);
    
    // Add volatility
    all_indicators.extend(volatility);
    
    // Add time features
    all_indicators.extend(time_features);

    // Stack all indicators
    let columns: Vec<Column> = all_indicators.into_iter().map(|s| s.into_column()).collect();
    let result = df.hstack(&columns)?;

    // Print column names for debugging
    println!("Columns after hstack: {:?}", result.get_column_names());

    // Return the result
    Ok(result)
}