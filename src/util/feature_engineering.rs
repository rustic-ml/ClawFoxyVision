// External crates
use polars::prelude::*;

// Use the ta-lib-in-rust library for technical indicators

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
    
    // Ensure all price columns are Float64
    for col_name in ["open", "high", "low", "close", "volume"].iter() {
        if result_df.column(col_name).is_ok() {
            // Convert to Float64 if needed
            let series = result_df.column(col_name)?;
            let f64_series = series.cast(&DataType::Float64)?;
            result_df.with_column(f64_series)?;
        }
    }
    
    // Add simple moving averages for key indicators
    add_simple_sma(&mut result_df, "close", 20, "sma_20")?;
    add_simple_sma(&mut result_df, "close", 50, "sma_50")?;
    
    // Add exponential moving average (approximated as simple for now)
    add_simple_sma(&mut result_df, "close", 20, "ema_20")?;
    
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
    
    // Calculate RSI 14 with a simplified approach
    add_rsi(&mut result_df, "close", 14, "rsi_14")?;
    
    // Add MACD (use simple SMA as approximation for now)
    add_simple_sma(&mut result_df, "close", 12, "ema_12")?;
    add_simple_sma(&mut result_df, "close", 26, "ema_26")?;
    
    // Create MACD series manually
    let ema_12_vals: Vec<Option<f64>> = result_df.column("ema_12")?.f64()?.into_iter().collect();
    let ema_26_vals: Vec<Option<f64>> = result_df.column("ema_26")?.f64()?.into_iter().collect();
    
    let mut macd = Vec::with_capacity(ema_12_vals.len());
    for i in 0..ema_12_vals.len() {
        if let (Some(e12), Some(e26)) = (ema_12_vals[i], ema_26_vals[i]) {
            macd.push(Some(e12 - e26));
        } else {
            macd.push(None);
        }
    }
    
    result_df.with_column(Series::new("macd".into(), macd.clone()))?;
    
    // Add signal line (9-day SMA of MACD)
    add_simple_sma_from_values(&mut result_df, &macd, 9, "macd_signal")?;
    
    // Bollinger Bands
    add_simple_sma(&mut result_df, "close", 20, "bb_middle")?;
    
    // Calculate ATR (Average True Range)
    add_atr(&mut result_df, 14, "atr_14")?;
    
    Ok(result_df)
}

/// Helper function to add Simple Moving Average
fn add_simple_sma(df: &mut DataFrame, column: &str, window: usize, new_name: &str) -> PolarsResult<()> {
    let series = df.column(column)?.f64()?;
    let values: Vec<Option<f64>> = series.into_iter().collect();
    let mut sma = Vec::with_capacity(values.len());
    
    // Fill with None for the first window-1 elements
    for _ in 0..window.min(values.len()) - 1 {
        sma.push(None);
    }
    
    // Calculate SMA for the rest
    for i in window - 1..values.len() {
        let window_vals: Vec<f64> = values[i + 1 - window..=i]
            .iter()
            .filter_map(|&v| v)
            .collect();
        
        if !window_vals.is_empty() {
            let mean = window_vals.iter().sum::<f64>() / window_vals.len() as f64;
            sma.push(Some(mean));
        } else {
            sma.push(None);
        }
    }
    
    df.with_column(Series::new(new_name.into(), sma))?;
    Ok(())
}

/// Helper function to add SMA from raw values
fn add_simple_sma_from_values(df: &mut DataFrame, values: &[Option<f64>], window: usize, new_name: &str) -> PolarsResult<()> {
    let mut sma = Vec::with_capacity(values.len());
    
    // Fill with None for the first window-1 elements
    for _ in 0..window.min(values.len()) - 1 {
        sma.push(None);
    }
    
    // Calculate SMA for the rest
    for i in window - 1..values.len() {
        let window_vals: Vec<f64> = values[i + 1 - window..=i]
            .iter()
            .filter_map(|&v| v)
            .collect();
        
        if !window_vals.is_empty() {
            let mean = window_vals.iter().sum::<f64>() / window_vals.len() as f64;
            sma.push(Some(mean));
        } else {
            sma.push(None);
        }
    }
    
    df.with_column(Series::new(new_name.into(), sma))?;
    Ok(())
}

/// Add RSI calculation
fn add_rsi(df: &mut DataFrame, column: &str, period: usize, new_name: &str) -> PolarsResult<()> {
    let close_vals: Vec<Option<f64>> = df.column(column)?.f64()?.into_iter().collect();
    
    // Calculate up/down movements
    let mut up = Vec::with_capacity(close_vals.len());
    let mut down = Vec::with_capacity(close_vals.len());
    
    // First value has no change
    up.push(None);
    down.push(None);
    
    for i in 1..close_vals.len() {
        if let (Some(curr), Some(prev)) = (close_vals[i], close_vals[i - 1]) {
            let change = curr - prev;
            if change > 0.0 {
                up.push(Some(change));
                down.push(Some(0.0));
            } else {
                up.push(Some(0.0));
                down.push(Some(-change));
            }
        } else {
            up.push(None);
            down.push(None);
        }
    }
    
    // Calculate SMA of up and down
    let mut avg_up = Vec::with_capacity(up.len());
    let mut avg_down = Vec::with_capacity(down.len());
    
    // Fill with None for the first period-1 elements
    for _ in 0..period.min(up.len()) {
        avg_up.push(None);
        avg_down.push(None);
    }
    
    // Calculate averages for the rest
    for i in period..up.len() {
        let up_window: Vec<f64> = up[i + 1 - period..=i]
            .iter()
            .filter_map(|&v| v)
            .collect();
        
        let down_window: Vec<f64> = down[i + 1 - period..=i]
            .iter()
            .filter_map(|&v| v)
            .collect();
        
        if !up_window.is_empty() && !down_window.is_empty() {
            let avg_u = up_window.iter().sum::<f64>() / up_window.len() as f64;
            let avg_d = down_window.iter().sum::<f64>() / down_window.len() as f64;
            
            avg_up.push(Some(avg_u));
            avg_down.push(Some(avg_d));
        } else {
            avg_up.push(None);
            avg_down.push(None);
        }
    }
    
    // Calculate RSI
    let mut rsi = Vec::with_capacity(avg_up.len());
    for i in 0..avg_up.len() {
        if let (Some(u), Some(d)) = (avg_up[i], avg_down[i]) {
            if d == 0.0 {
                rsi.push(Some(100.0));
            } else {
                let rs = u / d;
                rsi.push(Some(100.0 - (100.0 / (1.0 + rs))));
            }
        } else {
            rsi.push(None);
        }
    }
    
    df.with_column(Series::new(new_name.into(), rsi))?;
    Ok(())
}

/// Add ATR calculation
fn add_atr(df: &mut DataFrame, period: usize, new_name: &str) -> PolarsResult<()> {
    let high_vals: Vec<Option<f64>> = df.column("high")?.f64()?.into_iter().collect();
    let low_vals: Vec<Option<f64>> = df.column("low")?.f64()?.into_iter().collect();
    let close_vals: Vec<Option<f64>> = df.column("close")?.f64()?.into_iter().collect();
    
    // Calculate true range
    let mut tr = Vec::with_capacity(close_vals.len());
    
    // First value has no previous close
    tr.push(None);
    
    for i in 1..close_vals.len() {
        if let (Some(h), Some(l), Some(prev_c)) = (high_vals[i], low_vals[i], close_vals[i - 1]) {
            let h_l = h - l;
            let h_pc = (h - prev_c).abs();
            let l_pc = (l - prev_c).abs();
            
            tr.push(Some(h_l.max(h_pc).max(l_pc)));
        } else {
            tr.push(None);
        }
    }
    
    // Calculate ATR (simple moving average of true range)
    add_simple_sma_from_values(df, &tr, period, new_name)?;
    
    Ok(())
}