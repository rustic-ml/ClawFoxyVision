// External crates
use polars::frame::column::Column;
use polars::prelude::*;
use polars::prelude::PlSmallStr;

/// Calculates Simple Moving Average (SMA) using rolling_map
pub fn calculate_sma(df: &DataFrame, column: &str, window: usize) -> PolarsResult<Series> {
    let series = df.column(column)?.f64()?;
    let result = series.rolling_map(
        &|s: &Series| Series::new("".into(), [s.mean()]),
        RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        },
    )?;
    Ok(result)
}

/// Calculates Exponential Moving Average (EMA)
pub fn calculate_ema(df: &DataFrame, column: &str, window: usize) -> PolarsResult<Series> {
    let series = df.column(column)?.f64()?;
    let alpha = 2.0 / (window as f64 + 1.0);
    let mut ema_vec = Vec::with_capacity(series.len());
    let mut prev_ema = None;
    for (i, val) in series.into_no_null_iter().enumerate() {
        let ema = if i == 0 {
            val
        } else {
            let prev = prev_ema.unwrap();
            alpha * val + (1.0 - alpha) * prev
        };
        ema_vec.push(ema);
        prev_ema = Some(ema);
    }
    Ok(Series::new(PlSmallStr::from(column), ema_vec))
}

/// Calculates Relative Strength Index (RSI) using vectorized operations
pub fn calculate_rsi(df: &DataFrame, window: usize) -> PolarsResult<Series> {
    let close = df.column("close")?.f64()?;
    let close_slice1 = close.slice(1, close.len() - 1);
    let close_slice2 = close.slice(0, close.len() - 1);
    let diff = &close_slice1 - &close_slice2;
    let diff_vec: Vec<Option<f64>> = std::iter::once(None).chain(diff.into_iter()).collect();
    let diff = Float64Chunked::from_iter(diff_vec);
    let gain = diff.apply(|v| v.map(|x| if x > 0.0 { x } else { 0.0 }));
    let loss = diff.apply(|v| v.map(|x| if x < 0.0 { -x } else { 0.0 }));
    let avg_gain = gain.rolling_map(
        &|s: &Series| Series::new("".into(), [s.mean()]),
        RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        },
    )?;
    let avg_loss = loss.rolling_map(
        &|s: &Series| Series::new("".into(), [s.mean()]),
        RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        },
    )?;
    let rs = &avg_gain / &avg_loss;
    let rs = rs?;
    let rs_ca = rs.f64().unwrap();
    let rsi = rs_ca.apply(|v| v.map(|x| 100.0 - (100.0 / (1.0 + x))));
    Ok(rsi.into_series())
}

/// Calculates Moving Average Convergence Divergence (MACD)
pub fn calculate_macd(df: &DataFrame) -> PolarsResult<(Series, Series)> {
    let ema12 = calculate_ema(df, "close", 12)?;
    let ema26 = calculate_ema(df, "close", 26)?;
    let macd = (&ema12 - &ema26)?;
    let macd_df = DataFrame::new(vec![macd.clone().into()])?;
    let signal = calculate_ema(&macd_df, macd.name(), 9)?;
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

/// Calculates Bollinger Bands using rolling_map for mean and std
pub fn calculate_bollinger_bands(
    df: &DataFrame,
    window: usize,
    num_std: f64,
) -> PolarsResult<(Series, Series, Series)> {
    let close = df.column("close")?.f64()?;
    let sma = close.rolling_map(
        &|s: &Series| Series::new("".into(), [s.mean()]),
        RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        },
    )?;
    let std = close.rolling_map(
        &|s: &Series| {
            let values: Vec<f64> = s.f64().unwrap().into_iter().map(|v| v.unwrap_or(0.0)).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            Series::new("".into(), [variance.sqrt()])
        },
        RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        },
    )?;
    let upper_band = &sma + &(std.clone() * num_std);
    let lower_band = &sma - &(std * num_std);
    let upper_band = upper_band?;
    let lower_band = lower_band?;
    Ok((sma.into_series(), upper_band.into_series(), lower_band.into_series()))
}

/// Calculates Average True Range (ATR) using vectorized operations
pub fn calculate_atr(df: &DataFrame, window: usize) -> PolarsResult<Series> {
    let high = df.column("high")?.f64()?;
    let low = df.column("low")?.f64()?;
    let close = df.column("close")?.f64()?;
    let prev_close = close.shift(1);
    let tr = *&high - &*low;
    let tr2 = (*&high - &prev_close).apply(|v| v.map(|x| x.abs()));
    let tr3 = (*&low - &prev_close).apply(|v| v.map(|x| x.abs()));
    let tr = tr.zip_with(&tr2.gt(&tr), &tr2)?.zip_with(&tr3.gt(&tr), &tr3)?;
    let atr = tr.rolling_map(
        &|s: &Series| Series::new("".into(), [s.mean()]),
        RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        },
    )?;
    Ok(atr)
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

    // Stack all indicators
    let result = df.hstack(&[
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
    ])?;

    // Print column names for debugging
    println!("Columns after hstack: {:?}", result.get_column_names());

    // Return the result
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_processor::load_and_preprocess;

    #[test]
    fn test_add_technical_indicators() {
        let workspace_dir = std::env::current_dir().expect("Failed to get current directory");
        let full_path = workspace_dir.join("AAPL-ticker_minute_bars.csv");

        let result = load_and_preprocess(&full_path);
        assert!(result.is_ok());
        let mut df = result.unwrap();

        let result = add_technical_indicators(&mut df);
        assert!(result.is_ok());

        // Use the returned DataFrame for verification
        let df = result.unwrap();
        println!("Columns in final DataFrame: {:?}", df.get_column_names());

        // Verify that all features were added
        let new_features = [
            "sma_20",
            "sma_50",
            "ema_20",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "atr_14",
            "returns",
            "volume_sma_20",
            "volume_ema_20",
            "price_range",
        ];

        for feature in new_features {
            println!("Feature: {}", feature);
            println!("Column: {:?}", df.column(feature));
            assert!(
                df.column(feature).is_ok(),
                "Feature {} was not added",
                feature
            );
        }
    }
}
