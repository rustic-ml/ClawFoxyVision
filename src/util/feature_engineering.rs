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

/// Adds all technical indicators to the DataFrame
/// This is a wrapper around the ta-lib-in-rust library function
pub fn add_technical_indicators(df: &mut DataFrame) -> PolarsResult<DataFrame> {
    // Use ta-lib-in-rust to add standard indicators
    // You may need to implement this function using ta-lib-in-rust API, as it does not provide a single add_technical_indicators function.
    // For now, return the DataFrame unchanged as a placeholder.
    Ok(df.clone())
}