// External imports
use anyhow::Result;
use polars::prelude::*;
use chrono::{NaiveDateTime, Duration};
use rand::Rng;

/// Generate a test DataFrame with random data for testing
pub fn generate_test_dataframe(num_rows: usize) -> Result<DataFrame> {
    let mut rng = rand::thread_rng();
    
    // Create time series dates
    let base_date = NaiveDateTime::parse_from_str("2023-01-01 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let times: Vec<String> = (0..num_rows)
        .map(|i| (base_date + Duration::minutes(i as i64)).format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    
    // Generate random price data with realistic relationships
    let mut close_prices = Vec::with_capacity(num_rows);
    let mut open_prices = Vec::with_capacity(num_rows);
    let mut high_prices = Vec::with_capacity(num_rows);
    let mut low_prices = Vec::with_capacity(num_rows);
    let mut volume = Vec::with_capacity(num_rows);
    
    // Start with a base price around $100
    let mut current_price = 100.0 + (rng.gen::<f64>() * 50.0);
    
    for _ in 0..num_rows {
        // Random price movement between -1% and +1%
        let movement = (rng.gen::<f64>() * 2.0 - 1.0) * 0.01;
        current_price = current_price * (1.0 + movement);
        
        // Generate open, high, low with realistic relationships to close
        let open = current_price * (1.0 + (rng.gen::<f64>() * 0.01 - 0.005));
        let high = current_price.max(open) * (1.0 + rng.gen::<f64>() * 0.005);
        let low = current_price.min(open) * (1.0 - rng.gen::<f64>() * 0.005);
        
        // Add some random volume
        let vol = rng.gen::<u32>() % 100_000 + 10_000;
        
        // Push to vectors
        close_prices.push(current_price);
        open_prices.push(open);
        high_prices.push(high);
        low_prices.push(low);
        volume.push(vol as i64);
    }
    
    // Create a symbol column (all the same value)
    let symbol = vec!["AAPL".to_string(); num_rows];
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("time", times),
        Series::new("symbol", symbol),
        Series::new("close", close_prices),
        Series::new("open", open_prices),
        Series::new("high", high_prices),
        Series::new("low", low_prices),
        Series::new("volume", volume),
    ])?;
    
    Ok(df)
} 