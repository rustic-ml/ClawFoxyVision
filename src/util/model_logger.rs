use std::fs;
use std::path::{Path, PathBuf};
use chrono::Local;
use serde::{Serialize, Deserialize};
use std::io::Write;
use anyhow::Result;

#[derive(Serialize, Deserialize)]
pub struct ModelExperiment {
    pub timestamp: String,
    pub ticker: String,
    pub model_type: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub dropout: f64,
    pub training_days: i64,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub train_rmse: Option<f64>,
    pub test_rmse: Option<f64>,
    pub training_time_seconds: Option<f64>,
    pub notes: String,
}

impl ModelExperiment {
    pub fn new(
        ticker: &str, 
        model_type: &str,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f64,
        training_days: i64,
        batch_size: usize,
        learning_rate: f64,
    ) -> Self {
        Self {
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            ticker: ticker.to_string(),
            model_type: model_type.to_string(),
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
            training_days,
            batch_size,
            learning_rate,
            train_rmse: None,
            test_rmse: None,
            training_time_seconds: None,
            notes: "".to_string(),
        }
    }
    
    pub fn set_train_rmse(&mut self, rmse: f64) {
        self.train_rmse = Some(rmse);
    }
    
    pub fn set_test_rmse(&mut self, rmse: f64) {
        self.test_rmse = Some(rmse);
    }
    
    pub fn set_training_time(&mut self, seconds: f64) {
        self.training_time_seconds = Some(seconds);
    }
    
    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str("\n");
        }
        self.notes.push_str(note);
    }
    
    pub fn save(&self, experiment_dir: &Path) -> Result<PathBuf> {
        // Create directory if it doesn't exist
        fs::create_dir_all(experiment_dir)?;
        
        // Create a unique filename
        let filename = format!(
            "{}_{}_h{}_l{}_d{}_experiment.json",
            self.ticker,
            self.model_type,
            self.hidden_size,
            self.num_layers,
            (self.dropout * 100.0) as i32,
        );
        
        let file_path = experiment_dir.join(filename);
        
        // Serialize to JSON and save
        let json = serde_json::to_string_pretty(&self)?;
        let mut file = fs::File::create(&file_path)?;
        file.write_all(json.as_bytes())?;
        
        Ok(file_path)
    }
}

pub fn create_experiment_dir() -> Result<PathBuf> {
    let dir = Path::new("experiments").join(Local::now().format("%Y%m%d_%H%M%S").to_string());
    fs::create_dir_all(&dir)?;
    Ok(dir)
} 