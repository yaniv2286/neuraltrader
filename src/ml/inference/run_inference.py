#!/usr/bin/env python3
"""
NeuralTrader 2.0 - Model Inference Script
==========================================
Applies trained XGBoost model to processed data to generate scores.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from typing import List, Dict
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralTrader.Inference")

class ModelInference:
    """Apply trained model to processed data."""
    
    def __init__(self, processed_path: str = "data/processed", models_path: str = "models"):
        self.processed_path = Path(processed_path)
        self.models_path = Path(models_path)
        
        # Load feature metadata
        metadata_path = self.processed_path / "feature_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.feature_metadata = json.load(f)
        else:
            logger.warning("No feature metadata found")
            self.feature_metadata = {}
    
    def load_model(self) -> xgb.XGBRanker:
        """Load the trained XGBoost model."""
        # Try different model names
        model_names = ["neural_ranker_v1.json", "xgboost_ranker.json"]
        model_path = None
        
        for name in model_names:
            path = self.models_path / name
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model found in {self.models_path}")
        
        logger.info(f"Loading model from {model_path}")
        model = xgb.XGBRanker()
        model.load_model(str(model_path))
        logger.info("Model loaded successfully")
        return model
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed feature matrix."""
        data_path = self.processed_path / "master_feature_matrix.parquet"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found: {data_path}")
        
        logger.info(f"Loading processed data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data: {df.shape}")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for inference."""
        # Exclude non-feature columns
        exclude_cols = ['ticker', 'date', 'target', 'score']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        return feature_cols
    
    def predict_scores(self, model: xgb.XGBRanker, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Apply model to generate scores."""
        logger.info("Running model inference...")
        
        # Prepare data for inference
        X = df[feature_cols].copy()
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Predict scores
        scores = model.predict(X)
        
        # Add scores to dataframe
        df = df.copy()
        df['score'] = scores
        
        logger.info(f"Generated scores for {len(df)} rows")
        logger.info(f"Score range: {scores.min():.4f} to {scores.max():.4f}")
        
        return df
    
    def save_scored_data(self, df: pd.DataFrame, filename: str = "scored_data.parquet") -> None:
        """Save dataframe with scores."""
        output_path = self.processed_path / filename
        
        logger.info(f"Saving scored data to {output_path}")
        df.to_parquet(output_path, index=False)
        
        # Also save individual ticker files for backtester
        logger.info("Saving individual ticker files...")
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_path = self.processed_path / f"{ticker}.parquet"
            ticker_data.to_parquet(ticker_path, index=False)
        
        logger.info(f"Saved {len(df['ticker'].unique())} ticker files")
    
    def run(self) -> pd.DataFrame:
        """Run the complete inference pipeline."""
        try:
            # Load model and data
            model = self.load_model()
            df = self.load_processed_data()
            
            # Get feature columns
            feature_cols = self.get_feature_columns(df)
            
            # Generate scores
            df = self.predict_scores(model, df, feature_cols)
            
            # Save results
            self.save_scored_data(df)
            
            logger.info("Inference pipeline completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply trained model to processed data")
    parser.add_argument("--processed-path", default="data/processed", help="Path to processed data")
    parser.add_argument("--models-path", default="models", help="Path to models")
    
    args = parser.parse_args()
    
    # Run inference
    inference = ModelInference(
        processed_path=args.processed_path,
        models_path=args.models_path
    )
    
    df = inference.run()
    
    # Print summary
    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {len(df['ticker'].unique())}")
    print(f"Score range: {df['score'].min():.4f} to {df['score'].max():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
