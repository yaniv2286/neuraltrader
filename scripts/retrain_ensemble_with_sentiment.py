#!/usr/bin/env python3
"""
NeuralTrader Ensemble Model Training with Sentiment Integration
==============================================================

Enhanced model training that integrates economic, news, and social media sentiment
data to improve CAGR and reduce drawdown through multi-factor analysis.

SENTIMENT ENHANCEMENTS:
- Economic sentiment features (FRED data integration)
- News sentiment features (financial news analysis)
- Social media sentiment features (Twitter/Reddit/StockTwits)
- Sentiment-technical interaction features
- Combined sentiment consensus features
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import shutil
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sentiment_feature_engineer import SentimentFeatureEngineer
from scripts.data_manager import DataManager
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from src.sentiment.sentiment_integration import SentimentIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SentimentTrainer')

class SentimentEnhancedTrainer:
    """Enhanced model trainer with sentiment integration"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.sentiment_fe = SentimentFeatureEngineer(use_advanced_features=True, verbose=True)
        self.sentiment_integration = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.performance_metrics = {}
        
        # Initialize sentiment integration
        self._initialize_sentiment_integration()
        
        logger.info("SentimentEnhancedTrainer initialized")
    
    def _initialize_sentiment_integration(self):
        """Initialize sentiment integration for feature generation"""
        try:
            sentiment_config = {
                'enable_economic': True,
                'enable_news': True,
                'enable_social': True,
                'cache_ttl': 3600,
                'fred_api_key': os.getenv('FRED_API_KEY'),
                'news_api_key': os.getenv('NEWS_API_KEY'),
                'social_api_keys': {
                    'twitter_api_key': os.getenv('TWITTER_API_KEY'),
                    'reddit_api_key': os.getenv('REDDIT_API_KEY'),
                    'stocktwits_api_key': os.getenv('STOCKTWITS_API_KEY')
                }
            }
            
            self.sentiment_integration = SentimentIntegration(sentiment_config)
            logger.info("Sentiment integration initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment integration: {e}")
            self.sentiment_integration = None
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from parquet files"""
        try:
            logger.info("Loading training data...")
            
            cache_dir = Path("data/raw")
            
            if not cache_dir.exists():
                raise ValueError(f"Data directory not found: {cache_dir}")
            
            # Find all parquet files
            parquet_files = list(cache_dir.glob("*.parquet"))
            
            if not parquet_files:
                raise ValueError("No parquet files found in data directory")
            
            logger.info(f"Found {len(parquet_files)} parquet files")
            
            # Load and combine all data
            all_data = []
            
            for parquet_file in parquet_files[:50]:  # Limit to first 50 tickers for testing
                try:
                    ticker = parquet_file.stem
                    df = pd.read_parquet(parquet_file)
                    df['ticker'] = ticker
                    all_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {parquet_file}: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No data loaded from parquet files")
            
            market_data = pd.concat(all_data, ignore_index=True)
            
            # Ensure required columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
            for col in required_cols:
                if col not in market_data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Convert date column
            market_data['date'] = pd.to_datetime(market_data['date'])
            
            logger.info(f"Loaded {len(market_data)} rows of market data from {len(all_data)} tickers")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return pd.DataFrame()
    
    def generate_sentiment_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate sentiment features for training data"""
        try:
            if not self.sentiment_integration:
                logger.warning("No sentiment integration available, using mock data")
                return self._generate_mock_sentiment_data()
            
            logger.info("Generating sentiment features...")
            
            # Get date range from data
            if 'date' in data.columns:
                start_date = data['date'].min()
                end_date = data['date'].max()
            else:
                # Default to last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
            
            # Get comprehensive sentiment analysis
            sentiment_data = self.sentiment_integration.get_comprehensive_sentiment(
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"Generated sentiment data: {sentiment_data.get('overall_score', 0.0):.3f}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Failed to generate sentiment features: {e}")
            return self._generate_mock_sentiment_data()
    
    def _generate_mock_sentiment_data(self) -> Dict[str, Any]:
        """Generate mock sentiment data for testing"""
        import random
        
        # Generate realistic mock sentiment data
        economic_score = random.uniform(-0.3, 0.3)
        news_score = random.uniform(-0.4, 0.4)
        social_score = random.uniform(-0.5, 0.5)
        
        return {
            'overall_score': (economic_score + news_score + social_score) / 3,
            'overall_regime': 'NEUTRAL',
            'confidence': random.uniform(0.3, 0.8),
            'economic': {
                'score': economic_score,
                'regime': 'NEUTRAL',
                'momentum': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.3, 0.8)
            },
            'news': {
                'score': news_score,
                'regime': 'NEUTRAL',
                'momentum': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.3, 0.8),
                'articles_analyzed': random.randint(50, 200)
            },
            'social': {
                'score': social_score,
                'regime': 'NEUTRAL',
                'momentum': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.3, 0.8),
                'posts_analyzed': random.randint(100, 500)
            }
        }
    
    def prepare_training_data(self, market_data: pd.DataFrame, sentiment_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with sentiment features"""
        try:
            logger.info("Preparing training data with sentiment features...")
            
            # Group data by ticker
            all_features = []
            all_targets = []
            tickers = market_data['ticker'].unique()
            
            for i, ticker in enumerate(tickers):
                if i % 10 == 0:
                    logger.info(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
                
                ticker_data = market_data[market_data['ticker'] == ticker].copy()
                
                if len(ticker_data) < 100:  # Skip tickers with insufficient data
                    continue
                
                try:
                    # Create enhanced features with sentiment
                    features, target = self.sentiment_fe.create_enhanced_features(
                        ticker_data, sentiment_data, target_type='direction'
                    )
                    
                    if len(features) > 0 and len(target) > 0:
                        all_features.append(features)
                        all_targets.append(target)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {ticker}: {e}")
                    continue
            
            if not all_features:
                raise ValueError("No features generated for training")
            
            # Combine all features and targets
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_targets, ignore_index=True)
            
            # Remove any remaining NaN values
            X = X.fillna(0)
            y = y.fillna(0)
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble models with sentiment-enhanced features"""
        try:
            logger.info("Training ensemble models with sentiment features...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            models = {}
            
            # XGBoost
            logger.info("Training XGBoost model...")
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            models['xgboost'] = xgb_model
            
            # LightGBM
            logger.info("Training LightGBM model...")
            lgb_model = LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            models['lightgbm'] = lgb_model
            
            # Random Forest
            logger.info("Training Random Forest model...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            models['random_forest'] = rf_model
            
            # Evaluate models
            performance = {}
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                
                performance[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'features_used': X_train.shape[1]
                }
                
                logger.info(f"{name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}")
            
            self.models = models
            self.performance_metrics = performance
            
            return models, performance
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return {}, {}
    
    def save_models(self, output_dir: str = "models") -> None:
        """Save trained models with sentiment features"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                model_path = output_path / f"{name}_sentiment_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save scaler
            scaler_path = output_path / "sentiment_feature_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {scaler_path}")
            
            # Save feature names
            feature_path = output_path / "sentiment_feature_names.json"
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f)
            logger.info(f"Saved feature names to {feature_path}")
            
            # Save performance metrics
            perf_path = output_path / "sentiment_model_performance.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            logger.info(f"Saved performance metrics to {perf_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance by sentiment groups"""
        try:
            feature_importance = {}
            
            # Get feature groups
            feature_groups = self.sentiment_fe.get_feature_importance_groups()
            
            # Analyze each model
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Group importances by feature type
                    group_importance = {}
                    for group_name, feature_list in feature_groups.items():
                        group_importance[group_name] = {}
                        
                        for feature in feature_list:
                            if feature in self.feature_names:
                                idx = self.feature_names.index(feature)
                                group_importance[group_name][feature] = importances[idx]
                    
                    feature_importance[model_name] = group_importance
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {e}")
            return {}
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run complete sentiment-enhanced training pipeline"""
        try:
            logger.info("Starting sentiment-enhanced training pipeline...")
            
            # Load data
            market_data = self.load_training_data()
            if market_data.empty:
                raise ValueError("No market data available")
            
            # Generate sentiment features
            sentiment_data = self.generate_sentiment_features(market_data)
            
            # Prepare training data
            X, y = self.prepare_training_data(market_data, sentiment_data)
            if X.empty:
                raise ValueError("No training data prepared")
            
            # Train models
            models, performance = self.train_models(X, y)
            if not models:
                raise ValueError("No models trained")
            
            # Analyze feature importance
            feature_importance = self.analyze_feature_importance()
            
            # Save models
            self.save_models()
            
            # Generate summary
            summary = {
                'training_samples': len(X),
                'features': len(self.feature_names),
                'performance': performance,
                'feature_importance': feature_importance,
                'sentiment_data_summary': {
                    'overall_score': sentiment_data.get('overall_score', 0.0),
                    'overall_regime': sentiment_data.get('overall_regime', 'NEUTRAL'),
                    'confidence': sentiment_data.get('confidence', 0.0)
                }
            }
            
            logger.info("Sentiment-enhanced training pipeline completed successfully")
            
            return summary
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {}

def main():
    """Main execution function"""
    try:
        logger.info("Starting Sentiment-Enhanced Model Training...")
        
        trainer = SentimentEnhancedTrainer()
        results = trainer.run_training_pipeline()
        
        if results:
            logger.info("Training completed successfully!")
            logger.info(f"Results: {json.dumps(results, indent=2)}")
        else:
            logger.error("Training failed!")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
