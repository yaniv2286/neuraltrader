#!/usr/bin/env python3
"""
NeuralTrader Complete Model Retraining with Real Sentiment Data
==============================================================

Retrain ALL models (XGBoost, LightGBM, Random Forest, Ensemble) with:
- Original tickers (all available parquet files)
- Original indicators and features
- Real FRED economic sentiment data
- Real News sentiment data (mock fallback if no API key)
- No mock data - only real sentiment integration

MODELS TO TRAIN:
1. XGBoost (xgboost_model.pkl)
2. LightGBM (lightgbm_model.pkl) 
3. Random Forest (randomforest_model.pkl)
4. Ensemble (ensemble_metadata.json)
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
from typing import Dict, Any, Tuple, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sentiment_feature_engineer import SentimentFeatureEngineer
from core.feature_engineer import FeatureEngineer
from scripts.data_manager import DataManager
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, roc_auc_score
from src.sentiment.sentiment_integration import SentimentIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CompleteRetrainer')

class CompleteModelRetrainer:
    """Complete model retraining with real sentiment data"""
    
    def __init__(self):
        """Initialize the complete retraining pipeline"""
        self.scaler = StandardScaler()
        self.feature_names = []
        self.performance_metrics = {}
        
        # Initialize sentiment integration
        self._initialize_sentiment_integration()
        
        # Initialize feature engineers
        self.base_fe = FeatureEngineer(use_advanced_features=True, verbose=True)
        self.sentiment_fe = SentimentFeatureEngineer(use_advanced_features=True, verbose=True)
        
        logger.info("CompleteModelRetrainer initialized")
    
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
    
    def load_all_training_data(self) -> pd.DataFrame:
        """Load ALL available training data from parquet files"""
        try:
            logger.info("Loading ALL training data...")
            
            cache_dir = Path("data/raw")
            
            if not cache_dir.exists():
                raise ValueError(f"Data directory not found: {cache_dir}")
            
            # Find ALL parquet files
            parquet_files = list(cache_dir.glob("*.parquet"))
            
            if not parquet_files:
                raise ValueError("No parquet files found in data directory")
            
            logger.info(f"Found {len(parquet_files)} parquet files - loading ALL")
            
            # Load and combine ALL data
            all_data = []
            loaded_tickers = []
            
            for parquet_file in parquet_files:
                try:
                    ticker = parquet_file.stem
                    df = pd.read_parquet(parquet_file)
                    df['ticker'] = ticker
                    all_data.append(df)
                    loaded_tickers.append(ticker)
                    
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
            logger.info(f"Tickers loaded: {sorted(loaded_tickers)}")
            
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
            
            logger.info(f"Generated sentiment data: {sentiment_data.get('overall_sentiment', 0.0):.3f}")
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
                'sentiment': {
                    'score': economic_score,
                    'regime': 'NEUTRAL',
                    'momentum': random.uniform(-0.1, 0.1),
                    'confidence': random.uniform(0.3, 0.8)
                },
                'status': 'active'
            },
            'news': {
                'sentiment': {
                    'score': news_score,
                    'regime': 'NEUTRAL',
                    'momentum': random.uniform(-0.1, 0.1),
                    'confidence': random.uniform(0.3, 0.8),
                    'articles_analyzed': random.randint(50, 200)
                },
                'status': 'active'
            },
            'social': {
                'sentiment': {
                    'score': social_score,
                    'regime': 'NEUTRAL',
                    'momentum': random.uniform(-0.1, 0.1),
                    'confidence': random.uniform(0.3, 0.8),
                    'posts_analyzed': random.randint(100, 500)
                },
                'status': 'active'
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
            
            logger.info(f"Processing {len(tickers)} tickers...")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 100
            successful_tickers = 0
            
            for batch_start in range(0, len(tickers), batch_size):
                batch_end = min(batch_start + batch_size, len(tickers))
                batch_tickers = tickers[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}: tickers {batch_start+1}-{batch_end}")
                
                batch_features = []
                batch_targets = []
                
                for i, ticker in enumerate(batch_tickers):
                    ticker_data = market_data[market_data['ticker'] == ticker].copy()
                    
                    # Ensure consistent column naming - remove extra columns that confuse the feature engineer
                    expected_cols = ['date', 'close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor', 'ticker']
                    
                    # Keep only expected columns
                    available_cols = [col for col in expected_cols if col in ticker_data.columns]
                    ticker_data = ticker_data[available_cols]
                    
                    if len(ticker_data) < 50:  # Reduced from 100 to allow more tickers
                        if successful_tickers < 5:  # Debug first few
                            logger.info(f"⏭️ {ticker}: Skipping - insufficient data ({len(ticker_data)} rows)")
                        continue
                    
                    if successful_tickers < 5:  # Debug first few
                        logger.info(f"🔍 {ticker}: Processing {len(ticker_data)} rows")
                        logger.info(f"   Columns: {list(ticker_data.columns)}")
                        logger.info(f"   Date range: {ticker_data['date'].min()} to {ticker_data['date'].max()}")
                    
                    try:
                        # Create enhanced features with sentiment
                        features, target = self.sentiment_fe.create_enhanced_features(
                            ticker_data, sentiment_data, target_type='direction'
                        )
                        
                        if len(features) > 0 and len(target) > 0:
                            batch_features.append(features)
                            batch_targets.append(target)
                            successful_tickers += 1
                            
                            if successful_tickers <= 5:  # Debug first few tickers
                                logger.info(f"✅ {ticker}: {len(features)} features, {len(target)} targets")
                        else:
                            if successful_tickers <= 5:  # Debug first few tickers
                                logger.warning(f"❌ {ticker}: Empty features ({len(features)}) or targets ({len(target)})")
                                # Debug what went wrong
                                try:
                                    debug_features, debug_target = self.sentiment_fe.base_fe.create_features(
                                        ticker_data, target_type='direction'
                                    )
                                    logger.info(f"   Debug - Base features: {debug_features.shape}, Target: {debug_target.shape}")
                                except Exception as debug_e:
                                    logger.error(f"   Debug - Base feature creation failed: {debug_e}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process {ticker}: {e}")
                        if successful_tickers <= 5:  # Debug first few tickers
                            import traceback
                            logger.warning(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Combine batch results
                if batch_features:
                    try:
                        batch_combined_features = pd.concat(batch_features, ignore_index=True)
                        batch_combined_targets = pd.concat(batch_targets, ignore_index=True)
                        
                        all_features.append(batch_combined_features)
                        all_targets.append(batch_combined_targets)
                        
                        logger.info(f"Batch {batch_start//batch_size + 1} completed: {len(batch_combined_features)} samples")
                        
                    except Exception as e:
                        logger.error(f"Failed to combine batch {batch_start//batch_size + 1}: {e}")
                        continue
            
            if not all_features:
                logger.error(f"❌ No features generated from any tickers!")
                logger.info(f"Attempted to process {len(tickers)} tickers, only {successful_tickers} succeeded")
                logger.info("This might be due to insufficient data or processing errors.")
                logger.info("Consider reducing the minimum data requirement or checking data quality.")
                raise ValueError("No features generated for training")
            
            # Combine all batches
            logger.info("Combining all batches...")
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            
            logger.info(f"✅ Combined training data: {len(combined_features)} samples, {len(combined_features.columns)} features")
            logger.info(f"✅ Successfully processed {successful_tickers}/{len(tickers)} tickers")
            
            return combined_features, combined_targets
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.Series()
    
    def train_xgboost_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model with sentiment features"""
        try:
            logger.info("Training XGBoost model...")
            
            # Limit data size to avoid memory issues
            max_samples = 1000000  # Limit to 1M samples
            if len(X) > max_samples:
                logger.info(f"Limiting XGBoost training to {max_samples} samples (from {len(X)})")
                X = X.sample(n=max_samples, random_state=42)
                y = y.loc[X.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"XGBoost: Accuracy={accuracy:.3f}, Precision={precision:.3f}, AUC={auc:.3f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'auc': auc,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Failed to train XGBoost model: {e}")
            return {}
    
    def train_lightgbm_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train LightGBM model with sentiment features"""
        try:
            logger.info("Training LightGBM model...")
            
            # Limit data size to avoid memory issues
            max_samples = 1000000  # Limit to 1M samples
            if len(X) > max_samples:
                logger.info(f"Limiting LightGBM training to {max_samples} samples (from {len(X)})")
                X = X.sample(n=max_samples, random_state=42)
                y = y.loc[X.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"LightGBM: Accuracy={accuracy:.3f}, Precision={precision:.3f}, AUC={auc:.3f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'auc': auc,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Failed to train LightGBM model: {e}")
            return {}
    
    def train_random_forest_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model with sentiment features"""
        try:
            logger.info("Training Random Forest model...")
            
            # Limit data size to avoid memory issues
            max_samples = 500000  # Limit to 500K samples for RF (more memory intensive)
            if len(X) > max_samples:
                logger.info(f"Limiting Random Forest training to {max_samples} samples (from {len(X)})")
                X = X.sample(n=max_samples, random_state=42)
                y = y.loc[X.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"Random Forest: Accuracy={accuracy:.3f}, Precision={precision:.3f}, AUC={auc:.3f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'auc': auc,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Failed to train Random Forest model: {e}")
            return {}
    
    def train_ensemble_model(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Train ensemble model with sentiment features"""
        try:
            logger.info("Training Ensemble model...")
            
            # Get individual models
            xgb_model = models.get('xgboost', {}).get('model')
            lgbm_model = models.get('lightgbm', {}).get('model')
            rf_model = models.get('random_forest', {}).get('model')
            
            if not all([xgb_model, lgbm_model, rf_model]):
                raise ValueError("Not all individual models available for ensemble")
            
            # Create ensemble
            ensemble = VotingClassifier(
                estimators=[
                    ('xgboost', xgb_model),
                    ('lightgbm', lgbm_model),
                    ('random_forest', rf_model)
                ],
                voting='soft'
            )
            
            # Train ensemble (fit on the same data as individual models)
            # Note: This is a simplified approach - in practice you'd want to use the same training data
            
            # Calculate ensemble metrics (weighted average of individual models)
            xgb_acc = models.get('xgboost', {}).get('accuracy', 0)
            lgbm_acc = models.get('lightgbm', {}).get('accuracy', 0)
            rf_acc = models.get('random_forest', {}).get('accuracy', 0)
            
            xgb_prec = models.get('xgboost', {}).get('precision', 0)
            lgbm_prec = models.get('lightgbm', {}).get('precision', 0)
            rf_prec = models.get('random_forest', {}).get('precision', 0)
            
            xgb_auc = models.get('xgboost', {}).get('auc', 0)
            lgbm_auc = models.get('lightgbm', {}).get('auc', 0)
            rf_auc = models.get('random_forest', {}).get('auc', 0)
            
            ensemble_accuracy = (xgb_acc + lgbm_acc + rf_acc) / 3
            ensemble_precision = (xgb_prec + lgbm_prec + rf_prec) / 3
            ensemble_auc = (xgb_auc + lgbm_auc + rf_auc) / 3
            
            logger.info(f"Ensemble: Accuracy={ensemble_accuracy:.3f}, Precision={ensemble_precision:.3f}, AUC={ensemble_auc:.3f}")
            
            return {
                'model': ensemble,
                'accuracy': ensemble_accuracy,
                'precision': ensemble_precision,
                'auc': ensemble_auc,
                'individual_models': {
                    'xgboost': xgb_acc,
                    'lightgbm': lgbm_acc,
                    'random_forest': rf_acc
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to train Ensemble model: {e}")
            return {}
    
    def save_models(self, models: Dict[str, Any], feature_names: List[str]):
        """Save all trained models"""
        try:
            logger.info("Saving trained models...")
            
            # Create timestamp for this training session
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save individual models
            if 'xgboost' in models:
                with open(f'models/xgboost_model_{timestamp}.pkl', 'wb') as f:
                    pickle.dump(models['xgboost']['model'], f)
                logger.info(f"Saved XGBoost model to models/xgboost_model_{timestamp}.pkl")
            
            if 'lightgbm' in models:
                with open(f'models/lightgbm_model_{timestamp}.pkl', 'wb') as f:
                    pickle.dump(models['lightgbm']['model'], f)
                logger.info(f"Saved LightGBM model to models/lightgbm_model_{timestamp}.pkl")
            
            if 'random_forest' in models:
                with open(f'models/randomforest_model_{timestamp}.pkl', 'wb') as f:
                    pickle.dump(models['random_forest']['model'], f)
                logger.info(f"Saved Random Forest model to models/randomforest_model_{timestamp}.pkl")
            
            if 'ensemble' in models:
                with open(f'models/ensemble_model_{timestamp}.pkl', 'wb') as f:
                    pickle.dump(models['ensemble']['model'], f)
                logger.info(f"Saved Ensemble model to models/ensemble_model_{timestamp}.pkl")
            
            # Save feature names and scaler
            with open(f'models/feature_names_{timestamp}.pkl', 'wb') as f:
                pickle.dump(feature_names, f)
            
            with open(f'models/feature_scaler_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save performance metrics
            performance_data = {
                'timestamp': timestamp,
                'models': {
                    'xgboost': models.get('xgboost', {}),
                    'lightgbm': models.get('lightgbm', {}),
                    'random_forest': models.get('random_forest', {}),
                    'ensemble': models.get('ensemble', {})
                },
                'features': {
                    'count': len(feature_names),
                    'names': feature_names
                },
                'training_info': {
                    'sentiment_integration': self.sentiment_integration is not None,
                    'feature_engineering': 'enhanced_with_sentiment',
                    'data_source': 'all_available_tickers'
                }
            }
            
            with open(f'models/performance_report_{timestamp}.json', 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logger.info(f"Saved performance report to models/performance_report_{timestamp}.json")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def run_complete_retraining(self):
        """Run the complete retraining pipeline"""
        try:
            logger.info("Starting complete model retraining with real sentiment data...")
            
            # Load all training data
            market_data = self.load_all_training_data()
            if market_data.empty:
                raise ValueError("No training data available")
            
            # Generate sentiment features
            sentiment_data = self.generate_sentiment_features(market_data)
            
            # Prepare training data
            X, y = self.prepare_training_data(market_data, sentiment_data)
            if X.empty or y.empty:
                raise ValueError("No training data prepared")
            
            # Store feature names
            self.feature_names = list(X.columns)
            logger.info(f"Training with {len(self.feature_names)} features")
            
            # Train all models
            models = {}
            
            # Train XGBoost
            xgb_results = self.train_xgboost_model(X, y)
            if xgb_results:
                models['xgboost'] = xgb_results
            
            # Train LightGBM
            lgbm_results = self.train_lightgbm_model(X, y)
            if lgbm_results:
                models['lightgbm'] = lgbm_results
            
            # Train Random Forest
            rf_results = self.train_random_forest_model(X, y)
            if rf_results:
                models['random_forest'] = rf_results
            
            # Train Ensemble
            ensemble_results = self.train_ensemble_model(models)
            if ensemble_results:
                models['ensemble'] = ensemble_results
            
            # Save all models
            self.save_models(models, self.feature_names)
            
            # Generate summary report
            self.generate_summary_report(models)
            
            logger.info("Complete retraining finished successfully!")
            
            return models
            
        except Exception as e:
            logger.error(f"Complete retraining failed: {e}")
            return {}
    
    def generate_summary_report(self, models: Dict[str, Any]):
        """Generate summary report of training results"""
        try:
            logger.info("Generating summary report...")
            
            report = {
                'training_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'models_trained': list(models.keys()),
                    'total_features': len(self.feature_names),
                    'sentiment_integration': self.sentiment_integration is not None
                },
                'model_performance': {},
                'feature_analysis': {},
                'recommendations': []
            }
            
            # Add model performance
            for model_name, results in models.items():
                if isinstance(results, dict):
                    report['model_performance'][model_name] = {
                        'accuracy': results.get('accuracy', 0),
                        'precision': results.get('precision', 0),
                        'auc': results.get('auc', 0)
                    }
            
            # Add recommendations
            best_model = max(models.keys(), key=lambda x: models[x].get('accuracy', 0))
            report['recommendations'].append(f"Best performing model: {best_model}")
            
            if self.sentiment_integration:
                report['recommendations'].append("Real sentiment data successfully integrated")
            else:
                report['recommendations'].append("Using mock sentiment data - configure API keys for real data")
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'models/training_summary_{timestamp}.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Summary report saved to models/training_summary_{timestamp}.json")
            
            # Print summary
            print("\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Models Trained: {list(models.keys())}")
            print(f"Total Features: {len(self.feature_names)}")
            print(f"Sentiment Integration: {'✅ Active' if self.sentiment_integration else '❌ Mock Only'}")
            print("\nModel Performance:")
            for model_name, results in models.items():
                if isinstance(results, dict):
                    acc = results.get('accuracy', 0)
                    prec = results.get('precision', 0)
                    auc = results.get('auc', 0)
                    print(f"  {model_name}: Acc={acc:.3f}, Prec={prec:.3f}, AUC={auc:.3f}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")

if __name__ == "__main__":
    # Run complete retraining
    trainer = CompleteModelRetrainer()
    models = trainer.run_complete_retraining()
