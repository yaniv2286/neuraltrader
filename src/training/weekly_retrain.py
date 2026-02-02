"""
NeuralTrader Weekly Retrain Script
==================================

Performs Walk-Forward retraining of the XGBoost model every Saturday at 10:00 EST.
Updates the model with the latest historical data including current week's results.

Features:
- Walk-Forward retraining with full historical dataset
- Model validation and performance tracking
- Automatic model backup and versioning
- Retraining logs and performance metrics
- Integration with existing model cache system

Usage:
    python src/training/weekly_retrain.py
    
    # Or for manual retraining
    python src/training/weekly_retrain.py --force
"""

import os
import sys
import logging
import argparse
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pytz

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.model_cache import ModelCache
from src.data.data_store import DataStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weekly_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyRetrainer:
    """
    Weekly Model Retrainer for NeuralTrader
    Performs Walk-Forward retraining with full historical data
    """
    
    def __init__(self):
        """Initialize the weekly retrainer"""
        self.eastern = pytz.timezone('US/Eastern')
        self.israel = pytz.timezone('Asia/Jerusalem')
        
        # Model and data paths
        self.models_dir = 'models'
        self.cache_dir = 'models/cache'
        self.data_dir = 'src/data/cache/tiingo'
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize components
        self.model_cache = ModelCache()
        self.data_store = DataStore()
        
        # S&P 100 universe
        self.sp100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B', 'UNH',
            'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'CVX', 'HD', 'ABBV', 'MRK', 'LLY', 'PEP', 'KO',
            'COST', 'TMO', 'CSCO', 'PFE', 'MCD', 'CRM', 'BAC', 'ADBE', 'WMT', 'CMCSA', 'DIS', 'NFLX',
            'ABT', 'VZ', 'ORCL', 'TXN', 'AMD', 'LIN', 'PM', 'UPS', 'NKE', 'HON', 'UNP', 'RTX', 'INTU',
            'LOW', 'SPGI', 'MS', 'QCOM', 'COP', 'IBM', 'GE', 'AMAT', 'CAT', 'GS', 'ISRG', 'DE', 'BKNG',
            'ELV', 'PLD', 'SBUX', 'MDT', 'BLK', 'GILD', 'TJX', 'NOW', 'ADP', 'C', 'MMC', 'AMT', 'REGN',
            'MO', 'PYPL', 'CB', 'CI', 'ADI', 'MDLZ', 'VRTX', 'ZTS', 'SYK', 'CME', 'AMGN', 'FISV', 'SLB',
            'T', 'LMT', 'MU', 'CVS', 'DUK', 'ITW', 'EQIX', 'ANTM', 'CL', 'ICE', 'SHERW'
        ]
        
        logger.info("Weekly Retrainer initialized")
        logger.info(f"S&P 100 Universe: {len(self.sp100_tickers)} tickers")
    
    def check_retraining_schedule(self) -> bool:
        """
        Check if it's time for weekly retraining (Saturday 10:00 EST)
        
        Returns:
            True if it's time to retrain, False otherwise
        """
        try:
            now_eastern = datetime.now(self.eastern)
            
            # Check if it's Saturday
            if now_eastern.weekday() != 5:  # 5 = Saturday
                return False
            
            # Check if it's 10:00 AM ± 1 hour
            target_hour = 10
            current_hour = now_eastern.hour
            
            if abs(current_hour - target_hour) <= 1:
                logger.info(f"Within retraining window: {now_eastern.strftime('%Y-%m-%d %H:%M')} EST")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retraining schedule: {e}")
            return False
    
    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load full historical dataset for retraining
        
        Returns:
            Dictionary of ticker -> DataFrame
        """
        try:
            logger.info("Loading historical data for retraining...")
            
            data = {}
            loaded_count = 0
            
            for ticker in self.sp100_tickers:
                try:
                    # Load from cache
                    cache_file = os.path.join(self.data_dir, f"{ticker.lower()}.parquet")
                    
                    if os.path.exists(cache_file):
                        df = pd.read_parquet(cache_file)
                        
                        if not df.empty:
                            data[ticker] = df
                            loaded_count += 1
                            
                            if loaded_count % 20 == 0:
                                logger.info(f"Loaded {loaded_count} tickers...")
                    else:
                        logger.warning(f"No data file found for {ticker}")
                
                except Exception as e:
                    logger.error(f"Error loading {ticker}: {e}")
                    continue
            
            logger.info(f"Loaded historical data: {loaded_count}/{len(self.sp100_tickers)} tickers")
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return {}
    
    def prepare_features_and_targets(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and targets for training
        
        Args:
            data: Dictionary of ticker -> DataFrame
            
        Returns:
            Tuple of (features, targets)
        """
        try:
            logger.info("Preparing features and targets...")
            
            all_features = []
            all_targets = []
            
            for ticker, df in data.items():
                if df.empty or len(df) < 100:
                    continue
                
                # Calculate features (simplified version)
                df_features = self._calculate_features(df)
                
                # Calculate targets (next week return)
                df_targets = self._calculate_targets(df)
                
                # Combine features and targets
                combined = pd.concat([df_features, df_targets], axis=1)
                combined = combined.dropna()
                
                if not combined.empty:
                    features = combined.drop('target', axis=1)
                    targets = combined['target']
                    
                    all_features.append(features)
                    all_targets.append(targets)
            
            if not all_features:
                raise ValueError("No valid data for training")
            
            # Combine all tickers
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_targets, ignore_index=True)
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features and targets: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for training"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price features
            features['returns_1d'] = df['close'].pct_change(1)
            features['returns_5d'] = df['close'].pct_change(5)
            features['returns_20d'] = df['close'].pct_change(20)
            
            # Moving averages
            features['sma_10'] = df['close'].rolling(10).mean()
            features['sma_20'] = df['close'].rolling(20).mean()
            features['sma_50'] = df['close'].rolling(50).mean()
            
            # Price relative to moving averages
            features['price_vs_sma10'] = (df['close'] - features['sma_10']) / features['sma_10']
            features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50']
            
            # Volatility
            features['volatility_10'] = features['returns_1d'].rolling(10).std()
            features['volatility_20'] = features['returns_1d'].rolling(20).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Volume features
            features['volume_sma_10'] = df['volume'].rolling(10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_10']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr_14'] = true_range.rolling(14).mean()
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return pd.DataFrame()
    
    def _calculate_targets(self, df: pd.DataFrame) -> pd.Series:
        """Calculate targets for training (next week return)"""
        try:
            # Calculate next week's return (5 trading days)
            targets = df['close'].pct_change(5).shift(-5)
            targets.name = 'target'
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating targets: {e}")
            return pd.Series()
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """
        Train XGBoost model with Walk-Forward validation
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Trained XGBoost model
        """
        try:
            logger.info("Training XGBoost model...")
            
            # Walk-Forward validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            best_score = -np.inf
            best_model = None
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                
                logger.info(f"Fold {fold + 1}: R² = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            logger.info(f"Best validation R²: {best_score:.4f}")
            
            # Retrain on full dataset
            final_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            final_model.fit(X, y)
            
            return final_model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def save_model(self, model: xgb.XGBRegressor, version: str) -> str:
        """
        Save trained model with versioning
        
        Args:
            model: Trained model
            version: Model version
            
        Returns:
            Path to saved model
        """
        try:
            # Create versioned filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"xgboost_model_v{version}_{timestamp}.pkl"
            filepath = os.path.join(self.models_dir, filename)
            
            # Save model
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved: {filepath}")
            
            # Update latest symlink
            latest_path = os.path.join(self.models_dir, 'latest_model.pkl')
            if os.path.exists(latest_path):
                os.remove(latest_path)
            
            os.symlink(filepath, latest_path)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def evaluate_model(self, model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X: Features
            y: Targets
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Predictions
            y_pred = model.predict(X)
            
            # Metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Direction accuracy
            y_direction = np.sign(y)
            y_pred_direction = np.sign(y_pred)
            direction_accuracy = np.mean(y_direction == y_pred_direction)
            
            metrics = {
                'mse': mse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'samples': len(y),
                'features': X.shape[1]
            }
            
            logger.info(f"Model Performance:")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def run_retraining(self, force: bool = False) -> bool:
        """
        Run the weekly retraining process
        
        Args:
            force: Force retraining regardless of schedule
            
        Returns:
            True if retraining successful, False otherwise
        """
        try:
            logger.info("Starting weekly retraining...")
            
            # Check schedule (unless forced)
            if not force and not self.check_retraining_schedule():
                logger.info("Not within retraining schedule")
                return False
            
            # Load historical data
            data = self.load_historical_data()
            if not data:
                logger.error("No data available for retraining")
                return False
            
            # Prepare features and targets
            X, y = self.prepare_features_and_targets(data)
            if X.empty:
                logger.error("No valid training data")
                return False
            
            # Train model
            model = self.train_model(X, y)
            if model is None:
                logger.error("Model training failed")
                return False
            
            # Evaluate model
            metrics = self.evaluate_model(model, X, y)
            
            # Save model
            model_path = self.save_model(model, "6.1")
            if not model_path:
                logger.error("Failed to save model")
                return False
            
            # Log completion
            logger.info("Weekly retraining completed successfully")
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Performance: R²={metrics.get('r2', 0):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in weekly retraining: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NeuralTrader Weekly Retrainer')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of schedule')
    args = parser.parse_args()
    
    try:
        retrainer = WeeklyRetrainer()
        
        success = retrainer.run_retraining(force=args.force)
        
        if success:
            logger.info("🎉 Weekly retraining completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Weekly retraining failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Weekly retraining interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
