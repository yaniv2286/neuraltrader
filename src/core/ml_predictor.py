"""
ML Predictor - Generates predictions using trained models.
Wires XGBoost (and other models) into the BacktestEngine.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.features.technical_indicators import generate_all_features, get_feature_columns
from src.models.cpu_models.xgboost_model import XGBoostModel
from src.core.data_store import get_data_store


class MLPredictor:
    """
    Generates ML predictions for all tickers.
    Uses XGBoost as primary model with comprehensive technical indicators.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.models: Dict[str, XGBoostModel] = {}
        self.feature_columns: List[str] = []
        self.data_store = get_data_store()
        
    def train_models(
        self,
        tickers: List[str],
        start_date: str = '2010-01-01',
        end_date: str = '2024-12-31',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict[str, dict]:
        """
        Train models for all tickers.
        
        Returns:
            Dictionary of ticker -> training metrics
        """
        results = {}
        
        print(f"🤖 Training {self.model_type.upper()} models for {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers):
            try:
                metrics = self._train_single_ticker(ticker, start_date, end_date, train_ratio, val_ratio)
                if metrics:
                    results[ticker] = metrics
                    status = "✅" if metrics['test_dir'] > 52 else "⚠️"
                    print(f"   {status} {ticker}: Dir={metrics['test_dir']:.1f}%, R²={metrics['test_r2']:.4f}")
            except Exception as e:
                print(f"   ❌ {ticker}: {str(e)[:50]}")
        
        print(f"\n📊 Trained {len(results)}/{len(tickers)} models successfully")
        return results
    
    def _train_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        train_ratio: float,
        val_ratio: float
    ) -> Optional[dict]:
        """Train model for a single ticker."""
        # Load data
        df = self.data_store.get_ticker_data(ticker, start_date, end_date)
        if df is None or len(df) < 500:
            return None
        
        # Generate features
        df = generate_all_features(df)
        if len(df) < 300:
            return None
        
        # Create target: next day's LOG return (better for prediction)
        df['future_return'] = np.log(df['close'].shift(-1) / df['close'])
        df['target'] = df['future_return']  # Predict actual return
        df = df.dropna()
        
        if len(df) < 200:
            return None
        
        # Get feature columns
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return']
        self.feature_columns = [c for c in df.columns if c not in exclude]
        
        # Split data - use TIME-BASED split (no shuffling for time series!)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Prepare X, y
        X_train = train_df[self.feature_columns].values
        y_train = train_df['target'].values
        X_val = val_df[self.feature_columns].values
        y_val = val_df['target'].values
        X_test = test_df[self.feature_columns].values
        y_test = test_df['target'].values
        
        # Train model with early stopping
        model = XGBoostModel()
        
        # Scale validation set for early stopping
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model.fit(X_train, y_train)
        
        # Store model
        self.models[ticker] = model
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate direction accuracy (what matters for trading)
        y_train_dir = np.sign(y_train)
        y_val_dir = np.sign(y_val)
        y_test_dir = np.sign(y_test)
        
        train_dir = np.mean(np.sign(train_pred) == y_train_dir) * 100
        val_dir = np.mean(np.sign(val_pred) == y_val_dir) * 100
        test_dir = np.mean(np.sign(test_pred) == y_test_dir) * 100
        
        # R² (for regression interpretation)
        from sklearn.metrics import r2_score
        test_r2 = r2_score(y_test, test_pred)
        
        return {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'test_dir': test_dir,
            'test_r2': test_r2,
            'samples': len(df),
            'features': len(self.feature_columns)
        }
    
    def generate_predictions(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions for all tickers over date range.
        
        Returns:
            Dictionary of ticker -> DataFrame with 'prediction' column
        """
        predictions = {}
        
        print(f"📈 Generating predictions for {len(tickers)} tickers...")
        
        for ticker in tickers:
            if ticker not in self.models:
                continue
            
            try:
                pred_df = self._predict_single_ticker(ticker, start_date, end_date)
                if pred_df is not None and len(pred_df) > 0:
                    predictions[ticker] = pred_df
            except Exception as e:
                pass  # Skip failed predictions
        
        print(f"   ✅ Generated predictions for {len(predictions)} tickers")
        return predictions
    
    def _predict_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Generate predictions for a single ticker."""
        model = self.models.get(ticker)
        if model is None:
            return None
        
        # Load data
        df = self.data_store.get_ticker_data(ticker, start_date, end_date)
        if df is None or len(df) < 50:
            return None
        
        # Generate features
        df = generate_all_features(df)
        if len(df) < 20:
            return None
        
        # Get features
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        # Predict
        X = df[feature_cols].values
        predictions = model.predict(X)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'prediction': predictions,
            'confidence': np.abs(predictions)  # Use absolute value as confidence proxy
        }, index=df.index)
        
        return result
    
    def get_model_summary(self) -> dict:
        """Get summary of trained models."""
        return {
            'total_models': len(self.models),
            'model_type': self.model_type,
            'feature_count': len(self.feature_columns),
            'tickers': list(self.models.keys())
        }


def train_and_predict(
    tickers: List[str],
    train_start: str = '2010-01-01',
    train_end: str = '2023-12-31',
    predict_start: str = '2020-01-01',
    predict_end: str = '2024-12-31'
) -> Tuple[MLPredictor, Dict[str, pd.DataFrame]]:
    """
    Convenience function to train models and generate predictions.
    
    Args:
        tickers: List of tickers to process
        train_start: Training data start date
        train_end: Training data end date
        predict_start: Prediction start date
        predict_end: Prediction end date
    
    Returns:
        (predictor, predictions) tuple
    """
    predictor = MLPredictor()
    
    # Train models
    training_results = predictor.train_models(tickers, train_start, train_end)
    
    # Generate predictions
    predictions = predictor.generate_predictions(tickers, predict_start, predict_end)
    
    return predictor, predictions


if __name__ == "__main__":
    # Test the predictor
    from src.core.data_store import get_data_store
    
    store = get_data_store()
    tickers = store.available_tickers[:10]  # Test with 10 tickers
    
    predictor, predictions = train_and_predict(
        tickers=tickers,
        train_start='2015-01-01',
        train_end='2023-12-31',
        predict_start='2020-01-01',
        predict_end='2024-12-31'
    )
    
    print(f"\n📊 Summary:")
    print(f"   Models trained: {len(predictor.models)}")
    print(f"   Predictions generated: {len(predictions)}")
    print(f"   Features used: {len(predictor.feature_columns)}")
