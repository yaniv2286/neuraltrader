#!/usr/bin/env python3
"""
NeuralTrader 2.0 - XGBoost Ranker Training
==========================================

Train XGBoost Ranker model for 25% ARR mission using walk-forward validation.

Features:
- Time-series split with walk-forward validation
- XGBRanker with rank:ndcg objective
- Date-based grouping for stock ranking
- Top-10 accuracy evaluation
- Model persistence
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score
from scipy.stats import spearmanr
import json
import joblib
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralTrader.RankerTrainer")

class RankerTrainer:
    """
    XGBoost Ranker trainer with walk-forward validation.
    """
    
    def __init__(self, data_path: str = "data/processed/master_feature_matrix.parquet"):
        self.data_path = Path(data_path)
        self.models_path = Path("models")
        self.reports_path = Path("reports")
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.params = {
            'objective': 'rank:ndcg',
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'ndcg'
        }
        
        logger.info("RankerTrainer initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Models path: {self.models_path}")
        logger.info(f"Reports path: {self.reports_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the feature matrix."""
        logger.info("Loading feature matrix...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Feature matrix not found: {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded data: {df.shape}")
        
        # Convert index to numeric if it's not already
        if not isinstance(df.index, (pd.RangeIndex)):
            # If it's a DatetimeIndex, convert to numeric days
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = (df.index - df.index.min()).days
            else:
                # Try to convert to numeric
                df.index = pd.to_numeric(df.index, errors='coerce')
        
        # Remove any remaining NaN values
        df = df.dropna()
        logger.info(f"After NaN removal: {df.shape}")
        
        return df
    
    def create_time_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits based on specific years.
        
        Args:
            df: Feature matrix with numeric index (days since start)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating time-based splits...")
        
        # Convert numeric index back to datetime for filtering
        if isinstance(df.index, pd.RangeIndex):
            # Assume the index represents days from some start date
            # Use 2004-01-01 as reference start
            start_date = pd.to_datetime('2004-01-01')
            df.index = start_date + pd.to_timedelta(df.index, unit='D')
        
        # Define split years
        train_end = pd.to_datetime('2020-12-31')
        val_end = pd.to_datetime('2022-12-31')
        
        # Create splits
        train_df = df[df.index <= train_end]
        val_df = df[(df.index > train_end) & (df.index <= val_end)]
        test_df = df[df.index > val_end]
        
        logger.info(f"Train set: {train_df.shape} ({train_df.index.min().date()} to {train_df.index.max().date()})")
        logger.info(f"Validation set: {val_df.shape} ({val_df.index.min().date()} to {val_df.index.max().date()})")
        logger.info(f"Test set: {test_df.shape} ({test_df.index.min().date()} to {test_df.index.max().date()})")
        
        return train_df, val_df, test_df
    
    def prepare_ranker_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare data for XGBoost Ranker training.
        
        Args:
            df: Feature matrix with datetime index
        
        Returns:
            Tuple of (features, groups, targets)
        """
        logger.info("Preparing ranker data...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'ticker', 'date']]
        X = df[feature_cols]
        y = df['target']
        
        # Create groups for ranking (group by date)
        if 'date' in df.columns:
            groups = df.groupby('date').size().values
        else:
            # If no explicit date column, use index grouping
            # Group by year-month
            df['year_month'] = df.index.to_period('M')
            groups = df.groupby('year_month').size().values
        
        logger.info(f"Features: {X.shape}, Groups: {len(groups)}, Target range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, groups, y
    
    def train_ranker(self, X_train: pd.DataFrame, groups_train: np.ndarray, y_train: np.ndarray,
                    X_val: pd.DataFrame, groups_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRanker:
        """
        Train XGBoost Ranker with validation.
        
        Args:
            X_train, groups_train, y_train: Training data
            X_val, groups_val, y_val: Validation data
        
        Returns:
            Trained XGBoost Ranker model
        """
        logger.info("Training XGBoost Ranker...")
        
        # Create ranker
        ranker = xgb.XGBRanker(**self.params)
        
        # Train with early stopping
        ranker.fit(
            X_train, y_train, groups_train,
            eval_set=[(X_val, y_val)],
            eval_group=[groups_val],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Get best score
        best_score = ranker.best_score
        best_iteration = ranker.best_iteration
        
        logger.info(f"Best validation NDCG: {best_score:.4f} at iteration {best_iteration}")
        
        return ranker
    
    def evaluate_ranker(self, ranker: xgb.XGBRanker, X_test: pd.DataFrame, groups_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate trained ranker on test set.
        
        Args:
            ranker: Trained XGBoost Ranker
            X_test, groups_test, y_test: Test data
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating ranker...")
        
        # Make predictions
        y_pred = ranker.predict(X_test)
        
        # Calculate metrics
        # Spearman correlation (ranking quality)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        
        # Top-K accuracy (how often top-ranked stocks are actually good)
        top_k_accuracies = {}
        for k in [5, 10, 20]:
            top_k_acc = self.calculate_top_k_accuracy(y_test, y_pred, groups_test, k)
            top_k_accuracies[f'top_{k}_accuracy'] = top_k_acc
        
        # Overall accuracy (directional)
        direction_acc = accuracy_score(y_test > 0, y_pred > 0)
        
        results = {
            'spearman_correlation': spearman_corr,
            'directional_accuracy': direction_acc,
            **top_k_accuracies
        }
        
        logger.info("Evaluation results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int) -> float:
        """
        Calculate top-K accuracy for ranking.
        
        Args:
            y_true: True target values
            y_pred: Predicted scores
            groups: Group indices for ranking
            k: Top-K to consider
        
        Returns:
            Top-K accuracy
        """
        correct = 0
        total = 0
        
        # Get unique group boundaries
        group_boundaries = []
        start_idx = 0
        for group_size in groups:
            end_idx = start_idx + group_size
            group_boundaries.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Calculate top-K accuracy for each group
        for start_idx, end_idx in group_boundaries:
            if end_idx - start_idx < k:
                continue  # Skip groups smaller than k
            
            # Get true and predicted for this group
            y_true_group = y_true[start_idx:end_idx]
            y_pred_group = y_pred[start_idx:end_idx]
            
            # Get top-K indices by prediction
            top_k_pred_idx = np.argsort(y_pred_group)[-k:]
            
            # Check if any of top-K predictions are actually in top-K true values
            top_k_true_idx = np.argsort(y_true_group)[-k:]
            
            # Calculate accuracy
            intersection = len(set(top_k_pred_idx) & set(top_k_true_idx))
            accuracy = intersection / k
            
            correct += intersection
            total += k
        
        return correct / total if total > 0 else 0.0
    
    def save_model(self, ranker: xgb.XGBRanker, results: Dict):
        """Save trained model and results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = self.models_path / f"neural_ranker_v1_{timestamp}.json"
        ranker.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save results
        results_path = self.reports_path / f"ranker_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
        
        # Save feature importance
        if hasattr(ranker, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': ranker.get_booster().feature_names,
                'importance': ranker.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = self.reports_path / f"feature_importance_{timestamp}.csv"
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
    
    def run_training(self):
        """Run complete ranker training pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING XGBOOST RANKER TRAINING")
        logger.info("=" * 50)
        
        try:
            # Load data
            df = self.load_data()
            
            # Create time splits
            train_df, val_df, test_df = self.create_time_splits(df)
            
            # Prepare ranker data
            X_train, groups_train, y_train = self.prepare_ranker_data(train_df)
            X_val, groups_val, y_val = self.prepare_ranker_data(val_df)
            X_test, groups_test, y_test = self.prepare_ranker_data(test_df)
            
            # Train ranker
            ranker = self.train_ranker(X_train, groups_train, y_train, X_val, groups_val, y_val)
            
            # Evaluate ranker
            results = self.evaluate_ranker(ranker, X_test, groups_test, y_test)
            
            # Save model and results
            self.save_model(ranker, results)
            
            logger.info("=" * 50)
            logger.info("RANKER TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return ranker, results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

if __name__ == "__main__":
    trainer = RankerTrainer()
    ranker, results = trainer.run_training()
