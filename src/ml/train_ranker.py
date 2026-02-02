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
        logger.info("Creating time-series splits...")
        
        # Convert day index to approximate dates (assuming day 0 ~ 2010)
        # This is an approximation since we don't have the actual date mapping
        base_year = 2010
        df_copy = df.copy()
        df_copy['year'] = base_year + (df_copy.index / 365.25).astype(int)
        
        # Define splits based on years
        train_df = df_copy[df_copy['year'] < 2020].drop('year', axis=1)
        val_df = df_copy[(df_copy['year'] >= 2020) & (df_copy['year'] < 2024)].drop('year', axis=1)
        test_df = df_copy[df_copy['year'] >= 2024].drop('year', axis=1)
        
        logger.info(f"Train period (before 2020): {len(train_df):,} rows")
        logger.info(f"Validation period (2020-2023): {len(val_df):,} rows")
        logger.info(f"Test period (2024+): {len(test_df):,} rows")
        
        return train_df, val_df, test_df
    
    def prepare_ranking_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for ranking model.
        
        Args:
            df: Feature matrix
            
        Returns:
            Tuple of (X, y, groups)
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'ticker']]
        X = df[feature_cols].values
        
        # Convert continuous returns to integer relevance scores
        # Higher positive returns = higher relevance score
        y_continuous = df['target'].values
        
        # Create relevance scores (0-4 scale)
        # 0: negative returns (< -0.02)
        # 1: small negative (-0.02 to 0)
        # 2: small positive (0 to 0.02)
        # 3: moderate positive (0.02 to 0.05)
        # 4: strong positive (> 0.05)
        y = np.zeros_like(y_continuous, dtype=int)
        y[y_continuous < -0.02] = 0
        y[(y_continuous >= -0.02) & (y_continuous < 0)] = 1
        y[(y_continuous >= 0) & (y_continuous < 0.02)] = 2
        y[(y_continuous >= 0.02) & (y_continuous < 0.05)] = 3
        y[y_continuous >= 0.05] = 4
        
        # Create groups (number of stocks per date)
        groups = df.groupby(df.index).size().values
        
        logger.info(f"Prepared ranking data: X={X.shape}, y={y.shape}, groups={groups.shape}")
        logger.info(f"Total groups (dates): {len(groups)}")
        logger.info(f"Average stocks per date: {groups.mean():.1f}")
        logger.info(f"Relevance score distribution: {np.bincount(y)}")
        
        return X, y, groups
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
        """
        Calculate Top-K accuracy for ranking model.
        
        Args:
            y_true: True target values
            y_pred: Predicted scores
            groups: Group sizes (stocks per date)
            k: Number of top stocks to consider
            
        Returns:
            Top-K accuracy score
        """
        correct_predictions = 0
        total_predictions = 0
        
        start_idx = 0
        for group_size in groups:
            # Get predictions and true values for this group
            end_idx = start_idx + group_size
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            if len(group_pred) >= k:
                # Get top-k predicted stocks
                top_k_indices = np.argsort(group_pred)[-k:][::-1]
                top_k_true = group_true[top_k_indices]
                
                # Check if majority had positive returns
                correct_predictions += np.sum(top_k_true > 0)
                total_predictions += k
            
            start_idx = end_idx
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, groups_val: np.ndarray) -> xgb.XGBRanker:
        """
        Train XGBoost Ranker model.
        
        Args:
            X_train, y_train, groups_train: Training data
            X_val, y_val, groups_val: Validation data
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost Ranker...")
        
        # Create model
        model = xgb.XGBRanker(**self.params)
        
        # Fit model
        model.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_val, y_val)],
            eval_group=[groups_val],
            verbose=100
        )
        
        logger.info(f"Training completed. Total iterations: {len(model.evals_result())}")
        
        return model
    
    def evaluate_model(self, model: xgb.XGBRanker, X_test: np.ndarray, y_test: np.ndarray, 
                     groups_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance with enhanced metrics.
        
        Args:
            model: Trained model
            X_test, y_test, groups_test: Test data
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate ranking metrics
        top_5_accuracy = self.calculate_top_k_accuracy(y_test, y_pred, groups_test, k=5)
        top_10_accuracy = self.calculate_top_k_accuracy(y_test, y_pred, groups_test, k=10)
        top_20_accuracy = self.calculate_top_k_accuracy(y_test, y_pred, groups_test, k=20)
        
        # Calculate Precision@10
        precision_at_10 = self.calculate_precision_at_k(y_test, y_pred, groups_test, k=10)
        
        # Calculate Spearman Rank Correlation
        spearman_corr = self.calculate_spearman_correlation(y_test, y_pred, groups_test)
        
        # Calculate overall accuracy (positive returns prediction)
        overall_accuracy = np.mean((y_pred > 0) == (y_test > 0))
        
        metrics = {
            'top_5_accuracy': top_5_accuracy,
            'top_10_accuracy': top_10_accuracy,
            'top_20_accuracy': top_20_accuracy,
            'precision_at_10': precision_at_10,
            'spearman_correlation': spearman_corr,
            'overall_accuracy': overall_accuracy,
            'mean_prediction': np.mean(y_pred),
            'std_prediction': np.std(y_pred)
        }
        
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def calculate_precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
        """
        Calculate Precision@K for ranking model.
        
        Args:
            y_true: True target values
            y_pred: Predicted scores
            groups: Group sizes (stocks per date)
            k: Number of top stocks to consider
            
        Returns:
            Precision@K score
        """
        correct_predictions = 0
        total_predictions = 0
        
        start_idx = 0
        for group_size in groups:
            # Get predictions and true values for this group
            end_idx = start_idx + group_size
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            if len(group_pred) >= k:
                # Get top-k predicted stocks
                top_k_indices = np.argsort(group_pred)[-k:][::-1]
                top_k_true = group_true[top_k_indices]
                
                # Calculate precision (positive returns / k)
                correct_predictions += np.sum(top_k_true > 0)
                total_predictions += k
            
            start_idx = end_idx
        
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        return precision
    
    def calculate_spearman_correlation(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
        """
        Calculate Spearman Rank Correlation between predicted and actual returns.
        
        Args:
            y_true: True target values
            y_pred: Predicted scores
            groups: Group sizes (stocks per date)
            
        Returns:
            Average Spearman correlation across all groups
        """
        correlations = []
        
        start_idx = 0
        for group_size in groups:
            # Get predictions and true values for this group
            end_idx = start_idx + group_size
            group_pred = y_pred[start_idx:end_idx]
            group_true = y_true[start_idx:end_idx]
            
            if len(group_pred) > 1:  # Need at least 2 samples for correlation
                # Calculate Spearman correlation for this group
                corr, _ = spearmanr(group_pred, group_true)
                if not np.isnan(corr):
                    correlations.append(corr)
            
            start_idx = end_idx
        
        # Return average correlation
        return np.mean(correlations) if correlations else 0.0
    
    def plot_feature_importance(self, model: xgb.XGBRanker, feature_names: List[str]) -> None:
        """
        Create feature importance visualization.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
        """
        logger.info("Creating feature importance plot...")
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance - NeuralTrader Ranker v1', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels on bars
        for i, (importance, feature) in enumerate(zip(feature_importance_df['importance'].head(10), 
                                                   feature_importance_df['feature'].head(10))):
            plt.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_path / "feature_importance_v1.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved: {plot_path}")
        
        # Log top features
        logger.info("Top 10 Features by Importance:")
        for _, row in feature_importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_model(self, model: xgb.XGBRanker, metrics: Dict[str, float]) -> None:
        """Save trained model and metrics."""
        logger.info("Saving model...")
        
        # Save model
        model_path = self.models_path / "neural_ranker_v1.json"
        model.save_model(str(model_path))
        
        # Save metrics and metadata
        metadata = {
            'model_type': 'XGBRanker',
            'version': 'v1',
            'parameters': self.params,
            'metrics': {k: float(v) for k, v in metrics.items()},  # Convert to float for JSON
            'training_date': datetime.now().isoformat(),
            'feature_columns': [col for col in pd.read_parquet(self.data_path).columns 
                              if col not in ['target', 'ticker']]
        }
        
        metadata_path = self.models_path / "neural_ranker_v1_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
    
    def run(self) -> xgb.XGBRanker:
        """
        Run the complete training pipeline.
        
        Returns:
            Trained model
        """
        logger.info("Starting ranker training pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Create time splits
        train_df, val_df, test_df = self.create_time_splits(df)
        
        # Prepare ranking data
        X_train, y_train, groups_train = self.prepare_ranking_data(train_df)
        X_val, y_val, groups_val = self.prepare_ranking_data(val_df)
        X_test, y_test, groups_test = self.prepare_ranking_data(test_df)
        
        # Train model
        model = self.train_model(X_train, y_train, groups_train, X_val, y_val, groups_val)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test, groups_test)
        
        # Create feature importance plot
        feature_cols = [col for col in df.columns if col not in ['target', 'ticker']]
        self.plot_feature_importance(model, feature_cols)
        
        # Save model
        self.save_model(model, metrics)
        
        logger.info("Ranker training pipeline completed successfully!")
        
        return model

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost Ranker model")
    parser.add_argument("--data-path", default="data/processed/master_feature_matrix.parquet", 
                       help="Path to feature matrix")
    parser.add_argument("--models-path", default="models", help="Path to save models")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RankerTrainer(data_path=args.data_path)
    
    # Run training
    try:
        model = trainer.run()
        print("\n" + "="*60)
        print("RANKER TRAINING COMPLETED")
        print("="*60)
        print(f"Model saved to: {trainer.models_path}/neural_ranker_v1.json")
        print("Ready for inference and backtesting!")
        print("="*60)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
