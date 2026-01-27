"""
Model Trainer - Optimized for Stock Prediction
Handles log returns, feature selection, and proper train/val/test splits
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Optimized model trainer for stock prediction
    - Uses log returns instead of raw prices
    - Implements feature selection
    - Proper time-series splits
    - Regularization
    """
    
    def __init__(self, use_log_returns: bool = True, n_features: int = 25):
        """
        Initialize trainer
        
        Args:
            use_log_returns: Use log returns instead of raw prices (recommended)
            n_features: Number of top features to select (default 25)
        """
        self.use_log_returns = use_log_returns
        self.n_features = n_features
        self.selected_features = None
        self.feature_importances = None
        
    def prepare_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            feature_cols: List of feature columns (if None, auto-detect)
            
        Returns:
            Tuple of (X, y)
        """
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            # Exclude target and non-numeric columns
            exclude_cols = [target_col, 'low', 'volume', 'market_regime']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols].copy()
        
        # Prepare target
        if self.use_log_returns:
            # Calculate log returns
            prices = df[target_col]
            y = np.log(prices / prices.shift(1))
            
            # Remove first row (NaN from shift)
            X = X.iloc[1:]
            y = y.iloc[1:]
            
            print(f"   📊 Using LOG RETURNS as target (stationary)")
        else:
            y = df[target_col]
            print(f"   📊 Using RAW PRICES as target (non-stationary)")
        
        # Fill NaN values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        return X, y
    
    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'correlation'
    ) -> List[str]:
        """
        Select top N most important features
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Selection method ('correlation', 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        if method == 'correlation':
            # Calculate correlation with target
            correlations = X_train.corrwith(y_train).abs()
            top_features = correlations.nlargest(self.n_features).index.tolist()
            
            self.feature_importances = correlations.sort_values(ascending=False)
            
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
            mi_series = pd.Series(mi_scores, index=X_train.columns)
            top_features = mi_series.nlargest(self.n_features).index.tolist()
            
            self.feature_importances = mi_series.sort_values(ascending=False)
        
        self.selected_features = top_features
        
        print(f"   🔧 Selected top {len(top_features)} features using {method}")
        print(f"   📊 Top 10 features:")
        for i, feat in enumerate(top_features[:10], 1):
            importance = self.feature_importances[feat]
            print(f"      {i:2}. {feat:25} : {importance:.4f}")
        
        return top_features
    
    def create_time_series_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple:
        """
        Create time-series aware train/val/test splits
        
        Args:
            X: Features
            y: Target
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Time-series split (no shuffling)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        print(f"\n   📊 Data splits:")
        print(f"      Train: {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
        print(f"      Val:   {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
        print(f"      Test:  {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # R² scores
        results['train_r2'] = r2_score(y_train, y_train_pred)
        results['val_r2'] = r2_score(y_val, y_val_pred)
        results['test_r2'] = r2_score(y_test, y_test_pred)
        
        # RMSE
        results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        results['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # MAE
        results['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        results['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        results['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        
        # Directional accuracy
        results['train_dir'] = np.mean((y_train > 0) == (y_train_pred > 0)) * 100
        results['val_dir'] = np.mean((y_val > 0) == (y_val_pred > 0)) * 100
        results['test_dir'] = np.mean((y_test > 0) == (y_test_pred > 0)) * 100
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results in a nice format"""
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        print(f"\n{'Metric':<20} {'Train':>12} {'Val':>12} {'Test':>12}")
        print("-" * 70)
        
        print(f"{'R² Score':<20} {results['train_r2']:>12.4f} {results['val_r2']:>12.4f} {results['test_r2']:>12.4f}")
        
        if self.use_log_returns:
            print(f"{'RMSE (log ret)':<20} {results['train_rmse']:>12.6f} {results['val_rmse']:>12.6f} {results['test_rmse']:>12.6f}")
            print(f"{'MAE (log ret)':<20} {results['train_mae']:>12.6f} {results['val_mae']:>12.6f} {results['test_mae']:>12.6f}")
        else:
            print(f"{'RMSE ($)':<20} ${results['train_rmse']:>11.2f} ${results['val_rmse']:>11.2f} ${results['test_rmse']:>11.2f}")
            print(f"{'MAE ($)':<20} ${results['train_mae']:>11.2f} ${results['val_mae']:>11.2f} ${results['test_mae']:>11.2f}")
        
        print(f"{'Direction Acc':<20} {results['train_dir']:>11.2f}% {results['val_dir']:>11.2f}% {results['test_dir']:>11.2f}%")
        
        # Diagnosis
        print("\n" + "="*70)
        print("DIAGNOSIS")
        print("="*70)
        
        issues = []
        
        if results['train_r2'] > 0.95:
            issues.append("⚠️ High train R² - possible overfitting")
        
        if results['test_r2'] < 0:
            issues.append("🔴 CRITICAL: Test R² < 0 (worse than baseline)")
        elif results['test_r2'] < 0.1:
            issues.append("⚠️ Low test R² - poor generalization")
        elif results['test_r2'] > 0.3:
            issues.append("✅ Good test R² - model generalizes well")
        
        if results['test_dir'] < 52:
            issues.append("🔴 CRITICAL: Direction accuracy < 52% (worse than random)")
        elif results['test_dir'] < 55:
            issues.append("⚠️ Low direction accuracy - barely better than random")
        elif results['test_dir'] > 55:
            issues.append("✅ Good direction accuracy - model predicts trends")
        
        gap = results['train_r2'] - results['test_r2']
        if gap > 0.5:
            issues.append(f"🔴 SEVERE overfitting gap: {gap:.2f}")
        elif gap > 0.3:
            issues.append(f"⚠️ Overfitting gap: {gap:.2f}")
        else:
            issues.append(f"✅ Good generalization gap: {gap:.2f}")
        
        for issue in issues:
            print(f"   {issue}")
        
        return results
