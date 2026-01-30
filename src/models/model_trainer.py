"""
Model Trainer - Enhanced for Foundation Strengthening
Handles ensemble models, regime-aware training, and advanced evaluation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .ensemble_model import EnsembleModel, create_ensemble_for_regime

class ModelTrainer:
    """
    Enhanced model trainer for stock prediction
    - Supports ensemble models
    - Regime-aware training
    - Advanced evaluation metrics
    - Cross-validation
    """
    
    def __init__(self, use_ensemble: bool = False, use_regime_aware: bool = False):
        """
        Initialize trainer
        
        Args:
            use_ensemble: Use ensemble models
            use_regime_aware: Use regime-aware training
        """
        self.use_ensemble = use_ensemble
        self.use_regime_aware = use_regime_aware
        self.selected_features = None
        self.feature_importances = None
        self.regime_models = {}
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """
        Evaluate model on train, validation, and test sets
        
        Args:
            model: Trained model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Calculate direction accuracy
        train_dir = np.mean(np.sign(train_pred) == np.sign(y_train)) * 100
        val_dir = np.mean(np.sign(val_pred) == np.sign(y_val)) * 100
        test_dir = np.mean(np.sign(test_pred) == np.sign(y_test)) * 100
        
        # Generalization gap
        gen_gap = abs(train_r2 - test_r2)
        
        return {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_dir': train_dir,
            'val_dir': val_dir,
            'test_dir': test_dir,
            'gen_gap': gen_gap
        }
        
    def prepare_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'target',
        feature_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training with regime detection
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns (if None, auto-detect)
            
        Returns:
            Tuple of (X, y)
        """
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = [target_col, 'date', 'ticker']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove non-numeric columns
        numeric_features = []
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                numeric_features.append(col)
        
        X = df[numeric_features].copy()
        y = df[target_col].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Regime string
        """
        if 'high_vol' in df.columns and 'trend_regime' in df.columns:
            # Use regime features if available
            latest = df.iloc[-1]
            
            if latest['high_vol']:
                return 'high_volatility'
            elif latest['low_vol']:
                return 'low_volatility'
            elif latest['trend_regime'] == 1:
                if 'strong_uptrend' in df.columns and latest['strong_uptrend']:
                    return 'uptrend'
                else:
                    return 'sideways'
            else:
                if 'strong_downtrend' in df.columns and latest['strong_downtrend']:
                    return 'downtrend'
                else:
                    return 'sideways'
        else:
            # Fallback to simple volatility detection
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            long_vol = returns.rolling(50).std().iloc[-1]
            
            vol_ratio = volatility / long_vol
            
            if vol_ratio > 1.5:
                return 'high_volatility'
            elif vol_ratio < 0.7:
                return 'low_volatility'
            else:
                return 'sideways'
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime: str = None,
        validation_split: float = 0.2
    ) -> Tuple[object, Dict]:
        """
        Train model with regime awareness
        
        Args:
            X: Feature DataFrame
            y: Target Series
            regime: Market regime (if None, will detect)
            validation_split: Validation split ratio
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        # Detect regime if not provided
        if regime is None and self.use_regime_aware:
            # Simple regime detection based on target volatility
            target_vol = y.rolling(20).std().iloc[-1] if len(y) > 20 else y.std()
            long_vol = y.rolling(min(50, len(y))).std().iloc[-1] if len(y) > 50 else y.std()
            
            if target_vol > long_vol * 1.5:
                regime = 'high_volatility'
            elif target_vol < long_vol * 0.7:
                regime = 'low_volatility'
            else:
                regime = 'sideways'
        
        # Create model
        if self.use_ensemble:
            model = create_ensemble_for_regime(regime or 'sideways')
        else:
            from .cpu_models.xgboost_model import XGBoostModel
            model = XGBoostModel(max_depth=3, learning_rate=0.01, n_estimators=200)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # Direction accuracy
        train_dir = np.mean(np.sign(train_pred) == np.sign(y_train)) * 100
        val_dir = np.mean(np.sign(val_pred) == np.sign(y_val)) * 100
        
        # Generalization gap
        gen_gap = abs(train_r2 - val_r2)
        
        training_info = {
            'regime': regime,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_dir': train_dir,
            'val_dir': val_dir,
            'gen_gap': gen_gap,
            'model_type': type(model).__name__,
            'is_ensemble': hasattr(model, 'models'),
            'num_features': len(X.columns)
        }
        
        # Store model for regime
        if regime:
            self.regime_models[regime] = model
        
        return model, training_info
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        regime: str = None
    ) -> Dict:
        """
        Perform time series cross-validation
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            regime: Market regime
            
        Returns:
            Dictionary with CV results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'train_r2_scores': [],
            'val_r2_scores': [],
            'train_dir_scores': [],
            'val_dir_scores': [],
            'gen_gaps': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model, info = self.train_model(X_train, y_train, regime, validation_split=0.0)
            
            # Evaluate
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            val_dir = np.mean(np.sign(val_pred) == np.sign(y_val)) * 100
            
            cv_results['train_r2_scores'].append(info['train_r2'])
            cv_results['val_r2_scores'].append(val_r2)
            cv_results['train_dir_scores'].append(info['train_dir'])
            cv_results['val_dir_scores'].append(val_dir)
            cv_results['gen_gaps'].append(info['gen_gap'])
        
        # Calculate averages
        cv_results['mean_val_r2'] = np.mean(cv_results['val_r2_scores'])
        cv_results['mean_val_dir'] = np.mean(cv_results['val_dir_scores'])
        cv_results['std_val_r2'] = np.std(cv_results['val_r2_scores'])
        cv_results['std_val_dir'] = np.std(cv_results['val_dir_scores'])
        cv_results['mean_gen_gap'] = np.mean(cv_results['gen_gaps'])
        
        return cv_results
    
    def select_best_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int = 50
    ) -> List[str]:
        """
        Select best features using ensemble importance
        
        Args:
            X: Feature DataFrame
            y: Target Series
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        # Train ensemble model
        model, _ = self.train_model(X, y, validation_split=0.2)
        
        # Get feature importances
        if hasattr(model, 'models'):
            # Ensemble model - get average importance
            all_importances = []
            for base_model in model.models:
                if hasattr(base_model, 'get_feature_importance'):
                    importance_df = base_model.get_feature_importance()
                    if importance_df is not None:
                        all_importances.append(importance_df.set_index('feature')['importance'])
            
            if all_importances:
                avg_importance = pd.concat(all_importances, axis=1).mean(axis=1)
                importance_df = pd.DataFrame({
                    'feature': avg_importance.index,
                    'importance': avg_importance.values
                }).sort_values('importance', ascending=False)
            else:
                importance_df = None
        else:
            # Single model
            importance_df = model.get_feature_importance()
        
        if importance_df is not None:
            # Select top features
            top_features = importance_df.head(max_features)['feature'].tolist()
            self.selected_features = top_features
            self.feature_importances = importance_df
            return top_features
        else:
            # Fallback to all features
            self.selected_features = X.columns.tolist()
            return X.columns.tolist()
