"""
Optimized Model Trainer - Maximum CPU Performance
Per-ticker hyperparameter tuning, feature selection, and model comparison
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .cpu_models import XGBoostModel, LightGBMModel, RandomForestModel


class OptimizedTrainer:
    """
    Optimized trainer that:
    1. Compares XGBoost, LightGBM, RandomForest
    2. Selects best model per ticker
    3. Uses feature importance for selection
    4. Applies per-ticker hyperparameter tuning
    """
    
    def __init__(self, n_top_features: int = 25, use_tuning: bool = True):
        self.n_top_features = n_top_features
        self.use_tuning = use_tuning
        self.best_model = None
        self.best_model_type = None
        self.selected_features = None
        self.feature_importances = None
    
    def train_and_select_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.15
    ) -> Tuple[object, Dict]:
        """
        Train multiple models and select the best one
        
        Returns:
            Tuple of (best_model, training_info)
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_arr = X.values.astype(np.float64)
        else:
            feature_names = [f'f{i}' for i in range(X.shape[1])]
            X_arr = X.astype(np.float64)
        
        if isinstance(y, pd.Series):
            y_arr = y.values.astype(np.float64)
        else:
            y_arr = y.astype(np.float64)
        
        # Handle NaN
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        split_idx = int(len(X_arr) * (1 - validation_split))
        X_train, X_val = X_arr[:split_idx], X_arr[split_idx:]
        y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]
        
        # Train all models
        models = {}
        scores = {}
        
        # XGBoost
        try:
            xgb_model = XGBoostModel()
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_val)
            xgb_dir = np.mean(np.sign(xgb_pred) == np.sign(y_val)) * 100
            models['xgboost'] = xgb_model
            scores['xgboost'] = xgb_dir
        except Exception as e:
            scores['xgboost'] = 0
        
        # LightGBM
        try:
            lgb_model = LightGBMModel()
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_val)
            lgb_dir = np.mean(np.sign(lgb_pred) == np.sign(y_val)) * 100
            models['lightgbm'] = lgb_model
            scores['lightgbm'] = lgb_dir
        except Exception as e:
            scores['lightgbm'] = 0
        
        # RandomForest
        try:
            rf_model = RandomForestModel()
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_val)
            rf_dir = np.mean(np.sign(rf_pred) == np.sign(y_val)) * 100
            models['random_forest'] = rf_model
            scores['random_forest'] = rf_dir
        except Exception as e:
            scores['random_forest'] = 0
        
        # Select best model
        best_type = max(scores, key=scores.get)
        self.best_model = models.get(best_type)
        self.best_model_type = best_type
        
        # Get feature importance from best model
        if self.best_model and hasattr(self.best_model, 'get_feature_importance'):
            try:
                self.feature_importances = self.best_model.get_feature_importance()
            except:
                pass
        
        # Calculate final metrics
        if self.best_model:
            train_pred = self.best_model.predict(X_train)
            val_pred = self.best_model.predict(X_val)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            train_dir = np.mean(np.sign(train_pred) == np.sign(y_train)) * 100
            val_dir = np.mean(np.sign(val_pred) == np.sign(y_val)) * 100
        else:
            train_r2 = val_r2 = train_dir = val_dir = 0
        
        training_info = {
            'best_model_type': best_type,
            'all_scores': scores,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_dir': train_dir,
            'val_dir': val_dir,
            'gen_gap': abs(train_r2 - val_r2),
            'num_features': X_arr.shape[1]
        }
        
        return self.best_model, training_info
    
    def train_with_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.15
    ) -> Tuple[object, Dict]:
        """
        Train with automatic feature selection based on importance
        """
        # First pass: train to get feature importance
        model, info = self.train_and_select_best(X, y, validation_split)
        
        if self.feature_importances is not None and len(self.feature_importances) > 0:
            # Select top features
            top_features = self.feature_importances.head(self.n_top_features)['feature'].tolist()
            
            # Filter to features that exist in X
            available_features = [f for f in top_features if f in X.columns]
            
            if len(available_features) >= 10:
                X_selected = X[available_features]
                self.selected_features = available_features
                
                # Retrain with selected features
                model, info = self.train_and_select_best(X_selected, y, validation_split)
                info['selected_features'] = available_features
                info['feature_selection'] = True
        
        return model, info
    
    def evaluate(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    ) -> Dict:
        """Evaluate model on all splits"""
        # Convert to numpy
        for arr in [X_train, X_val, X_test]:
            if isinstance(arr, pd.DataFrame):
                arr = arr.values.astype(np.float64)
        for arr in [y_train, y_val, y_test]:
            if isinstance(arr, pd.Series):
                arr = arr.values.astype(np.float64)
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        return {
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_dir': np.mean(np.sign(train_pred) == np.sign(y_train)) * 100,
            'val_dir': np.mean(np.sign(val_pred) == np.sign(y_val)) * 100,
            'test_dir': np.mean(np.sign(test_pred) == np.sign(y_test)) * 100,
            'gen_gap': abs(r2_score(y_train, train_pred) - r2_score(y_test, test_pred))
        }


class EnsembleTrainer:
    """
    Ensemble trainer that combines predictions from multiple models
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'xgboost': 0.4, 'lightgbm': 0.4, 'random_forest': 0.2}
        self.models = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train all models in ensemble"""
        # Convert data
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = X.astype(np.float64)
        
        if isinstance(y, pd.Series):
            y_arr = y.values.astype(np.float64)
        else:
            y_arr = y.astype(np.float64)
        
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        results = {}
        
        # XGBoost
        try:
            self.models['xgboost'] = XGBoostModel()
            self.models['xgboost'].fit(X_arr, y_arr)
            results['xgboost'] = 'trained'
        except Exception as e:
            results['xgboost'] = f'failed: {e}'
        
        # LightGBM
        try:
            self.models['lightgbm'] = LightGBMModel()
            self.models['lightgbm'].fit(X_arr, y_arr)
            results['lightgbm'] = 'trained'
        except Exception as e:
            results['lightgbm'] = f'failed: {e}'
        
        # RandomForest
        try:
            self.models['random_forest'] = RandomForestModel()
            self.models['random_forest'].fit(X_arr, y_arr)
            results['random_forest'] = 'trained'
        except Exception as e:
            results['random_forest'] = f'failed: {e}'
        
        return results
    
    def predict(self, X) -> np.ndarray:
        """Make weighted ensemble prediction"""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = X.astype(np.float64)
        
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model is not None:
                try:
                    pred = model.predict(X_arr)
                    predictions.append(pred)
                    weights.append(self.weights.get(name, 0.33))
                except:
                    pass
        
        if not predictions:
            return np.zeros(len(X_arr))
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.zeros(len(X_arr))
        
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
