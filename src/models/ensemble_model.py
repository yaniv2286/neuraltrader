"""
Ensemble Model System - Combines multiple models for better performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from .cpu_models.base_cpu_model import BaseCPUModel
from .cpu_models.xgboost_model import XGBoostModel
from .cpu_models.random_forest_model import RandomForestModel

class EnsembleModel(BaseCPUModel):
    """Ensemble model that combines multiple base models"""
    
    def __init__(self, models: List[BaseCPUModel] = None, weights: List[float] = None):
        """
        Initialize ensemble model
        
        Args:
            models: List of base models
            weights: Weights for each model (default: equal weights)
        """
        super().__init__("ensemble")
        
        if models is None:
            # Default models
            self.models = [
                XGBoostModel(max_depth=3, learning_rate=0.01, n_estimators=200),
                XGBoostModel(max_depth=5, learning_rate=0.05, n_estimators=100),
                RandomForestModel(n_estimators=200, max_depth=10)
            ]
        else:
            self.models = models
            
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            self.weights = weights
            
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.model_performances = {}
        
    def _create_model(self, **kwargs):
        """Create ensemble (not used for ensemble)"""
        pass
    
    def get_model_params(self) -> Dict:
        """Get model parameters"""
        return {
            'models': [type(m).__name__ for m in self.models],
            'weights': self.weights,
            'num_models': len(self.models)
        }
    
    def fit(self, X, y, **fit_params):
        """Fit all models in the ensemble"""
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Fit each model
        for i, model in enumerate(self.models):
            try:
                model.fit(X, y, **fit_params)
                print(f"✅ Model {i+1}/{len(self.models)} ({type(model).__name__}) trained")
            except Exception as e:
                print(f"❌ Model {i+1} failed: {e}")
                # Remove failed model
                self.models.pop(i)
                self.weights.pop(i)
        
        if not self.models:
            raise ValueError("All models failed to train")
            
        # Re-normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def predict_proba(self, X) -> np.ndarray:
        """Make probability predictions (if supported by all models)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                predictions.append(pred)
            else:
                # Fallback to binary predictions
                pred = model.predict(X)
                # Convert to probability format
                prob_pred = np.zeros((len(pred), 2))
                prob_pred[pred > 0, 1] = 1
                prob_pred[pred <= 0, 0] = 1
                predictions.append(prob_pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def evaluate_individual_models(self, X, y) -> Dict:
        """Evaluate individual model performances"""
        performances = {}
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                r2 = model.score(X, y)
                direction_acc = np.mean(np.sign(pred) == np.sign(y)) * 100
                
                performances[f'model_{i+1}'] = {
                    'type': type(model).__name__,
                    'r2': r2,
                    'direction_accuracy': direction_acc,
                    'weight': self.weights[i]
                }
            except Exception as e:
                performances[f'model_{i+1}'] = {
                    'type': type(model).__name__,
                    'error': str(e),
                    'weight': self.weights[i]
                }
        
        return performances
    
    def update_weights(self, X_val, y_val):
        """Update model weights based on validation performance"""
        performances = []
        
        for model in self.models:
            try:
                pred = model.predict(X_val)
                # Use direction accuracy as performance metric
                acc = np.mean(np.sign(pred) == np.sign(y_val))
                performances.append(max(acc, 0.1))  # Minimum weight
            except:
                performances.append(0.1)  # Minimum weight for failed models
        
        # Update weights based on performance
        total_perf = sum(performances)
        self.weights = [p / total_perf for p in performances]
        
        print("Updated model weights:")
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            print(f"  {type(model).__name__}: {weight:.3f}")

class DynamicModelSelector:
    """Dynamically selects best model based on market conditions"""
    
    def __init__(self):
        self.models = {}
        self.regime_performance = {}
        
    def add_model(self, name: str, model: BaseCPUModel, regimes: List[str] = None):
        """Add a model for specific regimes"""
        if regimes is None:
            regimes = ['all']
        
        for regime in regimes:
            if regime not in self.models:
                self.models[regime] = []
            self.models[regime].append((name, model))
    
    def select_model(self, X, regime: str = 'all') -> Tuple[BaseCPUModel, str]:
        """Select best model for current regime"""
        if regime not in self.models:
            regime = 'all'
        
        if not self.models[regime]:
            # Fallback to all models
            regime = 'all'
        
        # For now, return first model (can be enhanced with performance tracking)
        name, model = self.models[regime][0]
        return model, name
    
    def update_performance(self, regime: str, model_name: str, performance: float):
        """Update performance tracking"""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {}
        
        self.regime_performance[regime][model_name] = performance

def create_ensemble_for_regime(regime: str) -> EnsembleModel:
    """Create optimized ensemble for specific market regime"""
    
    if regime == 'high_volatility':
        # Conservative models for high volatility
        models = [
            XGBoostModel(max_depth=2, learning_rate=0.01, n_estimators=300, reg_alpha=2.0),
            RandomForestModel(n_estimators=300, max_depth=8, min_samples_split=20)
        ]
        weights = [0.6, 0.4]  # Favor XGBoost
        
    elif regime == 'low_volatility':
        # More aggressive models for low volatility
        models = [
            XGBoostModel(max_depth=5, learning_rate=0.05, n_estimators=200),
            XGBoostModel(max_depth=3, learning_rate=0.02, n_estimators=400),
            RandomForestModel(n_estimators=200, max_depth=12)
        ]
        weights = [0.4, 0.3, 0.3]
        
    elif regime == 'uptrend':
        # Trend-following models
        models = [
            XGBoostModel(max_depth=4, learning_rate=0.03, n_estimators=250),
            RandomForestModel(n_estimators=250, max_depth=10)
        ]
        weights = [0.6, 0.4]
        
    elif regime == 'downtrend':
        # Mean-reversion models
        models = [
            XGBoostModel(max_depth=3, learning_rate=0.01, n_estimators=300, reg_alpha=1.5),
            RandomForestModel(n_estimators=300, max_depth=8, min_samples_split=15)
        ]
        weights = [0.7, 0.3]
        
    else:  # default/sideways
        # Balanced models
        models = [
            XGBoostModel(max_depth=3, learning_rate=0.02, n_estimators=200),
            XGBoostModel(max_depth=5, learning_rate=0.04, n_estimators=150),
            RandomForestModel(n_estimators=200, max_depth=10)
        ]
        weights = [0.4, 0.3, 0.3]
    
    return EnsembleModel(models, weights)
