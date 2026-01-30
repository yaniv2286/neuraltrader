"""
Optimized LightGBM Model for CPU
CPU-optimized LightGBM for stock prediction - often faster and better than XGBoost
"""

import lightgbm as lgb
import numpy as np
from .base_cpu_model import BaseCPUModel
from typing import Dict, Any

class LightGBMModel(BaseCPUModel):
    """
    CPU-optimized LightGBM model for stock prediction
    Optimized hyperparameters for financial time series
    """
    
    def __init__(self, **kwargs):
        default_params = self.get_model_params()
        default_params.update(kwargs)
        super().__init__("LightGBM", **default_params)
    
    def _create_model(self, **kwargs):
        """Create LightGBM model with optimized parameters"""
        self.model = lgb.LGBMRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.02),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.7),
            reg_alpha=kwargs.get('reg_alpha', 0.5),
            reg_lambda=kwargs.get('reg_lambda', 1.0),
            min_child_samples=kwargs.get('min_child_samples', 20),
            num_leaves=kwargs.get('num_leaves', 15),
            objective=kwargs.get('objective', 'regression'),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1),
            verbose=kwargs.get('verbose', -1),
            force_col_wise=True  # CPU optimization
        )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get optimized parameters for CPU LightGBM"""
        return {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'num_leaves': 15,
            'objective': 'regression',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def fit(self, X, y, **fit_params):
        """Fit LightGBM model"""
        if self.model is None:
            self._create_model(**self.model_params)
        
        X_scaled, y = self.prepare_data(X, y, fit_scaler=True)
        
        self.model.fit(X_scaled, y, **fit_params)
        self.is_fitted = True
        return self
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names or [f'f{i}' for i in range(len(importance))]
        
        import pandas as pd
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
