"""
LightGBM model wrapper
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from .base_cpu_model import BaseCPUModel

class LightGBMModel(BaseCPUModel):
    """LightGBM model wrapper"""
    
    def __init__(self, **kwargs):
        super().__init__("lightgbm")
        self.params = {
            'max_depth': 3,
            'learning_rate': 0.02,
            'n_estimators': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
        self.params.update(kwargs)
        self.model = lgb.LGBMClassifier(**self.params)
    
    def fit(self, X, y):
        """Fit the LightGBM model"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            if hasattr(self, 'feature_names'):
                X = X[self.feature_names].values
            else:
                X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions"""
        if isinstance(X, pd.DataFrame):
            if hasattr(self, 'feature_names'):
                # Keep as DataFrame with proper column names for LightGBM
                X = X[self.feature_names]
            else:
                # Keep as DataFrame with original column names
                pass  # X is already a DataFrame
        return self.model.predict_proba(X)
