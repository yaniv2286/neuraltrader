"""
XGBoost model wrapper
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from .base_cpu_model import BaseCPUModel

class XGBoostModel(BaseCPUModel):
    """XGBoost model wrapper"""
    
    def __init__(self, **kwargs):
        super().__init__("xgboost")
        self.params = {
            'max_depth': 3,
            'learning_rate': 0.02,
            'n_estimators': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': 0
        }
        self.params.update(kwargs)
        self.model = xgb.XGBClassifier(**self.params)
    
    def fit(self, X, y):
        """Fit the XGBoost model"""
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
                X = X[self.feature_names].values
            else:
                X = X.values
        return self.model.predict_proba(X)
