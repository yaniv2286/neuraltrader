"""
Random Forest model wrapper
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from .base_cpu_model import BaseCPUModel

class RandomForestModel(BaseCPUModel):
    """Random Forest model wrapper"""
    
    def __init__(self, **kwargs):
        super().__init__("random_forest")
        self.params = {
            'max_depth': 3,
            'n_estimators': 200,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        }
        self.params.update(kwargs)
        self.model = RandomForestClassifier(**self.params)
    
    def fit(self, X, y):
        """Fit the Random Forest model"""
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
