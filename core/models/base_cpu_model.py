"""
Base CPU model class
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseCPUModel(ABC):
    """Base class for CPU models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Make probability predictions"""
        pass
