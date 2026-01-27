"""
Base CPU Model Class
Common interface for all CPU-optimized models
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import joblib
import os

class BaseCPUModel(ABC):
    """
    Base class for CPU-optimized models
    Provides common functionality and interface
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.model_params = kwargs
        
    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying model instance"""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get default model parameters"""
        pass
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for training/prediction
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tuple of (X_scaled, y)
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values
        
        # Data validation
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            total_values = X.size
            nan_pct = (nan_count / total_values) * 100
            
            # XGBoost and RandomForest can handle NaN, but warn if excessive
            if nan_pct > 10:
                import warnings
                warnings.warn(
                    f"High percentage of NaN values in features: {nan_pct:.2f}%. "
                    f"Model will handle these internally, but consider imputation for better performance.",
                    UserWarning
                )
        
        if np.isinf(X).any():
            raise ValueError("Input contains infinite values. Please clean your data.")
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        """
        Fit the model
        
        Args:
            X: Feature matrix
            y: Target vector
            **fit_params: Additional fitting parameters
            
        Returns:
            Self
        """
        # Input validation
        if X is None or (hasattr(X, 'size') and X.size == 0):
            raise ValueError("X cannot be empty")
        
        if y is None or (hasattr(y, 'size') and y.size == 0):
            raise ValueError("y cannot be empty")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Check dimensions match
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {len(X)}, y: {len(y)}"
            )
        
        # Create model if not exists
        if self.model is None:
            self._create_model(**self.model_params)
        
        # Prepare data
        X_scaled, y = self.prepare_data(X, y, fit_scaler=True)
        
        # Fit model
        self.model.fit(X_scaled, y, **fit_params)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled, _ = self.prepare_data(X, fit_scaler=False)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions (if supported)
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"{self.model_name} does not support probability predictions")
        
        # Prepare data
        X_scaled, _ = self.prepare_data(X, fit_scaler=False)
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Directional accuracy (important for trading)
        if len(y) > 1:
            true_direction = np.diff(y) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_acc = np.mean(true_direction == pred_direction)
        else:
            directional_acc = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'directional_accuracy': directional_acc
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if supported)
        
        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        else:
            feature_names = self.feature_names
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data['model_params']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✅ Model loaded from {filepath}")
    
    def clone(self):
        """Create a clone of the model"""
        clone = self.__class__(**self.model_params)
        clone.model_name = self.model_name
        return clone
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'num_features': len(self.feature_names) if self.feature_names else None
        }
