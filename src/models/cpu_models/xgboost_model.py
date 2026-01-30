"""
Optimized XGBoost Model for CPU
CPU-optimized XGBoost for stock prediction
"""

import xgboost as xgb
import numpy as np
from .base_cpu_model import BaseCPUModel
from typing import Dict, Any

class XGBoostModel(BaseCPUModel):
    """
    CPU-optimized XGBoost model for stock prediction
    Optimized hyperparameters for financial time series
    """
    
    def __init__(self, **kwargs):
        default_params = self.get_model_params()
        default_params.update(kwargs)
        super().__init__("XGBoost", **default_params)
    
    def _create_model(self, **kwargs):
        """Create XGBoost model with optimized parameters"""
        self.model = xgb.XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 300),
            max_depth=kwargs.get('max_depth', 4),
            learning_rate=kwargs.get('learning_rate', 0.03),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.7),
            reg_alpha=kwargs.get('reg_alpha', 0.1),      # L1 regularization
            reg_lambda=kwargs.get('reg_lambda', 1.0),     # L2 regularization
            min_child_weight=kwargs.get('min_child_weight', 1),
            gamma=kwargs.get('gamma', 0.1),
            objective=kwargs.get('objective', 'reg:squarederror'),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1),
            tree_method=kwargs.get('tree_method', 'hist'),  # CPU-optimized
            grow_policy=kwargs.get('grow_policy', 'lossguide')
        )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get optimized parameters for CPU XGBoost - Phase 4 winning config"""
        return {
            'n_estimators': 200,            # Optimal for regularization
            'max_depth': 3,                  # Shallower trees prevent overfitting
            'learning_rate': 0.02,           # Smaller learning rate for stability
            'subsample': 0.8,                # Subsample for each boosting round
            'colsample_bytree': 0.7,         # Feature subsampling
            'reg_alpha': 0.5,                # L1 regularization (increased)
            'reg_lambda': 1.0,               # L2 regularization
            'min_child_weight': 3,           # Increased for regularization
            'gamma': 0.1,                    # Minimum loss reduction
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',           # CPU-optimized histogram method
            'grow_policy': 'lossguide'       # Faster training
        }
    
    def fit(self, X, y, **fit_params):
        """
        Fit XGBoost model with early stopping
        """
        # Create model if not exists
        if self.model is None:
            self._create_model(**self.model_params)
        
        # Prepare data
        X_scaled, y = self.prepare_data(X, y, fit_scaler=True)
        
        # Set up early stopping if validation data provided
        eval_set = fit_params.pop('eval_set', None)
        early_stopping_rounds = fit_params.pop('early_stopping_rounds', None)
        verbose = fit_params.pop('verbose', False)
        
        # Fit model with early stopping only if eval_set is provided
        if eval_set is not None and early_stopping_rounds is not None:
            self.model.fit(
                X_scaled, y,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                **fit_params
            )
        else:
            # Simple fit without early stopping
            self.model.fit(X_scaled, y, **fit_params)
        
        self.is_fitted = True
        return self
    
    def get_feature_importance(self, type='weight'):
        """
        Get feature importance with different types
        
        Args:
            type: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_dict = self.model.get_booster().get_score(importance_type=type)
        
        # Convert to arrays
        feature_names = self.feature_names or [f'f{i}' for i in range(len(importance_dict))]
        importance = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        
        import pandas as pd
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def plot_learning_curves(self):
        """Plot learning curves if evaluation data was used"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if not hasattr(self.model, 'evals_result_'):
            print("No evaluation results available. Train with eval_set to see learning curves.")
            return
        
        results = self.model.evals_result_
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            for i, (key, values) in enumerate(results.items()):
                plt.plot(values['rmse'], label=f'{key} RMSE')
            
            plt.xlabel('Boosting Round')
            plt.ylabel('RMSE')
            plt.title('XGBoost Learning Curves')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def predict_with_contributions(self, X):
        """
        Make predictions with SHAP-like feature contributions
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, contributions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Prepare data
        X_scaled, _ = self.prepare_data(X, fit_scaler=False)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Get feature contributions (simplified version)
        contributions = self.model.predict(X_scaled, pred_contribs=True)
        
        return predictions, contributions
