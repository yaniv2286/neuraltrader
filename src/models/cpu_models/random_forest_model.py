"""
Optimized Random Forest Model for CPU
CPU-optimized hyperparameters for stock prediction
"""

from sklearn.ensemble import RandomForestRegressor
from .base_cpu_model import BaseCPUModel
from typing import Dict, Any

class RandomForestModel(BaseCPUModel):
    """
    CPU-optimized Random Forest model for stock prediction
    Optimized hyperparameters for financial time series
    """
    
    def __init__(self, **kwargs):
        default_params = self.get_model_params()
        default_params.update(kwargs)
        super().__init__("RandomForest", **default_params)
    
    def _create_model(self, **kwargs):
        """Create Random Forest model with optimized parameters"""
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 10),
            min_samples_leaf=kwargs.get('min_samples_leaf', 5),
            max_features=kwargs.get('max_features', 'sqrt'),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1),  # Use all CPU cores
            bootstrap=kwargs.get('bootstrap', True),
            oob_score=kwargs.get('oob_score', True)
        )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get optimized parameters for CPU Random Forest"""
        return {
            'n_estimators': 200,        # Good balance of performance vs speed
            'max_depth': 10,            # Prevent overfitting
            'min_samples_split': 10,    # Require more samples for splits
            'min_samples_leaf': 5,      # Smaller leaf nodes for granular predictions
            'max_features': 'sqrt',     # Reduce overfitting
            'random_state': 42,         # Reproducibility
            'n_jobs': -1,               # Use all CPU cores
            'bootstrap': True,           # Bootstrap sampling
            'oob_score': True           # Out-of-bag scoring
        }
    
    def get_oob_score(self) -> float:
        """Get out-of-bag score (built-in cross-validation)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        else:
            raise ValueError("Model was not trained with oob_score=True")
    
    def predict_with_confidence(self, X) -> tuple:
        """
        Make predictions with confidence intervals using tree variance
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate mean and std
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence interval
        confidence_interval = 1.96 * std_pred
        
        return mean_pred, confidence_interval
