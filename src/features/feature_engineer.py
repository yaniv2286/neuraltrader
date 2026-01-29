"""
Feature Engineer - Consolidated feature engineering
Basic feature creation for stock prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .indicators import apply_indicators

class FeatureEngineer:
    """Simple feature engineering for stock prediction"""
    
    def __init__(self):
        self.feature_count = 0
    
    def create_features(self, data: pd.DataFrame, target_type: str = 'log_returns') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create basic features for model training
        
        Args:
            data: OHLCV data
            target_type: Type of target variable ('log_returns', 'returns', 'direction')
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            # Add basic technical indicators
            df = apply_indicators(df)
            
            # Create target variable BEFORE dropping NaNs
            target = self._create_target(df, target_type)
            
            # Combine features and target, then drop NaNs together
            # This ensures alignment between X and y
            combined_df = df.copy()
            combined_df['target'] = target
            
            # Drop rows where any feature or target is NaN
            combined_df = combined_df.dropna()
            
            # Separate features and target
            # Remove price columns to avoid data leakage
            price_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
            feature_cols = [col for col in combined_df.columns if col not in price_cols]
            
            # Ensure only numeric features are selected
            numeric_features = []
            for col in feature_cols:
                if combined_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_features.append(col)
                else:
                    print(f"Skipping non-numeric feature: {col} (dtype: {combined_df[col].dtype})")
            
            X = combined_df[numeric_features]
            y = combined_df['target']
            
            # Convert to float32 to avoid type issues
            X = X.astype('float32')
            y = y.astype('float32')
            
            self.feature_count = len(feature_cols)
            
            return X, y
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return None, None
    
    def _create_target(self, df: pd.DataFrame, target_type: str) -> pd.Series:
        """Create target variable"""
        if target_type == 'log_returns':
            # Log returns of next day
            returns = np.log(df['close']).diff().shift(-1)
            return returns  # Don't drop NaNs here - handled in main method
        elif target_type == 'returns':
            # Simple returns
            returns = df['close'].pct_change().shift(-1)
            return returns  # Don't drop NaNs here - handled in main method
        elif target_type == 'direction':
            # Direction (up/down)
            returns = df['close'].pct_change().shift(-1)
            direction = (returns > 0).astype(int)
            return direction  # Don't drop NaNs here - handled in main method
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
    
    def get_feature_count(self) -> int:
        """Get number of features created"""
        return self.feature_count
    
    def get_feature_names(self, data: pd.DataFrame) -> list:
        """Get list of feature names"""
        X, _ = self.create_features(data)
        if X is not None:
            return X.columns.tolist()
        return []
