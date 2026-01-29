"""
Feature Selector Module
Removes redundant and low-value features to prevent overfitting
"""

import pandas as pd
import numpy as np
from typing import List, Set

class FeatureSelector:
    """
    Intelligent feature selection to remove redundant features
    """
    
    # Features identified as redundant (correlation > 0.95)
    REDUNDANT_FEATURES = {
        'high',           # Identical to low/close (keep low, close)
        'open',           # Identical to low/close (keep low, close)
        'bb_middle',      # Identical to ma_20 (keep ma_20)
        'sma_20',         # Identical to ma_20 (keep ma_20)
        'sma_50',         # Identical to ma_50 (keep ma_50)
        'sma_200',        # Identical to ma_200 (keep ma_200)
        'return_1d',      # Identical to returns (keep returns)
        'hist_volatility',# Identical to volatility_20 (keep volatility_20)
        'price_momentum_5',   # Identical to return_5d (keep return_5d)
        'price_momentum_20',  # Identical to return_20d (keep return_20d)
        # Phase 3 identified redundant features (correlation > 0.95)
        'close',          # Highly correlated with low (0.995)
        'ema_20',         # Highly correlated with ma_20 (0.995)
        'ma_10',          # Highly correlated with ma_5 (0.987)
        'ma_5',           # Highly correlated with low (0.983)
        'price_to_ma_5',  # Highly correlated with ma_5 (0.959)
        'support_level',  # Highly correlated with ma_5 (0.959)
        # Additional redundant features from fast mode test
        'bb_lower',       # Highly correlated with ma_20 (0.993)
        'bb_upper',       # Highly correlated with ma_20 (0.994)
        'ma_20',          # Highly correlated with bb_upper (0.994)
        'resistance_level', # Highly correlated with low (0.992)
        'volatility_pct', # Highly correlated with other volatility features
    }
    
    def __init__(self):
        """Initialize feature selector"""
        self.removed_features = []
        self.kept_features = []
    
    def remove_redundant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove redundant features from DataFrame
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with redundant features removed
        """
        # Find which redundant features are actually in the DataFrame
        features_to_remove = [col for col in self.REDUNDANT_FEATURES if col in df.columns]
        
        if features_to_remove:
            self.removed_features = features_to_remove
            df_clean = df.drop(columns=features_to_remove)
            self.kept_features = list(df_clean.columns)
            
            print(f"   🔧 Removed {len(features_to_remove)} redundant features")
            print(f"   ✅ Kept {len(self.kept_features)} unique features")
            
            return df_clean
        else:
            self.kept_features = list(df.columns)
            return df
    
    def get_feature_importance_threshold(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'close',
        min_correlation: float = 0.01
    ) -> List[str]:
        """
        Identify features with very low correlation to target
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            min_correlation: Minimum absolute correlation threshold
            
        Returns:
            List of weak features
        """
        if target_col not in df.columns:
            return []
        
        # Calculate correlation with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        correlations = df[feature_cols].corrwith(df[target_col]).abs()
        weak_features = correlations[correlations < min_correlation].index.tolist()
        
        return weak_features
    
    def select_features(
        self,
        df: pd.DataFrame,
        remove_redundant: bool = True,
        remove_weak: bool = False,
        target_col: str = 'close',
        min_correlation: float = 0.01
    ) -> pd.DataFrame:
        """
        Main feature selection pipeline
        
        Args:
            df: DataFrame with all features
            remove_redundant: Remove redundant features (default True)
            remove_weak: Remove weak features (default False)
            target_col: Target column for weak feature detection
            min_correlation: Minimum correlation threshold for weak features
            
        Returns:
            DataFrame with selected features
        """
        df_selected = df.copy()
        
        # Step 1: Remove redundant features
        if remove_redundant:
            df_selected = self.remove_redundant_features(df_selected)
        
        # Step 2: Optionally remove weak features
        if remove_weak:
            weak_features = self.get_feature_importance_threshold(
                df_selected, 
                target_col, 
                min_correlation
            )
            
            if weak_features:
                print(f"   ⚠️ Found {len(weak_features)} weak features (correlation < {min_correlation})")
                df_selected = df_selected.drop(columns=weak_features)
                self.removed_features.extend(weak_features)
        
        return df_selected
    
    def get_summary(self) -> dict:
        """Get summary of feature selection"""
        return {
            'total_removed': len(self.removed_features),
            'total_kept': len(self.kept_features),
            'removed_features': self.removed_features,
            'kept_features': self.kept_features
        }


def remove_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to remove redundant features
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with redundant features removed
    """
    selector = FeatureSelector()
    return selector.remove_redundant_features(df)
