"""
Feature Engineer - Consolidated feature engineering
Advanced feature creation for stock prediction with CPU optimization
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .indicators import apply_indicators
from .advanced_features import (
    add_volatility_features,
    add_momentum_features,
    add_price_action_features,
    add_drawdown_features,
    add_lag_features
)

class FeatureEngineer:
    """Advanced feature engineering for stock prediction - CPU optimized"""
    
    def __init__(self, use_advanced_features: bool = True):
        self.feature_count = 0
        self.use_advanced_features = use_advanced_features
    
    def create_features(self, data: pd.DataFrame, target_type: str = 'log_returns') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create advanced features for model training
        
        Args:
            data: OHLCV data
            target_type: Type of target variable ('log_returns', 'returns', 'direction', 'multi_horizon')
            
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
            
            # Add advanced features if enabled
            if self.use_advanced_features:
                df = self._add_advanced_features(df)
            
            # Add market regime features
            df = self._add_regime_features(df)
            
            # Create target variable BEFORE dropping NaNs
            target = self._create_target(df, target_type)
            
            # Combine features and target, then drop NaNs together
            combined_df = df.copy()
            combined_df['target'] = target
            
            # Drop rows where any feature or target is NaN
            combined_df = combined_df.dropna()
            
            # Remove non-feature columns
            exclude_cols = ['target', 'date', 'Date', 'datetime', 'Datetime']
            features = combined_df.drop([c for c in exclude_cols if c in combined_df.columns], axis=1)
            target = combined_df['target']
            
            self.feature_count = len(features.columns)
            
            return features, target
            
        except Exception as e:
            print(f"Error in feature creation: {e}")
            raise
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all advanced features for maximum CPU performance"""
        # Volatility features (ATR, rolling volatility)
        df = add_volatility_features(df, window=14)
        
        # Momentum features (momentum, ROC)
        df = add_momentum_features(df, window=10)
        
        # Price action features (body size, wicks, engulfing)
        df = add_price_action_features(df)
        
        # Drawdown features
        df = add_drawdown_features(df, window=20)
        
        # Lag features (1-day lag of key indicators)
        df = add_lag_features(df, lag=1)
        
        # Additional momentum indicators
        df = self._add_extra_momentum(df)
        
        # MACD
        df = self._add_macd(df)
        
        # Bollinger Bands
        df = self._add_bollinger_bands(df)
        
        return df
    
    def _add_extra_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extra momentum indicators"""
        close = df['close']
        
        # Multiple timeframe momentum
        df['momentum_5'] = close - close.shift(5)
        df['momentum_10'] = close - close.shift(10)
        df['momentum_20'] = close - close.shift(20)
        
        # Rate of change
        df['roc_5'] = close.pct_change(5)
        df['roc_10'] = close.pct_change(10)
        df['roc_20'] = close.pct_change(20)
        
        # Price relative to moving averages
        sma_10 = close.rolling(10).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        df['price_sma10_ratio'] = close / sma_10
        df['price_sma50_ratio'] = close / sma_50
        df['price_sma200_ratio'] = close / sma_200
        
        # Moving average crossovers
        df['sma10_sma50_cross'] = (sma_10 > sma_50).astype(int)
        df['sma50_sma200_cross'] = (sma_50 > sma_200).astype(int)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        close = df['close']
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD crossover signals
        df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & 
                              (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        close = df['close']
        
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_middle'] = sma_20
        
        # Bollinger Band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band signals
        df['bb_overbought'] = (close > df['bb_upper']).astype(int)
        df['bb_oversold'] = (close < df['bb_lower']).astype(int)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        # Volatility regime
        returns = df['close'].pct_change()
        volatility_20 = returns.rolling(20).std()
        volatility_50 = returns.rolling(50).std()
        
        df['vol_regime'] = volatility_20 / volatility_50
        df['high_vol'] = (df['vol_regime'] > 1.5).astype(int)
        df['low_vol'] = (df['vol_regime'] < 0.7).astype(int)
        
        # Trend regime
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        df['trend_regime'] = (sma_20 > sma_50).astype(int)
        df['strong_uptrend'] = (df['close'] > sma_20 * 1.05).astype(int)
        df['strong_downtrend'] = (df['close'] < sma_20 * 0.95).astype(int)
        
        # Market efficiency
        df['price_efficiency'] = df['close'] / sma_20
        
        return df
    
    def _create_target(self, df: pd.DataFrame, target_type: str = 'log_returns') -> pd.Series:
        """Create advanced target variables"""
        if target_type == 'multi_horizon':
            # Multi-horizon direction prediction
            target_1d = np.sign(df['close'].shift(-1) / df['close'] - 1)
            target_5d = np.sign(df['close'].shift(-5) / df['close'] - 1)
            target_10d = np.sign(df['close'].shift(-10) / df['close'] - 1)
            
            # Weighted combination (more weight to near-term)
            target = (0.5 * target_1d + 0.3 * target_5d + 0.2 * target_10d)
            
        elif target_type == 'volatility_adjusted':
            # Volatility-adjusted returns
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            target = (returns.shift(-1) / volatility).fillna(0)
            
        elif target_type == 'regime_aware':
            # Regime-aware targets
            high_vol_mask = df['high_vol'] == 1
            low_vol_mask = df['low_vol'] == 1
            
            # In high volatility, predict smaller moves
            base_target = np.sign(df['close'].shift(-1) / df['close'] - 1)
            target = base_target.copy()
            target[high_vol_mask] *= 0.5  # Reduce target magnitude in high vol
            target[low_vol_mask] *= 1.5   # Increase target magnitude in low vol
            
        elif target_type == 'log_returns':
            # Log returns of next day
            returns = np.log(df['close']).diff().shift(-1)
            target = returns  # Don't drop NaNs here - handled in main method
        elif target_type == 'returns':
            # Simple returns
            returns = df['close'].pct_change().shift(-1)
            target = returns  # Don't drop NaNs here - handled in main method
        elif target_type == 'direction':
            # Direction (up/down)
            returns = df['close'].pct_change().shift(-1)
            direction = (returns > 0).astype(int)
            target = direction  # Don't drop NaNs here - handled in main method
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return target
    
    def get_feature_count(self) -> int:
        """Get number of features created"""
        return self.feature_count
    
    def get_feature_names(self, data: pd.DataFrame) -> list:
        """Get list of feature names"""
        X, _ = self.create_features(data)
        if X is not None:
            return X.columns.tolist()
        return []
