import pandas as pd
import numpy as np

def add_volatility_features(df, price_col='close', window=20):
    """Add volatility-based features for dynamic thresholds"""
    # ATR (Average True Range) for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df[price_col].shift())
    low_close = np.abs(df['low'] - df[price_col].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=window).mean()
    
    # Volatility as percentage of price
    df['volatility_pct'] = (df['atr'] / df[price_col]) * 100
    
    # Dynamic threshold multiplier based on volatility
    df['vol_multiplier'] = np.where(
        df['volatility_pct'] > df['volatility_pct'].quantile(0.75),
        1.5,  # High volatility - increase thresholds
        np.where(
            df['volatility_pct'] < df['volatility_pct'].quantile(0.25),
            0.5,  # Low volatility - decrease thresholds
            1.0   # Normal volatility
        )
    )
    
    return df
