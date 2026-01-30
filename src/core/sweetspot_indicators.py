"""
Sweet Spot Indicators - Standalone Implementation
==================================================

Calculates the specific indicators needed for VT Sweet Spot strategy:
- SMAs (25, 50, 100, 200)
- Volume average (30-day)
- Daily Stochastic (10, 3, 3)
- Weekly Stochastic (19, 4, 4)

This is a standalone implementation to avoid dependencies on complex feature generation.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_sma(data: pd.DataFrame, periods: list = [25, 50, 100, 200]) -> pd.DataFrame:
    """Calculate Simple Moving Averages."""
    df = data.copy()
    for period in periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_volume_avg(data: pd.DataFrame, period: int = 30) -> pd.DataFrame:
    """Calculate volume moving average."""
    df = data.copy()
    df['volume_avg_30'] = df['volume'].rolling(window=period).mean()
    return df


def calculate_stochastic(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
    prefix: str = 'stoch'
) -> pd.DataFrame:
    """
    Calculate Stochastic %K and %D.
    
    Args:
        data: DataFrame with OHLC data
        k_period: Period for %K calculation
        d_period: Period for %D (SMA of %K)
        smooth: Smoothing period for %K
        prefix: Column name prefix
    
    Returns:
        DataFrame with stochastic columns added
    """
    df = data.copy()
    
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Smooth %K if needed
    if smooth > 1:
        stoch_k = stoch_k.rolling(window=smooth).mean()
    
    # Calculate %D (moving average of %K)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    df[f'{prefix}_k'] = stoch_k
    df[f'{prefix}_d'] = stoch_d
    
    return df


def calculate_weekly_stochastic(
    data: pd.DataFrame,
    k_period: int = 19,
    d_period: int = 4,
    smooth: int = 4
) -> pd.DataFrame:
    """
    Calculate weekly stochastic from daily data.
    
    Args:
        data: DataFrame with daily OHLC data (must have 'date' column or DatetimeIndex)
        k_period: Period for %K calculation
        d_period: Period for %D (SMA of %K)
        smooth: Smoothing period for %K
    
    Returns:
        DataFrame with weekly stochastic columns added
    """
    df = data.copy()
    
    # Ensure we have a datetime index
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Resample to weekly
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Calculate stochastic on weekly bars
    low_min = weekly['low'].rolling(window=k_period).min()
    high_max = weekly['high'].rolling(window=k_period).max()
    
    stoch_k = 100 * (weekly['close'] - low_min) / (high_max - low_min)
    
    # Smooth %K
    if smooth > 1:
        stoch_k = stoch_k.rolling(window=smooth).mean()
    
    # Calculate %D
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    # Map back to daily (forward fill)
    stoch_k_daily = stoch_k.reindex(df.index, method='ffill')
    stoch_d_daily = stoch_d.reindex(df.index, method='ffill')
    
    # Reset index if original had date column
    if 'date' in data.columns:
        df = df.reset_index()
        df['stoch_weekly_k'] = stoch_k_daily.values
        df['stoch_weekly_d'] = stoch_d_daily.values
    else:
        df['stoch_weekly_k'] = stoch_k_daily
        df['stoch_weekly_d'] = stoch_d_daily
    
    return df


def add_sweetspot_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all Sweet Spot indicators to data.
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all Sweet Spot indicators added
    """
    df = data.copy()
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add SMAs
    df = calculate_sma(df, periods=[25, 50, 100, 200])
    
    # Add volume average
    df = calculate_volume_avg(df, period=30)
    
    # Add daily stochastic
    df = calculate_stochastic(df, k_period=10, d_period=3, smooth=3, prefix='stoch_daily')
    
    # Add weekly stochastic
    df = calculate_weekly_stochastic(df, k_period=19, d_period=4, smooth=4)
    
    return df


def is_in_sweetspot(row: pd.Series, debug: bool = False) -> bool:
    """
    Check if a row meets all Sweet Spot criteria.
    
    Args:
        row: Series with all required indicators
        debug: If True, print why signal was rejected
    
    Returns:
        True if in Sweet Spot, False otherwise
    """
    try:
        # Check for NaN values
        required_cols = ['close', 'sma_25', 'sma_50', 'sma_100', 'sma_200',
                        'volume', 'volume_avg_30',
                        'stoch_daily_k', 'stoch_daily_d',
                        'stoch_weekly_k', 'stoch_weekly_d']
        
        for col in required_cols:
            if pd.isna(row[col]):
                if debug:
                    print(f"         ❌ NaN value in {col}")
                return False
        
        # 1. Power Stock: Price above ALL SMAs
        if not (row['close'] > row['sma_25'] and
                row['close'] > row['sma_50'] and
                row['close'] > row['sma_100'] and
                row['close'] > row['sma_200']):
            if debug:
                print(f"         ❌ Power Stock failed: close={row['close']:.2f}, sma_25={row['sma_25']:.2f}, sma_50={row['sma_50']:.2f}, sma_100={row['sma_100']:.2f}, sma_200={row['sma_200']:.2f}")
            return False
        
        # 2. Volume Fuel: Volume >= 30-day average
        if not (row['volume'] >= row['volume_avg_30']):
            if debug:
                print(f"         ❌ Volume failed: {row['volume']:.0f} < {row['volume_avg_30']:.0f}")
            return False
        
        # 3. Daily Stochastic: Both %K and %D >= 80
        if not (row['stoch_daily_k'] >= 80 and row['stoch_daily_d'] >= 80):
            if debug:
                print(f"         ❌ Daily Stoch failed: K={row['stoch_daily_k']:.1f}, D={row['stoch_daily_d']:.1f}")
            return False
        
        # 4. Weekly Stochastic: Both %K and %D >= 80
        if not (row['stoch_weekly_k'] >= 80 and row['stoch_weekly_d'] >= 80):
            if debug:
                print(f"         ❌ Weekly Stoch failed: K={row['stoch_weekly_k']:.1f}, D={row['stoch_weekly_d']:.1f}")
            return False
        
        # All criteria met
        if debug:
            print(f"         ✅ PASSED Sweet Spot!")
        return True
        
    except (KeyError, TypeError) as e:
        if debug:
            print(f"         ❌ Error: {e}")
        return False


def should_exit_sweetspot(row: pd.Series) -> bool:
    """
    Check if should exit Sweet Spot position.
    
    Args:
        row: Series with stochastic indicators
    
    Returns:
        True if should exit, False otherwise
    """
    try:
        # Exit if daily stochastic drops below 80 (either %K or %D)
        if pd.isna(row['stoch_daily_k']) or pd.isna(row['stoch_daily_d']):
            return True
        
        return row['stoch_daily_k'] < 80 or row['stoch_daily_d'] < 80
        
    except (KeyError, TypeError):
        return True
