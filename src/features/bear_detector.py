import pandas as pd
import numpy as np

def detect_bear_market(df, price_col='close', lookback=252):
    """
    Detect bear market conditions
    Returns bear_strength (0-1, higher = worse bear market)
    """
    if len(df) < lookback:
        return 0.0
    
    # Long-term trend
    sma_200 = df[price_col].rolling(window=200).mean()
    price_below_200 = (df[price_col].iloc[-1] < sma_200.iloc[-1])
    
    # Recent performance
    returns_6m = df[price_col].pct_change(126).iloc[-1]
    returns_3m = df[price_col].pct_change(63).iloc[-1]
    returns_1m = df[price_col].pct_change(21).iloc[-1]
    
    # Moving average cross (death cross)
    sma_50 = df[price_col].rolling(window=50).mean()
    death_cross = sma_50.iloc[-1] < sma_200.iloc[-1]
    
    # Volatility spike
    volatility = df[price_col].pct_change().rolling(21).std()
    vol_spike = volatility.iloc[-1] > volatility.iloc[-252:].quantile(0.8)
    
    # Bear market strength score
    bear_score = 0.0
    
    if price_below_200:
        bear_score += 0.3
    
    if returns_6m < -0.20:  # -20% in 6 months
        bear_score += 0.3
    elif returns_6m < -0.10:
        bear_score += 0.2
    
    if returns_3m < -0.15:  # -15% in 3 months
        bear_score += 0.2
    
    if death_cross:
        bear_score += 0.1
    
    if vol_spike:
        bear_score += 0.1
    
    return min(bear_score, 1.0)

def get_bear_market_strategy(bear_strength):
    """
    Determine strategy based on bear market strength
    """
    if bear_strength > 0.7:
        return "CASH_PRESERVATION"  # No new positions, exit existing
    elif bear_strength > 0.4:
        return "DEFENSIVE"  # Only high-conviction trades
    else:
        return "NORMAL"  # Standard strategy
