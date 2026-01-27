import pandas as pd
import numpy as np

def is_bull_market(df, price_col='close', window=63):
    """
    Simple bull market detector
    Returns True if in a bull market (strong uptrend)
    """
    if len(df) < window:
        return True
    
    # Check if price is above moving averages
    sma_20 = df[price_col].rolling(window=20).mean()
    sma_50 = df[price_col].rolling(window=50).mean()
    
    # Recent momentum
    recent_return = df[price_col].pct_change(window)
    
    # Bull market conditions
    above_sma_20 = df[price_col].iloc[-1] > sma_20.iloc[-1]
    above_sma_50 = df[price_col].iloc[-1] > sma_50.iloc[-1]
    positive_momentum = recent_return.iloc[-1] > 0.02  # 2% in last period
    
    return above_sma_20 and above_sma_50 and positive_momentum

def get_adaptive_thresholds(df, base_buy=0.01, base_sell=0.01):
    """
    Adjust thresholds based on market volatility
    """
    # Calculate recent volatility
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(21).std().iloc[-1]
    
    # Adjust thresholds based on volatility
    if volatility > 0.02:  # High volatility
        return base_buy * 1.5, base_sell * 1.5
    elif volatility < 0.01:  # Low volatility
        return base_buy * 0.7, base_sell * 0.7
    else:
        return base_buy, base_sell
