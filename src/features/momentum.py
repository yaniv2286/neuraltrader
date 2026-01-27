import pandas as pd
import numpy as np

def add_momentum_features(df, price_col='close'):
    """Add momentum/trend features to help with market timing"""
    # 200-day SMA for long-term trend
    df['sma_200'] = df[price_col].rolling(window=200).mean()
    df['above_sma_200'] = (df[price_col] > df['sma_200']).astype(int)
    
    # 50-day SMA for medium-term trend
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    df['above_sma_50'] = (df[price_col] > df['sma_50']).astype(int)
    
    # RSI for momentum strength
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_above_50'] = (df['rsi'] > 50).astype(int)
    
    # MACD for trend direction
    exp1 = df[price_col].ewm(span=12).mean()
    exp2 = df[price_col].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Combined trend signal (1 = strong uptrend, 0 = neutral/downtrend)
    df['trend_signal'] = (
        df['above_sma_200'].astype(int) + 
        df['above_sma_50'].astype(int) + 
        df['rsi_above_50'].astype(int) + 
        df['macd_bullish'].astype(int)
    ) / 4.0
    
    return df

def should_trade(row, min_trend_strength=0.75):
    """Determine if we should trade based on trend"""
    return row.get('trend_signal', 0) >= min_trend_strength
