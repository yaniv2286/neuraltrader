import pandas as pd
import numpy as np

def detect_market_regime(df, price_col='close', lookback=252):
    """
    Detect market regime: BULL, BEAR, or SIDEWAYS
    Returns regime signal and strength
    """
    # Long-term trend (200-day SMA)
    sma_200 = df[price_col].rolling(window=200).mean()
    sma_50 = df[price_col].rolling(window=50).mean()
    
    # Regime classification
    price_above_200 = df[price_col] > sma_200
    price_above_50 = df[price_col] > sma_50
    sma_50_above_200 = sma_50 > sma_200
    
    # Momentum
    returns_1m = df[price_col].pct_change(21)
    returns_3m = df[price_col].pct_change(63)
    
    # Volatility
    volatility = df[price_col].pct_change().rolling(21).std()
    
    regime = pd.Series(index=df.index, dtype='object')
    regime_strength = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 200:
            regime.iloc[i] = 'SIDEWAYS'
            regime_strength.iloc[i] = 0.5
            continue
            
        # Bull market conditions
        if (price_above_200.iloc[i] and 
            price_above_50.iloc[i] and 
            sma_50_above_200.iloc[i] and
            returns_3m.iloc[i] > 0.05):
            regime.iloc[i] = 'BULL'
            regime_strength.iloc[i] = min(1.0, (returns_3m.iloc[i] / 0.10))
        
        # Bear market conditions
        elif (not price_above_200.iloc[i] and 
              not price_above_50.iloc[i] and
              returns_3m.iloc[i] < -0.05):
            regime.iloc[i] = 'BEAR'
            regime_strength.iloc[i] = min(1.0, abs(returns_3m.iloc[i] / 0.10))
        
        # Sideways/neutral
        else:
            regime.iloc[i] = 'SIDEWAYS'
            regime_strength.iloc[i] = 0.5
    
    return regime, regime_strength

def get_regime_adjusted_thresholds(regime, regime_strength, base_buy=0.01, base_sell=0.01):
    """
    Adjust thresholds based on market regime
    """
    buy_adj = base_buy.copy() if hasattr(base_buy, 'copy') else base_buy
    sell_adj = base_sell.copy() if hasattr(base_sell, 'copy') else base_sell
    
    if regime == 'BULL':
        # More aggressive in bull markets
        buy_adj = base_buy * (1.0 + 0.5 * regime_strength)
        sell_adj = base_sell * (1.0 + 0.3 * regime_strength)
    elif regime == 'BEAR':
        # Very conservative in bear markets
        buy_adj = base_buy * (2.0 + 2.0 * regime_strength)
        sell_adj = base_sell * 0.5
    else:  # SIDEWAYS
        # Neutral approach
        buy_adj = base_buy * 1.2
        sell_adj = base_sell * 1.2
    
    return buy_adj, sell_adj
