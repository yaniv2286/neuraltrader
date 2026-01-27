import pandas as pd
import numpy as np

# Sector ETFs for rotation
SECTOR_ETFS = {
    'TECH': 'XLK',    # Technology
    'HEALTH': 'XLV',  # Healthcare
    'FINANCE': 'XLF', # Financial
    'CONSUMER': 'XLY', # Consumer Discretionary
    'UTILITIES': 'XLU', # Utilities
    'ENERGY': 'XLE',   # Energy
    'MATERIALS': 'XLB', # Materials
    'INDUSTRIAL': 'XLI', # Industrial
    'REITS': 'XLRE',   # Real Estate
    'STAPLES': 'XLP',  # Consumer Staples
    'COMMUNICATION': 'XLC', # Communication
}

def get_sector_momentum(sector_data, lookback=63):
    """
    Calculate momentum scores for each sector
    Returns dict of sector -> momentum_score
    """
    momentum_scores = {}
    
    for sector, ticker in SECTOR_ETFS.items():
        if ticker in sector_data:
            df = sector_data[ticker]
            if len(df) > lookback:
                # Calculate momentum (risk-adjusted return)
                returns = df['close'].pct_change().dropna()
                total_return = (df['close'].iloc[-1] / df['close'].iloc[-lookback]) - 1
                volatility = returns.tail(lookback).std()
                
                # Sharpe-like momentum score
                momentum_score = total_return / (volatility + 1e-6)
                momentum_scores[sector] = momentum_score
    
    return momentum_scores

def get_sector_allocation(regime, momentum_scores, bear_strength=0):
    """
    Determine sector allocation based on market regime and momentum
    Returns dict of sector -> allocation_weight
    """
    allocation = {}
    
    if regime == "BEAR" or bear_strength > 0.6:
        # Bear market: Defensive sectors
        defensive_sectors = ['HEALTH', 'UTILITIES', 'STAPLES', 'CONSUMER']
        for sector in defensive_sectors:
            if sector in momentum_scores and momentum_scores[sector] > 0:
                allocation[sector] = 0.25  # Equal weight among defensive
    
    elif regime == "BULL":
        # Bull market: Growth sectors
        growth_sectors = ['TECH', 'FINANCE', 'CONSUMER', 'INDUSTRIAL']
        # Sort by momentum
        sorted_sectors = sorted(
            [(s, momentum_scores.get(s, 0)) for s in growth_sectors],
            key=lambda x: x[1],
            reverse=True
        )
        # Top 4 get weights
        for i, (sector, score) in enumerate(sorted_sectors[:4]):
            if score > 0:
                allocation[sector] = 0.25
    
    else:  # SIDEWAYS
        # Balanced approach
        all_sectors = list(SECTOR_ETFS.keys())
        sorted_sectors = sorted(
            [(s, momentum_scores.get(s, 0)) for s in all_sectors],
            key=lambda x: x[1],
            reverse=True
        )
        # Top 6 get weights
        for i, (sector, score) in enumerate(sorted_sectors[:6]):
            if score > 0:
                allocation[sector] = 0.167  # ~1/6
    
    return allocation

def rotate_sectors(current_allocation, new_allocation, rebalance_threshold=0.1):
    """
    Determine if rebalancing is needed
    Returns list of sectors to buy/sell
    """
    trades = []
    
    for sector in set(current_allocation.keys()) | set(new_allocation.keys()):
        current_weight = current_allocation.get(sector, 0)
        new_weight = new_allocation.get(sector, 0)
        
        if abs(new_weight - current_weight) > rebalance_threshold:
            if new_weight > current_weight:
                trades.append(('BUY', sector, new_weight - current_weight))
            else:
                trades.append(('SELL', sector, current_weight - new_weight))
    
    return trades
