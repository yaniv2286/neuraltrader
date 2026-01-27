import pandas as pd
import numpy as np
import yfinance as yf

def fetch_vix_data(start_date, end_date):
    """Fetch VIX data"""
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        return vix
    except:
        # If VIX fails, create synthetic from SPY
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        returns = spy['Close'].pct_change()
        # Synthetic VIX = rolling std * sqrt(252) * 100
        synthetic_vix = returns.rolling(21).std() * np.sqrt(252) * 100
        synthetic_vix = pd.DataFrame({'Close': synthetic_vix}, index=spy.index)
        return synthetic_vix

def fetch_interest_rates(start_date, end_date):
    """Fetch 10-year Treasury yield"""
    try:
        # Try 10-year Treasury yield (^TNX)
        rates = yf.download('^TNX', start=start_date, end=end_date, progress=False)
        return rates
    except:
        # Fallback: create synthetic rates based on market conditions
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Default 2% with some variation
        rates = pd.DataFrame({
            'Close': 2.0 + np.random.normal(0, 0.5, len(dates))
        }, index=dates)
        return rates

def calculate_macro_signals(df, vix_data, rates_data):
    """
    Calculate macro signals for market timing
    Returns dict with macro indicators
    """
    signals = {}
    
    # VIX signals
    if vix_data is not None and len(vix_data) > 0:
        vix = vix_data['Close'].reindex(df.index, method='ffill')
        
        # VIX percentiles (historical context)
        vix_percentile = vix.rolling(252).rank(pct=True)
        
        # High VIX (> 80th percentile) = fear
        signals['vix_fear'] = (vix_percentile > 0.8).astype(int)
        
        # Low VIX (< 20th percentile) = complacency
        signals['vix_complacency'] = (vix_percentile < 0.2).astype(int)
        
        # VIX level
        signals['vix_level'] = vix
        
        # VIX change (momentum)
        signals['vix_change'] = vix.pct_change(5)  # 5-day change
    
    # Interest rate signals
    if rates_data is not None and len(rates_data) > 0:
        rates = rates_data['Close'].reindex(df.index, method='ffill')
        
        # Rate trend
        rates_ma = rates.rolling(63).mean()
        signals['rates_rising'] = (rates > rates_ma).astype(int)
        
        # Rate level
        signals['rate_level'] = rates
        
        # Rate change
        signals['rate_change'] = rates.pct_change(21)  # Monthly change
        
        # High rates (> 4%) = headwind for stocks
        signals['high_rates'] = (rates > 4.0).astype(int)
        
        # Rate cuts/bullish
        signals['rate_cutting'] = (rates < rates_ma).astype(int)
    
    # Combined macro risk score
    risk_factors = []
    if 'vix_fear' in signals:
        risk_factors.append(signals['vix_fear'])
    if 'high_rates' in signals:
        risk_factors.append(signals['high_rates'])
    
    if risk_factors:
        signals['macro_risk_score'] = sum(risk_factors) / len(risk_factors)
    else:
        signals['macro_risk_score'] = 0.5  # Neutral
    
    # Macro regime
    macro_regime = 'NEUTRAL'
    if signals.get('vix_fear', 0) > 0:
        macro_regime = 'RISK_OFF'
    elif signals.get('vix_complacency', 0) > 0 and signals.get('rate_cutting', 0) > 0:
        macro_regime = 'RISK_ON'
    
    signals['macro_regime'] = macro_regime
    
    return signals

def get_macro_adjusted_allocation(base_allocation, macro_signals):
    """
    Adjust sector allocation based on macro conditions
    """
    adjusted = base_allocation.copy()
    
    # High VIX: Reduce risk, increase defensives
    if macro_signals.get('vix_fear', 0) > 0:
        # Reduce tech, increase utilities/staples
        if 'TECH' in adjusted:
            adjusted['TECH'] *= 0.5
        if 'UTILITIES' in adjusted:
            adjusted['UTILITIES'] *= 1.5
        if 'STAPLES' in adjusted:
            adjusted['STAPLES'] *= 1.5
    
    # High rates: Reduce rate-sensitive sectors
    if macro_signals.get('high_rates', 0) > 0:
        if 'FINANCE' in adjusted:
            adjusted['FINANCE'] *= 0.7
        if 'REITS' in adjusted:
            adjusted['REITS'] *= 0.6
        if 'UTILITIES' in adjusted:
            adjusted['UTILITIES'] *= 1.3  # Utilities benefit from high rates
    
    # Normalize weights
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v/total for k, v in adjusted.items()}
    
    return adjusted
