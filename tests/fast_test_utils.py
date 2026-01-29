#!/usr/bin/env python3
"""
Fast Test Utilities - Smart 5-ticker selection for quick testing
Provides diverse, high-quality ticker selection for fast development testing
"""

import os
import glob
import pandas as pd
import random
from typing import List, Dict, Set

def get_all_available_tickers() -> List[str]:
    """Get all available tickers from cache"""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    
    available_tickers = []
    for file in cache_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            available_tickers.append(ticker)
    
    return sorted(list(set(available_tickers)))

def check_ticker_quality(ticker: str) -> bool:
    """Check if ticker has sufficient quality data"""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    cache_files = glob.glob(os.path.join(cache_dir, f'{ticker}_*.csv'))
    
    if not cache_files:
        return False
    
    try:
        # Load and check data quality
        df = pd.read_csv(cache_files[0])
        
        # Minimum quality checks
        if len(df) < 1000:  # At least 1000 rows
            return False
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for too many missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 5:  # More than 5% missing
            return False
        
        return True
        
    except Exception:
        return False

def get_quality_tickers() -> List[str]:
    """Get all tickers that pass quality checks"""
    all_tickers = get_all_available_tickers()
    quality_tickers = []
    
    print(f"🔍 Checking quality for {len(all_tickers)} tickers...")
    
    for ticker in all_tickers:
        if check_ticker_quality(ticker):
            quality_tickers.append(ticker)
    
    print(f"   ✅ {len(quality_tickers)} tickers passed quality checks")
    return quality_tickers

def get_fast_test_tickers(strategic: bool = True) -> List[str]:
    """Get 5 diverse tickers for fast testing"""
    
    quality_tickers = get_quality_tickers()
    
    if len(quality_tickers) < 5:
        print(f"⚠️ Only {len(quality_tickers)} quality tickers available")
        return quality_tickers
    
    if strategic:
        return get_strategic_selection(quality_tickers)
    else:
        return get_random_selection(quality_tickers)

def get_strategic_selection(quality_tickers: List[str]) -> List[str]:
    """Get strategic selection - 1 ticker from each major sector"""
    
    # Define sector mappings
    sectors = {
        'tech': ['MSFT', 'GOOGL', 'NVDA', 'AMD', 'AAPL', 'META', 'AMZN', 'TSLA', 'NFLX', 'INTC'],
        'finance': ['JPM', 'BAC', 'V', 'MA', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'C'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'ABT', 'LLY', 'TMO', 'CVS', 'DHR'],
        'consumer': ['WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'KO', 'PEP', 'LOW', 'TGT'],
        'energy': ['XOM', 'CVX', 'COP', 'BP', 'SHEL'],
        'industrial': ['CAT', 'BA', 'GE', 'HON', 'MMM', 'DE', 'UPS', 'RTX'],
        'etf': ['SPY', 'QQQ', 'VTI', 'VOO', 'IWM', 'GLD', 'SLV', 'AGG'],
        'crypto': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'COIN']
    }
    
    selected = []
    
    # Try to get 1 from each major sector
    priority_sectors = ['tech', 'finance', 'healthcare', 'consumer', 'energy']
    
    for sector in priority_sectors:
        if sector in sectors:
            for ticker in sectors[sector]:
                if ticker in quality_tickers and ticker not in selected:
                    selected.append(ticker)
                    break
        
        if len(selected) >= 5:
            break
    
    # If we don't have 5 yet, add more from remaining sectors
    if len(selected) < 5:
        remaining_sectors = ['industrial', 'etf', 'crypto']
        for sector in remaining_sectors:
            if sector in sectors:
                for ticker in sectors[sector]:
                    if ticker in quality_tickers and ticker not in selected:
                        selected.append(ticker)
                        break
            
            if len(selected) >= 5:
                break
    
    # If still don't have 5, add random quality tickers
    if len(selected) < 5:
        remaining = [t for t in quality_tickers if t not in selected]
        selected.extend(random.sample(remaining, min(5 - len(selected), len(remaining))))
    
    return selected[:5]

def get_random_selection(quality_tickers: List[str]) -> List[str]:
    """Get random selection of 5 quality tickers"""
    return random.sample(quality_tickers, min(5, len(quality_tickers)))

def get_diverse_sample(sample_size: int = 5) -> List[str]:
    """Get diverse sample of tickers across sectors"""
    quality_tickers = get_quality_tickers()
    
    if len(quality_tickers) <= sample_size:
        return quality_tickers
    
    # Define sector mappings
    sectors = {
        'tech': ['MSFT', 'GOOGL', 'NVDA', 'AMD', 'AAPL', 'META', 'AMZN', 'TSLA', 'NFLX', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'CRM'],
        'finance': ['JPM', 'BAC', 'V', 'MA', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'C', 'COF', 'PRU', 'MET', 'AIG'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'ABT', 'LLY', 'TMO', 'CVS', 'DHR', 'MDT', 'ISRG', 'REGN', 'BIIB'],
        'consumer': ['WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'KO', 'PEP', 'LOW', 'TGT', 'AMZN', 'TSLA'],
        'energy': ['XOM', 'CVX', 'COP', 'BP', 'SHEL'],
        'industrial': ['CAT', 'BA', 'GE', 'HON', 'MMM', 'DE', 'UPS', 'RTX', 'GWW', 'FAST'],
        'etf': ['SPY', 'QQQ', 'VTI', 'VOO', 'IWM', 'GLD', 'SLV', 'AGG', 'ACWI', 'EFA', 'VT'],
        'crypto': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'COIN']
    }
    
    selected = []
    
    # Try to get balanced representation
    for sector, tickers in sectors.items():
        for ticker in tickers:
            if ticker in quality_tickers and ticker not in selected:
                selected.append(ticker)
                break
        
        if len(selected) >= sample_size:
            break
    
    # If we don't have enough, add more from remaining quality tickers
    if len(selected) < sample_size:
        remaining = [t for t in quality_tickers if t not in selected]
        needed = sample_size - len(selected)
        selected.extend(random.sample(remaining, min(needed, len(remaining))))
    
    return selected[:sample_size]

def print_fast_test_info(tickers: List[str]):
    """Print information about the fast test selection"""
    print(f"\n🚀 FAST TEST SELECTION ({len(tickers)} tickers)")
    print("="*50)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"   {i}. {ticker}")
    
    # Check sector diversity
    sectors = {
        'tech': ['MSFT', 'GOOGL', 'NVDA', 'AMD', 'AAPL', 'META', 'AMZN', 'TSLA', 'NFLX', 'INTC'],
        'finance': ['JPM', 'BAC', 'V', 'MA', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'C'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'ABT', 'LLY', 'TMO', 'CVS', 'DHR'],
        'consumer': ['WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'KO', 'PEP', 'LOW', 'TGT'],
        'energy': ['XOM', 'CVX', 'COP', 'BP', 'SHEL'],
        'industrial': ['CAT', 'BA', 'GE', 'HON', 'MMM', 'DE', 'UPS', 'RTX'],
        'etf': ['SPY', 'QQQ', 'VTI', 'VOO', 'IWM', 'GLD', 'SLV', 'AGG'],
        'crypto': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'COIN']
    }
    
    sector_count = {}
    for ticker in tickers:
        for sector, sector_tickers in sectors.items():
            if ticker in sector_tickers:
                sector_count[sector] = sector_count.get(sector, 0) + 1
                break
    
    print(f"\n📊 Sector Coverage:")
    for sector, count in sector_count.items():
        print(f"   {sector.capitalize()}: {count} ticker(s)")
    
    print(f"\n✅ Benefits:")
    print(f"   • Fast testing ({len(tickers)} vs 130+ tickers)")
    print(f"   • Diverse sector representation")
    print(f"   • High-quality data only")
    print(f"   • Consistent and reproducible")

if __name__ == "__main__":
    # Test the fast selection
    print("🔧 TESTING FAST TICKER SELECTION")
    print("="*50)
    
    fast_tickers = get_fast_test_tickers()
    print_fast_test_info(fast_tickers)
