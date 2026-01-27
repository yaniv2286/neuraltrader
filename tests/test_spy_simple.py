#!/usr/bin/env python3
"""
Simple test for SPY with 20 years of data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_spy_data():
    """Load SPY 20-year data"""
    try:
        df = pd.read_csv('data/cache/tiingo/SPY_1d_20y.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print(f"✅ Loaded SPY data: {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
        return df
    except Exception as e:
        print(f"❌ Error loading SPY data: {e}")
        return None

def basic_analysis(df):
    """Basic analysis of SPY data"""
    print("\n=== SPY Basic Analysis ===")
    
    # Basic stats
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total trading days: {len(df)}")
    print(f"Price range: ${df['1d_close'].min():.2f} - ${df['1d_close'].max():.2f}")
    print(f"Current price: ${df['1d_close'].iloc[-1]:.2f}")
    print(f"Average daily volume: {df['1d_volume'].mean():,.0f}")
    
    # Returns
    df['daily_return'] = df['1d_close'].pct_change()
    print(f"Average daily return: {df['daily_return'].mean()*100:.3f}%")
    print(f"Daily volatility: {df['daily_return'].std()*100:.3f}%")
    
    # Annual returns
    annual_returns = df['daily_return'].groupby(df.index.year).mean() * 252
    print(f"Average annual return: {annual_returns.mean()*100:.2f}%")
    print(f"Annual volatility: {df['daily_return'].std()*np.sqrt(252)*100:.2f}%")
    
    # Best/worst years
    best_year = annual_returns.idxmax()
    worst_year = annual_returns.idxmin()
    print(f"Best year: {best_year} ({annual_returns.max()*100:.1f}%)")
    print(f"Worst year: {worst_year} ({annual_returns.min()*100:.1f}%)")
    
    return df

def simple_moving_average_test(df):
    """Simple moving average crossover test"""
    print("\n=== Simple MA Crossover Test ===")
    
    # Calculate moving averages
    df['MA_50'] = df['1d_close'].rolling(window=50).mean()
    df['MA_200'] = df['1d_close'].rolling(window=200).mean()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['MA_50'] > df['MA_200'], 'signal'] = 1  # Buy
    df.loc[df['MA_50'] < df['MA_200'], 'signal'] = -1  # Sell
    
    # Calculate returns
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    
    # Performance metrics
    total_return = (1 + df['strategy_return']).prod() - 1
    buy_hold_return = (df['1d_close'].iloc[-1] / df['1d_close'].iloc[0]) - 1
    
    print(f"Strategy total return: {total_return*100:.2f}%")
    print(f"Buy & hold return: {buy_hold_return*100:.2f}%")
    print(f"Strategy annual return: {((1+total_return)**(1/20)-1)*100:.2f}%")
    print(f"Buy & hold annual return: {((1+buy_hold_return)**(1/20)-1)*100:.2f}%")
    
    # Sharpe ratio
    sharpe_strategy = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)
    sharpe_buy_hold = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)
    
    print(f"Strategy Sharpe ratio: {sharpe_strategy:.2f}")
    print(f"Buy & hold Sharpe ratio: {sharpe_buy_hold:.2f}")
    
    # Win rate
    winning_days = (df['strategy_return'] > 0).sum()
    total_days = len(df[df['strategy_return'] != 0])
    win_rate = winning_days / total_days * 100 if total_days > 0 else 0
    print(f"Win rate: {win_rate:.1f}%")
    
    return df

def simple_volatility_test(df):
    """Simple volatility-based test"""
    print("\n=== Volatility Analysis ===")
    
    # Calculate rolling volatility
    df['rolling_vol'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
    
    # High/low volatility periods
    high_vol_threshold = df['rolling_vol'].quantile(0.8)
    low_vol_threshold = df['rolling_vol'].quantile(0.2)
    
    high_vol_periods = df[df['rolling_vol'] > high_vol_threshold]
    low_vol_periods = df[df['rolling_vol'] < low_vol_threshold]
    
    print(f"High volatility threshold: {high_vol_threshold*100:.1f}%")
    print(f"Low volatility threshold: {low_vol_threshold*100:.1f}%")
    print(f"High vol periods: {len(high_vol_periods)} days")
    print(f"Low vol periods: {len(low_vol_periods)} days")
    
    # Returns in different volatility regimes
    if len(high_vol_periods) > 0:
        high_vol_return = high_vol_periods['daily_return'].mean() * 100
        print(f"Average return in high vol: {high_vol_return:.3f}%")
    
    if len(low_vol_periods) > 0:
        low_vol_return = low_vol_periods['daily_return'].mean() * 100
        print(f"Average return in low vol: {low_vol_return:.3f}%")
    
    return df

def main():
    """Main test function"""
    print("🚀 Starting SPY Simple Test")
    print("=" * 50)
    
    # Load data
    df = load_spy_data()
    if df is None:
        return
    
    # Basic analysis
    df = basic_analysis(df)
    
    # Simple MA test
    df = simple_moving_average_test(df)
    
    # Volatility test
    df = simple_volatility_test(df)
    
    print("\n✅ SPY Simple Test Complete!")
    
    # Save results summary
    summary = {
        'total_days': len(df),
        'start_date': df.index.min().date(),
        'end_date': df.index.max().date(),
        'current_price': df['1d_close'].iloc[-1],
        'avg_daily_return': df['daily_return'].mean(),
        'daily_volatility': df['daily_return'].std(),
        'total_return': (df['1d_close'].iloc[-1] / df['1d_close'].iloc[0]) - 1
    }
    
    print(f"\n📊 Summary saved to test results")
    return summary

if __name__ == "__main__":
    results = main()
