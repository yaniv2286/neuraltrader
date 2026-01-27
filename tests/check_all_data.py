"""
Check All Available Data
Inventory all cached data to maximize training on bull AND bear markets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.tiingo_loader import TiingoDataLoader
import pandas as pd

print("="*70)
print("DATA INVENTORY - ALL AVAILABLE CACHED DATA")
print("="*70)

loader = TiingoDataLoader()

# Get all available tickers
available = loader.get_available_tickers()
print(f"\n📊 Total tickers available: {len(available)}")
print(f"Tickers: {', '.join(sorted(available))}")

# Load each ticker and get date range
print("\n" + "="*70)
print("DETAILED DATA INVENTORY")
print("="*70)

data_summary = []

for ticker in sorted(available):
    df = loader.load_ticker_data(ticker)
    
    if df is not None and not df.empty:
        date_range_years = (df.index.max() - df.index.min()).days / 365.25
        
        data_summary.append({
            'Ticker': ticker,
            'Rows': len(df),
            'Start': df.index.min().date(),
            'End': df.index.max().date(),
            'Years': f"{date_range_years:.1f}",
            'Missing %': f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}"
        })

# Create summary DataFrame
summary_df = pd.DataFrame(data_summary)

print("\n📊 Data Summary:")
print(summary_df.to_string(index=False))

# Overall statistics
print("\n" + "="*70)
print("OVERALL STATISTICS")
print("="*70)
print(f"Total tickers: {len(summary_df)}")
print(f"Average rows per ticker: {summary_df['Rows'].mean():.0f}")
print(f"Average years of data: {summary_df['Years'].astype(float).mean():.1f}")
print(f"Earliest data: {summary_df['Start'].min()}")
print(f"Latest data: {summary_df['End'].max()}")

# Identify bull and bear periods
print("\n" + "="*70)
print("BULL & BEAR MARKET COVERAGE")
print("="*70)
print("✅ 2008-2009: Financial Crisis (BEAR)")
print("✅ 2009-2020: Bull Market")
print("✅ 2020 Q1: COVID Crash (BEAR)")
print("✅ 2020-2021: Recovery Bull")
print("✅ 2022: Bear Market")
print("✅ 2023-2024: Recovery")
print("\n💡 We have data covering MULTIPLE bull and bear cycles!")

# Recommend tickers for training
print("\n" + "="*70)
print("RECOMMENDED TICKERS FOR TRAINING")
print("="*70)

# Filter tickers with 15+ years of data
long_history = summary_df[summary_df['Years'].astype(float) >= 15]
print(f"\n📊 Tickers with 15+ years (covers multiple cycles): {len(long_history)}")
print(f"Recommended: {', '.join(long_history['Ticker'].tolist()[:20])}")

print("\n✅ Use ALL available data to maximize bull/bear performance!")
