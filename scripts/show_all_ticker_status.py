#!/usr/bin/env python3
"""
Show All Ticker Status Script
Displays status for all 116 tickers with "Data already exists" messages
"""

import os
import glob

def get_target_tickers():
    """Get the complete target list of 116 tickers"""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    ticker_list_path = os.path.join(cache_dir, 'info', 'lists', 'tickers_to_download.txt')
    
    tickers = []
    with open(ticker_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                tickers.append(line)
    return sorted(tickers)

def get_existing_tickers():
    """Get tickers we already have"""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    existing_tickers = set()
    
    for file in cache_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            existing_tickers.add(ticker)
    
    return existing_tickers

def main():
    """Show status for all tickers"""
    print("="*70)
    print("TICKER STATUS REPORT - ALL 116 TICKERS")
    print("="*70)
    
    target_tickers = get_target_tickers()
    existing_tickers = get_existing_tickers()
    
    print(f"\n📊 Total Target Tickers: {len(target_tickers)}")
    print(f"✅ Have Data: {len(existing_tickers)}")
    print(f"❌ Missing Data: {len(target_tickers) - len(existing_tickers)}")
    
    print(f"\n📋 DETAILED STATUS:")
    print("="*50)
    
    exists_count = 0
    missing_count = 0
    
    for ticker in target_tickers:
        if ticker in existing_tickers:
            print(f"✅ {ticker}: Data already exists")
            exists_count += 1
        else:
            print(f"❌ {ticker}: Missing data")
            missing_count += 1
    
    print("="*50)
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Data exists: {exists_count} tickers")
    print(f"   ❌ Missing data: {missing_count} tickers")
    print(f"   📈 Progress: {exists_count}/{len(target_tickers)} ({exists_count/len(target_tickers):.1%})")
    
    if missing_count > 0:
        print(f"\n🎯 Missing tickers to download:")
        missing_tickers = [t for t in target_tickers if t not in existing_tickers]
        for ticker in missing_tickers:
            print(f"   {ticker}")
    else:
        print(f"\n🎉 All tickers downloaded!")

if __name__ == "__main__":
    main()
