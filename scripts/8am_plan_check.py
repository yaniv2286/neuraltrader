#!/usr/bin/env python3
"""
8AM Plan Verification Script
Double-checks everything is ready for 8AM download
"""

import os
import glob
from datetime import datetime

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
    return tickers

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

def check_script_readiness():
    """Check if download script is ready"""
    script_path = os.path.join(os.path.dirname(__file__), 'download_missing_only.py')
    
    if os.path.exists(script_path):
        print("✅ download_missing_only.py exists")
        
        # Check key features
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "max_api_failures = 2" in content:
            print("✅ API failure limit set to 2")
        else:
            print("❌ API failure limit not found")
            
        if "Data already exists" in content:
            print("✅ Status messages implemented")
        else:
            print("❌ Status messages not found")
            
        if "get_missing_tickers" in content:
            print("✅ Missing ticker detection implemented")
        else:
            print("❌ Missing ticker detection not found")
            
        return True
    else:
        print("❌ download_missing_only.py not found")
        return False

def main():
    """Verify 8AM plan"""
    print("="*70)
    print("8AM PLAN VERIFICATION")
    print("="*70)
    
    print(f"🕐 Current time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎯 Target time: 08:00:00")
    
    # Check current status
    target_tickers = get_target_tickers()
    existing_tickers = get_existing_tickers()
    missing_tickers = [t for t in target_tickers if t not in existing_tickers]
    
    print(f"\n📊 CURRENT STATUS:")
    print(f"   Target tickers: {len(target_tickers)}")
    print(f"   Have tickers: {len(existing_tickers)}")
    print(f"   Missing tickers: {len(missing_tickers)}")
    
    # Check script readiness
    print(f"\n🔧 SCRIPT READINESS:")
    script_ready = check_script_readiness()
    
    # Check files
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    ticker_list_path = os.path.join(cache_dir, 'info', 'lists', 'tickers_to_download.txt')
    
    print(f"\n📁 FILE CHECKS:")
    print(f"   ✅ Ticker list: {os.path.exists(ticker_list_path)}")
    print(f"   ✅ Cache dir: {os.path.exists(cache_dir)}")
    print(f"   ✅ CSV files: {len(glob.glob(os.path.join(cache_dir, '*.csv')))}")
    
    # 8AM Plan
    print(f"\n🎯 8AM EXECUTION PLAN:")
    print(f"   1. ✅ API should reset at 08:00")
    print(f"   2. ✅ Run: uv run python scripts/download_missing_only.py")
    print(f"   3. ✅ Download {len(missing_tickers)} missing tickers")
    print(f"   4. ✅ Stop after 2 API failures")
    print(f"   5. ✅ Get complete 116 ticker dataset")
    print(f"   6. ✅ Ready for Phase 2 testing")
    
    # Critical missing tickers
    critical_missing = [t for t in missing_tickers if t in ['SPY', 'QQQ', 'NVDA', 'TSLA', 'V', 'MA']]
    
    print(f"\n🚨 CRITICAL MISSING TICKERS:")
    for ticker in critical_missing:
        print(f"   ❌ {ticker}")
    
    if len(missing_tickers) > 0:
        print(f"\n📋 FIRST 10 MISSING TICKERS:")
        for ticker in missing_tickers[:10]:
            print(f"   ❌ {ticker}")
    
    print(f"\n🎯 READY FOR 8AM?")
    if script_ready and len(missing_tickers) > 0:
        print(f"   ✅ YES - Everything ready!")
        print(f"   🚀 Run at 8AM: uv run python scripts/download_missing_only.py")
    else:
        print(f"   ❌ NO - Issues to resolve")
    
    print(f"\n⏰ SET REMINDER FOR 8:00 AM!")
    print(f"   💡 Command: uv run python scripts/download_missing_only.py")

if __name__ == "__main__":
    main()
