#!/usr/bin/env python3
"""
Smart Download Test Script
Tests API status and downloads a few tickers safely
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time
import glob

# Tiingo API configuration
TIINGO_TOKEN = "72e14af10f4c32db4a7631275929617481aed281"
BASE_URL = "https://api.tiingo.com/tiingo/daily"

# Output directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
os.makedirs(CACHE_DIR, exist_ok=True)

def test_api_status():
    """Test if API is working with a simple request"""
    print("🔍 Testing API status...")
    
    # Test with a simple request
    test_url = f"{BASE_URL}/SPY/prices"
    params = {
        'token': TIINGO_TOKEN,
        'startDate': '2026-01-26',
        'endDate': '2026-01-27',
        'format': 'csv',
        'resampleFreq': 'daily'
    }
    
    try:
        response = requests.get(test_url, params=params, timeout=10)
        
        if response.status_code == 200:
            if len(response.text) > 100 and 'date' in response.text:
                print("✅ API is working!")
                return True
            else:
                print("❌ API returned invalid data")
                return False
        else:
            print(f"❌ API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def get_target_tickers():
    """Get the complete target list of 116 tickers"""
    ticker_list_path = os.path.join(CACHE_DIR, 'info', 'lists', 'tickers_to_download.txt')
    
    tickers = []
    with open(ticker_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                tickers.append(line)
    return sorted(tickers)

def get_existing_tickers():
    """Get tickers we already have"""
    cache_files = glob.glob(os.path.join(CACHE_DIR, '*.csv'))
    existing_tickers = set()
    
    for file in cache_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            existing_tickers.add(ticker)
    
    return existing_tickers

def download_ticker_safe(ticker):
    """Download ticker with safety checks"""
    print(f"📊 Testing {ticker}...")
    
    # Format ticker for API
    api_ticker = ticker
    if ticker.endswith('USD'):
        api_ticker = ticker.replace('-', '_')
    
    # Build URL for recent data (small request)
    url = f"{BASE_URL}/{api_ticker}/prices"
    params = {
        'token': TIINGO_TOKEN,
        'startDate': '2026-01-26',
        'endDate': '2026-01-27',
        'format': 'csv',
        'resampleFreq': 'daily'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            if len(response.text) > 100 and 'date' in response.text:
                print(f"   ✅ {ticker}: API working")
                return True
            else:
                print(f"   ❌ {ticker}: Invalid response")
                return False
        else:
            print(f"   ❌ {ticker}: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ {ticker}: Error {e}")
        return False

def main():
    """Smart test and download"""
    print("="*70)
    print("SMART DOWNLOAD TEST - API STATUS & SAMPLE DOWNLOAD")
    print("="*70)
    
    # Test API first
    if not test_api_status():
        print("\n❌ API is not working - cannot proceed")
        return
    
    # Get ticker status
    target_tickers = get_target_tickers()
    existing_tickers = get_existing_tickers()
    
    print(f"\n📊 Target tickers: {len(target_tickers)}")
    print(f"✅ Have tickers: {len(existing_tickers)}")
    print(f"❌ Missing tickers: {len(target_tickers) - len(existing_tickers)}")
    
    # Show some existing tickers
    print(f"\n✅ Sample existing tickers:")
    for ticker in sorted(list(existing_tickers))[:5]:
        print(f"   {ticker}: Data already exists")
    
    # Find some missing tickers to test
    missing_tickers = [t for t in target_tickers if t not in existing_tickers]
    
    if len(missing_tickers) == 0:
        print("\n🎉 All tickers already downloaded!")
        return
    
    print(f"\n🎯 Testing {min(5, len(missing_tickers))} missing tickers:")
    test_tickers = missing_tickers[:5]
    
    success_count = 0
    for ticker in test_tickers:
        if download_ticker_safe(ticker):
            success_count += 1
        time.sleep(1)  # Rate limiting
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Successful: {success_count}/{len(test_tickers)}")
    print(f"   ❌ Failed: {len(test_tickers) - success_count}/{len(test_tickers)}")
    
    if success_count == len(test_tickers):
        print(f"\n🎉 API is working well!")
        print(f"🚀 Ready to download remaining {len(missing_tickers)} tickers")
        print(f"💡 Run: uv run python scripts/download_missing_only.py")
    else:
        print(f"\n⚠️ API having issues")
        print(f"💡 Wait for better API status before full download")

if __name__ == "__main__":
    main()
