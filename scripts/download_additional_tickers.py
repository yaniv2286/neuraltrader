"""
Download Additional Tickers from Tiingo
Expands dataset with international indices, crypto, currencies, commodities, futures, and more stocks
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time

# Tiingo API configuration
TIINGO_TOKEN = "72e14af10f4c32db4a7631275929617481aed281"
BASE_URL = "https://api.tiingo.com/tiingo/daily"

# Output directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
os.makedirs(CACHE_DIR, exist_ok=True)

# New tickers to download
NEW_TICKERS = {
    # International Indices
    'DAX': 'DAX (German Stock Index)',
    
    # Cryptocurrencies (via crypto tickers)
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    
    # Commodities & ETFs
    'GLD': 'Gold ETF',
    'SLV': 'Silver ETF',
    'USO': 'Oil ETF',
    'UNG': 'Natural Gas ETF',
    
    # Currency ETFs
    'UUP': 'US Dollar Index',
    'FXE': 'Euro Currency',
    'FXY': 'Japanese Yen',
    
    # Volatility & Futures
    'VIX': 'Volatility Index',
    'SPY': 'S&P 500 ETF',
    'QQQ': 'NASDAQ ETF',
    'IWM': 'Russell 2000 ETF',
    
    # Additional Tech Stocks
    'TXN': 'Texas Instruments',
    'QCOM': 'Qualcomm',
    'MU': 'Micron Technology',
    'AMAT': 'Applied Materials',
    'LRCX': 'Lam Research',
    'SNPS': 'Synopsys',
    'CDNS': 'Cadence Design',
    'MRVL': 'Marvell Technology',
    'ADI': 'Analog Devices',
    'MCHP': 'Microchip Technology',
    
    # Financial Services
    'BLK': 'BlackRock',
    'SCHW': 'Charles Schwab',
    'MS': 'Morgan Stanley',
    'C': 'Citigroup',
    'AXP': 'American Express',
    
    # Healthcare
    'UNH': 'UnitedHealth',
    'JNJ': 'Johnson & Johnson',
    'LLY': 'Eli Lilly',
    'ABBV': 'AbbVie',
    'TMO': 'Thermo Fisher',
    
    # Consumer
    'COST': 'Costco',
    'WMT': 'Walmart',
    'HD': 'Home Depot',
    'MCD': 'McDonald\'s',
    'SBUX': 'Starbucks',
    
    # Industrial
    'BA': 'Boeing',
    'CAT': 'Caterpillar',
    'GE': 'General Electric',
    'MMM': '3M',
    'HON': 'Honeywell',
}

def download_ticker(ticker, description):
    """
    Download historical data for a ticker from Tiingo
    """
    print(f"\n📊 Downloading {ticker} ({description})...")
    
    try:
        # Build request URL
        url = f"{BASE_URL}/{ticker}/prices"
        params = {
            'token': TIINGO_TOKEN,
            'startDate': '2004-01-01',
            'endDate': datetime.now().strftime('%Y-%m-%d'),
            'format': 'csv'
        }
        
        # Make request
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            # Save to file
            filename = f"{ticker.replace('-', '_')}_1d_20y_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(CACHE_DIR, filename)
            
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Verify data
            df = pd.read_csv(filepath)
            print(f"   ✅ Downloaded {len(df)} days of data")
            print(f"   📅 Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
            
            return True
            
        elif response.status_code == 404:
            print(f"   ⚠️ Ticker not found on Tiingo")
            return False
            
        else:
            print(f"   ❌ Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """
    Download all new tickers
    """
    print("="*70)
    print("DOWNLOADING ADDITIONAL TICKERS FROM TIINGO")
    print("="*70)
    
    print(f"\n🎯 Downloading {len(NEW_TICKERS)} new tickers")
    print(f"📁 Saving to: {CACHE_DIR}")
    
    success_count = 0
    fail_count = 0
    
    for ticker, description in NEW_TICKERS.items():
        if download_ticker(ticker, description):
            success_count += 1
        else:
            fail_count += 1
        
        # Rate limiting - wait 1 second between requests
        time.sleep(1)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n✅ Successfully downloaded: {success_count} tickers")
    print(f"❌ Failed: {fail_count} tickers")
    print(f"\n📊 Total tickers in cache: {len(os.listdir(CACHE_DIR))} files")
    
    # List all available tickers
    import glob
    cache_files = glob.glob(os.path.join(CACHE_DIR, '*.csv'))
    all_tickers = sorted(list(set([os.path.basename(f).split('_')[0].upper() for f in cache_files])))
    
    print(f"\n📋 All available tickers ({len(all_tickers)}):")
    print(', '.join(all_tickers))


if __name__ == "__main__":
    main()
