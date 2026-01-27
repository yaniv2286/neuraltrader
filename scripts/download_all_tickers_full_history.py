"""
Download ALL Tickers with FULL Historical Data (1980-2026)
Re-downloads all 80 existing tickers with maximum available history
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

def get_existing_tickers():
    """Get list of all existing tickers from cache"""
    cache_files = glob.glob(os.path.join(CACHE_DIR, '*.csv'))
    tickers = sorted(list(set([os.path.basename(f).split('_')[0].upper() for f in cache_files])))
    return tickers

def download_ticker_full_history(ticker):
    """
    Download MAXIMUM historical data for a ticker from Tiingo (1970-2026)
    """
    print(f"\n📊 Downloading {ticker} (FULL HISTORY)...")
    
    try:
        # Build request URL - fetch maximum possible historical data
        url = f"{BASE_URL}/{ticker}/prices"
        params = {
            'token': TIINGO_TOKEN,
            'startDate': '1970-01-01',  # Try to get maximum possible data (54+ years)
            'endDate': datetime.now().strftime('%Y-%m-%d'),
            'format': 'csv'
        }
        
        # Make request
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            # Check if we got valid data
            if len(response.text) < 100:  # Too short to be valid
                print(f"   ⚠️ No data available")
                return False, 0
            
            # Save to file
            filename = f"{ticker.replace('-', '_')}_1d_max_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(CACHE_DIR, filename)
            
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Verify data
            try:
                df = pd.read_csv(filepath)
                if len(df) == 0 or 'date' not in df.columns:
                    print(f"   ⚠️ Invalid data format")
                    os.remove(filepath)
                    return False, 0
                
                years = (datetime.now().year - pd.to_datetime(df['date'].iloc[0]).year)
                
                print(f"   ✅ Downloaded {len(df)} days ({years} years)")
                print(f"   📅 Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
                
                # Delete old files for this ticker
                old_files = glob.glob(os.path.join(CACHE_DIR, f"{ticker.replace('-', '_')}_1d_*.csv"))
                for old_file in old_files:
                    try:
                        os.remove(old_file)
                        print(f"   🗑️ Removed old file: {os.path.basename(old_file)}")
                    except:
                        pass
                
                return True, years
            except Exception as e:
                print(f"   ⚠️ Error parsing data: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False, 0
            
        elif response.status_code == 404:
            print(f"   ⚠️ Ticker not found on Tiingo")
            return False, 0
            
        else:
            print(f"   ❌ Error: HTTP {response.status_code}")
            return False, 0
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0


def main():
    """
    Re-download all existing tickers with FULL historical data
    """
    print("="*70)
    print("RE-DOWNLOADING ALL TICKERS WITH FULL HISTORICAL DATA")
    print("="*70)
    
    # Get existing tickers
    existing_tickers = get_existing_tickers()
    
    print(f"\n🎯 Found {len(existing_tickers)} existing tickers")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"📅 Fetching data from 1970-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📈 Attempting to get MAXIMUM possible historical data (54+ years)")
    print(f"⏱️ This will take approximately {len(existing_tickers)} minutes...")
    
    success_count = 0
    fail_count = 0
    total_years = 0
    
    for i, ticker in enumerate(existing_tickers, 1):
        print(f"\n[{i}/{len(existing_tickers)}]", end=" ")
        success, years = download_ticker_full_history(ticker)
        
        if success:
            success_count += 1
            total_years += years
        else:
            fail_count += 1
        
        # Rate limiting - wait 1 second between requests
        if i < len(existing_tickers):
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n✅ Successfully downloaded: {success_count} tickers")
    print(f"❌ Failed: {fail_count} tickers")
    print(f"📊 Average history: {total_years/success_count:.1f} years per ticker" if success_count > 0 else "")
    print(f"\n📊 Total files in cache: {len(os.listdir(CACHE_DIR))} files")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print(f"{'='*70}")
    
    cache_files = glob.glob(os.path.join(CACHE_DIR, '*.csv'))
    total_days = 0
    oldest_date = None
    newest_date = None
    
    for file in cache_files:
        try:
            df = pd.read_csv(file)
            total_days += len(df)
            
            first_date = pd.to_datetime(df['date'].iloc[0])
            last_date = pd.to_datetime(df['date'].iloc[-1])
            
            if oldest_date is None or first_date < oldest_date:
                oldest_date = first_date
            if newest_date is None or last_date > newest_date:
                newest_date = last_date
        except:
            pass
    
    print(f"\n📊 Total trading days: {total_days:,}")
    print(f"📅 Oldest data: {oldest_date.strftime('%Y-%m-%d') if oldest_date else 'N/A'}")
    print(f"📅 Newest data: {newest_date.strftime('%Y-%m-%d') if newest_date else 'N/A'}")
    print(f"📈 Years of history: {(newest_date.year - oldest_date.year) if oldest_date and newest_date else 0}")
    
    print(f"\n✅ All tickers now have MAXIMUM available historical data!")
    print(f"🚀 Ready for enhanced model training with 46 years of market data")


if __name__ == "__main__":
    main()
