#!/usr/bin/env python3
"""
Download Only Missing Tickers Script
Downloads only the tickers we don't have yet (smart approach)
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
        # Extract ticker from filename like "AAL_1d_full_20260128.csv"
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            # Handle crypto tickers
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            existing_tickers.add(ticker)
    
    return existing_tickers

def get_missing_tickers():
    """Get only the tickers we need to download"""
    target_tickers = get_target_tickers()
    existing_tickers = get_existing_tickers()
    
    missing_tickers = [t for t in target_tickers if t not in existing_tickers]
    return missing_tickers

def validate_response_content(response_text):
    """Validate that response contains valid CSV data"""
    # Check for API error messages
    if "Error:" in response_text or "allocation" in response_text:
        return False, "API error in response"
    
    # Check minimum length
    if len(response_text) < 100:
        return False, "Response too short"
    
    # Check for CSV format
    lines = response_text.strip().split('\n')
    if len(lines) < 2:
        return False, "Not enough lines for CSV"
    
    # Check for required columns
    header = lines[0].lower()
    required_columns = ['date', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in header]
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    return True, "Valid CSV data"

def download_ticker_full_history(ticker):
    """Download full historical data for a ticker with validation"""
    print(f"📊 Downloading {ticker} (FULL HISTORY)...")
    
    # Format ticker for API (handle crypto)
    api_ticker = ticker
    if ticker.endswith('USD'):
        api_ticker = ticker.replace('-', '_')
    
    # Build URL for maximum historical data
    url = f"{BASE_URL}/{api_ticker}/prices"
    params = {
        'token': TIINGO_TOKEN,
        'startDate': '1970-01-01',
        'endDate': datetime.now().strftime('%Y-%m-%d'),
        'format': 'csv',
        'resampleFreq': 'daily'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            # Validate response content before saving
            is_valid, validation_msg = validate_response_content(response.text)
            
            if not is_valid:
                print(f"   ❌ Invalid response: {validation_msg}")
                return False, validation_msg
            
            # Create filename with today's date
            filename = f"{ticker.replace('-', '_')}_1d_full_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(CACHE_DIR, filename)
            
            # Save the file
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Load and verify the data
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                years = (df.index.max() - df.index.min()).days / 365.25
                print(f"   ✅ Downloaded {len(df)} days ({years:.1f} years)")
                print(f"   📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
                
                # Clean up old files for this ticker
                old_files = glob.glob(os.path.join(CACHE_DIR, f"{ticker.replace('-', '_')}_1d_*.csv"))
                for old_file in old_files:
                    if old_file != filepath:
                        os.remove(old_file)
                
                return True, years
                
            except Exception as e:
                print(f"   ❌ Error parsing data: {e}")
                # Remove invalid file
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False, f"Parse error: {e}"
                
        else:
            print(f"   ❌ Error: HTTP {response.status_code}")
            return False, 0
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def main():
    """Download only missing tickers"""
    print("="*70)
    print("DOWNLOADING ONLY MISSING TICKERS (SMART APPROACH)")
    print("="*70)
    
    # Get all tickers and check status
    target_tickers = get_target_tickers()
    existing_tickers = get_existing_tickers()
    
    print(f"\n📊 Target tickers: {len(target_tickers)}")
    print(f"✅ Have tickers: {len(existing_tickers)}")
    
    # Check each ticker and print status
    missing_tickers = []
    for ticker in target_tickers:
        if ticker in existing_tickers:
            print(f"✅ {ticker}: Data already exists")
        else:
            missing_tickers.append(ticker)
    
    print(f"❌ Missing tickers: {len(missing_tickers)}")
    
    if len(missing_tickers) == 0:
        print("\n🎉 All tickers already downloaded!")
        return
    
    print(f"\n🎯 Downloading only the {len(missing_tickers)} missing tickers:")
    print(f"   First 10: {', '.join(missing_tickers[:10])}")
    if len(missing_tickers) > 10:
        print(f"   ... and {len(missing_tickers) - 10} more")
    
    print(f"\n📁 Cache directory: {CACHE_DIR}")
    print(f"📅 Fetching data from 1970-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    print(f"⏱️ This will take approximately {len(missing_tickers)} minutes...")
    
    success_count = 0
    fail_count = 0
    total_years = 0
    api_failure_count = 0
    max_api_failures = 2
    
    for i, ticker in enumerate(missing_tickers, 1):
        print(f"\n[{i}/{len(missing_tickers)}]", end=" ")
        success, years = download_ticker_full_history(ticker)
        
        if success:
            success_count += 1
            total_years += years
            api_failure_count = 0  # Reset on success
        else:
            fail_count += 1
            # Check if this was an API error
            if "API error" in str(years) or "allocation" in str(years):
                api_failure_count += 1
                print(f"\n🚨 API FAILURE #{api_failure_count}")
                
                if api_failure_count >= max_api_failures:
                    print(f"\n🛑 STOPPING - {max_api_failures} consecutive API failures")
                    print(f"   Reason: API not responding")
                    print(f"   Recommendation: Try again later")
                    break
                else:
                    print(f"   Will retry (failure {api_failure_count}/{max_api_failures})")
        
        # Rate limiting - wait 1 second between requests
        if i < len(missing_tickers) and api_failure_count < max_api_failures:
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    
    if api_failure_count >= max_api_failures:
        print(f"\n🚨 DOWNLOAD STOPPED - API NOT RESPONDING")
        print(f"   ✅ Successfully downloaded: {success_count} tickers")
        print(f"   ❌ Failed: {fail_count} tickers (API error)")
        print(f"   📊 Average history: {total_years/success_count:.1f} years per ticker" if success_count > 0 else "")
        print(f"   🔄 Try again later when API is working")
    else:
        print(f"\n✅ All missing tickers downloaded successfully!")
        print(f"   ✅ Successfully downloaded: {success_count} tickers")
        print(f"   ❌ Failed: {fail_count} tickers")
        print(f"   📊 Average history: {total_years/success_count:.1f} years per ticker" if success_count > 0 else "")
    
    # Final status
    final_existing = len(get_existing_tickers())
    target_count = len(get_target_tickers())
    print(f"\n🎯 FINAL STATUS:")
    print(f"   📊 Total tickers in cache: {final_existing}")
    print(f"   🎯 Target tickers: {target_count}")
    print(f"   📈 Progress: {final_existing}/{target_count} ({final_existing/target_count:.1%})")
    
    if final_existing >= len(get_target_tickers()):
        print(f"   🎉 COMPLETE! All {len(get_target_tickers())} tickers downloaded!")
        print(f"   🚀 Ready for comprehensive Phase 2 testing!")
    else:
        remaining = len(get_target_tickers()) - final_existing
        print(f"   ⏳ Remaining: {remaining} tickers")

if __name__ == "__main__":
    main()
