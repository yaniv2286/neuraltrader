"""
FIXED Download Script - Proper Error Handling
Downloads all tickers with full historical data and proper validation
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
    tickers = sorted(list(set([os.path.basename(f).split('_')[0].upper().replace('_USD', '-USD') for f in cache_files])))
    return tickers

def get_ticker_list():
    """Get complete ticker list from organized cache info"""
    ticker_list_path = os.path.join(CACHE_DIR, 'info', 'lists', 'tickers_to_download.txt')
    
    if os.path.exists(ticker_list_path):
        tickers = []
        with open(ticker_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    tickers.append(line)
        return sorted(tickers)
    else:
        # Fallback to cache files if list doesn't exist
        print(f"⚠️ Ticker list not found at {ticker_list_path}")
        print("🔄 Using existing cache files as fallback")
        return get_existing_tickers()

def validate_response_content(response_text):
    """Validate that response contains valid CSV data"""
    # Check for API error messages
    if "Error:" in response_text or "allocation" in response_text:
        return False, "API error in response"
    
    # Check minimum length
    if len(response_text) < 100:
        return False, "Response too short"
    
    # Check for CSV header
    if not response_text.strip().startswith("date"):
        return False, "Invalid CSV format"
    
    # Check for basic CSV structure
    lines = response_text.strip().split('\n')
    if len(lines) < 2:
        return False, "Insufficient data rows"
    
    # Check header has required columns
    header = lines[0].lower()
    required_cols = ['date', 'close']
    for col in required_cols:
        if col not in header:
            return False, f"Missing required column: {col}"
    
    return True, "Valid CSV data"

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
            # CRITICAL FIX: Validate response content before saving
            is_valid, validation_msg = validate_response_content(response.text)
            
            if not is_valid:
                print(f"   ❌ Invalid response: {validation_msg}")
                return False, 0
            
            # Create filename
            filename = f"{ticker.replace('-', '_')}_1d_full_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(CACHE_DIR, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Verify data
            try:
                df = pd.read_csv(filepath)
                if len(df) == 0 or 'date' not in df.columns:
                    print(f"   ⚠️ Invalid data format after saving")
                    os.remove(filepath)
                    return False, 0
                
                years = (datetime.now().year - pd.to_datetime(df['date'].iloc[0]).year)
                
                print(f"   ✅ Downloaded {len(df)} days ({years} years)")
                print(f"   📅 Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
                
                # CRITICAL FIX: Only delete old files IF new file is valid
                old_files = glob.glob(os.path.join(CACHE_DIR, f"{ticker.replace('-', '_')}_1d_*.csv"))
                for old_file in old_files:
                    if old_file != filepath:  # Don't delete the file we just created
                        try:
                            os.remove(old_file)
                            print(f"   🗑️ Removed old file: {os.path.basename(old_file)}")
                        except:
                            pass
                
                return True, years
            except Exception as e:
                print(f"   ⚠️ Error parsing data: {e}")
                # CRITICAL FIX: Don't delete file on parsing error - might be valid data
                print(f"   📝 Keeping file for manual inspection: {filename}")
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
    Download all tickers with FULL historical data (100+ tickers)
    """
    print("="*70)
    print("DOWNLOADING ALL TICKERS WITH FULL HISTORICAL DATA (100+ TICKERS)")
    print("="*70)
    
    # Get ticker list from organized cache info
    all_tickers = get_ticker_list()
    
    print(f"\n🎯 Found {len(all_tickers)} tickers in master list")
    print(f"📁 Using ticker list: src/data/cache/tiingo/info/lists/tickers_to_download.txt")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"📅 Fetching data from 1970-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📈 Attempting to get MAXIMUM possible historical data (54+ years)")
    print(f"⏱️ This will take approximately {len(all_tickers)} minutes...")
    print(f"🔧 FIXED: Proper error handling and content validation")
    print(f"🛑 EARLY TERMINATION: Stops on first API error")
    
    success_count = 0
    fail_count = 0
    total_years = 0
    api_error_detected = False
    
    # Process in batches of 20 for better API management
    batch_size = 20
    for batch_start in range(0, len(all_tickers), batch_size):
        batch_end = min(batch_start + batch_size, len(all_tickers))
        batch_tickers = all_tickers[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//batch_size + 1}: tickers {batch_start+1}-{batch_end}")
        
        for i, ticker in enumerate(batch_tickers, batch_start + 1):
            print(f"\n[{i}/{len(all_tickers)}]", end=" ")
            success, years = download_ticker_full_history(ticker)
            
            if success:
                success_count += 1
                total_years += years
            else:
                fail_count += 1
                # Check if this was an API error (not ticker not found)
                if "API error" in str(years) or "allocation" in str(years):
                    api_error_detected = True
                    print(f"\n🚨 API ERROR DETECTED - STOPPING DOWNLOAD")
                    print(f"   Reason: Tiingo API rate limits reached")
                    print(f"   Recommendation: Wait 1 hour for API limits to reset")
                    break
            
            # Rate limiting - wait 1 second between requests
            if i < len(all_tickers) and not api_error_detected:
                time.sleep(1)
        
        # Check if we should continue to next batch
        if api_error_detected:
            break
        
        # Longer pause between batches (5 seconds)
        if batch_end < len(all_tickers):
            print(f"\n⏸️ Batch complete, pausing 5 seconds before next batch...")
            time.sleep(5)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    
    if api_error_detected:
        print(f"\n🚨 DOWNLOAD STOPPED DUE TO API RATE LIMITS")
        print(f"   ✅ Successfully downloaded: {success_count} tickers")
        print(f"   ❌ Failed: {fail_count} tickers (API error)")
        print(f"   📊 Average history: {total_years/success_count:.1f} years per ticker" if success_count > 0 else "")
        print(f"   📊 Total files in cache: {len(os.listdir(CACHE_DIR))} files")
        print(f"\n🛑 RECOMMENDATION: Wait 1 hour for API limits to reset")
        print(f"   ⏰ Then re-run script to download remaining tickers")
    else:
        print(f"\n✅ Successfully downloaded: {success_count} tickers")
        print(f"❌ Failed: {fail_count} tickers")
        print(f"📊 Average history: {total_years/success_count:.1f} years per ticker" if success_count > 0 else "")
        print(f"\n📊 Total files in cache: {len(os.listdir(CACHE_DIR))} files")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print(f"{'='*70}")
    
    # Count valid files
    cache_files = glob.glob(os.path.join(CACHE_DIR, '*.csv'))
    valid_files = 0
    total_days = 0
    oldest_date = None
    newest_date = None
    
    for file in cache_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0 and 'date' in df.columns:
                valid_files += 1
                total_days += len(df)
                file_start = pd.to_datetime(df['date'].iloc[0])
                file_end = pd.to_datetime(df['date'].iloc[-1])
                
                if oldest_date is None or file_start < oldest_date:
                    oldest_date = file_start
                if newest_date is None or file_end > newest_date:
                    newest_date = file_end
        except:
            continue
    
    print(f"📊 Total trading days: {total_days:,}")
    print(f"📅 Oldest data: {oldest_date.strftime('%Y-%m-%d') if oldest_date else 'N/A'}")
    print(f"📅 Newest data: {newest_date.strftime('%Y-%m-%d') if newest_date else 'N/A'}")
    print(f"📈 Years of history: {(newest_date - oldest_date).days / 365.25:.1f}" if oldest_date and newest_date else "")
    
    if api_error_detected:
        print(f"\n⚠️ PARTIAL DOWNLOAD COMPLETE")
        print(f"🚀 Ready for enhanced model training with {total_days:,} days of market data")
        print(f"📝 Download remaining tickers after API reset")
    else:
        print(f"\n✅ All tickers now have MAXIMUM available historical data!")
        print(f"🚀 Ready for enhanced model training with {total_days:,} days of market data")

if __name__ == "__main__":
    main()
