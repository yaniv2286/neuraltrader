"""
Clean Tiingo Cache - Remove Duplicates and Keep Only Full Data Files
Keeps only the most comprehensive data file for each ticker
"""

import os
import glob
import pandas as pd
from collections import defaultdict
import shutil

def clean_tiingo_cache():
    """
    Clean the Tiingo cache directory by removing duplicates and keeping only the best file for each ticker
    """
    print("🧹 Cleaning Tiingo Cache Directory...")
    
    # Get cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    print(f"📊 Found {len(csv_files)} CSV files")
    
    # Group files by ticker
    ticker_files = defaultdict(list)
    for file in csv_files:
        ticker = os.path.basename(file).split('_')[0].upper()
        ticker_files[ticker].append(file)
    
    print(f"📈 Found {len(ticker_files)} unique tickers")
    
    # Statistics
    duplicates_count = sum(1 for files in ticker_files.values() if len(files) > 1)
    print(f"🔄 Found {duplicates_count} tickers with duplicate files")
    
    # Clean each ticker
    kept_files = []
    removed_files = []
    total_days_kept = 0
    
    for ticker, files in ticker_files.items():
        print(f"\n📊 Processing {ticker} ({len(files)} files)...")
        
        if len(files) == 1:
            # Only one file, keep it
            best_file = files[0]
            print(f"   ✅ Only one file: {os.path.basename(best_file)}")
        else:
            # Multiple files, find the best one
            best_file = None
            max_days = 0
            earliest_date = None
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    if len(df) > max_days:
                        max_days = len(df)
                        best_file = file
                        earliest_date = df['date'].iloc[0]
                    elif len(df) == max_days and earliest_date:
                        # If same length, check earliest date
                        file_date = df['date'].iloc[0]
                        if file_date < earliest_date:
                            best_file = file
                            earliest_date = file_date
                except Exception as e:
                    print(f"   ⚠️ Error reading {os.path.basename(file)}: {e}")
                    continue
            
            if best_file:
                print(f"   🏆 Best file: {os.path.basename(best_file)} ({max_days} days)")
                
                # Remove other files
                for file in files:
                    if file != best_file:
                        try:
                            os.remove(file)
                            removed_files.append(file)
                            print(f"   🗑️ Removed: {os.path.basename(file)}")
                        except Exception as e:
                            print(f"   ❌ Error removing {os.path.basename(file)}: {e}")
            else:
                print(f"   ❌ No valid files found for {ticker}")
                continue
        
        # Verify the kept file
        if best_file:
            try:
                df = pd.read_csv(best_file)
                days = len(df)
                start_date = df['date'].iloc[0]
                end_date = df['date'].iloc[-1]
                years = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year)
                
                print(f"   📈 Data: {days} days ({years} years)")
                print(f"   📅 Range: {start_date} to {end_date}")
                
                kept_files.append(best_file)
                total_days_kept += days
                
            except Exception as e:
                print(f"   ❌ Error verifying {os.path.basename(best_file)}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("CACHE CLEANING COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Kept files: {len(kept_files)}")
    print(f"   🗑️ Removed files: {len(removed_files)}")
    print(f"   📈 Total trading days: {total_days_kept:,}")
    print(f"   📈 Average days per ticker: {total_days_kept//len(kept_files) if kept_files else 0}")
    
    # Show final file list
    print(f"\n📋 Final files kept:")
    for file in sorted(kept_files):
        ticker = os.path.basename(file).split('_')[0].upper()
        try:
            df = pd.read_csv(file)
            days = len(df)
            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]
            years = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year)
            print(f"   {ticker:8s}: {days:5d} days ({years:2d} years) {start_date} to {end_date}")
        except:
            print(f"   {ticker:8s}: Error reading file")
    
    # Check for any remaining issues
    remaining_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    print(f"\n🔍 Verification: {len(remaining_files)} files remaining in cache")
    
    # Check if we have the expected number of tickers
    expected_tickers = 80  # You mentioned 80 tickers
    if len(kept_files) < expected_tickers:
        print(f"⚠️  Warning: Only {len(kept_files)} tickers found (expected {expected_tickers})")
    else:
        print(f"✅ All {len(kept_files)} tickers processed successfully")
    
    print(f"\n🎯 Cache directory is now clean and optimized!")
    print(f"📁 Location: {cache_dir}")

if __name__ == "__main__":
    clean_tiingo_cache()
