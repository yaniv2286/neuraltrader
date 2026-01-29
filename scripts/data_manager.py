#!/usr/bin/env python3
"""
Generic Data Download Module for NeuralTrader
Handles downloading, updating, and managing ticker data
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Tiingo API configuration
TIINGO_TOKEN = "72e14af10f4c32db4a7631275929617481aed281"
BASE_URL = "https://api.tiingo.com/tiingo/daily"

# Output directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
os.makedirs(CACHE_DIR, exist_ok=True)

class DataManager:
    """Generic data management class for all ticker operations"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.api_token = TIINGO_TOKEN
        self.base_url = BASE_URL
        
    def get_existing_tickers(self, exclude_crypto: bool = True) -> List[str]:
        """Get list of already downloaded tickers"""
        existing = []
        pattern = os.path.join(self.cache_dir, "*_1d_full_*.csv")
        
        for file_path in glob.glob(pattern):
            ticker = os.path.basename(file_path).split("_1d_full_")[0]
            if exclude_crypto:
                crypto_tickers = ['BTC', 'ETH', 'ADA', 'SOL', 'COIN']
                if ticker not in crypto_tickers:
                    existing.append(ticker)
            else:
                existing.append(ticker)
        
        return sorted(existing)
    
    def get_target_tickers(self, exclude_crypto: bool = True) -> List[str]:
        """Get list of target tickers from configuration file"""
        ticker_file = os.path.join(self.cache_dir, 'info', 'lists', 'tickers_to_download.txt')
        target = []
        
        if os.path.exists(ticker_file):
            with open(ticker_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if exclude_crypto:
                            crypto_tickers = ['BTC', 'ETH', 'ADA', 'SOL', 'COIN']
                            if line not in crypto_tickers:
                                target.append(line)
                        else:
                            target.append(line)
        
        return sorted(target)
    
    def download_ticker(self, ticker: str, start_date: str = "2000-01-01", 
                       end_date: str = None) -> Optional[int]:
        """Download data for a single ticker"""
        try:
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")
            
            url = f"{self.base_url}/{ticker}/prices"
            params = {
                'token': self.api_token,
                'startDate': start_date,
                'endDate': end_date,
                'format': 'csv',
                'resampleFreq': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                from io import StringIO
                data = pd.read_csv(StringIO(response.text))
                
                if not data.empty:
                    filename = f"{ticker}_1d_full_{datetime.now().strftime('%Y%m%d')}.csv"
                    filepath = os.path.join(self.cache_dir, filename)
                    data.to_csv(filepath, index=False)
                    return len(data)
                else:
                    return 0
            else:
                print(f"     HTTP {response.status_code}: {response.text[:100]}")
                return None
                
        except Exception as e:
            print(f"     Error: {str(e)}")
            return None
    
    def download_missing_tickers(self, exclude_crypto: bool = True, 
                                rate_limit: float = 0.5) -> Dict[str, any]:
        """Download only missing tickers"""
        existing = self.get_existing_tickers(exclude_crypto)
        target = self.get_target_tickers(exclude_crypto)
        missing = [t for t in target if t not in existing]
        
        if not missing:
            return {"success": 0, "failed": 0, "total": 0, "message": "All tickers already downloaded"}
        
        print(f"Downloading {len(missing)} missing tickers...")
        
        success_count = 0
        failed_tickers = []
        
        for i, ticker in enumerate(missing, 1):
            print(f"[{i}/{len(missing)}] Downloading {ticker}...")
            
            days = self.download_ticker(ticker)
            
            if days and days > 0:
                print(f"   ✅ Success: {days} days")
                success_count += 1
            else:
                print(f"   ❌ Failed")
                failed_tickers.append(ticker)
            
            if i < len(missing):
                time.sleep(rate_limit)
        
        return {
            "success": success_count,
            "failed": len(failed_tickers),
            "total": len(missing),
            "failed_tickers": failed_tickers,
            "message": f"Downloaded {success_count}/{len(missing)} tickers"
        }
    
    def update_all_tickers(self, exclude_crypto: bool = True, 
                          rate_limit: float = 0.5) -> Dict[str, any]:
        """Update all existing tickers with latest data"""
        tickers = self.get_existing_tickers(exclude_crypto)
        
        if not tickers:
            return {"success": 0, "failed": 0, "total": 0, "message": "No tickers to update"}
        
        print(f"Updating {len(tickers)} tickers...")
        
        success_count = 0
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Updating {ticker}...")
            
            days = self.download_ticker(ticker)
            
            if days and days > 0:
                print(f"   ✅ Success: {days} days")
                success_count += 1
            else:
                print(f"   ❌ Failed")
                failed_tickers.append(ticker)
            
            if i < len(tickers):
                time.sleep(rate_limit)
        
        return {
            "success": success_count,
            "failed": len(failed_tickers),
            "total": len(tickers),
            "failed_tickers": failed_tickers,
            "message": f"Updated {success_count}/{len(tickers)} tickers"
        }
    
    def get_status(self, exclude_crypto: bool = True) -> Dict[str, any]:
        """Get current status of ticker data"""
        existing = self.get_existing_tickers(exclude_crypto)
        target = self.get_target_tickers(exclude_crypto)
        missing = [t for t in target if t not in existing]
        
        return {
            "existing": len(existing),
            "target": len(target),
            "missing": len(missing),
            "missing_tickers": missing,
            "completion_pct": (len(existing) / len(target) * 100) if target else 0,
            "message": f"{len(existing)}/{len(target)} tickers downloaded"
        }
    
    def clean_old_files(self, days_to_keep: int = 7) -> int:
        """Clean old data files, keeping only recent ones"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        removed_count = 0
        
        for file_path in glob.glob(os.path.join(self.cache_dir, "*.csv")):
            if os.path.getmtime(file_path) < cutoff_date:
                os.remove(file_path)
                removed_count += 1
        
        return removed_count

def main():
    """Command line interface for data management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralTrader Data Management')
    parser.add_argument('action', choices=['status', 'download-missing', 'update-all', 'clean'], 
                       help='Action to perform')
    parser.add_argument('--exclude-crypto', action='store_true', default=True,
                       help='Exclude crypto tickers')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Rate limit between API calls (seconds)')
    parser.add_argument('--days-to-keep', type=int, default=7,
                       help='Days to keep when cleaning (default: 7)')
    
    args = parser.parse_args()
    
    dm = DataManager()
    
    if args.action == 'status':
        status = dm.get_status(args.exclude_crypto)
        print(f"Status: {status['message']}")
        print(f"Completion: {status['completion_pct']:.1f}%")
        if status['missing_tickers']:
            print(f"Missing: {', '.join(status['missing_tickers'][:10])}")
            if len(status['missing_tickers']) > 10:
                print(f"... and {len(status['missing_tickers']) - 10} more")
    
    elif args.action == 'download-missing':
        result = dm.download_missing_tickers(args.exclude_crypto, args.rate_limit)
        print(f"Result: {result['message']}")
        if result['failed_tickers']:
            print(f"Failed: {', '.join(result['failed_tickers'])}")
    
    elif args.action == 'update-all':
        result = dm.update_all_tickers(args.exclude_crypto, args.rate_limit)
        print(f"Result: {result['message']}")
        if result['failed_tickers']:
            print(f"Failed: {', '.join(result['failed_tickers'])}")
    
    elif args.action == 'clean':
        removed = dm.clean_old_files(args.days_to_keep)
        print(f"Cleaned {removed} old files")

if __name__ == "__main__":
    main()
