#!/usr/bin/env python3
"""
Generic Data Download Module for NeuralTrader
Handles downloading, updating, and managing ticker data
"""

import requests
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import time
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Tiingo API configuration
TIINGO_TOKEN = os.getenv('TIINGO_API_KEY')
if not TIINGO_TOKEN:
    raise ValueError("TIINGO_API_KEY not found in environment variables. Please set it in .env file.")
BASE_URL = "https://api.tiingo.com/tiingo/daily"

# Output directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache', 'tiingo')
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
        
        # Try both naming patterns
        patterns = [
            os.path.join(self.cache_dir, "*_1d_full_*.csv"),
            os.path.join(self.cache_dir, "*_sp100_emergency_*.csv")
        ]
        
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                filename = os.path.basename(file_path)
                
                # Extract ticker from different naming patterns
                if "_1d_full_" in filename:
                    ticker = filename.split("_1d_full_")[0]
                elif "_sp100_emergency_" in filename:
                    ticker = filename.split("_sp100_emergency_")[0]
                else:
                    # Skip files that don't match expected patterns
                    continue
                
                if exclude_crypto:
                    crypto_tickers = ['BTC', 'ETH', 'ADA', 'SOL', 'COIN']
                    if ticker not in crypto_tickers and ticker not in existing:
                        existing.append(ticker)
                else:
                    if ticker not in existing:
                        existing.append(ticker)
        
        return sorted(existing)
    
    def get_target_tickers(self, exclude_crypto: bool = True) -> List[str]:
        """Get list of target tickers from configuration file"""
        target = []
        
        # Use config/tickers.txt specifically - fix path construction
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        ticker_file = os.path.join(project_root, 'config', 'tickers.txt')
        
        print(f"[DEBUG] Looking for ticker file: {ticker_file}")
        
        if os.path.exists(ticker_file):
            print(f"[DEBUG] Found ticker file, reading...")
            with open(ticker_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if exclude_crypto:
                            crypto_tickers = ['BTC', 'ETH', 'ADA', 'SOL', 'COIN']
                            if line not in crypto_tickers:
                                target.append(line)
                        else:
                            target.append(line)
            print(f"[DEBUG] Read {len(target)} tickers from config file")
        else:
            print(f"[ERROR] Ticker file not found: {ticker_file}")
            # Fallback to cache directory list
            ticker_file = os.path.join(self.cache_dir, 'info', 'lists', 'tickers_to_download.txt')
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
        """Download only missing tickers by performing physical disk audit"""
        print(f"[SYNC] Performing physical disk audit of {self.cache_dir}...")
        
        # Get target tickers from config file only - bypass all internal lists
        target = self.get_target_tickers(exclude_crypto)
        
        print(f"[INFO] Found {len(target)} tickers in config/tickers.txt. Performing physical disk check...")
        
        # Get current date for expected file naming
        from datetime import datetime
        current_date = datetime.now().strftime('%Y%m%d')
        
        missing = []
        existing_count = 0
        
        # Get list of files in cache directory for pure disk-based checking
        try:
            cache_files = os.listdir(self.cache_dir)
        except OSError as e:
            print(f"[ERROR] Cannot read cache directory {self.cache_dir}: {e}")
            cache_files = []
        
        for ticker in target:
            # Pure disk-based check - look for any file starting with ticker_1d_full_
            ticker_exists = any(f.startswith(f"{ticker}_1d_full_") for f in cache_files)
            
            print(f"[DEBUG] Physical disk check for {ticker}... Found: {ticker_exists}")
            
            if ticker_exists:
                existing_count += 1
            else:
                # MUST be added to download queue if file doesn't exist physically
                missing.append(ticker)
                print(f"[DEBUG] Physical check: {ticker} file missing. Adding to queue.")
        
        print(f"[AUDIT] Physical disk audit complete:")
        print(f"[AUDIT]   Total tickers in config: {len(target)}")
        print(f"[AUDIT]   Files found on disk: {existing_count}")
        print(f"[AUDIT]   Files missing: {len(missing)}")
        
        if not missing:
            print(f"[COMPLETE] All tickers verified on disk. No downloads needed.")
            return {
                "success": 0, 
                "failed": 0, 
                "total": 0, 
                "failed_tickers": [],
                "message": "All tickers already downloaded"
            }
        
        print(f"[DOWNLOAD] Queueing {len(missing)} missing tickers for download:")
        print(f"[DOWNLOAD] Missing: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
        
        success_count = 0
        failed_tickers = []
        
        for i, ticker in enumerate(missing, 1):
            print(f"[{i}/{len(missing)}] Downloading {ticker}...")
            
            days = self.download_ticker(ticker)
            
            if days and days > 0:
                print(f"   [OK] Success: {days} days")
                success_count += 1
            else:
                print(f"   [FAIL] Failed")
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
            return {
                "success": 0, 
                "failed": 0, 
                "total": 0, 
                "failed_tickers": [],
                "message": "No tickers to update"
            }
        
        print(f"Updating {len(tickers)} tickers...")
        
        success_count = 0
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Updating {ticker}...")
            
            days = self.download_ticker(ticker)
            
            if days and days > 0:
                print(f"   [OK] Success: {days} days")
                success_count += 1
            else:
                print(f"   [FAIL] Failed")
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
    
    def create_market_filters(self) -> Dict:
        """Create standardized market_filters.json for Golden Shield"""
        print("Creating market filter data...")
        
        # Output file
        output_file = os.path.join(os.path.dirname(self.cache_dir), 'market_filters.json')
        
        # Get SPY and VIX data
        spy_data = self._load_ticker_data('SPY')
        vix_data = self._load_ticker_data('VIX')
        
        if vix_data is None or vix_data.empty:
            # Try VXX as fallback
            vix_data = self._load_ticker_data('VXX')
            
            # Check if VXX data is valid
            if vix_data is not None and not vix_data.empty:
                if len(vix_data) < 20:
                    print("❌ CRITICAL FAIL: VXX data has insufficient rows (< 20) for calculations")
                    print("❌ Please ensure VXX data has at least 20 trading days")
                    return {'success': False, 'error': 'Insufficient VXX data - less than 20 rows'}
        
        # Check SPY data validity
        if spy_data is not None and not spy_data.empty:
            if len(spy_data) < 20:
                print("❌ CRITICAL FAIL: SPY data has insufficient rows (< 20) for calculations")
                print("❌ Please ensure SPY data has at least 20 trading days")
                return {'success': False, 'error': 'Insufficient SPY data - less than 20 rows'}
        
        # Check VIX data validity
        if vix_data is not None and not vix_data.empty:
            if len(vix_data) < 20:
                print("❌ CRITICAL FAIL: VIX data has insufficient rows (< 20) for calculations")
                print("❌ Please ensure VIX data has at least 20 trading days")
                return {'success': False, 'error': 'Insufficient VIX data - less than 20 rows'}
        
        market_data = {}
        
        # Process SPY data
        if spy_data is not None and not spy_data.empty:
            spy_standardized = self._standardize_market_data(spy_data, 'SPY')
            if spy_standardized is not None:
                market_data['SPY'] = spy_standardized
                print(f"[OK] Processed SPY: {len(spy_standardized)} records")
        
        # Process VIX data
        if vix_data is not None and not vix_data.empty:
            vix_standardized = self._standardize_market_data(vix_data, 'VIX')
            if vix_standardized is not None:
                market_data['VIX'] = vix_standardized
                print(f"[OK] Processed VIX: {len(vix_standardized)} records")
        
        # FAIL FAST if no real data available
        if not market_data:
            print("[ERROR] CRITICAL FAIL: No real market data available for SPY/VIX")
            print("[ERROR] Connection Error: Unable to load market data from cache")
            print("[ERROR] Please run 'python scripts/data_manager.py download-missing --ticker SPY' first")
            return {'success': False, 'error': 'Connection Error - No market data available'}
        
        # Save to JSON
        try:
            with open(output_file, 'w') as f:
                json.dump(market_data, f, indent=2, default=str)
            print(f"[OK] Market filters saved to {output_file}")
            return {'success': True, 'file': output_file, 'records': len(market_data)}
        except Exception as e:
            print(f"[ERROR] Error saving market filters: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker data from cache"""
        # Try different naming patterns
        patterns = [
            os.path.join(self.cache_dir, f"{ticker}_1d_full_*.csv"),
            os.path.join(self.cache_dir, f"{ticker}_sp100_emergency_*.csv"),
            os.path.join(self.cache_dir, f"{ticker}*.csv")  # Fallback
        ]
        
        files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                break
        
        if not files:
            return None
        
        try:
            df = pd.read_csv(files[0])
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.rename(columns={'date': 'Date'}, inplace=True)
            return df
        except Exception as e:
            print(f"❌ Error loading {ticker}: {e}")
            return None
    
    def _standardize_market_data(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Standardize market data for Golden Shield"""
        records = []
        
        # Ensure the index is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print(f"❌ Unable to convert index to datetime for {ticker}")
                return None
        
        for idx, row in df.iterrows():
            record = {
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'price': float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                'vix_value': float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                'sma_20': None,
                'sma_200': None,
                'sma_20_distance': None,
                'sma_200_distance': None,
                'vix_1d_change': None,
                'vix_5d_ma': None,
                'vix_spike_20d': None,
                'ticker': ticker
            }
            records.append(record)
        
        # Calculate indicators for SPY
        if ticker == 'SPY' and len(records) > 0:
            prices = [r['price'] for r in records if r['price'] is not None]
            if len(prices) >= 20:
                # Calculate SMAs
                sma_20 = pd.Series(prices).rolling(window=20, min_periods=1).mean()
                sma_200 = pd.Series(prices).rolling(window=200, min_periods=1).mean()
                
                for i, record in enumerate(records):
                    if i < len(sma_20) and i < len(sma_200):
                        record['sma_20'] = float(sma_20.iloc[i])
                        record['sma_200'] = float(sma_200.iloc[i])
                        
                        if record['sma_20'] and record['sma_200']:
                            record['sma_20_distance'] = (record['price'] - record['sma_20']) / record['sma_20']
                            record['sma_200_distance'] = (record['price'] - record['sma_200']) / record['sma_200']
        
        # Calculate indicators for VIX
        if ticker in ['VIX', 'VXX'] and len(records) > 0:
            vix_values = [r['vix_value'] for r in records if r['vix_value'] is not None]
            if len(vix_values) >= 2:
                # Calculate changes
                vix_changes = pd.Series(vix_values).pct_change()
                vix_5d_ma = pd.Series(vix_values).rolling(window=5, min_periods=1).mean()
                vix_spike_20d = vix_changes.rolling(window=20, min_periods=1).max()
                
                for i, record in enumerate(records):
                    if i < len(vix_changes) and i < len(vix_5d_ma):
                        record['vix_1d_change'] = float(vix_changes.iloc[i])
                        record['vix_5d_ma'] = float(vix_5d_ma.iloc[i])
                        if i < len(vix_spike_20d):
                            record['vix_spike_20d'] = float(vix_spike_20d.iloc[i])
        
        return records
    
    def _create_sample_market_data(self) -> Dict:
        """Create sample market data for testing"""
        print("Creating sample market data...")
        
        # Create date range
        dates = pd.date_range(start='2004-01-01', end='2026-02-07', freq='D')
        
        market_data = {}
        
        # Create SPY sample data
        spy_records = []
        base_price = 100.0
        for i, date in enumerate(dates):
            price = base_price * (1 + 0.0001 * np.sin(i * 0.1))
            sma_20 = price * (1 + 0.02 * np.sin(i * 0.05))
            sma_200 = price * (1 + 0.01 * np.sin(i * 0.01))
            
            spy_records.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': price,
                'sma_20': sma_20,
                'sma_200': sma_200,
                'sma_20_distance': (price - sma_20) / sma_20,
                'sma_200_distance': (price - sma_200) / sma_200,
                'ticker': 'SPY'
            })
        
        market_data['SPY'] = spy_records
        
        # Create VIX sample data
        vix_records = []
        base_vix = 20.0
        for i, date in enumerate(dates):
            vix_value = base_vix * (1 + 0.01 * np.sin(i * 0.2))
            vix_change = 0.01 * np.sin(i * 0.2)
            
            vix_records.append({
                'date': date.strftime('%Y-%m-%d'),
                'vix_value': vix_value,
                'vix_1d_change': vix_change,
                'vix_5d_ma': vix_value,
                'vix_spike_20d': abs(vix_change),
                'ticker': 'VIX'
            })
        
        market_data['VIX'] = vix_records
        
        print(f"[OK] Created sample data: {len(spy_records)} SPY records, {len(vix_records)} VIX records")
        return market_data

def main():
    """Command line interface for data management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralTrader Data Management')
    parser.add_argument('action', choices=['status', 'download-missing', 'update-all', 'clean', 'market-filters'], 
                       help='Action to perform')
    parser.add_argument('--exclude-crypto', action='store_true', default=True,
                       help='Exclude crypto tickers')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Rate limit between API calls (seconds)')
    parser.add_argument('--days-to-keep', type=int, default=7,
                       help='Days to keep when cleaning (default: 7)')
    parser.add_argument('--ticker', type=str,
                       help='Specific ticker to download (for single ticker operations)')
    
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
        if args.ticker:
            # Single ticker download
            print(f"Downloading single ticker: {args.ticker}")
            result = dm.download_ticker(args.ticker)
            if result:
                print(f"[OK] Successfully downloaded {args.ticker}: {result} records")
            else:
                print(f"[FAIL] Failed to download {args.ticker}")
        else:
            # Download missing tickers
            result = dm.download_missing_tickers(args.exclude_crypto, args.rate_limit)
            
            # Handle both string and dictionary results
            if isinstance(result, str):
                print(f"Result: {result}")
            elif isinstance(result, dict):
                print(f"Result: {result.get('message', 'Operation completed')}")
                if result.get('failed_tickers'):
                    print(f"Failed: {', '.join(result['failed_tickers'])}")
            else:
                print(f"Unexpected result type: {type(result)}")
    
    elif args.action == 'update-all':
        result = dm.update_all_tickers(args.exclude_crypto, args.rate_limit)
        
        # Handle both string and dictionary results
        if isinstance(result, str):
            print(f"Result: {result}")
        elif isinstance(result, dict):
            print(f"Result: {result.get('message', 'Operation completed')}")
            if result.get('failed_tickers'):
                print(f"Failed: {', '.join(result['failed_tickers'])}")
        else:
            print(f"Unexpected result type: {type(result)}")
    
    elif args.action == 'clean':
        removed = dm.clean_old_files(args.days_to_keep)
        print(f"Cleaned {removed} old files")
    
    elif args.action == 'market-filters':
        result = dm.create_market_filters()
        if result['success']:
            print(f"[OK] Market filters created: {result['file']}")
        else:
            print(f"[FAIL] Failed to create market filters: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
