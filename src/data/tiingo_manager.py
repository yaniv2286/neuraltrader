"""
Tiingo Data Manager
===================

Primary data source for NeuralTrader using Tiingo API.
Handles batch processing, rate limiting, and adjClose calculations.

Features:
- 50 stocks per batch processing
- adjClose for dividend/split adjustments
- Rate limiting compliance
- Comprehensive error handling
- Cache management
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

class TiingoManager:
    """
    Tiingo API Data Manager for NeuralTrader
    Primary data source for historical market data
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = None):
        """
        Initialize Tiingo Manager
        
        Args:
            api_key: Tiingo API key (from environment if None)
            cache_dir: Cache directory for data storage
        """
        self.logger = logging.getLogger(__name__)
        
        # API Configuration
        self.api_key = api_key or os.getenv('TIINGO_API_KEY')
        if not self.api_key:
            raise ValueError("Tiingo API key required. Set TIINGO_API_KEY environment variable.")
        
        self.base_url = "https://api.tiingo.com/tiingo"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        })
        
        # Cache Configuration
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'tiingo'
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate Limiting (Tiingo: 500 requests/minute, 50 stocks per batch)
        self.batch_size = 50
        self.requests_per_minute = 500
        self.request_interval = 60.0 / self.requests_per_minute  # ~0.12 seconds between requests
        
        # S&P 100 Universe (can be expanded)
        self.sp100_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'PYPL', 'ADBE', 'CRM', 'NFLX', 'CMCSA',
            'INTC', 'CSCO', 'PEP', 'COST', 'AVGO', 'TXN', 'KO', 'NKE', 'ABT', 'DHR',
            'ACN', 'MRK', 'VZ', 'MDT', 'XOM', 'CVX', 'BAC', 'WFC', 'LIN', 'LLY',
            'DIS', 'WMT', 'IBM', 'GS', 'CAT', 'RTX', 'HON', 'UNP', 'UPS', 'MMM',
            'T', 'BA', 'GE', 'AMD', 'SBUX', 'QCOM', 'NOW', 'F', 'PLD', 'AMGN',
            'C', 'GILD', 'MDT', 'ISRG', 'BDX', 'ZTS', 'EL', 'ADP', 'ICE', 'CB',
            'SPGI', 'SCHW', 'AON', 'CME', 'MO', 'EQIX', 'CL', 'ANTM'
        ]
        
        self.logger.info(f"[TIINGO] Tiingo Manager initialized")
        self.logger.info(f"[TIINGO] Cache directory: {self.cache_dir}")
        self.logger.info(f"[TIINGO] Batch size: {self.batch_size}")
        self.logger.info(f"[TIINGO] Universe: {len(self.sp100_tickers)} tickers")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make authenticated request to Tiingo API with rate limiting
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data or None if error
        """
        try:
            # Rate limiting
            time.sleep(self.request_interval)
            
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                self.logger.warning(f"[TIINGO] Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                self.logger.error(f"[TIINGO] API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"[TIINGO] Request failed: {e}")
            return None
    
    def get_ticker_data(self, ticker: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for a single ticker
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Default to last 5 years if no dates specified
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'resampleFreq': 'daily'
            }
            
            endpoint = f"daily/{ticker}/prices"
            data = self._make_request(endpoint, params)
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Rename columns to standard format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'adjClose': 'Adj Close',
                'volume': 'Volume',
                'divCash': 'Dividend',
                'splitFactor': 'Split Factor'
            })
            
            # Set Date as index
            df = df.set_index('Date')
            
            # Sort by date
            df = df.sort_index()
            
            # Use adjClose for all calculations (accounts for dividends/splits)
            if 'Adj Close' in df.columns:
                df['Price'] = df['Adj Close']
            else:
                df['Price'] = df['Close']
                self.logger.warning(f"[TIINGO] {ticker}: No adjClose available, using close price")
            
            self.logger.info(f"[TIINGO] {ticker}: Loaded {len(df)} days of data")
            return df
            
        except Exception as e:
            self.logger.error(f"[TIINGO] Error loading {ticker}: {e}")
            return None
    
    def get_batch_data(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple tickers in batches
        
        Args:
            tickers: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary of ticker -> DataFrame
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(tickers), self.batch_size):
            batch_tickers = tickers[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"[TIINGO] Processing batch {batch_num}/{total_batches}: {batch_tickers}")
            
            for ticker in batch_tickers:
                try:
                    df = self.get_ticker_data(ticker, start_date, end_date)
                    if df is not None:
                        results[ticker] = df
                    else:
                        self.logger.warning(f"[TIINGO] {ticker}: No data available")
                        
                except Exception as e:
                    self.logger.error(f"[TIINGO] {ticker}: Error in batch processing: {e}")
                    continue
            
            # Brief pause between batches
            if i + self.batch_size < len(tickers):
                time.sleep(1)
        
        self.logger.info(f"[TIINGO] Batch processing complete: {len(results)}/{len(tickers)} tickers loaded")
        return results
    
    def get_sp100_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all S&P 100 tickers
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary of ticker -> DataFrame
        """
        self.logger.info(f"[TIINGO] Fetching S&P 100 data for {len(self.sp100_tickers)} tickers")
        return self.get_batch_data(self.sp100_tickers, start_date, end_date)
    
    def save_to_cache(self, data: Dict[str, pd.DataFrame], prefix: str = "tiingo_data"):
        """
        Save data to cache
        
        Args:
            data: Dictionary of ticker -> DataFrame
            prefix: File prefix for cache files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for ticker, df in data.items():
            try:
                cache_file = self.cache_dir / f"{ticker}_{prefix}_{timestamp}.csv"
                df.to_csv(cache_file)
                self.logger.debug(f"[TIINGO] Cached {ticker}: {cache_file}")
            except Exception as e:
                self.logger.error(f"[TIINGO] Error caching {ticker}: {e}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'tickers': list(data.keys()),
            'prefix': prefix,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = self.cache_dir / f"{prefix}_{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"[TIINGO] Cached {len(data)} tickers with prefix {prefix}_{timestamp}")
    
    def load_from_cache(self, prefix: str = "tiingo_data", timestamp: str = None) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load data from cache
        
        Args:
            prefix: File prefix for cache files
            timestamp: Specific timestamp to load (latest if None)
            
        Returns:
            Dictionary of ticker -> DataFrame or None
        """
        try:
            if timestamp:
                metadata_file = self.cache_dir / f"{prefix}_{timestamp}_metadata.json"
            else:
                # Find latest metadata file
                metadata_files = list(self.cache_dir.glob(f"{prefix}_*_metadata.json"))
                if not metadata_files:
                    return None
                metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            timestamp = metadata['timestamp']
            tickers = metadata['tickers']
            
            data = {}
            for ticker in tickers:
                cache_file = self.cache_dir / f"{ticker}_{prefix}_{timestamp}.csv"
                if cache_file.exists():
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    data[ticker] = df
                else:
                    self.logger.warning(f"[TIINGO] Cache file missing: {cache_file}")
            
            self.logger.info(f"[TIINGO] Loaded {len(data)} tickers from cache")
            return data
            
        except Exception as e:
            self.logger.error(f"[TIINGO] Error loading from cache: {e}")
            return None
    
    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get latest prices for specified tickers
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary of ticker -> latest price
        """
        prices = {}
        
        # Get last 5 days of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        data = self.get_batch_data(tickers, start_date, end_date)
        
        for ticker, df in data.items():
            if not df.empty:
                # Use adjClose (Price column) for latest price
                latest_price = df['Price'].iloc[-1]
                prices[ticker] = latest_price
            else:
                self.logger.warning(f"[TIINGO] No data for {ticker}")
        
        return prices
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        Validate data quality for all tickers
        
        Args:
            data: Dictionary of ticker -> DataFrame
            
        Returns:
            Dictionary of ticker -> quality status
        """
        quality_report = {}
        
        for ticker, df in data.items():
            try:
                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price']
                has_required = all(col in df.columns for col in required_cols)
                
                # Check for data continuity
                has_gaps = df['Price'].isnull().any()
                
                # Check minimum data length
                sufficient_data = len(df) >= 252  # At least 1 year
                
                # Check for price anomalies
                price_anomaly = (df['Price'] <= 0).any()
                
                quality_report[ticker] = (
                    has_required and 
                    not has_gaps and 
                    sufficient_data and 
                    not price_anomaly
                )
                
                if not quality_report[ticker]:
                    self.logger.warning(f"[TIINGO] Quality issues for {ticker}: "
                                      f"Required={has_required}, Gaps={has_gaps}, "
                                      f"Length={sufficient_data}, Anomaly={price_anomaly}")
                
            except Exception as e:
                self.logger.error(f"[TIINGO] Error validating {ticker}: {e}")
                quality_report[ticker] = False
        
        return quality_report
