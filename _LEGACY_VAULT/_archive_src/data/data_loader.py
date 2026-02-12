"""
Tiingo-Only Data Loader
=====================

EMERGENCY OVERRIDE: Force Tiingo usage, disable YFinance
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Tiingo-Only Data Loader
    EMERGENCY: YFinance disabled, Tiingo forced
    """
    
    def __init__(self):
        """Initialize with Tiingo Manager only"""
        try:
            from src.data.tiingo_manager import TiingoManager
            self.tiingo_manager = TiingoManager()
            self.logger = logging.getLogger(__name__)
            self.logger.info("[LOADER] Tiingo-only data loader initialized")
        except:
            self.tiingo_manager = None
            self.logger = logging.getLogger(__name__)
            self.logger.warning("[LOADER] Tiingo manager not available, using cache only")
    
    def load_data(self, tickers: List[str] = None, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load data using Tiingo only, with cache fallback
        
        Args:
            tickers: List of tickers (default: S&P 100)
            start_date: Start date (default: 2004-01-01)
            end_date: End date (default: today)
            
        Returns:
            Dictionary of ticker -> DataFrame
        """
        if tickers is None:
            tickers = self.get_available_tickers()
        
        if start_date is None:
            start_date = '2004-01-01'
        
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"[LOAD] Loading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Try to load from cache first
        cached_data = self.load_from_cache(tickers, start_date, end_date)
        if cached_data:
            self.logger.info(f"[CACHE] Loaded {len(cached_data)} tickers from cache")
            return cached_data
        
        # If cache fails and Tiingo manager is available, try API
        if self.tiingo_manager:
            self.logger.info("[API] Cache miss, fetching from Tiingo API...")
            return self.tiingo_manager.fetch_daily_data(tickers, start_date=start_date, end_date=end_date)
        
        self.logger.error("[ERROR] No data available - cache empty and API unavailable")
        return {}
    
    def load_from_cache(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load data from cache files"""
        cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'tiingo'
        
        results = {}
        
        for ticker in tickers:
            # Look for cache files
            cache_files = list(cache_dir.glob(f"{ticker}_*.csv")) + list(cache_dir.glob(f"{ticker}_*.parquet"))
            
            if cache_files:
                # Use the most recent cache file
                latest_file = max(cache_files, key=lambda x: x.stat().st_mtime)
                
                try:
                    if latest_file.suffix == '.csv':
                        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                    else:  # parquet
                        df = pd.read_parquet(latest_file)
                    
                    # Filter by date range
                    if 'Date' in df.index.names or df.index.name == 'Date':
                        df.index = pd.to_datetime(df.index)
                    
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df_filtered = df[mask]
                    
                    if len(df_filtered) > 0:
                        results[ticker] = df_filtered
                        self.logger.debug(f"[CACHE] {ticker}: Loaded {len(df_filtered)} rows from cache")
                    else:
                        self.logger.warning(f"[CACHE] {ticker}: No data in date range {start_date} to {end_date}")
                        
                except Exception as e:
                    self.logger.error(f"[CACHE] {ticker}: Error loading cache file: {e}")
            else:
                self.logger.warning(f"[CACHE] {ticker}: No cache file found")
        
        return results
    
    def get_available_tickers(self) -> List[str]:
        """Get available tickers from Tiingo cache"""
        cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'tiingo'
        cache_files = list(cache_dir.glob("*.csv")) + list(cache_dir.glob("*.parquet"))
        
        tickers = []
        for file_path in cache_files:
            ticker = file_path.stem.upper()
            # Extract ticker from filename (remove prefix and suffix)
            parts = ticker.split('_')
            if parts and parts[0] not in ['feature_metadata', 'master_feature_matrix', 'scored_data', 'sp100']:
                tickers.append(parts[0])
        
        self.logger.info(f"[CACHE] Found {len(tickers)} cached tickers")
        return tickers
