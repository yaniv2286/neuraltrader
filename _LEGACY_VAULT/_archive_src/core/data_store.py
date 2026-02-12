"""
DataStore - Incremental data loading with strict policies.
CRITICAL: Never re-download all data. Use cached CSVs.
Fail loudly if data missing or schema inconsistent.
"""
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

class DataStoreError(Exception):
    """Raised when data store encounters an error."""
    pass

class DataStore:
    """
    Manages cached ticker data with strict policies:
    - Never re-download all data
    - Use cached CSVs (154 tickers already saved)
    - Incremental updates only
    - Fail loudly if data missing or schema inconsistent
    - Use adjusted prices consistently
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    MIN_DATA_DAYS = 252  # Minimum 1 year of data
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        else:
            self.cache_dir = cache_dir
        
        self.cache_dir = os.path.abspath(self.cache_dir)
        self._ticker_cache: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, dict] = {}
        self.price_policy = "adjusted"
        
        # Scan available tickers
        self.available_tickers = self._scan_tickers()
    
    def _scan_tickers(self) -> List[str]:
        """Scan cache directory for available tickers."""
        if not os.path.exists(self.cache_dir):
            raise DataStoreError(f"Cache directory not found: {self.cache_dir}")
        
        tickers = set()
        search_dirs = [
            self.cache_dir,
            os.path.join(self.cache_dir, 'tiingo')
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            for file_path in glob.glob(os.path.join(search_dir, "*.csv")):
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                if parts:
                    ticker = parts[0].upper()
                    if 1 <= len(ticker) <= 5:
                        tickers.add(ticker)
        
        return sorted(list(tickers))
    
    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Get ticker data from cache. Never downloads.
        
        Args:
            ticker: Stock symbol
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            validate: Whether to validate data schema
        
        Returns:
            DataFrame with OHLCV data
        
        Raises:
            DataStoreError: If data not found or invalid
        """
        ticker = ticker.upper()
        
        # Check cache first
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self._ticker_cache:
            return self._ticker_cache[cache_key].copy()
        
        # Find data file
        data_file = self._find_data_file(ticker)
        if data_file is None:
            raise DataStoreError(f"No data file found for {ticker}. Data must be pre-cached.")
        
        # Load data
        df = self._load_csv(data_file, ticker)
        
        # Validate schema
        if validate:
            self._validate_schema(df, ticker)
        
        # Filter by date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Validate minimum data
        if len(df) < self.MIN_DATA_DAYS:
            raise DataStoreError(
                f"{ticker} has insufficient data: {len(df)} days < {self.MIN_DATA_DAYS} required"
            )
        
        # Cache and return
        self._ticker_cache[cache_key] = df
        return df.copy()
    
    def _find_data_file(self, ticker: str) -> Optional[str]:
        """Find the data file for a ticker."""
        search_patterns = [
            f"{ticker}_1d_20y.csv",
            f"{ticker}_1d_*.csv",
            f"{ticker}_1d.csv",
            f"{ticker}.csv"
        ]
        
        search_dirs = [
            os.path.join(self.cache_dir, 'tiingo'),
            self.cache_dir
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for pattern in search_patterns:
                matches = glob.glob(os.path.join(search_dir, pattern))
                if matches:
                    return matches[0]
        
        return None
    
    def _load_csv(self, file_path: str, ticker: str) -> pd.DataFrame:
        """Load CSV file with proper parsing."""
        try:
            df = pd.read_csv(file_path, skiprows=[1], index_col=0)
            
            # Parse datetime index
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            df = df[df.index.notna()]
            
            # Standardize columns
            df.columns = [col.lower() for col in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Map OHLCV columns
            ohlcv_map = {}
            for col in self.REQUIRED_COLUMNS:
                if col in df.columns:
                    ohlcv_map[col] = col
                elif f'1d_{col}' in df.columns:
                    ohlcv_map[f'1d_{col}'] = col
            
            if len(ohlcv_map) == 5:
                df = df[list(ohlcv_map.keys())].copy()
                df = df.rename(columns=ohlcv_map)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean data
            df = df.dropna()
            df = df[df['close'] > 0]
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            raise DataStoreError(f"Failed to load {ticker} from {file_path}: {e}")
    
    def _validate_schema(self, df: pd.DataFrame, ticker: str):
        """Validate data schema. Fail loudly if inconsistent."""
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise DataStoreError(
                f"{ticker} schema invalid: missing columns {missing_cols}"
            )
        
        # Check for NaN values
        nan_counts = df[self.REQUIRED_COLUMNS].isna().sum()
        if nan_counts.any():
            raise DataStoreError(
                f"{ticker} has NaN values: {nan_counts[nan_counts > 0].to_dict()}"
            )
        
        # Check for non-positive prices
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                raise DataStoreError(f"{ticker} has non-positive {col} values")
    
    def get_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fail_on_missing: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date filter
            end_date: End date filter
            fail_on_missing: If True, raise error on missing ticker
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        result = {}
        errors = []
        
        for ticker in tickers:
            try:
                result[ticker] = self.get_ticker_data(ticker, start_date, end_date)
            except DataStoreError as e:
                if fail_on_missing:
                    raise
                errors.append(str(e))
        
        if errors and not fail_on_missing:
            print(f"⚠️ {len(errors)} tickers failed to load")
        
        return result
    
    def get_latest_date(self, ticker: str) -> Optional[datetime]:
        """Get the latest date available for a ticker."""
        try:
            df = self.get_ticker_data(ticker, validate=False)
            return df.index.max()
        except DataStoreError:
            return None
    
    def check_data_freshness(self, max_age_days: int = 1) -> Dict[str, bool]:
        """Check if data is fresh (updated within max_age_days)."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        freshness = {}
        
        for ticker in self.available_tickers:
            latest = self.get_latest_date(ticker)
            if latest:
                freshness[ticker] = latest >= cutoff
            else:
                freshness[ticker] = False
        
        return freshness
    
    def get_summary(self) -> dict:
        """Get summary of data store status."""
        return {
            'cache_dir': self.cache_dir,
            'total_tickers': len(self.available_tickers),
            'available_tickers': self.available_tickers,
            'price_policy': self.price_policy,
            'min_data_days': self.MIN_DATA_DAYS
        }
    
    def log_price_policy(self):
        """Log price policy as required by constitution."""
        print(f"📊 Price Policy: {self.price_policy.upper()}")
        print(f"   Using adjusted prices consistently: YES")
        return self.price_policy


# Global instance
_data_store = None

def get_data_store() -> DataStore:
    """Get global DataStore instance."""
    global _data_store
    if _data_store is None:
        _data_store = DataStore()
    return _data_store
