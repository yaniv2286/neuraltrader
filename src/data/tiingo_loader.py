"""
Tiingo Data Loader
Loads data from cached Tiingo CSV files (no external dependencies)
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

class TiingoDataLoader:
    """
    Loads data from cached Tiingo CSV files
    No external API dependencies
    """
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            # Default to src/data/cache directory
            self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        else:
            self.cache_dir = cache_dir
        
        self.available_tickers = self._scan_available_tickers()
    
    def _scan_available_tickers(self) -> List[str]:
        """Scan for available ticker data files - prioritize 20-year files"""
        if not os.path.exists(self.cache_dir):
            print(f"⚠️ Cache directory not found: {self.cache_dir}")
            return []
        
        # Scan for all CSV files in cache directory AND tiingo subdirectory
        csv_files = glob.glob(os.path.join(self.cache_dir, "*.csv"))
        csv_files.extend(glob.glob(os.path.join(self.cache_dir, "tiingo", "*.csv")))
        tickers = set()
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Skip non-data files
            if filename in ['data_summary.txt']:
                continue
            
            # Extract ticker from filename (handle various naming patterns)
            # Examples: AAPL_1d_20y.csv, AAPL_1d_hash.csv, AAPL.csv
            parts = filename.split('_')
            if len(parts) > 0:
                ticker = parts[0].upper()
                # Only add valid ticker symbols (2-5 characters, all uppercase)
                if 2 <= len(ticker) <= 5 and ticker.isalpha():
                    tickers.add(ticker)
        
        return sorted(list(tickers))
    
    def load_ticker_data(self, ticker: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load data for a specific ticker from cache
        
        Args:
            ticker: Stock symbol
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        # Find the data file - prioritize 20-year files for maximum historical data
        data_file = None
        
        # Search for files matching ticker pattern - PRIORITIZE 20-year data
        # Check both cache root and tiingo subdirectory
        search_patterns = [
            f"{ticker}_1d_20y.csv",      # 20-year data (PRIORITY for bull/bear cycles)
            f"{ticker}_1d_*.csv",        # Hash-suffixed files
            f"{ticker}_1d.csv",          # Standard daily
            f"{ticker}.csv"              # Fallback
        ]
        
        search_dirs = [
            os.path.join(self.cache_dir, "tiingo"),  # Check tiingo subdirectory first
            self.cache_dir                            # Then check cache root
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for pattern in search_patterns:
                matches = glob.glob(os.path.join(search_dir, pattern))
                if matches:
                    # Use the first match and stop searching
                    data_file = matches[0]
                    break
            if data_file:
                break  # Found a file, stop searching other directories
        
        if data_file is None:
            print(f"❌ No data file found for {ticker}")
            return None
        
        try:
            # Load the data - skip the Ticker row (row 1) if it exists
            df = pd.read_csv(data_file, skiprows=[1], index_col=0)
            
            # Convert index to datetime first (handles timezone-aware strings)
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            
            # Remove timezone info (make timezone-naive for consistency)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Remove any rows with invalid dates
            df = df[df.index.notna()]
            
            # Remove rows where index string contains 'datetime' (header artifacts)
            # This must be done after datetime conversion to avoid Series ambiguity
            try:
                valid_dates_mask = ~df.index.to_series().astype(str).str.contains('datetime', case=False, na=False).values
                df = df.iloc[valid_dates_mask]
            except Exception as e:
                # If this fails, just skip the datetime filtering
                pass
            
            # Standardize column names to lowercase first
            df.columns = [col.lower() for col in df.columns]
            
            # Remove duplicate columns (keep first occurrence)
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Try to find OHLCV columns with various naming patterns
            ohlcv_map = {}
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                # Look for exact match first
                if base_col in df.columns:
                    ohlcv_map[base_col] = base_col
                # Look for prefixed version (e.g., '1d_open')
                elif f'1d_{base_col}' in df.columns:
                    ohlcv_map[f'1d_{base_col}'] = base_col
            
            # Select and rename only the OHLCV columns we found
            if len(ohlcv_map) == 5:
                df = df[list(ohlcv_map.keys())].copy()
                df = df.rename(columns=ohlcv_map)
            else:
                # Fallback: just keep columns that match OHLCV names
                available_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
                if len(available_cols) == 5:
                    df = df[available_cols].copy()
                else:
                    print(f"⚠️ Could not find all OHLCV columns for {ticker}")
                    print(f"   Available columns: {list(df.columns)}")
                    return None
            
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Sort by date
            df = df.sort_index()
            
            # Basic data validation
            if df.empty:
                print(f"❌ No data available for {ticker} in specified date range")
                return None
            
            # Check for data quality issues
            if df.isnull().any().any():
                print(f"⚠️ {ticker} has missing values, cleaning...")
                df = df.dropna()
            
            # Check for zero/negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    print(f"⚠️ {ticker} has non-positive {col} values, cleaning...")
                    df = df[df[col] > 0]
            
            print(f"✅ Loaded {ticker}: {len(df)} days from {df.index.min().date()} to {df.index.max().date()}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {ticker}: {e}")
            return None
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers"""
        return self.available_tickers.copy()
    
    def load_multiple_tickers(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers
        
        Args:
            tickers: List of stock symbols
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data_dict = {}
        
        for ticker in tickers:
            if ticker in self.available_tickers:
                df = self.load_ticker_data(ticker, start_date, end_date)
                if df is not None:
                    data_dict[ticker] = df
            else:
                print(f"❌ {ticker} not available in cache")
        
        print(f"📊 Loaded {len(data_dict)}/{len(tickers)} tickers successfully")
        return data_dict
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary of available data"""
        if not self.available_tickers:
            return {"total_tickers": 0, "message": "No data available"}
        
        # Sample a few tickers to get date range
        sample_tickers = self.available_tickers[:min(5, len(self.available_tickers))]
        date_ranges = []
        
        for ticker in sample_tickers:
            df = self.load_ticker_data(ticker)
            if df is not None:
                date_ranges.append((df.index.min(), df.index.max()))
        
        if date_ranges:
            min_date = min(d[0] for d in date_ranges)
            max_date = max(d[1] for d in date_ranges)
        else:
            min_date = max_date = None
        
        return {
            "total_tickers": len(self.available_tickers),
            "available_tickers": self.available_tickers,
            "date_range": f"{min_date.date()} to {max_date.date()}" if min_date else "Unknown",
            "cache_directory": self.cache_dir
        }

# Global instance for easy access
tiingo_loader = TiingoDataLoader()

def load_tiingo_data(ticker: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Convenience function to load Tiingo data
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    return tiingo_loader.load_ticker_data(ticker, start_date, end_date)

def get_available_tickers() -> List[str]:
    """Get list of available tickers from cache"""
    return tiingo_loader.get_available_tickers()

if __name__ == "__main__":
    # Test the loader
    print("🧪 Testing Tiingo Data Loader")
    print("=" * 40)
    
    loader = TiingoDataLoader()
    summary = loader.get_data_summary()
    
    print(f"📊 Available tickers: {summary['total_tickers']}")
    print(f"📅 Date range: {summary['date_range']}")
    print(f"📁 Cache: {summary['cache_directory']}")
    
    if summary['total_tickers'] > 0:
        # Test loading a sample ticker
        sample_ticker = summary['available_tickers'][0]
        df = loader.load_ticker_data(sample_ticker, "2023-01-01", "2023-12-31")
        
        if df is not None:
            print(f"\n✅ Sample data for {sample_ticker}:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head())
