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
            # Default to project's data cache
            self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data_cache', 'tiingo')
        else:
            self.cache_dir = cache_dir
        
        self.available_tickers = self._scan_available_tickers()
    
    def _scan_available_tickers(self) -> List[str]:
        """Scan for available ticker data files"""
        if not os.path.exists(self.cache_dir):
            print(f"⚠️ Cache directory not found: {self.cache_dir}")
            return []
        
        csv_files = glob.glob(os.path.join(self.cache_dir, "*.csv"))
        tickers = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Extract ticker from filename (handle various naming patterns)
            ticker = filename.split('_')[0].split('.')[0]
            tickers.append(ticker)
        
        return sorted(list(set(tickers)))
    
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
        # Find the data file
        file_patterns = [
            f"{ticker}_1d_20y.csv",
            f"{ticker}_1d.csv", 
            f"{ticker}.csv"
        ]
        
        data_file = None
        for pattern in file_patterns:
            potential_file = os.path.join(self.cache_dir, pattern)
            if os.path.exists(potential_file):
                data_file = potential_file
                break
        
        if data_file is None:
            print(f"❌ No data file found for {ticker}")
            return None
        
        try:
            # Load the data
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            # Standardize column names
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'close' in col_lower and 'close' not in df.columns:
                    col_mapping['close'] = col
                elif 'open' in col_lower and 'open' not in df.columns:
                    col_mapping['open'] = col
                elif 'high' in col_lower and 'high' not in df.columns:
                    col_mapping['high'] = col
                elif 'low' in col_lower and 'low' not in df.columns:
                    col_mapping['low'] = col
                elif 'volume' in col_lower and 'volume' not in df.columns:
                    col_mapping['volume'] = col
            
            if col_mapping:
                df = df.rename(columns=col_mapping)
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing columns for {ticker}: {missing_cols}")
                return None
            
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
