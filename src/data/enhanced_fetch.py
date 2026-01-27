"""
Enhanced Data Fetcher with Multiple Data Sources
Supports 20+ years of historical data from various providers
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataFetcher:
    """Enhanced data fetcher supporting multiple sources for 20+ years of data"""
    
    def __init__(self, preferred_source: str = 'tiingo'):
        """
        Initialize enhanced data fetcher
        
        Args:
            preferred_source: 'tiingo', 'yfinance', 'polygon', or 'auto'
        """
        self.preferred_source = preferred_source
        self.tiingo_api_key = os.getenv('TIINGO_API_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        
        # Initialize Tiingo fetcher if API key is available
        self.tiingo_fetcher = None
        if self.tiingo_api_key:
            try:
                from tiingo_fetch import TiingoDataFetcher
                self.tiingo_fetcher = TiingoDataFetcher(self.tiingo_api_key)
                print("✓ Tiingo API configured")
            except ImportError:
                print("⚠ Tiingo fetcher not available")
            except Exception as e:
                print(f"⚠ Tiingo initialization failed: {e}")
                self.tiingo_fetcher = None
    
    def fetch_20_years_data(self, ticker: str, source: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch 20 years of data using the best available source
        
        Args:
            ticker: Stock ticker symbol
            source: Force specific source ('tiingo', 'yfinance', 'polygon')
        
        Returns:
            DataFrame with OHLCV data
        """
        source = source or self.preferred_source
        
        # Try Tiingo first (most reliable for 20+ years)
        if source == 'tiingo' or (source == 'auto' and self.tiingo_fetcher):
            try:
                print(f"Fetching 20 years of {ticker} data from Tiingo...")
                df = self.tiingo_fetcher.fetch_20_years_data(ticker)
                df = self._standardize_columns(df, ticker, '1d')
                print(f"✓ Successfully fetched {len(df)} days from Tiingo")
                return df
            except Exception as e:
                print(f"✗ Tiingo failed: {e}")
        
        # Try Yahoo Finance (free, but may not have full 20 years)
        if source == 'yfinance' or source == 'auto':
            try:
                print(f"Fetching {ticker} data from Yahoo Finance...")
                df = self._fetch_yfinance_20y(ticker)
                print(f"✓ Successfully fetched {len(df)} days from Yahoo Finance")
                return df
            except Exception as e:
                print(f"✗ Yahoo Finance failed: {e}")
        
        raise ValueError(f"Failed to fetch data for {ticker} from any source")
    
    def _fetch_yfinance_20y(self, ticker: str) -> pd.DataFrame:
        """Fetch 20 years from Yahoo Finance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20*365)
        
        # Download data
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Standardize columns
        data = self._standardize_columns(data, ticker, '1d')
        
        return data
    
    def _standardize_columns(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        """Standardize column names to match existing format"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Map common column names to our standard format
        column_mapping = {
            'Open': f'{timeframe}_open',
            'High': f'{timeframe}_high',
            'Low': f'{timeframe}_low',
            'Close': f'{timeframe}_close',
            'Volume': f'{timeframe}_volume',
            'Adj Close': f'{timeframe}_close',  # Use adjusted close as close
            'open': f'{timeframe}_open',
            'high': f'{timeframe}_high',
            'low': f'{timeframe}_low',
            'close': f'{timeframe}_close',
            'volume': f'{timeframe}_volume',
            'adjClose': f'{timeframe}_close',
            'adjHigh': f'{timeframe}_high',
            'adjLow': f'{timeframe}_low',
            'adjOpen': f'{timeframe}_open',
            'adjVolume': f'{timeframe}_volume'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        required_cols = [f'{timeframe}_open', f'{timeframe}_high', f'{timeframe}_low', f'{timeframe}_close', f'{timeframe}_volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Keep only required columns
        df = df[required_cols]
        
        return df
    
    def fetch_multiple_tickers_20y(self, tickers: List[str], source: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch 20 years of data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            source: Force specific source
        
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        results = {}
        
        for ticker in tickers:
            try:
                print(f"\n{'='*50}")
                print(f"Processing {ticker}...")
                df = self.fetch_20_years_data(ticker, source)
                results[ticker] = df
                
                # Save to cache
                self._save_to_cache(df, ticker, '1d')
                
            except Exception as e:
                print(f"✗ Failed to fetch {ticker}: {e}")
                continue
        
        return results
    
    def _save_to_cache(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """Save data to cache"""
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', '20y')
        os.makedirs(cache_dir, exist_ok=True)
        
        filename = f"{ticker}_{timeframe}_20y.csv"
        filepath = os.path.join(cache_dir, filename)
        
        df.to_csv(filepath)
        print(f"✓ Saved to cache: {filepath}")
    
    def load_from_cache(self, ticker: str, timeframe: str = '1d') -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', '20y')
        filename = f"{ticker}_{timeframe}_20y.csv"
        filepath = os.path.join(cache_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"✓ Loaded {len(df)} days from cache for {ticker}")
            return df
        
        return None
    
    def get_data_summary(self, df: pd.DataFrame, ticker: str) -> Dict:
        """Get summary statistics for the data"""
        return {
            'ticker': ticker,
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'total_days': len(df),
            'trading_days': len(df),
            'price_range': {
                'min': df['1d_close'].min(),
                'max': df['1d_close'].max(),
                'mean': df['1d_close'].mean()
            },
            'volume_stats': {
                'mean': df['1d_volume'].mean(),
                'max': df['1d_volume'].max()
            },
            'years_covered': (df.index.max() - df.index.min()).days / 365.25
        }

def setup_api_keys():
    """Guide for setting up API keys"""
    print("Data Source Setup for 20+ Years Historical Data")
    print("=" * 60)
    print()
    print("1. TIIINGO (Recommended - $10/month)")
    print("   - Go to https://www.tiingo.com/")
    print("   - Sign up for free account")
    print("   - Get API key from dashboard")
    print("   - Set environment: set TIINGO_API_KEY=your_key")
    print("   - Free tier: 500 requests/day")
    print("   - Paid tier: $10/month unlimited")
    print()
    print("2. YAHOO FINANCE (Free)")
    print("   - No API key required")
    print("   - May not have full 20 years for all stocks")
    print("   - Good for testing")
    print()
    print("3. POLYGON.IO ($99/month)")
    print("   - Professional grade data")
    print("   - Full 20+ years history")
    print("   - Set environment: set POLYGON_API_KEY=your_key")
    print()
    print("Recommendation: Start with Yahoo Finance, upgrade to Tiingo for reliable 20+ years")
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Check available data sources
    fetcher = EnhancedDataFetcher()
    
    # Test with a popular stock
    ticker = 'AAPL'
    
    print(f"Testing data fetch for {ticker}...")
    print("=" * 50)
    
    try:
        # Try to load from cache first
        df = fetcher.load_from_cache(ticker)
        
        if df is None:
            # Fetch from API
            df = fetcher.fetch_20_years_data(ticker)
            
            # Save to cache
            fetcher._save_to_cache(df, ticker, '1d')
        
        # Show summary
        summary = fetcher.get_data_summary(df, ticker)
        
        print(f"\n✓ Successfully fetched data for {ticker}")
        print(f"  Date Range: {summary['start_date']} to {summary['end_date']}")
        print(f"  Total Days: {summary['total_days']}")
        print(f"  Years Covered: {summary['years_covered']:.1f}")
        print(f"  Price Range: ${summary['price_range']['min']:.2f} to ${summary['price_range']['max']:.2f}")
        print(f"  Mean Price: ${summary['price_range']['mean']:.2f}")
        print(f"  Mean Volume: {summary['volume_stats']['mean']:,.0f}")
        
        print(f"\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nPlease setup API keys:")
        setup_api_keys()
