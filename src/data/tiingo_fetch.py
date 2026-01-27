"""
Tiingo Data Fetcher
API Key required: Get free at https://www.tiingo.com/
Free tier: 500 requests/day, 20+ years of data
Paid tier: $10/month, unlimited requests
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, List
import os

class TiingoDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tiingo data fetcher
        
        Args:
            api_key: Tiingo API key. If None, tries to get from environment variable TIINGO_API_KEY
        """
        self.api_key = api_key or os.getenv('TIINGO_API_KEY')
        if not self.api_key:
            raise ValueError("Tiingo API key required. Set TIIINGO_API_KEY environment variable or pass api_key parameter")
        
        self.base_url = "https://api.tiingo.com/tiingo/daily"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
    
    def fetch_price_data(self, ticker: str, start_date: str, end_date: str, 
                         resample_freq: str = 'daily') -> pd.DataFrame:
        """
        Fetch price data from Tiingo
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            resample_freq: 'daily', 'weekly', 'monthly'
        
        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/{ticker}/prices"
        
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'resampleFreq': resample_freq,
            'columns': 'open,high,low,close,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                raise ValueError(f"No data returned for {ticker}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to match existing format
            df.columns = [col.replace('adj', '') for col in df.columns]
            
            # Sort by date
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching data from Tiingo: {e}")
    
    def fetch_multiple_tickers(self, tickers: List[str], start_date: str, 
                              end_date: str, resample_freq: str = 'daily') -> dict:
        """
        Fetch data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            resample_freq: 'daily', 'weekly', 'monthly'
        
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Fetching {ticker}...")
                df = self.fetch_price_data(ticker, start_date, end_date, resample_freq)
                results[ticker] = df
                
                # Rate limiting (free tier: 50 requests/minute)
                time.sleep(1.2)
                
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        
        return results
    
    def fetch_20_years_data(self, ticker: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch 20 years of data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date (default: today)
        
        Returns:
            DataFrame with 20 years of OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_date = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
        
        print(f"Fetching 20 years of data for {ticker} ({start_date} to {end_date})")
        return self.fetch_price_data(ticker, start_date, end_date)
    
    def save_to_cache(self, df: pd.DataFrame, ticker: str, timeframe: str = 'daily'):
        """Save data to cache file"""
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', 'tiingo')
        os.makedirs(cache_dir, exist_ok=True)
        
        filename = f"{ticker}_{timeframe}_20y.csv"
        filepath = os.path.join(cache_dir, filename)
        
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        return filepath
    
    def load_from_cache(self, ticker: str, timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', 'tiingo')
        filename = f"{ticker}_{timeframe}_20y.csv"
        filepath = os.path.join(cache_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded {len(df)} days of data from cache for {ticker}")
            return df
        
        return None

def setup_tiingo():
    """Setup Tiingo API key"""
    print("Tiingo Setup")
    print("=" * 50)
    print("1. Go to https://www.tiingo.com/")
    print("2. Sign up for free account")
    print("3. Get your API key from dashboard")
    print("4. Set environment variable: set TIINGO_API_KEY=your_key_here")
    print("   Or add it to your .env file")
    print()
    print("Free tier: 500 requests/day (enough for testing)")
    print("Paid tier: $10/month for unlimited requests")
    print("=" * 50)

# Example usage
if __name__ == "__main__":
    # Check if API key is set
    import os
    if not os.getenv('TIINGO_API_KEY'):
        setup_tiingo()
    else:
        # Example: Fetch 20 years of AAPL data
        fetcher = TiingoDataFetcher()
        
        try:
            # Try to load from cache first
            df = fetcher.load_from_cache('AAPL')
            
            if df is None:
                # Fetch from API
                df = fetcher.fetch_20_years_data('AAPL')
                
                # Save to cache
                fetcher.save_to_cache(df, 'AAPL')
            
            print(f"\nAAPL 20-year data summary:")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Total days: {len(df)}")
            print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
        except Exception as e:
            print(f"Error: {e}")
