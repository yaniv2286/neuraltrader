"""
NeuralTrader Data Manager - The Sentry
=====================================

Responsible for fetching, validating, and managing market data for paper trading.
Implements safety checks and data quality validation.

Features:
- Daily OHLCV data fetching with split/dividend adjustments
- Missing data detection and ticker exclusion
- ATR calculation for risk management validation
- Data quality assurance and logging

Usage:
    from src.trading.data_manager import DataManager
    
    dm = DataManager()
    data = dm.fetch_daily_data(['AAPL', 'MSFT', 'NVDA'])
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Manager - The Sentry
    Fetches and validates market data with safety checks
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """Initialize Data Manager with Alpaca API"""
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper_url = 'https://paper-api.alpaca.markets'
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.paper_url
        )
        
        # Curated ticker universe (Phase 5 optimized)
        self.curated_tickers = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'UNH']
        
        logger.info("Data Manager initialized")
        logger.info(f"Curated universe: {self.curated_tickers}")
    
    def fetch_daily_data(self, tickers: List[str] = None, days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV data with safety checks
        
        Args:
            tickers: List of tickers to fetch (defaults to curated universe)
            days_back: Number of days of historical data to fetch
            
        Returns:
            Dictionary of ticker -> DataFrame with OHLCV data
        """
        if tickers is None:
            tickers = self.curated_tickers
        
        logger.info(f"Fetching daily data for {len(tickers)} tickers")
        
        data = {}
        valid_tickers = []
        excluded_tickers = []
        
        for ticker in tickers:
            try:
                df = self._fetch_ticker_data(ticker, days_back)
                
                if df is not None and not df.empty:
                    # Safety check: Validate data quality
                    if self._validate_data_quality(ticker, df):
                        data[ticker] = df
                        valid_tickers.append(ticker)
                        logger.info(f"✅ {ticker}: {len(df)} days of data")
                    else:
                        excluded_tickers.append(ticker)
                        logger.warning(f"❌ {ticker}: Failed data quality validation")
                else:
                    excluded_tickers.append(ticker)
                    logger.warning(f"❌ {ticker}: No data fetched")
                    
            except Exception as e:
                excluded_tickers.append(ticker)
                logger.error(f"❌ {ticker}: Error fetching data - {e}")
        
        # Log summary
        logger.info(f"Data fetch summary: {len(valid_tickers)} valid, {len(excluded_tickers)} excluded")
        if excluded_tickers:
            logger.warning(f"Excluded tickers: {excluded_tickers}")
        
        return data
    
    def _fetch_ticker_data(self, ticker: str, days_back: int) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker"""
        try:
            # Get calendar for trading days
            calendar = self.api.get_calendar(
                start=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d')
            )
            
            # Calculate actual trading days needed
            trading_days = len([day for day in calendar if day.date.weekday() < 5])
            
            # Fetch bars with adjustment='all' for splits/dividends
            bars = self.api.get_bars(
                symbol=ticker,
                timeframe=TimeFrame.Day,
                limit=trading_days + 5,  # Buffer for holidays
                adjustment='all'  # Important: handles splits and dividends
            ).df
            
            if bars.empty:
                return None
            
            # Convert to standard format
            df = bars.reset_index()
            df = df.rename(columns={
                'timestamp': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def _validate_data_quality(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Validate data quality with safety checks
        
        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV data
            
        Returns:
            True if data passes validation, False otherwise
        """
        try:
            # Check minimum data requirements
            if len(df) < 14:  # Need at least 14 days for ATR
                logger.warning(f"{ticker}: Insufficient data ({len(df)} days < 14)")
                return False
            
            # Check for missing days (allow 1 day gap)
            df_sorted = df.sort_values('date')
            date_diffs = df_sorted['date'].diff().dt.days
            
            # Count gaps > 1 day (excluding weekends)
            large_gaps = date_diffs[date_diffs > 1].count()
            if large_gaps > 1:
                logger.warning(f"{ticker}: Too many data gaps ({large_gaps} gaps > 1 day)")
                return False
            
            # Check for null values
            if df.isnull().any().any():
                logger.warning(f"{ticker}: Contains null values")
                return False
            
            # Check for zero or negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    logger.warning(f"{ticker}: Contains non-positive {col} values")
                    return False
            
            # Check price consistency (high >= low, etc.)
            if not (df['high'] >= df['low']).all():
                logger.warning(f"{ticker}: High < Low inconsistency")
                return False
            
            if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
                logger.warning(f"{ticker}: High price inconsistency")
                return False
            
            if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
                logger.warning(f"{ticker}: Low price inconsistency")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating {ticker}: {e}")
            return False
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for validation
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR calculation period
            
        Returns:
            Series with ATR values
        """
        try:
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    def validate_atr_consistency(self, ticker: str, df: pd.DataFrame, expected_atr: float = None) -> bool:
        """
        Validate ATR calculation against expected value
        
        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV data
            expected_atr: Expected ATR value (from pre-calculated scores)
            
        Returns:
            True if ATR is consistent, False otherwise
        """
        try:
            calculated_atr = self.calculate_atr(df)
            
            if calculated_atr.empty:
                logger.warning(f"{ticker}: Could not calculate ATR")
                return False
            
            latest_atr = calculated_atr.iloc[-1]
            
            if expected_atr is not None:
                # Check consistency (allow 10% tolerance)
                tolerance = 0.10
                diff = abs(latest_atr - expected_atr) / expected_atr
                
                if diff > tolerance:
                    logger.warning(f"{ticker}: ATR inconsistency - Calculated: {latest_atr:.4f}, Expected: {expected_atr:.4f}")
                    return False
                else:
                    logger.info(f"{ticker}: ATR consistent - {latest_atr:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ATR for {ticker}: {e}")
            return False
    
    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get latest prices for tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of ticker -> latest price
        """
        prices = {}
        
        for ticker in tickers:
            try:
                # Get latest trade
                trade = self.api.get_latest_trade(ticker)
                prices[ticker] = float(trade.price)
                
            except Exception as e:
                logger.error(f"Error getting latest price for {ticker}: {e}")
                prices[ticker] = None
        
        return prices
    
    def get_market_status(self) -> Dict[str, any]:
        """
        Get current market status
        
        Returns:
            Dictionary with market status information
        """
        try:
            clock = self.api.get_clock()
            
            return {
                'is_open': clock.is_open,
                'timestamp': clock.timestamp,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'current_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_open': False,
                'timestamp': None,
                'next_open': None,
                'next_close': None,
                'current_time': datetime.now()
            }
    
    def log_data_summary(self, data: Dict[str, pd.DataFrame]):
        """Log summary of fetched data"""
        logger.info("=== Data Summary ===")
        
        for ticker, df in data.items():
            if not df.empty:
                latest_price = df['close'].iloc[-1]
                atr = self.calculate_atr(df).iloc[-1] if not self.calculate_atr(df).empty else 0
                
                logger.info(f"{ticker}: {len(df)} days, Latest: ${latest_price:.2f}, ATR: {atr:.4f}")
        
        logger.info(f"Total tickers with valid data: {len(data)}")

# Usage example
if __name__ == "__main__":
    # Test the data manager
    dm = DataManager()
    
    # Fetch data
    data = dm.fetch_daily_data()
    
    # Log summary
    dm.log_data_summary(data)
    
    # Get market status
    status = dm.get_market_status()
    logger.info(f"Market status: {status}")
