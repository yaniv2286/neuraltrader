"""
NeuralTrader YFinance Data Manager - IST Scheduled Data Fetching
==============================================================

Handles market data fetching with IST scheduling for shadow trading.
Captures previous day's data and opening trends at 16:45 IST (09:45 EST).

Features:
- S&P 100 universe data fetching with yfinance
- IST scheduling (16:45 IST / 09:45 EST) for optimal data capture
- Previous day's data and opening trends analysis
- Data quality validation and caching
- Timezone-aware scheduling system

Usage:
    from src.data.yfinance_manager import YFinanceManager
    
    yfm = YFinanceManager()
    data = yfm.fetch_scheduled_data()
"""

import os
import logging
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import schedule
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YFinanceManager:
    """
    YFinance Data Manager with IST Scheduling
    Handles market data fetching with timezone-aware scheduling
    """
    
    def __init__(self):
        """Initialize YFinance Manager with IST scheduling"""
        # Timezone setup
        self.ist = pytz.timezone('Asia/Jerusalem')
        self.est = pytz.timezone('US/Eastern')
        
        # S&P 100 ticker universe
        self.sp100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'UNH',
            'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'CVX', 'HD', 'ABBV', 'MRK', 'LLY', 'PEP', 'KO',
            'COST', 'TMO', 'CSCO', 'PFE', 'MCD', 'CRM', 'BAC', 'ADBE', 'WMT', 'CMCSA', 'DIS', 'NFLX',
            'ABT', 'VZ', 'ORCL', 'TXN', 'AMD', 'LIN', 'PM', 'UPS', 'NKE', 'HON', 'UNP', 'RTX', 'INTU',
            'LOW', 'SPGI', 'MS', 'QCOM', 'COP', 'IBM', 'GE', 'AMAT', 'CAT', 'GS', 'ISRG', 'DE', 'BKNG',
            'ELV', 'PLD', 'SBUX', 'MDT', 'BLK', 'GILD', 'TJX', 'NOW', 'ADP', 'C', 'MMC', 'AMT', 'REGN',
            'MO', 'PYPL', 'CB', 'CI', 'ADI', 'MDLZ', 'VRTX', 'ZTS', 'SYK', 'CME', 'AMGN', 'FISV', 'SLB',
            'T', 'LMT', 'MU', 'CVS', 'DUK', 'ITW', 'EQIX', 'ANTM', 'CL', 'ICE', 'SHERW'
        ]
        
        # Data cache directory
        self.cache_dir = 'data/cache/yfinance'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("YFinance Manager initialized")
        logger.info(f"S&P 100 Universe: {len(self.sp100_tickers)} tickers")
        logger.info(f"IST Schedule: 16:45 IST (09:45 EST) for data fetching")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def fetch_scheduled_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch scheduled data at 16:45 IST (09:45 EST)
        Captures previous day's data and opening trends
        
        Returns:
            Dictionary of ticker -> DataFrame with OHLCV data
        """
        try:
            logger.info("🕐 Starting scheduled data fetch (16:45 IST / 09:45 EST)")
            
            # Get current time in IST
            now_ist = datetime.now(self.ist)
            logger.info(f"Current IST time: {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")
            
            # Fetch data for the last 5 trading days to capture trends
            data = self.fetch_daily_data(self.sp100_tickers, period="5d")
            
            if not data:
                logger.error("❌ No data fetched during scheduled run")
                return {}
            
            # Analyze opening trends
            opening_trends = self._analyze_opening_trends(data)
            
            # Cache the data
            self._cache_data(data, opening_trends)
            
            logger.info(f"✅ Scheduled data fetch completed: {len(data)} tickers")
            logger.info(f"   Opening trends captured: {len(opening_trends)} tickers")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error in scheduled data fetch: {e}")
            return {}
    
    def fetch_daily_data(self, tickers: List[str] = None, period: str = "5d") -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV data with safety checks
        
        Args:
            tickers: List of tickers to fetch (defaults to S&P 100)
            period: Data period (default: 5d for recent trends)
            
        Returns:
            Dictionary of ticker -> DataFrame with OHLCV data
        """
        if tickers is None:
            tickers = self.sp100_tickers
        
        logger.info(f"Fetching daily data for {len(tickers)} tickers (period: {period})")
        
        data = {}
        valid_tickers = []
        failed_tickers = []
        
        # Fetch data in batches to avoid API limits
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            try:
                # Download batch data
                batch_data = yf.download(batch, period=period, progress=False)
                
                if batch_data.empty:
                    logger.warning(f"No data fetched for batch {i//batch_size + 1}")
                    continue
                
                # Process each ticker in the batch
                for ticker in batch:
                    try:
                        # Extract ticker data (multi-level index handling)
                        if 'Close' in batch_data.columns:
                            ticker_data = batch_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                            
                            # Handle multi-level columns
                            if isinstance(ticker_data.columns, pd.MultiIndex):
                                ticker_data = ticker_data.xs(ticker, axis=1, level=1)
                            
                            # Reset index to get dates as a column
                            ticker_data.reset_index(inplace=True)
                            ticker_data.rename(columns={'Date': 'date'}, inplace=True)
                            
                            # Ensure required columns exist
                            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                            ticker_data.columns = [col.lower() for col in ticker_data.columns]
                            
                            if all(col in ticker_data.columns for col in required_cols):
                                # Safety check: Validate data quality
                                if self._validate_data_quality(ticker, ticker_data):
                                    data[ticker] = ticker_data
                                    valid_tickers.append(ticker)
                                    logger.info(f"✅ {ticker}: {len(ticker_data)} days of data")
                                else:
                                    failed_tickers.append(ticker)
                                    logger.warning(f"❌ {ticker}: Failed data quality validation")
                            else:
                                failed_tickers.append(ticker)
                                logger.warning(f"❌ {ticker}: Missing required columns")
                        else:
                            failed_tickers.append(ticker)
                            logger.warning(f"❌ {ticker}: No Close data available")
                    
                    except Exception as e:
                        failed_tickers.append(ticker)
                        logger.error(f"❌ {ticker}: Error processing data - {e}")
                
                # Small delay between batches
                if i + batch_size < len(tickers):
                    time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                continue
        
        # Log summary
        logger.info(f"Data fetch summary: {len(valid_tickers)} valid, {len(failed_tickers)} failed")
        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers[:10]}...")  # Show first 10
        
        return data
    
    def _analyze_opening_trends(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Analyze opening trends for all tickers
        
        Args:
            data: Dictionary of ticker -> DataFrame
            
        Returns:
            Dictionary of ticker -> opening trend analysis
        """
        try:
            logger.info("📈 Analyzing opening trends...")
            
            trends = {}
            
            for ticker, df in data.items():
                if len(df) < 2:
                    continue
                
                try:
                    # Calculate opening gap
                    latest_open = df['open'].iloc[-1]
                    previous_close = df['close'].iloc[-2]
                    opening_gap = (latest_open - previous_close) / previous_close
                    
                    # Calculate intraday trend
                    latest_close = df['close'].iloc[-1]
                    intraday_return = (latest_close - latest_open) / latest_open
                    
                    # Calculate volume ratio
                    latest_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume'].mean()
                    volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Determine trend direction
                    if opening_gap > 0.01:  # > 1% gap up
                        trend_direction = "STRONG_UP"
                    elif opening_gap > 0.005:  # > 0.5% gap up
                        trend_direction = "UP"
                    elif opening_gap < -0.01:  # < -1% gap down
                        trend_direction = "STRONG_DOWN"
                    elif opening_gap < -0.005:  # < -0.5% gap down
                        trend_direction = "DOWN"
                    else:
                        trend_direction = "FLAT"
                    
                    trends[ticker] = {
                        'opening_gap': opening_gap,
                        'intraday_return': intraday_return,
                        'volume_ratio': volume_ratio,
                        'trend_direction': trend_direction,
                        'latest_open': latest_open,
                        'latest_close': latest_close,
                        'previous_close': previous_close,
                        'timestamp': datetime.now(self.ist).isoformat()
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing trends for {ticker}: {e}")
                    continue
            
            logger.info(f"Opening trends analyzed: {len(trends)} tickers")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing opening trends: {e}")
            return {}
    
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
            if len(df) < 2:  # Need at least 2 days for trend analysis
                logger.warning(f"{ticker}: Insufficient data ({len(df)} days < 2)")
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
            
            # Check price consistency
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
    
    def _cache_data(self, data: Dict[str, pd.DataFrame], trends: Dict[str, Dict]):
        """Cache data and trends to files"""
        try:
            # Cache timestamp
            cache_timestamp = datetime.now(self.ist).strftime('%Y%m%d_%H%M')
            
            # Cache data
            for ticker, df in data.items():
                cache_file = os.path.join(self.cache_dir, f"{ticker}_{cache_timestamp}.csv")
                df.to_csv(cache_file, index=False)
            
            # Cache trends
            trends_file = os.path.join(self.cache_dir, f"trends_{cache_timestamp}.json")
            import json
            with open(trends_file, 'w') as f:
                json.dump(trends, f, indent=2)
            
            logger.info(f"Data cached with timestamp: {cache_timestamp}")
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get latest prices for specified tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of ticker -> latest price
        """
        try:
            logger.info(f"Fetching latest prices for {len(tickers)} tickers")
            
            # Fetch data for today
            data = yf.download(tickers, period="1d", progress=False)
            
            if data.empty:
                logger.warning("No data fetched for latest prices")
                return {}
            
            prices = {}
            for ticker in tickers:
                try:
                    if 'Close' in data.columns:
                        # Handle single ticker case
                        if isinstance(data['Close'], pd.Series):
                            price = data['Close'].iloc[-1]
                        else:
                            # Handle multiple tickers
                            price = data['Close'][ticker].iloc[-1]
                        
                        if pd.notna(price) and price > 0:
                            prices[ticker] = float(price)
                        else:
                            logger.warning(f"Invalid price for {ticker}: {price}")
                    else:
                        logger.warning(f"No Close data for {ticker}")
                
                except Exception as e:
                    logger.error(f"Error getting price for {ticker}: {e}")
                    continue
            
            logger.info(f"Fetched latest prices: {len(prices)} tickers")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return {}
    
    def start_scheduler(self):
        """Start the IST scheduler for automated data fetching"""
        try:
            logger.info("🕐 Starting IST scheduler...")
            
            # Schedule data fetch at 16:45 IST (09:45 EST)
            schedule.every().day.at("16:45").do(self.fetch_scheduled_data)
            
            logger.info("✅ Scheduler started - Daily fetch at 16:45 IST (09:45 EST)")
            
            # Run the scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in scheduler: {e}")
    
    def get_market_status(self) -> Dict:
        """
        Get current market status with timezone awareness
        
        Returns:
            Dictionary with market status information
        """
        try:
            now_ist = datetime.now(self.ist)
            now_est = datetime.now(self.est)
            
            # Check if market is open (9:30 AM - 4:00 PM EST, Mon-Fri)
            is_weekday = now_est.weekday() < 5  # 0-4 = Mon-Fri
            market_open = now_est.replace(hour=9, minute=30)
            market_close = now_est.replace(hour=16, minute=0)
            
            is_market_hours = is_weekday and market_open <= now_est <= market_close
            
            # Check if it's data fetch time (16:45 IST / 09:45 EST)
            fetch_time_est = now_est.replace(hour=9, minute=45)
            is_fetch_time = is_weekday and abs((now_est - fetch_time_est).total_seconds()) < 300  # Within 5 minutes
            
            return {
                'timestamp_ist': now_ist.strftime('%Y-%m-%d %H:%M:%S IST'),
                'timestamp_est': now_est.strftime('%Y-%m-%d %H:%M:%S EST'),
                'is_weekday': is_weekday,
                'is_market_open': is_market_hours,
                'is_fetch_time': is_fetch_time,
                'market_open_time': market_open.strftime('%H:%M EST'),
                'market_close_time': market_close.strftime('%H:%M EST'),
                'fetch_time_est': fetch_time_est.strftime('%H:%M EST'),
                'fetch_time_ist': fetch_time_est.astimezone(self.ist).strftime('%H:%M IST')
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {}

# Usage example
if __name__ == "__main__":
    # Test the YFinance manager
    try:
        yfm = YFinanceManager()
        
        # Test market status
        status = yfm.get_market_status()
        logger.info(f"Market status: {status}")
        
        # Test scheduled data fetch
        data = yfm.fetch_scheduled_data()
        logger.info(f"Scheduled data: {len(data)} tickers")
        
        # Test latest prices
        prices = yfm.get_latest_prices(['AAPL', 'MSFT'])
        logger.info(f"Latest prices: {prices}")
        
    except Exception as e:
        logger.error(f"Error in YFinance manager test: {e}")
