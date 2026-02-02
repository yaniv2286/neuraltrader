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
import numpy as np
import pytz
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
        
        # S&P 100 ticker universe (Phase 6.1 production)
        self.sp100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B', 'UNH',
            'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'CVX', 'HD', 'ABBV', 'MRK', 'LLY', 'PEP', 'KO',
            'COST', 'TMO', 'CSCO', 'PFE', 'MCD', 'CRM', 'BAC', 'ADBE', 'WMT', 'CMCSA', 'DIS', 'NFLX',
            'ABT', 'VZ', 'ORCL', 'TXN', 'AMD', 'LIN', 'PM', 'UPS', 'NKE', 'HON', 'UNP', 'RTX', 'INTU',
            'LOW', 'SPGI', 'MS', 'QCOM', 'COP', 'IBM', 'GE', 'AMAT', 'CAT', 'GS', 'ISRG', 'DE', 'BKNG',
            'ELV', 'PLD', 'SBUX', 'MDT', 'BLK', 'GILD', 'TJX', 'NOW', 'ADP', 'C', 'MMC', 'AMT', 'REGN',
            'MO', 'PYPL', 'CB', 'CI', 'ADI', 'MDLZ', 'VRTX', 'ZTS', 'SYK', 'CME', 'AMGN', 'FISV', 'SLB',
            'T', 'LMT', 'MU', 'CVS', 'DUK', 'ITW', 'EQIX', 'ANTM', 'CL', 'ICE', 'SHERW'
        ]
        
        # Use S&P 100 as default universe
        self.curated_tickers = self.sp100_tickers
        
        logger.info("Data Manager initialized")
        logger.info(f"S&P 100 Universe: {len(self.sp100_tickers)} tickers")
    
    def calculate_53_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the full suite of 53 technical indicators for ML scoring
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 53 technical indicators added
        """
        try:
            if len(df) < 50:  # Need sufficient data for indicators
                logger.warning("Insufficient data for 53 indicators calculation")
                return df
            
            df_indicators = df.copy()
            
            # Price-based indicators (15)
            # Moving Averages
            df_indicators['sma_5'] = df['close'].rolling(window=5).mean()
            df_indicators['sma_10'] = df['close'].rolling(window=10).mean()
            df_indicators['sma_20'] = df['close'].rolling(window=20).mean()
            df_indicators['sma_50'] = df['close'].rolling(window=50).mean()
            df_indicators['ema_12'] = df['close'].ewm(span=12).mean()
            df_indicators['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Price relative to moving averages
            df_indicators['price_vs_sma5'] = (df['close'] - df_indicators['sma_5']) / df_indicators['sma_5']
            df_indicators['price_vs_sma20'] = (df['close'] - df_indicators['sma_20']) / df_indicators['sma_20']
            df_indicators['price_vs_sma50'] = (df['close'] - df_indicators['sma_50']) / df_indicators['sma_50']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df_indicators['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            df_indicators['bb_upper'] = df_indicators['bb_middle'] + (df['close'].rolling(window=bb_period).std() * bb_std)
            df_indicators['bb_lower'] = df_indicators['bb_middle'] - (df['close'].rolling(window=bb_period).std() * bb_std)
            df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
            df_indicators['bb_position'] = (df['close'] - df_indicators['bb_lower']) / (df_indicators['bb_upper'] - df_indicators['bb_lower'])
            
            # Volatility indicators (8)
            # ATR and related
            df_indicators['atr_14'] = self.calculate_atr(df, 14)
            df_indicators['atr_ratio'] = df_indicators['atr_14'] / df['close']
            
            # Historical volatility
            df_indicators['volatility_10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
            df_indicators['volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Price ranges
            df_indicators['high_low_ratio'] = df['high'] / df['low']
            df_indicators['close_open_ratio'] = df['close'] / df['open']
            df_indicators['price_range'] = (df['high'] - df['low']) / df['close']
            
            # Volume indicators (6)
            # Volume moving averages
            df_indicators['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df_indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df_indicators['volume_ratio'] = df['volume'] / df_indicators['volume_sma_20']
            
            # On-Balance Volume
            df_indicators['obv'] = (np.where(df['close'] > df['close'].shift(), df['volume'], 
                                          np.where(df['close'] < df['close'].shift(), -df['volume'], 0))).cumsum()
            df_indicators['obv_sma'] = df_indicators['obv'].rolling(window=10).mean()
            
            # Volume Price Trend
            df_indicators['vpt'] = (df['volume'] * (df['close'].pct_change())).cumsum()
            
            # Momentum indicators (12)
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_indicators['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df_indicators['macd'] = ema_12 - ema_26
            df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9).mean()
            df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
            
            # Stochastic Oscillator
            lowest_low = df['low'].rolling(window=14).min()
            highest_high = df['high'].rolling(window=14).max()
            df_indicators['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df_indicators['stoch_d'] = df_indicators['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df_indicators['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            
            # Rate of Change
            df_indicators['roc_5'] = df['close'].pct_change(5) * 100
            df_indicators['roc_10'] = df['close'].pct_change(10) * 100
            
            # Commodity Channel Index
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df_indicators['cci'] = (tp - sma_tp) / (0.015 * mad)
            
            # Pattern indicators (12)
            # Price patterns
            df_indicators['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df_indicators['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            df_indicators['inside_day'] = ((df['high'] < df['high'].shift(1)) & 
                                         (df['low'] > df['low'].shift(1))).astype(int)
            df_indicators['outside_day'] = ((df['high'] > df['high'].shift(1)) & 
                                          (df['low'] < df['low'].shift(1))).astype(int)
            
            # Gap indicators
            df_indicators['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            df_indicators['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
            
            # Doji patterns
            body_size = abs(df['close'] - df['open'])
            df_indicators['doji'] = (body_size < (df['high'] - df['low']) * 0.1).astype(int)
            
            # Hammer/Hanging Man
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            body = abs(df['close'] - df['open'])
            df_indicators['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)).astype(int)
            
            # Engulfing patterns
            bullish_engulfing = ((df['open'].shift(1) > df['close'].shift(1)) &  # Previous red candle
                               (df['close'] > df['open']) &  # Current green candle
                               (df['open'] < df['close'].shift(1)) &  # Open below previous close
                               (df['close'] > df['open'].shift(1)))  # Close above previous open
            df_indicators['bullish_engulfing'] = bullish_engulfing.astype(int)
            
            bearish_engulfing = ((df['open'].shift(1) < df['close'].shift(1)) &  # Previous green candle
                               (df['close'] < df['open']) &  # Current red candle
                               (df['open'] > df['close'].shift(1)) &  # Open above previous close
                               (df['close'] < df['open'].shift(1)))  # Close below previous open
            df_indicators['bearish_engulfing'] = bearish_engulfing.astype(int)
            
            # Trend indicators (10)
            # ADX (simplified)
            df_indicators['adx'] = self._calculate_adx(df, 14)
            
            # Trend strength
            df_indicators['trend_strength'] = abs(df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            
            # Price momentum
            df_indicators['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df_indicators['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df_indicators['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Acceleration
            df_indicators['acceleration'] = df_indicators['momentum_5'] - df_indicators['momentum_5'].shift(1)
            
            # Support/Resistance levels
            df_indicators['resistance_distance'] = (df['high'].rolling(20).max() - df['close']) / df['close']
            df_indicators['support_distance'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
            
            # Seasonal patterns
            df_indicators['day_of_week'] = df.index.dayofweek
            df_indicators['month'] = df.index.month
            
            logger.info(f"Calculated 53 indicators for {len(df_indicators)} data points")
            return df_indicators
            
        except Exception as e:
            logger.error(f"Error calculating 53 indicators: {e}")
            return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed values
            atr = tr.rolling(window=period).mean()
            plus_di = pd.Series(plus_dm).rolling(window=period).mean() / atr * 100
            minus_di = pd.Series(minus_dm).rolling(window=period).mean() / atr * 100
            
            # ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.Series(index=df.index, data=0)
    
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
        Get current market status with timezone enforcement
        
        Returns:
            Dictionary with market status information
        """
        try:
            # Get current time in US/Eastern timezone
            eastern = pytz.timezone('US/Eastern')
            now_eastern = datetime.now(eastern)
            
            # Get market clock
            clock = self.api.get_clock()
            
            # Check if within trading window (9:45 AM - 3:45 PM EST)
            trading_start = now_eastern.replace(hour=9, minute=45, second=0, microsecond=0)
            trading_end = now_eastern.replace(hour=15, minute=45, second=0, microsecond=0)
            
            in_trading_window = trading_start <= now_eastern <= trading_end
            
            return {
                'is_open': clock.is_open,
                'timestamp': clock.timestamp,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'current_time_est': now_eastern,
                'current_time_ist': now_eastern.astimezone(pytz.timezone('Asia/Jerusalem')),
                'in_trading_window': in_trading_window,
                'trading_start_est': trading_start,
                'trading_end_est': trading_end,
                'israel_time': now_eastern.astimezone(pytz.timezone('Asia/Jerusalem')).strftime('%H:%M:%S'),
                'ny_time': now_eastern.strftime('%H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_open': False,
                'timestamp': None,
                'next_open': None,
                'next_close': None,
                'current_time_est': None,
                'current_time_ist': None,
                'in_trading_window': False,
                'trading_start_est': None,
                'trading_end_est': None,
                'israel_time': 'Unknown',
                'ny_time': 'Unknown'
            }
    
    def check_daily_execution_limit(self) -> bool:
        """
        Check if the bot has already executed today
        
        Returns:
            True if bot can execute, False if already executed today
        """
        try:
            # Check for execution log file
            execution_log_path = 'logs/daily_execution.log'
            
            if not os.path.exists(execution_log_path):
                return True  # No execution today yet
            
            # Read last execution date
            with open(execution_log_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return True
            
            # Get last execution date
            last_line = lines[-1].strip()
            if last_line:
                last_execution = datetime.strptime(last_line, '%Y-%m-%d').date()
                today = datetime.now().date()
                
                if last_execution == today:
                    logger.info("Bot already executed today")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking daily execution limit: {e}")
            return True  # Allow execution if error
    
    def log_daily_execution(self):
        """Log today's execution to prevent multiple executions"""
        try:
            execution_log_path = 'logs/daily_execution.log'
            os.makedirs(os.path.dirname(execution_log_path), exist_ok=True)
            
            today = datetime.now().date()
            
            with open(execution_log_path, 'a') as f:
                f.write(f"{today}\n")
            
            logger.info(f"Logged daily execution for {today}")
            
        except Exception as e:
            logger.error(f"Error logging daily execution: {e}")
    
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
