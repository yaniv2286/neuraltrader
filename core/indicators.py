"""
Core Technical Indicators - RSI, ATR, Volatility, Momentum
Protected module containing all technical indicator calculations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """
    Complete technical indicator library with:
    - Trend indicators (MA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, CCI)
    - Volatility indicators (Bollinger, ATR, Keltner)
    - Volume indicators (OBV, MFI, VWAP)
    - Support/Resistance levels
    """
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands - volatility indicator"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def bollinger_pct_b(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger %B - position within bands (0-1)"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close, period, std_dev)
        return (close - lower) / (upper - lower)
    
    @staticmethod
    def bollinger_bandwidth(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Bandwidth - volatility measure"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close, period, std_dev)
        return (upper - lower) / middle
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range - volatility"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR as percentage of price"""
        atr = TechnicalIndicators.atr(high, low, close, period)
        return (atr / close) * 100
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mad)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index - trend strength"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, 1) * period
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index - volume-weighted RSI"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, 
                           lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Dynamic Support and Resistance levels"""
        resistance = high.rolling(window=lookback).max()
        support = low.rolling(window=lookback).min()
        return support, resistance
    
    @staticmethod
    def distance_from_support(close: pd.Series, low: pd.Series, lookback: int = 20) -> pd.Series:
        """Distance from support as percentage"""
        support = low.rolling(window=lookback).min()
        return ((close - support) / support) * 100
    
    @staticmethod
    def distance_from_resistance(close: pd.Series, high: pd.Series, lookback: int = 20) -> pd.Series:
        """Distance from resistance as percentage"""
        resistance = high.rolling(window=lookback).max()
        return ((resistance - close) / close) * 100
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Classic Pivot Points"""
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        r1 = 2 * pivot - low.shift(1)
        s1 = 2 * pivot - high.shift(1)
        r2 = pivot + (high.shift(1) - low.shift(1))
        s2 = pivot - (high.shift(1) - low.shift(1))
        return pivot, r1, s1, r2, s2
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                         period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        middle = close.ewm(span=period, adjust=False).mean()
        atr = TechnicalIndicators.atr(high, low, close, period)
        upper = middle + (atr * atr_mult)
        lower = middle - (atr * atr_mult)
        return upper, middle, lower
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels - breakout indicator"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower
    
    @staticmethod
    def price_rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
        """Price Rate of Change (ROC)"""
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    @staticmethod
    def momentum(close: pd.Series, period: int = 10) -> pd.Series:
        """Price Momentum"""
        return close - close.shift(period)
    
    @staticmethod
    def trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
        """Trend strength: distance from MA as % of ATR"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (close - sma) / std
    
    @staticmethod
    def higher_highs_lower_lows(high: pd.Series, low: pd.Series, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
        """Detect higher highs and lower lows pattern"""
        hh = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
        ll = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
        return hh.astype(int), ll.astype(int)
    
    @staticmethod
    def volume_spike(volume: pd.Series, period: int = 20, threshold: float = 2.0) -> pd.Series:
        """Detect volume spikes"""
        avg_volume = volume.rolling(window=period).mean()
        return (volume / avg_volume) > threshold
    
    @staticmethod
    def price_velocity(close: pd.Series, period: int = 5) -> pd.Series:
        """Price velocity (rate of change smoothed)"""
        roc = TechnicalIndicators.price_rate_of_change(close, period)
        return roc.rolling(window=3).mean()
    
    @staticmethod
    def price_acceleration(close: pd.Series, period: int = 5) -> pd.Series:
        """Price acceleration (change in velocity)"""
        velocity = TechnicalIndicators.price_velocity(close, period)
        return velocity.diff()


# Volatility-specific functions
def calculate_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns)"""
    returns = close.pct_change()
    return returns.rolling(window=window).std()


def calculate_historical_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Calculate historical volatility"""
    # 🦅 NUCLEAR MATH SAFETY - Prevent log-zero crashes
    returns = np.log(close.div(close.shift(1).replace(0, np.nan))).fillna(0)
    volatility = returns.rolling(window=window).std()
    if annualize:
        volatility = volatility * np.sqrt(252)  # Annualize assuming 252 trading days
    return volatility


def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson volatility estimator (uses high-low range)"""
    # 🦅 NUCLEAR MATH SAFETY - Prevent log-zero crashes
    hl_ratio = np.log(high.div(low.replace(0, np.nan))).fillna(0)
    parkinson = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2))
    return parkinson.rolling(window=window).mean()


def calculate_garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Garman-Klass volatility estimator"""
    # 🦅 NUCLEAR MATH SAFETY - Prevent log-zero crashes
    hl = np.log(high.div(low.replace(0, np.nan))).fillna(0) ** 2
    co = np.log(close.div(open_.replace(0, np.nan))).fillna(0) ** 2
    gk = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(gk.rolling(window=window).mean())


# Momentum-specific functions
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_momentum_score(close: pd.Series, periods: list = [5, 10, 20, 60]) -> pd.Series:
    """Calculate composite momentum score across multiple periods"""
    momentum_scores = []
    for period in periods:
        momentum = (close / close.shift(period) - 1) * 100
        momentum_scores.append(momentum)
    
    # Average momentum across all periods
    return pd.concat(momentum_scores, axis=1).mean(axis=1)
