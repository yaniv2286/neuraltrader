"""
Comprehensive Technical Indicators Library
High-probability signals for entry/exit decisions.
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
    - Chart patterns (simplified)
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


def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all technical indicators for a DataFrame.
    Input: DataFrame with columns [open, high, low, close, volume]
    Output: DataFrame with all features added
    """
    df = df.copy()
    ti = TechnicalIndicators
    
    # Price data
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # === TREND INDICATORS ===
    df['sma_5'] = ti.sma(close, 5)
    df['sma_10'] = ti.sma(close, 10)
    df['sma_20'] = ti.sma(close, 20)
    df['sma_50'] = ti.sma(close, 50)
    df['sma_200'] = ti.sma(close, 200)
    
    df['ema_9'] = ti.ema(close, 9)
    df['ema_21'] = ti.ema(close, 21)
    df['ema_50'] = ti.ema(close, 50)
    
    # MA crossovers
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['price_above_sma_200'] = (close > df['sma_200']).astype(int)
    
    # MACD
    macd, signal, hist = ti.macd(close)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = hist
    df['macd_cross'] = (macd > signal).astype(int)
    
    # ADX
    adx, plus_di, minus_di = ti.adx(high, low, close)
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['adx_strong_trend'] = (adx > 25).astype(int)
    
    # === MOMENTUM INDICATORS ===
    df['rsi'] = ti.rsi(close, 14)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    stoch_k, stoch_d = ti.stochastic(high, low, close)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['stoch_oversold'] = (stoch_k < 20).astype(int)
    df['stoch_overbought'] = (stoch_k > 80).astype(int)
    
    df['cci'] = ti.cci(high, low, close)
    df['williams_r'] = ti.williams_r(high, low, close)
    df['roc'] = ti.price_rate_of_change(close, 10)
    df['momentum'] = ti.momentum(close, 10)
    df['price_velocity'] = ti.price_velocity(close)
    df['price_acceleration'] = ti.price_acceleration(close)
    
    # === VOLATILITY INDICATORS ===
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(close)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_pct_b'] = ti.bollinger_pct_b(close)
    df['bb_bandwidth'] = ti.bollinger_bandwidth(close)
    df['bb_squeeze'] = (df['bb_bandwidth'] < df['bb_bandwidth'].rolling(50).quantile(0.2)).astype(int)
    
    df['atr'] = ti.atr(high, low, close)
    df['atr_percent'] = ti.atr_percent(high, low, close)
    
    kc_upper, kc_middle, kc_lower = ti.keltner_channels(high, low, close)
    df['kc_upper'] = kc_upper
    df['kc_lower'] = kc_lower
    
    # === VOLUME INDICATORS ===
    df['obv'] = ti.obv(close, volume)
    df['obv_sma'] = ti.sma(df['obv'], 20)
    df['mfi'] = ti.mfi(high, low, close, volume)
    df['vwap'] = ti.vwap(high, low, close, volume)
    df['volume_sma'] = ti.sma(volume, 20)
    df['volume_ratio'] = volume / df['volume_sma']
    df['volume_spike'] = ti.volume_spike(volume).astype(int)
    
    # === SUPPORT/RESISTANCE ===
    support, resistance = ti.support_resistance(high, low, close)
    df['support'] = support
    df['resistance'] = resistance
    df['dist_from_support'] = ti.distance_from_support(close, low)
    df['dist_from_resistance'] = ti.distance_from_resistance(close, high)
    df['sr_position'] = (close - support) / (resistance - support)  # 0=at support, 1=at resistance
    
    pivot, r1, s1, r2, s2 = ti.pivot_points(high, low, close)
    df['pivot'] = pivot
    df['pivot_r1'] = r1
    df['pivot_s1'] = s1
    
    # === PATTERN DETECTION ===
    hh, ll = ti.higher_highs_lower_lows(high, low)
    df['higher_highs'] = hh
    df['lower_lows'] = ll
    df['trend_strength'] = ti.trend_strength(close)
    
    # === DERIVED FEATURES ===
    df['returns_1d'] = close.pct_change(1)
    df['returns_5d'] = close.pct_change(5)
    df['returns_20d'] = close.pct_change(20)
    df['log_returns'] = np.log(close / close.shift(1))
    
    # Volatility of returns
    df['volatility_5d'] = df['returns_1d'].rolling(5).std()
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()
    
    # Price position
    df['price_pct_from_high_52w'] = (close - high.rolling(252).max()) / high.rolling(252).max() * 100
    df['price_pct_from_low_52w'] = (close - low.rolling(252).min()) / low.rolling(252).min() * 100
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def get_feature_columns() -> list:
    """Get list of all feature column names (excluding OHLCV and target)"""
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date']
    # Generate on sample to get column names
    sample = pd.DataFrame({
        'open': [100]*300, 'high': [101]*300, 'low': [99]*300, 
        'close': [100.5]*300, 'volume': [1000000]*300
    })
    sample = generate_all_features(sample)
    return [c for c in sample.columns if c not in exclude]
