import pandas as pd
import numpy as np

def detect_support_resistance(df: pd.DataFrame, window: int = 10, tolerance: float = 0.01) -> pd.DataFrame:
    """
    Detect support and resistance levels. Handles both flat and MultiIndex columns.
    """
    df = df.copy()

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for col in df.columns]

    # Try to locate correct price columns
    close_col = [col for col in df.columns if 'close' in col.lower()][0]
    high_col = [col for col in df.columns if 'high' in col.lower()][0]
    low_col = [col for col in df.columns if 'low' in col.lower()][0]

    df['support_level'] = df[low_col].rolling(window=window, center=True).min()
    df['resistance_level'] = df[high_col].rolling(window=window, center=True).max()

    dist_to_support = (df[close_col] - df['support_level']) / df['support_level']
    dist_to_resist = (df['resistance_level'] - df[close_col]) / df['resistance_level']

    df['near_support'] = dist_to_support < tolerance
    df['near_resistance'] = dist_to_resist < tolerance

    df['near_support'] = df['near_support'].fillna(False)
    df['near_resistance'] = df['near_resistance'].fillna(False)

    return df


def detect_patterns(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Detects basic reversal patterns:
    - Double Top
    - Double Bottom
    - Head and Shoulders
    - Inverse Head and Shoulders
    """
    df = df.copy()

    # Flatten columns if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for c in df.columns]

    # Auto-detect close column
    close_col = [col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()]
    if not close_col:
        raise ValueError("No valid close column found for pattern detection.")
    close = df[close_col[0]]

    # Initialize columns
    df['double_top_flag'] = False
    df['double_bottom_flag'] = False
    df['head_and_shoulders_flag'] = False
    df['inverse_head_and_shoulders_flag'] = False

    for i in range(window * 2, len(df)):
        window_data = close[i - window*2:i]

        # Double Top
        peak1 = window_data[:window].max()
        peak2 = window_data[window:].max()
        valley = window_data[window//2:window + window//2].min()
        if abs(peak1 - peak2) / peak1 < 0.02 and valley < peak1 * 0.98:
            df.at[df.index[i], 'double_top_flag'] = True

        # Double Bottom
        low1 = window_data[:window].min()
        low2 = window_data[window:].min()
        mid = window_data[window//2:window + window//2].max()
        if abs(low1 - low2) / low1 < 0.02 and mid > low1 * 1.02:
            df.at[df.index[i], 'double_bottom_flag'] = True

        # H&S
        p = window_data.values
        if len(p) == window*2:
            l, m = window, window//2
            if p[m] > p[m-2] and p[m] > p[m+2] and p[m] > p[0] and p[m] > p[-1]:
                df.at[df.index[i], 'head_and_shoulders_flag'] = True
            if p[m] < p[m-2] and p[m] < p[m+2] and p[m] < p[0] and p[m] < p[-1]:
                df.at[df.index[i], 'inverse_head_and_shoulders_flag'] = True

    return df


def add_volatility_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Adds volatility features:
    - rolling std of log returns (rolling_volatility)
    - average true range (atr_14)
    """
    df = df.copy()

    # Flatten columns if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for col in df.columns]

    # Find columns
    close_col = [col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()][0]
    high_col = [col for col in df.columns if 'high' in col.lower()][0]
    low_col = [col for col in df.columns if 'low' in col.lower()][0]

    # Log return volatility
    log_return = np.log(df[close_col] / df[close_col].shift(1))
    df['rolling_volatility'] = log_return.rolling(window).std()

    # ATR
    high = df[high_col]
    low = df[low_col]
    prev_close = df[close_col].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df['atr_14'] = tr.rolling(window).mean()

    return df


def add_drawdown_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds drawdown-related features:
    - drawdown_pct: % drop from rolling max close
    - in_drawdown: True if current close is below recent max
    """
    df = df.copy()

    # Flatten MultiIndex if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for col in df.columns]

    # Find close column
    close_col = [col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()][0]

    # Rolling max and drawdown
    rolling_max = df[close_col].rolling(window=window, min_periods=1).max()
    drawdown = (df[close_col] - rolling_max) / rolling_max

    df['drawdown_pct'] = drawdown
    df['in_drawdown'] = df[close_col] < rolling_max
    df['in_drawdown'] = df['in_drawdown'].fillna(False)

    return df

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds candlestick price action features:
    - body_size
    - upper_wick, lower_wick
    - bullish_candle (bool)
    - bearish_engulfing (simple logic)
    """
    df = df.copy()

    # Flatten if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for col in df.columns]

    # Auto-detect OHLC
    open_col = [c for c in df.columns if 'open' in c.lower()][0]
    close_col = [c for c in df.columns if 'close' in c.lower() and 'adj' not in c.lower()][0]
    high_col = [c for c in df.columns if 'high' in c.lower()][0]
    low_col = [c for c in df.columns if 'low' in c.lower()][0]

    open_ = df[open_col]
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    # Body and wick sizes
    df['body_size'] = (close - open_).abs()
    df['upper_wick'] = high - close.where(close > open_, open_)
    df['lower_wick'] = open_.where(close > open_, close) - low

    # Candle sentiment
    df['bullish_candle'] = close > open_

    # Basic bearish engulfing (today's body > yesterday's and in opposite direction)
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    df['bearish_engulfing'] = (
        (prev_close > prev_open) &  # Yesterday bullish
        (close < open_) &           # Today bearish
        (open_ > prev_close) &
        (close < prev_open)
    ).fillna(False)

    return df

def add_lag_features(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Adds lagged versions of key columns: close, volume, rsi, regime
    Example: close_lag_1, rsi_lag_1
    """
    df = df.copy()

    # Flatten if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for col in df.columns]

    # Identify columns to lag
    to_lag = [c for c in df.columns if any(k in c.lower() for k in ['close', 'volume', 'rsi', 'regime']) and 'adj' not in c.lower()]

    for col in to_lag:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds datetime-based features:
    - hour_of_day
    - day_of_week
    - month
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for time features.")

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df

def add_momentum_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Adds momentum-based indicators:
    - momentum: price difference over N periods
    - roc: rate of change (% change over N periods)
    """
    df = df.copy()

    # Flatten MultiIndex if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(str(c) for c in col if c) for c in df.columns]

    # Find 'close' column
    close_col = [c for c in df.columns if 'close' in c.lower() and 'adj' not in c.lower()][0]
    close = df[close_col]

    df['momentum'] = close - close.shift(window)
    df['roc'] = close.pct_change(window)

    return df
