import pandas as pd

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0)
    direction = direction.gt(0).astype(int) - direction.lt(0).astype(int)
    obv = (direction * volume).cumsum()
    return obv

def calculate_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMA & EMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # RSI
    df['rsi'] = calculate_rsi(df['close'])

    # OBV (fixed version)
    df['obv'] = calculate_obv(df['close'], df['volume'])

    # VWAP
    if {'high', 'low', 'close', 'volume'}.issubset(df.columns):
        df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])

    df.dropna(inplace=True)
    return df
