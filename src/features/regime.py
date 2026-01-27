import pandas as pd

def detect_regime(df: pd.DataFrame, short_window=5, long_window=10) -> pd.Series:
    """
    Detects market regime using moving average crossover logic:
    - 'bull' if short MA > long MA
    - 'bear' if short MA < long MA
    - 'range' if close is near MA within small tolerance
    """
    df = df.copy()
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()

    tolerance = 0.01  # ~1% band for 'range'
    regime = []

    for i in range(len(df)):
        if pd.isna(df['sma_short'].iloc[i]) or pd.isna(df['sma_long'].iloc[i]):
            regime.append(None)
            continue

        short = df['sma_short'].iloc[i]
        long = df['sma_long'].iloc[i]
        close = df['close'].iloc[i]

        if abs(short - long) / long < tolerance:
            regime.append('range')
        elif short > long:
            regime.append('bull')
        else:
            regime.append('bear')

    return pd.Series(regime, index=df.index, name="regime")


def detect_multi_timeframe_regimes(df_dict: dict) -> pd.DataFrame:
    """
    Apply regime detection to multiple timeframes.
    Input: { '1d': df_1d, '1h': df_1h }
    Output: DataFrame with regime_1d, regime_1h columns
    """
    regime_df = pd.DataFrame(index=next(iter(df_dict.values())).index)

    for tf, df in df_dict.items():
        regime_series = detect_regime(df)
        regime_df[f"regime_{tf}"] = regime_series

    return regime_df
