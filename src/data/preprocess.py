import pandas as pd
from .fetch_price import fetch_price_data

def build_model_input(ticker, timeframes, start, end):
    """
    Builds a combined multi-timeframe feature set for a given ticker.

    Args:
        ticker (str): Stock symbol
        timeframes (list): e.g., ["1d", "1wk"]
        start (str): Start date (YYYY-MM-DD)
        end (str): End date (YYYY-MM-DD)

    Returns:
        DataFrame: Combined multi-timeframe features
    """
    df_dict = {}

    for tf in timeframes:
        print(f"[INFO] Fetching {ticker} data for {tf}...")
        df = fetch_price_data(ticker, interval=tf, start=start, end=end)

        if df.empty:
            print(f"[WARN] No data returned for {ticker} at {tf}")
            continue

        # Add timeframe prefix to column names
        df = df.add_prefix(f"{tf}_")
        df_dict[tf] = df

    if not df_dict:
        raise ValueError(f"No valid data could be retrieved for {ticker}.")

    # Align timeframes and merge
    combined = pd.concat(df_dict.values(), axis=1, join="outer")
    combined = combined.dropna()
    return combined
