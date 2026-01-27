import time
import yfinance as yf
import pandas as pd
import os
import hashlib

def safe_download(ticker, start, end, interval="1d", retries=5):
    import time
    import yfinance as yf

    last_exc = None

    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if not df.empty:
                return df
            else:
                print(f"[WARN] Empty dataframe for {ticker} (attempt {attempt+1})")
        except Exception as e:
            last_exc = e
            print(f"[WARN] Attempt {attempt+1} failed: {e}")

        wait_s = 10 * (attempt + 1)
        if last_exc is not None:
            msg = str(last_exc).lower()
            if "rate limited" in msg or "too many requests" in msg:
                wait_s = max(wait_s, 30 * (attempt + 1))
        time.sleep(wait_s)

    if last_exc is not None:
        raise ValueError(f"Failed to fetch data for {ticker} after {retries} retries: {last_exc}")
    raise ValueError(f"Failed to fetch data for {ticker} after {retries} retries")


def _cache_key(ticker: str, interval: str, start: str, end: str) -> str:
    raw = f"{ticker}|{interval}|{start}|{end}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_path(ticker: str, interval: str, start: str, end: str) -> str:
    key = _cache_key(ticker, interval, start, end)
    base = os.path.join(os.path.dirname(__file__), "cache")
    return os.path.join(base, f"{ticker}_{interval}_{key}.csv")


def fetch_price_data(ticker: str, interval: str = "1d", start: str = None, end: str = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical price data using yfinance.
    Normalize all timestamps to tz-naive (UTC).
    """
    try:
        cache_path = None
        if use_cache and start is not None and end is not None:
            cache_path = _cache_path(ticker, interval, start, end)
            if os.path.exists(cache_path):
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
                cached.index.name = "datetime"
                # Ensure numeric columns are numeric (avoid object dtypes)
                for col in cached.columns:
                    cached[col] = pd.to_numeric(cached[col], errors='coerce')
                return cached

        df = safe_download(ticker, start=start, end=end, interval=interval)

        df.dropna(inplace=True)
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        # Normalize datetime index
        df.index = df.index.tz_convert(None) if df.index.tz else df.index
        df.index.name = "datetime"

        if use_cache and cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)

        return df
    except Exception as e:
        print(f"[fetch_price_data] Error fetching {ticker} - {e}")
        return pd.DataFrame()
