"""
Universe hygiene and data-staleness checks shared by the live orchestrator
and the unified backtest.

Rule 3.1 (data freshness) was only enforced globally; this module enforces it
per ticker so that delisted / acquired / renamed symbols whose parquet stopped
updating can never be scored, bought, or held.
"""

import numpy as np
import pandas as pd

MAX_STALE_BDAYS    = 3            # last bar must be within 3 business days of as_of
MIN_PRICE          = 10.0         # $10 minimum price
MIN_DOLLAR_VOLUME  = 5_000_000    # 20d average dollar volume floor
DOLLAR_VOL_WINDOW  = 20


def is_otc_ticker(ticker: str) -> bool:
    """Five-letter symbols ending in F or Y are OTC / pink-sheet ADRs (AETUF, AKZOY, NTTYY)."""
    t = ticker.upper().split('-')[0]
    return len(t) == 5 and t[-1] in ('F', 'Y')


def last_bar_date(df: pd.DataFrame) -> pd.Timestamp:
    if df is None or len(df) == 0:
        return pd.NaT
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Timestamp(df.index[-1]).normalize()
    for col in ('date', 'Date'):
        if col in df.columns:
            return pd.Timestamp(df[col].iloc[-1]).normalize()
    return pd.NaT


def staleness_bdays(df: pd.DataFrame, as_of=None) -> int:
    """Business days between the last bar and as_of (today by default). Large number if unknown."""
    last = last_bar_date(df)
    if pd.isna(last):
        return 10_000
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    if last >= as_of:
        return 0
    return int(np.busday_count(last.date(), as_of.date()))


def is_stale(df: pd.DataFrame, as_of=None, max_bdays: int = MAX_STALE_BDAYS) -> bool:
    return staleness_bdays(df, as_of) > max_bdays


def avg_dollar_volume(df: pd.DataFrame, window: int = DOLLAR_VOL_WINDOW) -> float:
    cols = {c.lower(): c for c in df.columns}
    close = cols.get('adjclose') or cols.get('close')
    vol = cols.get('adjvolume') or cols.get('volume')
    if close is None or vol is None or len(df) < window:
        return 0.0
    tail = df.iloc[-window:]
    return float((tail[close] * tail[vol]).mean())


def universe_check(ticker: str, df: pd.DataFrame, as_of=None,
                   min_price: float = MIN_PRICE,
                   min_dollar_volume: float = MIN_DOLLAR_VOLUME,
                   max_stale_bdays: int = MAX_STALE_BDAYS):
    """
    Returns (ok, reason). reason is one of:
      OTC, STALE, NO_DATA, LOW_PRICE, LOW_DOLLAR_VOLUME, or '' when ok.
    """
    if is_otc_ticker(ticker):
        return False, 'OTC'
    if df is None or len(df) == 0:
        return False, 'NO_DATA'
    if is_stale(df, as_of, max_stale_bdays):
        return False, 'STALE'
    cols = {c.lower(): c for c in df.columns}
    close = cols.get('adjclose') or cols.get('close')
    if close is None:
        return False, 'NO_DATA'
    price = float(df[close].iloc[-1])
    if price < min_price:
        return False, 'LOW_PRICE'
    if avg_dollar_volume(df) < min_dollar_volume:
        return False, 'LOW_DOLLAR_VOLUME'
    return True, ''
