#!/usr/bin/env python3
"""
NEURALTRADER - UNIFIED BACKTEST ENGINE (2000-Present)
======================================================

Single pipeline for all modes: backtest / simulation / paper / live
Data source : data/raw/*.parquet  (2,183 tickers, pre-loaded at startup)
AI models   : 64-feature base ensemble (XGBoost + LightGBM + RF)
Risk laws   : ARCHITECTURE.md / Rule 2.1-2.4 enforced

Entry : AI confidence > 0.60
Exit  : 10% stop-loss | 30% take-profit | 20-day timeout (Phase 11.1)
Size  : 1% portfolio risk / annualized-vol * conf^2  (confidence-weighted)
CB    : 12% drawdown -> 8-day cooldown (Uncle Point)
Regime: SPY > 50-day SMA required for new entries

PERFORMANCE CACHE (data/cache/backtest/):
    - Confidence scores cached per ticker as .npz files
    - Cache key = hash(parquet_mtime + model_mtime)
    - Cold run (~18 min) -> Warm run (~25 sec)
    - Auto-invalidated when parquet or model files change
    - Use --no-cache to force full recompute

Usage:
    python scripts/run_full_backtest.py                     # full 26yr
    python scripts/run_full_backtest.py --start 2000-01-01  # explicit
    python scripts/run_full_backtest.py --tickers AAPL MSFT # subset
    python scripts/run_full_backtest.py --no-cache          # force recompute
"""

import os
import sys
import json
import hashlib
import argparse
import traceback
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RAW_DATA_DIR  = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR    = PROJECT_ROOT / 'models'
LOG_DIR       = PROJECT_ROOT / 'logs'
CACHE_DIR     = PROJECT_ROOT / 'data' / 'cache' / 'backtest'
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'backtest.log', mode='w', encoding='utf-8'),
    ],
)
logger = logging.getLogger('backtest')

# ---------------------------------------------------------------------------
# Production imports
# ---------------------------------------------------------------------------
from core.feature_engineer import FeatureEngineer
from core.ai_models import EnsemblePredictor


# ===========================================================================
# Disk cache helpers  (Cold: ~18 min -> Warm: ~25 sec)
# ===========================================================================

def _model_mtime_hash() -> str:
    """Hash of all model pkl mtimes — changes when models are retrained."""
    models_dir = PROJECT_ROOT / 'models'
    mtimes = sorted(str(f.stat().st_mtime) for f in models_dir.glob('*.pkl'))
    return hashlib.md5('|'.join(mtimes).encode()).hexdigest()[:8]


def _ticker_cache_key(parquet_path: Path, model_hash: str) -> str:
    """Per-ticker cache key = hash(parquet_mtime + model_hash)."""
    pmtime = str(parquet_path.stat().st_mtime)
    return hashlib.md5(f'{pmtime}|{model_hash}'.encode()).hexdigest()[:12]


def _cache_load(ticker: str, cache_key: str) -> Optional[dict]:
    """
    Load cached confidence scores for a ticker.
    Returns {'dates': np.ndarray, 'confidence': np.ndarray} or None on miss/stale.
    """
    meta_path = CACHE_DIR / f'{ticker}.meta'
    data_path = CACHE_DIR / f'{ticker}.npz'
    if not meta_path.exists() or not data_path.exists():
        return None
    try:
        if meta_path.read_text().strip() != cache_key:
            return None  # stale — parquet or model changed
        npz = np.load(data_path, allow_pickle=True)
        return {'dates': npz['dates'], 'confidence': npz['confidence'].astype(float)}
    except Exception:
        return None


def _cache_save(ticker: str, cache_key: str,
                dates: np.ndarray, confidence: np.ndarray):
    """Persist confidence scores to disk cache."""
    try:
        np.savez_compressed(CACHE_DIR / f'{ticker}.npz',
                            dates=dates, confidence=confidence)
        (CACHE_DIR / f'{ticker}.meta').write_text(cache_key)
    except Exception as e:
        logger.warning(f'[CACHE] Failed to save {ticker}: {e}')

# ---------------------------------------------------------------------------
# Constants  (ARCHITECTURE.md / Rule 2.1-2.4)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.44    # AI entry gate — Phase 12: p98 of score dist (base rate 31.9%)
STOP_LOSS_PCT        = 0.10    # 10% stop-loss exit
TAKE_PROFIT_PCT      = 0.30    # 30% take-profit — let winners run (was 20%)
MAX_HOLD_DAYS        = 20      # 20-day timeout — cut dead weight faster (was 30)
MAX_RISK_PER_TRADE   = 0.008   # 0.8% portfolio risk per trade — tuned for <12% DD
UNCLE_POINT_DD       = 0.10    # 10% drawdown circuit breaker — fires before 12% peak DD
COOLDOWN_DAYS        = 8       # cooldown days after uncle point
MIN_HISTORY_ROWS     = 60      # minimum rows to generate valid features
MIN_PRICE            = 5.0     # minimum stock price filter (blocks penny stocks)
MIN_AVG_VOLUME       = 100_000  # minimum 20-day avg daily volume (blocks illiquid stocks)
MAX_POSITIONS        = 20      # hard cap on concurrent positions (prevents overexposure)
SPY_REGIME_SMA       = 100     # SPY must be above this SMA to allow new entries (stronger bear filter)
RE_ENTRY_COOLDOWN    = 5       # days to wait before re-entering a stopped-out ticker
CONF_SIZE_POWER      = 2.0     # confidence^N multiplier on position size (rewards high-conf signals)


# ===========================================================================
# Data loader: pre-load all parquet files into memory at startup
# ===========================================================================

def load_all_parquet(data_dir: Path,
                     ticker_filter: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load all *.parquet files from data/raw/ into memory keyed by ticker."""
    parquet_files = sorted(data_dir.glob('*.parquet'))
    logger.info(f'[DATA] Found {len(parquet_files)} parquet files in {data_dir}')

    market_data: Dict[str, pd.DataFrame] = {}
    upper_filter = [t.upper() for t in ticker_filter] if ticker_filter else None

    for fpath in parquet_files:
        ticker = fpath.stem.upper()
        if upper_filter and ticker not in upper_filter:
            continue
        try:
            df = pd.read_parquet(fpath)
            # Normalise index to tz-naive DatetimeIndex
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=False)
                df = df.set_index('date')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=False)
                df = df.set_index('Date')
            else:
                df.index = pd.to_datetime(df.index, utc=False)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.sort_index(inplace=True)
            df.columns = [c.lower() for c in df.columns]
            if 'close' not in df.columns:
                continue   # skip tickers with no close price
            market_data[ticker] = df
        except Exception as e:
            logger.warning(f'[WARN] Could not load {fpath.name}: {e}')

    logger.info(f'[DATA] Loaded {len(market_data)} tickers into memory')
    return market_data


# ===========================================================================
# Feature generation helper
# ===========================================================================

def build_features_for_ticker(df: pd.DataFrame,
                               feature_engineer: FeatureEngineer,
                               expected_features: List[str]) -> Optional[pd.DataFrame]:
    """Generate full feature DataFrame for a ticker.  Returns None on failure."""
    if df is None or len(df) < MIN_HISTORY_ROWS:
        return None
    try:
        feat_df, _ = feature_engineer.create_features(df, target_type='direction')
        if feat_df is None or feat_df.empty:
            return None
        # Pad any missing columns with zeros
        for col in expected_features:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[expected_features]
        # Align date index
        feat_df.index = df.index[-len(feat_df):]
        return feat_df
    except Exception as e:
        logger.debug(f'[FEAT] error: {e}')
        return None


# ===========================================================================
# Unified Backtest Engine
# ===========================================================================

class UnifiedBacktest:
    """
    Unified backtest engine - identical logic used by
    backtest / simulation / paper / live modes.
    """

    def __init__(self,
                 start_date: str = '2000-01-01',
                 end_date: Optional[str] = None,
                 initial_cash: float = 100_000.0,
                 ticker_filter: Optional[List[str]] = None,
                 use_cache: bool = True):

        self.start_date    = pd.to_datetime(start_date)
        self.end_date      = pd.to_datetime(end_date) if end_date else pd.Timestamp.today().normalize()
        self.initial_cash  = initial_cash
        self.ticker_filter = ticker_filter
        self.use_cache     = use_cache

        # Portfolio state
        self.cash             = initial_cash
        self.portfolio_value  = initial_cash
        self.peak_value       = initial_cash
        self.cooldown_until: Optional[pd.Timestamp] = None

        # {ticker: {shares, entry_price, entry_date, entry_confidence}}
        self.positions: Dict[str, Dict] = {}
        self.trades:    List[Dict]      = []
        self.equity_curve: List[Dict]   = []
        # Per-ticker stop-loss cooldown: {ticker: earliest_re_entry_date}
        self._stopped_out: Dict[str, pd.Timestamp] = {}

        # AI + features
        logger.info('[INIT] Loading EnsemblePredictor (sentiment models)...')
        self.ensemble = EnsemblePredictor()
        self.feature_engineer = FeatureEngineer()
        self.expected_features: List[str] = self.ensemble.feature_names or []
        logger.info(f'[INIT] Ensemble loaded | {len(self.expected_features)} features')

        # AI Regime Classifier (Phase 12) — optional drop-in replacement for SPY SMA rule
        self._regime_clf    = None
        self._regime_scaler = None
        self._regime_meta   = None
        self._regime_feat_names: List[str] = []
        self._regime_cache: Dict[pd.Timestamp, int] = {}  # date -> regime code
        _clf_path    = MODELS_DIR / 'regime_classifier.pkl'
        _scaler_path = MODELS_DIR / 'regime_scaler.pkl'
        _meta_path   = MODELS_DIR / 'regime_classifier_meta.json'
        if _clf_path.exists() and _scaler_path.exists():
            try:
                import pickle as _pkl
                with open(_clf_path, 'rb') as f:
                    self._regime_clf = _pkl.load(f)
                with open(_scaler_path, 'rb') as f:
                    self._regime_scaler = _pkl.load(f)
                if _meta_path.exists():
                    import json as _json
                    self._regime_meta = _json.loads(_meta_path.read_text())
                    self._regime_feat_names = self._regime_meta.get('feature_names', [])
                    self._regime_thresholds = self._regime_meta.get('regime_thresholds',
                        {'0': None, '1': 0.72, '2': 0.65})
                logger.info('[INIT] AI Regime Classifier loaded (Phase 12)')
            except Exception as _e:
                logger.warning(f'[INIT] Regime classifier load failed: {_e} — using SMA fallback')
                self._regime_clf = None
        else:
            logger.info('[INIT] No regime_classifier.pkl found — using SPY SMA100 rule')

        # Data
        logger.info('[INIT] Pre-loading all parquet data...')
        self.market_data = load_all_parquet(RAW_DATA_DIR, ticker_filter)
        self.universe    = sorted(self.market_data.keys())
        logger.info(f'[INIT] Universe: {len(self.universe)} tickers')

        # Pre-compute features + confidence scores once for every ticker
        self._model_hash = _model_mtime_hash()
        cache_status = 'DISABLED' if not use_cache else 'ENABLED'
        logger.info(f'[INIT] Pre-computing confidence scores | cache={cache_status} | model_hash={self._model_hash}')
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.confidence_cache: Dict[str, pd.Series] = {}  # {ticker: Series(date->confidence)}
        self._precompute_features()

    # -----------------------------------------------------------------------
    # Pre-computation
    # -----------------------------------------------------------------------

    def _precompute_features(self):
        ok = skip = cache_hits = 0
        total = len(self.universe)

        for i, ticker in enumerate(self.universe):
            # --- Cache lookup ---
            if self.use_cache:
                parquet_path = RAW_DATA_DIR / f'{ticker}.parquet'
                if parquet_path.exists():
                    ckey   = _ticker_cache_key(parquet_path, self._model_hash)
                    cached = _cache_load(ticker, ckey)
                    if cached is not None:
                        dates_idx = pd.DatetimeIndex(
                            pd.to_datetime(cached['dates']).tz_localize(None)
                        )
                        self.confidence_cache[ticker] = pd.Series(
                            cached['confidence'], index=dates_idx
                        )
                        cache_hits += 1
                        ok += 1
                        continue

            # --- Cache miss: compute from scratch ---
            df   = self.market_data[ticker]
            feat = build_features_for_ticker(df, self.feature_engineer, self.expected_features)
            if feat is not None and not feat.empty:
                try:
                    scores = self.ensemble.predict_batch(feat)
                    self.confidence_cache[ticker] = pd.Series(scores, index=feat.index)
                    # Save to cache for next run
                    if self.use_cache:
                        parquet_path = RAW_DATA_DIR / f'{ticker}.parquet'
                        if parquet_path.exists():
                            ckey = _ticker_cache_key(parquet_path, self._model_hash)
                            _cache_save(
                                ticker, ckey,
                                feat.index.values.astype('datetime64[ns]'),
                                np.array(scores, dtype=np.float32)
                            )
                except Exception as e:
                    logger.warning(f'[FEAT] Batch score failed for {ticker}: {e}')
                    self.confidence_cache[ticker] = pd.Series(dtype=float)
                ok += 1
            else:
                skip += 1

            if (i + 1) % 500 == 0:
                logger.info(f'[FEAT] {i+1}/{total} | hits={cache_hits} computed={ok-cache_hits} skip={skip}')

        logger.info(f'[FEAT] Done: {ok} OK ({cache_hits} cache hits, {ok-cache_hits} computed), {skip} skipped')

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_price(self, ticker: str, as_of: pd.Timestamp) -> Optional[float]:
        df = self.market_data.get(ticker)
        if df is None:
            return None
        hist = df.loc[:as_of, 'close'].dropna()
        return float(hist.iloc[-1]) if not hist.empty else None

    def _calc_portfolio_value(self, as_of: pd.Timestamp) -> float:
        total = self.cash
        for ticker, pos in self.positions.items():
            price = self._get_price(ticker, as_of)
            if price:
                total += pos['shares'] * price
        return total

    def _record_equity(self, date: pd.Timestamp):
        self.portfolio_value = self._calc_portfolio_value(date)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        dd = (self.peak_value - self.portfolio_value) / self.peak_value
        self.equity_curve.append({
            'date': date, 'value': self.portfolio_value,
            'drawdown': dd, 'cash': self.cash
        })
        return dd

    def _calc_position_size(self, ticker: str, price: float,
                             as_of: pd.Timestamp, conf: float = 0.60) -> int:
        """
        Inverse-volatility sizing with confidence weighting.
        size = (1% risk / annualized_vol) * conf^CONF_SIZE_POWER
        High-confidence signals (0.90) get ~2.25x more size than borderline (0.60).
        """
        try:
            closes = self.market_data[ticker].loc[:as_of, 'close'].dropna()
            if len(closes) < 20 or price <= 0:
                return 0
            vol = closes.pct_change().tail(20).std() * np.sqrt(252)
            if vol <= 0:
                return 0
            risk_amount    = self.portfolio_value * MAX_RISK_PER_TRADE
            position_value = risk_amount / vol
            # Confidence multiplier: rewards strong signals, penalises borderline ones
            conf_mult      = (conf / CONFIDENCE_THRESHOLD) ** CONF_SIZE_POWER
            return max(0, int(position_value * conf_mult / price))
        except Exception:
            return 0

    def _build_regime_features_for_date(self, as_of: pd.Timestamp) -> Optional[np.ndarray]:
        """Build a single-row regime feature vector up to as_of date."""
        try:
            spy = self.market_data.get('SPY')
            vxx = self.market_data.get('VXX')
            if spy is None:
                return None

            close_spy = spy.loc[:as_of, 'close'].dropna()
            if len(close_spy) < 200:
                return None

            row = {}
            for w in [20, 50, 100, 200]:
                sma = close_spy.rolling(w).mean()
                ratio = close_spy.iloc[-1] / sma.iloc[-1] if sma.iloc[-1] > 0 else 1.0
                row[f'spy_sma{w}_ratio'] = float(ratio)

            for w in [5, 10, 20, 60]:
                if len(close_spy) > w:
                    row[f'spy_roc_{w}'] = float(close_spy.pct_change(w).iloc[-1])
                else:
                    row[f'spy_roc_{w}'] = 0.0

            spy_ret = close_spy.pct_change()
            vol20 = float(spy_ret.tail(20).std() * np.sqrt(252))
            row['spy_vol_20'] = vol20
            vol60 = spy_ret.tail(60).std() * np.sqrt(252)
            row['spy_vol_regime'] = vol20 / vol60 if vol60 > 0 else 1.0

            high_52w = close_spy.tail(252).max()
            row['spy_dd_52w'] = float((close_spy.iloc[-1] - high_52w) / high_52w) if high_52w > 0 else 0.0

            for w in [50, 100, 200]:
                sma = close_spy.rolling(w).mean().iloc[-1]
                row[f'spy_above_sma{w}'] = float(close_spy.iloc[-1] > sma)

            streak = close_spy.pct_change().tail(10)
            row['spy_streak'] = float((streak > 0).sum() / 10.0)

            if vxx is not None:
                close_vxx = vxx.loc[:as_of, 'close'].dropna()
                if len(close_vxx) > 20:
                    row['vxx_level']   = float(close_vxx.iloc[-1])
                    row['vxx_roc_5']   = float(close_vxx.pct_change(5).iloc[-1]) if len(close_vxx) > 5 else 0.0
                    row['vxx_roc_20']  = float(close_vxx.pct_change(20).iloc[-1]) if len(close_vxx) > 20 else 0.0
                    vxx_sma20 = close_vxx.tail(20).mean()
                    vxx_std20 = close_vxx.tail(20).std()
                    row['vxx_bb_pct']   = float((close_vxx.iloc[-1] - vxx_sma20) / (vxx_std20 + 1e-9))
                    row['vxx_above_bb'] = float(close_vxx.iloc[-1] > (vxx_sma20 + 2 * vxx_std20))
                else:
                    for k in ['vxx_level','vxx_roc_5','vxx_roc_20','vxx_bb_pct','vxx_above_bb']:
                        row[k] = 0.0
            else:
                for k in ['vxx_level','vxx_roc_5','vxx_roc_20','vxx_bb_pct','vxx_above_bb']:
                    row[k] = 0.0

            # Build vector in training feature order
            vec = np.array([row.get(f, 0.0) for f in self._regime_feat_names], dtype=np.float32)
            return vec.reshape(1, -1)
        except Exception as _e:
            logger.debug(f'[REGIME] Feature build failed: {_e}')
            return None

    def _get_regime(self, as_of: pd.Timestamp) -> int:
        """
        Returns regime code for as_of date:
          2 = BULL   (standard entries, threshold=0.65)
          1 = BEAR   (strict entries, threshold=0.72)
          0 = CRISIS (no new entries)
        Uses AI classifier if loaded, else SPY SMA100 fallback (returns 2 or 1).
        """
        # Cache lookup
        if as_of in self._regime_cache:
            return self._regime_cache[as_of]

        regime = 2  # default: BULL

        if self._regime_clf is not None and self._regime_feat_names:
            # AI classifier requires VXX data — fall back to SMA rule for pre-VXX dates
            vxx = self.market_data.get('VXX')
            vxx_available = (
                vxx is not None and
                len(vxx.loc[:as_of, 'close'].dropna()) > 20
            )
            if not vxx_available:
                regime = self._spy_sma_regime(as_of)
            else:
                vec = self._build_regime_features_for_date(as_of)
                if vec is not None:
                    try:
                        vec_s  = self._regime_scaler.transform(vec)
                        regime = int(self._regime_clf.predict(vec_s)[0])
                    except Exception as _e:
                        logger.debug(f'[REGIME] Classifier predict failed: {_e} — SMA fallback')
                        regime = self._spy_sma_regime(as_of)
                else:
                    regime = self._spy_sma_regime(as_of)
        else:
            regime = self._spy_sma_regime(as_of)

        self._regime_cache[as_of] = regime
        return regime

    def _spy_sma_regime(self, as_of: pd.Timestamp) -> int:
        """Legacy SPY SMA100 fallback: returns 2 (BULL) or 1 (BEAR)."""
        try:
            spy = self.market_data.get('SPY')
            if spy is None:
                return 2
            closes = spy.loc[:as_of, 'close'].dropna()
            if len(closes) < SPY_REGIME_SMA:
                return 2
            sma = closes.iloc[-SPY_REGIME_SMA:].mean()
            return 2 if float(closes.iloc[-1]) > sma else 1
        except Exception:
            return 2

    def _spy_regime_ok(self, as_of: pd.Timestamp) -> bool:
        """Legacy wrapper — kept for backward compatibility. Returns True if not CRISIS."""
        return self._get_regime(as_of) != 0

    def _regime_threshold(self, as_of: pd.Timestamp) -> float:
        """Return the confidence threshold for the current regime."""
        regime = self._get_regime(as_of)
        if self._regime_meta:
            thr = self._regime_thresholds.get(str(regime))
            if thr is None:  # CRISIS
                return 999.0  # effectively blocks all entries
            return float(thr)
        # SMA fallback: BEAR -> 0.46, BULL -> CONFIDENCE_THRESHOLD
        return 0.46 if regime == 1 else CONFIDENCE_THRESHOLD

    def _get_confidence(self, ticker: str, as_of: pd.Timestamp) -> float:
        scores = self.confidence_cache.get(ticker)
        if scores is None or scores.empty:
            return 0.0
        available = scores.loc[:as_of]
        if available.empty:
            return 0.0
        return float(available.iloc[-1])

    # -----------------------------------------------------------------------
    # Trade execution
    # -----------------------------------------------------------------------

    def _buy(self, ticker: str, date: pd.Timestamp, price: float, conf: float):
        shares = self._calc_position_size(ticker, price, date, conf)
        cost   = shares * price
        if shares <= 0 or cost > self.cash:
            return
        self.cash -= cost
        self.positions[ticker] = {
            'shares': shares, 'entry_price': price,
            'entry_date': date, 'entry_confidence': conf
        }
        self.trades.append({
            'date': date, 'ticker': ticker, 'action': 'BUY',
            'shares': shares, 'price': price, 'value': cost,
            'confidence': conf, 'pnl': None, 'reason': None
        })
        logger.info(f'[BUY]  {ticker} x{shares} @ ${price:.2f} conf={conf:.3f}')

    def _sell(self, ticker: str, date: pd.Timestamp, price: float, reason: str):
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return
        proceeds = pos['shares'] * price
        pnl      = proceeds - pos['shares'] * pos['entry_price']
        pnl_pct  = pnl / (pos['shares'] * pos['entry_price'])
        self.cash += proceeds
        self.trades.append({
            'date': date, 'ticker': ticker, 'action': 'SELL',
            'shares': pos['shares'], 'price': price, 'value': proceeds,
            'confidence': pos['entry_confidence'], 'pnl': pnl, 'reason': reason
        })
        logger.info(f'[SELL] {ticker} x{pos["shares"]} @ ${price:.2f} pnl={pnl_pct:+.1%} [{reason}]')

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        # Trading calendar from SPY (most complete) or first available ticker
        anchor = self.market_data.get('SPY', next(iter(self.market_data.values())))
        cal    = anchor.loc[self.start_date:self.end_date].index
        if len(cal) == 0:
            logger.error('[FATAL] No trading days found in date range')
            return

        total = len(cal)
        logger.info(f'[RUN] {total} trading days | {self.start_date.date()} -> {self.end_date.date()}')

        for i, date in enumerate(cal):

            # ---- Progress log every ~1 year --------------------------------
            if i % 250 == 0:
                logger.info(
                    f'[PROG] {i/total*100:.0f}% | {date.date()} | '
                    f'pos={len(self.positions)} | value=${self.portfolio_value:,.0f}'
                )

            # ---- Uncle Point circuit breaker (Rule 2.1) --------------------
            if self.cooldown_until and date < self.cooldown_until:
                self._record_equity(date)
                continue

            dd = self._record_equity(date)

            if dd > UNCLE_POINT_DD:
                logger.warning(f'[SHIELD] Uncle Point: {dd:.1%} DD. Cooldown {COOLDOWN_DAYS}d.')
                self.cooldown_until = date + pd.Timedelta(days=COOLDOWN_DAYS)
                for t in list(self.positions.keys()):
                    p = self._get_price(t, date)
                    if p:
                        self._sell(t, date, p, 'UNCLE_POINT')
                # Reset peak so cooldown days don't re-trigger the uncle point check
                self.peak_value = self.portfolio_value
                continue

            # ---- Exit existing positions ------------------------------------
            for ticker in list(self.positions.keys()):
                pos   = self.positions[ticker]
                price = self._get_price(ticker, date)
                if not price:
                    continue
                pnl_pct   = (price - pos['entry_price']) / pos['entry_price']
                hold_days = (date - pos['entry_date']).days

                if pnl_pct <= -STOP_LOSS_PCT:
                    self._sell(ticker, date, price, 'STOP_LOSS')
                    self._stopped_out[ticker] = date + pd.Timedelta(days=RE_ENTRY_COOLDOWN)
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    self._sell(ticker, date, price, 'TAKE_PROFIT')
                elif hold_days >= MAX_HOLD_DAYS:
                    self._sell(ticker, date, price, 'TIMEOUT')

            # ---- Scan for entries ------------------------------------------
            # Regime filter: CRISIS = no entries; BEAR = stricter threshold
            regime = self._get_regime(date)
            if regime == 0:  # CRISIS
                continue
            regime_thr = self._regime_threshold(date)

            for ticker in self.universe:
                if ticker in self.positions:
                    continue

                # Hard position cap
                if len(self.positions) >= MAX_POSITIONS:
                    break

                if self.cash < self.portfolio_value * 0.02:
                    break   # effectively out of cash

                # Per-ticker re-entry cooldown after stop-loss
                if ticker in self._stopped_out and date < self._stopped_out[ticker]:
                    continue

                conf = self._get_confidence(ticker, date)
                if conf < regime_thr:
                    continue

                price = self._get_price(ticker, date)
                if not price or price < MIN_PRICE:
                    continue

                # Volume liquidity filter (Rule: no illiquid stocks)
                df = self.market_data.get(ticker)
                if df is not None:
                    past = df.loc[:date]
                    if len(past) >= 20:
                        avg_vol = past['volume'].iloc[-20:].mean()
                        if avg_vol < MIN_AVG_VOLUME:
                            continue

                self._buy(ticker, date, price, conf)

        # ---- Final liquidation at last bar ---------------------------------
        last = cal[-1]
        for ticker in list(self.positions.keys()):
            p = self._get_price(ticker, last)
            if p:
                self._sell(ticker, last, p, 'END_OF_BACKTEST')

        logger.info(f'[DONE] {len(self.trades)} trades total')

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def report(self) -> Dict:
        if not self.equity_curve:
            logger.error('[REPORT] No equity curve recorded')
            return {}

        eq        = pd.DataFrame(self.equity_curve).set_index('date')
        final_val = eq['value'].iloc[-1]
        total_ret = (final_val - self.initial_cash) / self.initial_cash
        years     = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr      = (final_val / self.initial_cash) ** (1 / years) - 1 if years > 0 else 0.0
        max_dd    = eq['drawdown'].max()

        daily_ret = eq['value'].pct_change().dropna()
        sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                     if daily_ret.std() > 0 else 0.0)

        sells        = [t for t in self.trades if t['action'] == 'SELL' and t['pnl'] is not None]
        wins         = [t for t in sells if t['pnl'] > 0]
        win_rate     = len(wins) / len(sells) * 100 if sells else 0.0
        gross_win    = sum(t['pnl'] for t in wins)
        gross_loss   = abs(sum(t['pnl'] for t in sells if t['pnl'] <= 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

        metrics = {
            'period':        f"{eq.index[0].date()} to {eq.index[-1].date()}",
            'years':         round(years, 1),
            'initial_cash':  self.initial_cash,
            'final_value':   round(final_val, 2),
            'total_return':  f'{total_ret:.2%}',
            'cagr':          f'{cagr:.2%}',
            'max_drawdown':  f'{max_dd:.2%}',
            'sharpe':        round(sharpe, 2),
            'total_trades':  len(self.trades),
            'sell_trades':   len(sells),
            'win_rate':      f'{win_rate:.1f}%',
            'profit_factor': round(profit_factor, 2),
        }

        bar = '=' * 72
        print(f'\n{bar}')
        print('  NEURALTRADER - UNIFIED BACKTEST RESULTS')
        print(bar)
        for k, v in metrics.items():
            print(f'  {k:<20} {v}')
        print(bar)

        # Stress periods
        stress = {
            '2000-2002 Dot-Com':     ('2000-01-01', '2002-12-31'),
            '2008 Financial Crisis': ('2008-01-01', '2009-12-31'),
            '2020 COVID Crash':      ('2020-01-01', '2020-12-31'),
            '2022 Bear Market':      ('2022-01-01', '2022-12-31'),
        }
        print('\n  STRESS-TEST PERIODS')
        print(f"  {'Period':<28} {'Return':<12} {'Max DD'}")
        print(f"  {'-'*28} {'-'*12} {'-'*10}")
        for label, (s, e) in stress.items():
            sub = eq.loc[s:e, 'value']
            if len(sub) < 2:
                continue
            ret    = (sub.iloc[-1] / sub.iloc[0]) - 1
            sub_dd = eq.loc[s:e, 'drawdown'].max()
            print(f'  {label:<28} {ret:+.2%}       {sub_dd:.2%}')
        print(bar + '\n')

        # Save outputs
        reports_dir = PROJECT_ROOT / 'reports'
        reports_dir.mkdir(exist_ok=True)

        eq.to_csv(reports_dir / 'equity_curve.csv')
        pd.DataFrame(self.trades).to_csv(reports_dir / 'trade_log.csv', index=False)
        with open(reports_dir / 'backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f'[REPORT] Saved to {reports_dir}')
        return metrics


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='NeuralTrader Unified Backtest')
    parser.add_argument('--start',    default='2000-01-01')
    parser.add_argument('--end',      default=None)
    parser.add_argument('--cash',     default=100_000.0, type=float)
    parser.add_argument('--tickers',  nargs='*', default=None)
    parser.add_argument('--no-cache', action='store_true',
                        help='Force full recompute, ignore disk cache')
    args = parser.parse_args()

    logger.info('[START] NeuralTrader Unified Backtest Engine')
    bt = UnifiedBacktest(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        ticker_filter=args.tickers,
        use_cache=not args.no_cache,
    )
    bt.run()
    bt.report()


if __name__ == '__main__':
    main()
