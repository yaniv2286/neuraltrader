#!/usr/bin/env python3
"""
NeuralTrader - Walk-Forward Validation (the first honest test of the model)
=========================================================================

Why this exists
---------------
Every prior "backtest" number was in-sample:
  * retrain_phase12 / retrain_phase14 split by ROW INDEX after stacking tickers,
    i.e. by ticker alphabetically, NOT by time -> validation leaked the future.
  * The 2020-2025 Phase 16 backtest ran on dates INSIDE the training window.
  * retrain_phase14 labels compared an ADJUSTED entry price with RAW highs/lows,
    so every pre-split bar was LONG_WIN (82% of rows). The model learned garbage.

What this does
--------------
1. Builds features EXACTLY as the serving path does (FeatureEngineer on
   split/dividend-adjusted OHLCV, raw OHLCV columns dropped) so train == serve.
2. Builds triple-barrier labels on ADJUSTED high/low/close (bug fixed).
3. Caches X / y / date / ticker per row so splits can be made BY DATE.
4. For each fold: train on rows whose label window ends >= EMBARGO days before
   test_start, test on [test_start, test_end]. Reports OOS metrics that mean
   something: majority baseline, one-vs-rest AUC for LONG, precision@0.65 vs
   base rate (lift), per fold.
5. Saves each fold's ensemble to models/wf_fold{k}/ in the production format so
   run_full_backtest.py --models-dir can produce a TRUE out-of-sample backtest.

Usage
-----
    python scripts/walk_forward_validation.py --build-cache      # ~45 min, once
    python scripts/walk_forward_validation.py --train             # all folds
    python scripts/walk_forward_validation.py --train --folds 3   # only fold 3
"""

import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.universe_filter import is_otc_ticker  # noqa: E402

DATA_DIR  = PROJECT_ROOT / 'data' / 'raw'
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache' / 'walk_forward'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS   = PROJECT_ROOT / 'reports'
LOG_DIR   = PROJECT_ROOT / 'logs'
for d in (CACHE_DIR, REPORTS, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOG_DIR / 'walk_forward.log', mode='a', encoding='ascii')])
logger = logging.getLogger('WalkForward')

# --- Triple-barrier label config (same as live Phase 14, on ADJUSTED prices) ---
TP_PCT        = 0.05
STOP_ATR_MULT = 1.5
STOP_MIN      = 0.03
STOP_MAX      = 0.08
TIMEOUT_DAYS  = 10
ATR_WINDOW    = 20
MIN_ROWS      = 252

# Label at date t looks at t+1 .. t+TIMEOUT_DAYS. Purge/embargo so no training
# label can see any test-period price: 10 trading days ~ 15 calendar days, +5 safety.
EMBARGO_DAYS  = 20

RAW_OHLCV_COLS = {'Price', 'high', 'low', 'open', 'close', 'volume',
                  'Adj Close', 'Adj High', 'Adj Low', 'Adj Open', 'Adj Volume',
                  'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume',
                  'Dividend', 'Split Factor', 'divCash', 'splitFactor'}

FOLDS = [
    # (name, train_end,   test_start,   test_end)
    ('fold1', '2017-12-31', '2018-01-01', '2020-12-31'),
    ('fold2', '2020-12-31', '2021-01-01', '2023-12-31'),
    ('fold3', '2023-12-31', '2024-01-01', '2026-12-31'),
]

MAX_TRAIN_ROWS = 3_000_000   # stratified subsample cap per fold (speed; >> old 100K)
MAX_TEST_ROWS  = 1_500_000

XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=10, gamma=1.0,
                  objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                  n_jobs=-1, tree_method='hist', verbosity=0, random_state=42)
LGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                  reg_alpha=0.1, reg_lambda=1.0, objective='multiclass', num_class=3,
                  n_jobs=-1, verbosity=-1, random_state=42)
HGB_PARAMS = dict(max_iter=100, max_depth=5, learning_rate=0.08, min_samples_leaf=50,
                  l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=42)
WEIGHTS = {'xgboost': 0.40, 'lightgbm': 0.40, 'hgb': 0.20}


# ---------------------------------------------------------------------------
# Labels (FIXED: adjusted high/low/close throughout)
# ---------------------------------------------------------------------------

def _atr_pct_series(high, low, close):
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(ATR_WINDOW + 2, min_periods=14).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(close > 0, atr * STOP_ATR_MULT / close, STOP_MIN)
    pct = np.where(np.isnan(pct), STOP_MIN, pct)
    return np.clip(pct, STOP_MIN, STOP_MAX)


def triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    """+1 LONG_WIN (TP hit first), -1 SHORT_WIN (stop hit first), 0 NEUTRAL (timeout)."""
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    n = len(df)
    stop_pct = _atr_pct_series(high, low, close)
    labels = np.zeros(n, dtype=np.int8)
    valid  = np.zeros(n, dtype=bool)
    for t in range(n - TIMEOUT_DAYS):
        entry = close[t]
        if entry <= 0:
            continue
        upper = entry * (1.0 + TP_PCT)
        lower = entry * (1.0 - stop_pct[t])
        lab = 0
        for s in range(t + 1, t + 1 + TIMEOUT_DAYS):
            if high[s] >= upper:
                lab = 1
                break
            if low[s] <= lower:
                lab = -1
                break
        labels[t] = lab
        valid[t] = True
    return pd.Series(labels, index=df.index)[valid]


# ---------------------------------------------------------------------------
# Features: identical to serving (EnsemblePredictor.generate_features)
# ---------------------------------------------------------------------------

def load_adjusted(pf: Path) -> pd.DataFrame:
    df = pd.read_parquet(pf)
    if 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for raw, adj in (('close', 'adjClose'), ('open', 'adjOpen'), ('high', 'adjHigh'),
                     ('low', 'adjLow'), ('volume', 'adjVolume')):
        if adj in df.columns:
            df[raw] = df[adj]
    return df


def build_features(df: pd.DataFrame, fe) -> pd.DataFrame:
    feats, _ = fe.create_features(df, target_type='direction')
    if feats is None or feats.empty:
        return pd.DataFrame()
    feats = feats.select_dtypes(include=[np.number])
    feats = feats.drop(columns=[c for c in feats.columns if c in RAW_OHLCV_COLS], errors='ignore')
    return feats.replace([np.inf, -np.inf], np.nan).dropna()


def build_cache():
    from core.feature_engineer import FeatureEngineer
    fe = FeatureEngineer(use_advanced_features=True, verbose=False)
    files = sorted(DATA_DIR.glob('*.parquet'))
    logger.info(f'[CACHE] Building walk-forward cache from {len(files)} parquets...')
    X_parts, y_parts, d_parts, t_parts = [], [], [], []
    tickers, feature_names = [], None
    ok = skip = 0
    t0 = datetime.now()
    for i, pf in enumerate(files):
        tk = pf.stem.upper()
        if is_otc_ticker(tk):
            skip += 1
            continue
        try:
            df = load_adjusted(pf)
            if len(df) < MIN_ROWS or not {'high', 'low', 'close'}.issubset(df.columns):
                skip += 1
                continue
            feats = build_features(df, fe)
            if len(feats) < 50:
                skip += 1
                continue
            labels = triple_barrier_labels(df)
            common = feats.index.intersection(labels.index)
            if len(common) < 50:
                skip += 1
                continue
            Xt = feats.loc[common]
            if feature_names is None:
                feature_names = list(Xt.columns)
            else:
                Xt = Xt.reindex(columns=feature_names, fill_value=0.0)
            X_parts.append(Xt.values.astype(np.float32))
            y_parts.append(labels.loc[common].values.astype(np.int8))
            d_parts.append(common.values.astype('datetime64[D]').astype(np.int32))
            t_parts.append(np.full(len(common), len(tickers), dtype=np.int16))
            tickers.append(tk)
            ok += 1
            if ok % 100 == 0:
                rows = sum(len(x) for x in X_parts)
                el = (datetime.now() - t0).total_seconds()
                logger.info(f'[CACHE] {ok} tickers | {rows:,} rows | {el/60:.1f} min | ~{el/(i+1)*(len(files)-i-1)/60:.0f} min left')
        except Exception as e:
            skip += 1
            if skip <= 10:
                logger.warning(f'[CACHE] {tk}: {e}')
    X = np.vstack(X_parts); y = np.concatenate(y_parts)
    dates = np.concatenate(d_parts); tix = np.concatenate(t_parts)
    np.save(CACHE_DIR / 'X.npy', X); np.save(CACHE_DIR / 'y.npy', y)
    np.save(CACHE_DIR / 'dates.npy', dates); np.save(CACHE_DIR / 'ticker_idx.npy', tix)
    (CACHE_DIR / 'meta.json').write_text(json.dumps({
        'feature_names': feature_names, 'tickers': tickers, 'n_rows': int(len(y)),
        'label': f'TB TP={TP_PCT} stop=ATR{STOP_ATR_MULT}x[{STOP_MIN}-{STOP_MAX}] timeout={TIMEOUT_DAYS}d (adjusted OHLC)',
        'built': datetime.now().isoformat()}, indent=2))
    n = len(y)
    logger.info(f'[CACHE] Done: X={X.shape} ok={ok} skipped={skip} | '
                f'LONG={np.mean(y==1):.1%} NEUTRAL={np.mean(y==0):.1%} SHORT={np.mean(y==-1):.1%} | '
                f'{(datetime.now()-t0).total_seconds()/60:.1f} min')


def load_cache():
    X = np.load(CACHE_DIR / 'X.npy', mmap_mode='r')
    y = np.load(CACHE_DIR / 'y.npy')
    dates = np.load(CACHE_DIR / 'dates.npy').astype('datetime64[D]')
    meta = json.loads((CACHE_DIR / 'meta.json').read_text())
    return X, y, dates, meta


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _gather_rows(X, idx, chunk=400_000):
    """Sequential chunked gather from a memmap (random fancy-indexing a 3.8 GB file is ~1 MB/s)."""
    idx = np.sort(idx)
    out = np.empty((len(idx), X.shape[1]), dtype=np.float32)
    pos = 0
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        lo = np.searchsorted(idx, start); hi = np.searchsorted(idx, stop)
        if hi > lo:
            block = np.asarray(X[start:stop])
            out[pos:pos + hi - lo] = block[idx[lo:hi] - start]
            pos += hi - lo
    return out


def _stratified_subsample(idx, y, cap, seed=42):
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = int(round(cap * len(ci) / len(idx)))
        out.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.sort(np.concatenate(out))


def evaluate(proba: np.ndarray, y_true: np.ndarray, label: str) -> dict:
    from sklearn.metrics import roc_auc_score
    preds = proba.argmax(axis=1)
    acc = float((preds == y_true).mean())
    base = {int(c): float((y_true == c).mean()) for c in (0, 1, 2)}
    majority = max(base.values())
    long_p = proba[:, 1]
    auc_long = float(roc_auc_score((y_true == 1).astype(int), long_p))
    out = {'accuracy': round(acc, 4), 'majority_baseline': round(majority, 4),
           'base_rate': {k: round(v, 4) for k, v in base.items()}, 'auc_long_ovr': round(auc_long, 4)}
    for thr in (0.50, 0.65):
        sel = long_p >= thr
        n = int(sel.sum())
        prec = float((y_true[sel] == 1).mean()) if n else None
        out[f'long_prec@{thr:.2f}'] = None if prec is None else round(prec, 4)
        out[f'long_n@{thr:.2f}'] = n
        out[f'long_lift@{thr:.2f}'] = None if prec is None else round(prec / base[1], 3)
    # decile spread: realised LONG rate in top decile vs bottom decile of model score
    q = np.quantile(long_p, [0.1, 0.9])
    top = float((y_true[long_p >= q[1]] == 1).mean()); bot = float((y_true[long_p <= q[0]] == 1).mean())
    out['long_rate_top_decile'] = round(top, 4); out['long_rate_bottom_decile'] = round(bot, 4)
    logger.info(f'[{label}] acc={acc:.3f} (majority {majority:.3f}) | AUC_long={auc_long:.3f} | '
                f'P@0.65={out["long_prec@0.65"]} n={out["long_n@0.65"]} lift={out["long_lift@0.65"]} | '
                f'top/bottom decile LONG rate {top:.3f}/{bot:.3f}')
    return out


def train_fold(name, train_end, test_start, test_end, X, y, dates, meta):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    t0 = datetime.now()
    y3 = np.where(y == -1, 2, y).astype(np.int8)          # 0=NEUTRAL 1=LONG 2=SHORT (production order)
    cutoff = np.datetime64(pd.Timestamp(test_start) - pd.Timedelta(days=EMBARGO_DAYS), 'D')
    tr_idx = np.where(dates <= cutoff)[0]
    te_idx = np.where((dates >= np.datetime64(test_start, 'D')) & (dates <= np.datetime64(test_end, 'D')))[0]
    logger.info(f'[{name}] train <= {cutoff} ({len(tr_idx):,} rows) | test {test_start}..{test_end} ({len(te_idx):,} rows) | embargo {EMBARGO_DAYS}d')
    if len(tr_idx) < 100_000 or len(te_idx) < 10_000:
        raise RuntimeError(f'[{name}] insufficient rows')
    tr_idx = _stratified_subsample(tr_idx, y3, MAX_TRAIN_ROWS)
    te_idx = _stratified_subsample(te_idx, y3, MAX_TEST_ROWS, seed=7)
    Xtr = _gather_rows(X, tr_idx); ytr = y3[np.sort(tr_idx)]
    Xte = _gather_rows(X, te_idx); yte = y3[np.sort(te_idx)]
    logger.info(f'[{name}] rows gathered in {(datetime.now()-t0).total_seconds()/60:.1f} min')
    logger.info(f'[{name}] after subsample train={len(ytr):,} test={len(yte):,} | '
                f'train LONG={np.mean(ytr==1):.1%} NEU={np.mean(ytr==0):.1%} SHORT={np.mean(ytr==2):.1%} | '
                f'test LONG={np.mean(yte==1):.1%}')

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)
    counts = np.bincount(ytr, minlength=3)
    cw = len(ytr) / (3.0 * np.maximum(counts, 1))
    sw = cw[ytr]

    models, metrics, probas = {}, {}, {}
    for mname, mdl in (('xgboost', XGBClassifier(**XGB_PARAMS)),
                       ('lightgbm', LGBMClassifier(**LGB_PARAMS)),
                       ('hgb', HistGradientBoostingClassifier(**HGB_PARAMS))):
        logger.info(f'[{name}] training {mname}...')
        mdl.fit(Xtr_s, ytr, sample_weight=sw)
        p = mdl.predict_proba(Xte_s)
        models[mname] = mdl; probas[mname] = p
        metrics[mname] = evaluate(p, yte, f'{name}/{mname}')
    ens = sum(WEIGHTS[m] * probas[m] for m in probas) / sum(WEIGHTS.values())
    metrics['ensemble'] = evaluate(ens, yte, f'{name}/ENSEMBLE')

    out_dir = MODELS_DIR / f'wf_{name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(models['xgboost'], open(out_dir / 'xgboost_model.pkl', 'wb'))
    pickle.dump(models['lightgbm'], open(out_dir / 'lightgbm_model.pkl', 'wb'))
    pickle.dump(models['hgb'], open(out_dir / 'rf_model.pkl', 'wb'))
    pickle.dump(scaler, open(out_dir / 'feature_scaler.pkl', 'wb'))
    pickle.dump(meta['feature_names'], open(out_dir / 'feature_names.pkl', 'wb'))
    md = {'phase': f'WalkForward_{name}', 'label_type': 'triple_barrier_3class_adjusted',
          'classes': '0=NEUTRAL, 1=LONG_WIN, 2=SHORT_WIN', 'train_end': str(cutoff),
          'test_start': test_start, 'test_end': test_end, 'embargo_days': EMBARGO_DAYS,
          'n_train': int(len(ytr)), 'n_test': int(len(yte)), 'n_features': len(meta['feature_names']),
          'feature_names': meta['feature_names'], 'weights': WEIGHTS, 'metrics': metrics,
          'trained': datetime.now().isoformat()}
    pickle.dump(md, open(out_dir / 'ensemble_metadata.pkl', 'wb'))
    (out_dir / 'ensemble_metadata.json').write_text(json.dumps(md, indent=2))
    logger.info(f'[{name}] saved -> {out_dir} | {(datetime.now()-t0).total_seconds()/60:.1f} min')
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-cache', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--folds', nargs='*', type=int, default=None, help='1-based fold numbers')
    a = ap.parse_args()
    if a.build_cache or not (CACHE_DIR / 'X.npy').exists():
        build_cache()
    if a.train:
        X, y, dates, meta = load_cache()
        logger.info(f'[WF] cache rows={len(y):,} features={len(meta["feature_names"])} '
                    f'dates {dates.min()}..{dates.max()}')
        results = {}
        for k, (name, tr_end, te_s, te_e) in enumerate(FOLDS, 1):
            if a.folds and k not in a.folds:
                continue
            results[name] = train_fold(name, tr_end, te_s, te_e, X, y, dates, meta)
        out = REPORTS / 'walk_forward_metrics.json'
        prev = json.loads(out.read_text()) if out.exists() else {}
        prev.update(results)
        out.write_text(json.dumps(prev, indent=2))
        bar = '=' * 96
        print(f'\n{bar}\n  WALK-FORWARD OUT-OF-SAMPLE MODEL METRICS (ensemble)\n{bar}')
        print(f"  {'fold':7} {'test window':24} {'acc':>6} {'major':>6} {'AUC':>6} {'P@.65':>6} {'lift':>5} {'n@.65':>8} {'top/bot decile':>16}")
        for name, md in prev.items():
            e = md['metrics']['ensemble']
            print(f"  {name:7} {md['test_start']+'..'+md['test_end']:24} {e['accuracy']:6.3f} {e['majority_baseline']:6.3f} "
                  f"{e['auc_long_ovr']:6.3f} {str(e['long_prec@0.65']):>6} {str(e['long_lift@0.65']):>5} {e['long_n@0.65']:8d} "
                  f"{e['long_rate_top_decile']:7.3f}/{e['long_rate_bottom_decile']:.3f}")
        print(bar)


if __name__ == '__main__':
    main()
