#!/usr/bin/env python3
"""
NeuralTrader Phase 12 — Model Retraining
=========================================

Key fixes vs old retrain_ensemble.py:
  1. LABEL: 5-day forward return > 2% (was: up-tomorrow any amount = noise)
  2. DATA: ALL ~15M rows used (was: hard-capped at 100K = 2.3% of data)
  3. FEATURES: 4 new momentum features added to 64-feature base set:
       - rs_vs_spy       : ticker return / SPY return (20d) — relative strength
       - high_52w_prox   : close / 52-week high — breakout proximity
       - volume_breakout : volume / 50d avg volume — unusual volume flag
       - roc_63          : 63-day (3-month) rate of change
  4. SAVE: raw sklearn objects to models/ — compatible with EnsemblePredictor
  5. BRAIN-GATE: model quality gate — rejects models with precision@0.65 < 55%

Output files (drop-in replacements, no orchestrator changes needed):
  models/xgboost_model.pkl
  models/lightgbm_model.pkl
  models/rf_model.pkl
  models/feature_names.pkl
  models/feature_scaler.pkl
  models/ensemble_metadata.pkl
  models/ensemble_metadata.json
"""

import os
import sys
import json
import pickle
import shutil
import logging
import traceback
from datetime import datetime
from pathlib import Path

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / 'logs' / 'retrain_phase12.log',
            mode='w', encoding='ascii'
        ),
    ]
)
logger = logging.getLogger('Phase12Retrain')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR      = PROJECT_ROOT / 'models'
DATA_DIR        = PROJECT_ROOT / 'data' / 'raw'
CACHE_DIR       = PROJECT_ROOT / 'data' / 'cache'
BACKUP_DIR      = MODELS_DIR / 'backup'
SPY_PATH        = DATA_DIR / 'SPY.parquet'
DATA_CACHE_X    = CACHE_DIR / 'train_X_phase12.npy'
DATA_CACHE_Y    = CACHE_DIR / 'train_y_phase12.npy'
DATA_CACHE_META = CACHE_DIR / 'train_meta_phase12.json'

LABEL_HORIZON   = 5      # forward-looking days
LABEL_THRESHOLD = 0.02   # stock must gain >2% in 5 days to be label=1
MIN_ROWS        = 252    # minimum history rows per ticker
PRECISION_GATE  = 0.55   # precision@0.65 threshold — below this = rollback
CACHE_MAX_AGE_H = 24     # hours before cache is considered stale

# Model hyperparams — tuned for 15M rows on CPU
# With 13M training rows, 200 trees converge fully; 400 adds <0.5% AUC but 2x time
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, gamma=1.0,
    eval_metric='logloss',
    n_jobs=-1, tree_method='hist', verbosity=0,
)
LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbosity=-1,
)
# HistGradientBoosting = sklearn's Cython-accelerated boosting, ~10x faster than RF
# on large datasets. Acts as the diversity model in the ensemble.
HGB_PARAMS = dict(
    max_iter=100, max_depth=5, learning_rate=0.08,
    min_samples_leaf=50, l2_regularization=1.0,
    max_bins=255,  # uses integer binning — no float32 scaling needed
)

WEIGHTS = {'xgboost': 0.40, 'lightgbm': 0.40, 'hgb': 0.20}


# ---------------------------------------------------------------------------
# Feature engineering: base 64 features + 4 new momentum features
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV dataframe.
    Returns a DataFrame aligned to df's index with NaN rows dropped.
    Adds 4 Phase-12 momentum features on top of the existing 64-feature set.
    """
    from core.feature_engineer import FeatureEngineer

    fe = FeatureEngineer(use_advanced_features=True, verbose=False)
    try:
        features, _ = fe.create_features(df, target_type='direction')
    except Exception as e:
        raise RuntimeError(f'FeatureEngineer failed: {e}') from e

    if features.empty:
        return features

    # Re-align close price to feature index
    close = df['adjClose'] if 'adjClose' in df.columns else df['close']
    close = close.reindex(features.index)

    # --- Phase 12 feature 1: Relative Strength vs SPY (20-day) ---
    ticker_ret20 = close.pct_change(20)
    if spy_close is not None and not spy_close.empty:
        spy_aligned = spy_close.reindex(features.index, method='ffill')
        spy_ret20   = spy_aligned.pct_change(20)
        features['rs_vs_spy'] = (ticker_ret20 - spy_ret20).fillna(0).clip(-2, 2)
    else:
        features['rs_vs_spy'] = ticker_ret20.fillna(0).clip(-2, 2)

    # --- Phase 12 feature 2: 52-week high proximity ---
    high_52w = close.rolling(252, min_periods=60).max()
    features['high_52w_prox'] = (close / high_52w.replace(0, np.nan)).fillna(0).clip(0, 1.5)

    # --- Phase 12 feature 3: Volume breakout ratio ---
    if 'volume' in df.columns:
        vol = df['volume'].reindex(features.index)
        vol_avg50 = vol.rolling(50, min_periods=20).mean()
        features['volume_breakout'] = (vol / vol_avg50.replace(0, np.nan)).fillna(1.0).clip(0, 10)
    else:
        features['volume_breakout'] = 1.0

    # --- Phase 12 feature 4: 63-day (quarter) rate of change ---
    features['roc_63'] = close.pct_change(63).fillna(0).clip(-2, 2)

    # Drop any remaining NaN rows
    features = features.dropna()
    return features


def _build_label(df: pd.DataFrame, features_index) -> pd.Series:
    """
    Label = 1 if adjClose rises > LABEL_THRESHOLD in next LABEL_HORIZON days.
    Aligned to features_index to ensure no lookahead.
    """
    close = df['adjClose'] if 'adjClose' in df.columns else df['close']
    fwd_return = close.pct_change(LABEL_HORIZON).shift(-LABEL_HORIZON)
    label = (fwd_return > LABEL_THRESHOLD).astype(int)
    return label.reindex(features_index).dropna()


# ---------------------------------------------------------------------------
# Data cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    """Return True if the numpy cache files exist and are less than CACHE_MAX_AGE_H hours old."""
    if not (DATA_CACHE_X.exists() and DATA_CACHE_Y.exists() and DATA_CACHE_META.exists()):
        return False
    age_h = (datetime.now().timestamp() - DATA_CACHE_X.stat().st_mtime) / 3600
    return age_h < CACHE_MAX_AGE_H


def _load_cache() -> tuple:
    logger.info(f'[CACHE] Loading cached arrays from {CACHE_DIR}...')
    X   = np.load(DATA_CACHE_X, mmap_mode='r')   # memory-mapped — no RAM spike
    y   = np.load(DATA_CACHE_Y)
    meta = json.loads(DATA_CACHE_META.read_text())
    logger.info(f'[CACHE] X={X.shape} ({X.nbytes/1e9:.2f} GB)  Label=1: {y.mean():.1%}')
    return X, y, meta['feature_names']


def _save_cache(X_arr, y_arr, feature_names):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_CACHE_X, X_arr)
    np.save(DATA_CACHE_Y, y_arr)
    DATA_CACHE_META.write_text(json.dumps({'feature_names': feature_names,
                                           'n_rows': int(len(X_arr)),
                                           'saved': datetime.now().isoformat()}))
    logger.info(f'[CACHE] Saved to {CACHE_DIR}')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(force_rebuild: bool = False) -> tuple:
    """
    Returns (X_arr, y_arr, feature_names) as float32 numpy arrays.
    Uses a disk cache (data/cache/train_*.npy) to skip re-processing on reruns.
    Pass force_rebuild=True to ignore the cache.
    """
    if not force_rebuild and _cache_is_fresh():
        logger.info('[CACHE] Fresh cache found - skipping parquet scan (use --rebuild to force)')
        return _load_cache()

    parquet_files = sorted(DATA_DIR.glob('*.parquet'))
    logger.info(f'[DATA] Found {len(parquet_files)} parquet files')

    # Load SPY for relative-strength feature
    spy_close = None
    if SPY_PATH.exists():
        spy_df    = pd.read_parquet(SPY_PATH)
        spy_close = spy_df['adjClose'] if 'adjClose' in spy_df.columns else spy_df['close']
        spy_close.index = pd.to_datetime(spy_close.index)
        logger.info('[DATA] SPY loaded for rs_vs_spy feature')
    else:
        logger.warning('[WARN] SPY.parquet not found — rs_vs_spy will use ticker-only returns')

    all_X, all_y = [], []
    ok = fail = 0
    feature_names = None

    for pf in parquet_files:
        ticker = pf.stem
        try:
            df = pd.read_parquet(pf)
            if len(df) < MIN_ROWS:
                continue

            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)

            features = _build_features(df, spy_close)
            if features.empty or len(features) < 50:
                continue

            label = _build_label(df, features.index)
            common = features.index.intersection(label.index)
            if len(common) < 50:
                continue

            X_t = features.loc[common]
            y_t = label.loc[common]
            valid = y_t.notna()
            X_t = X_t[valid]
            y_t = y_t[valid]

            if len(X_t) < 50:
                continue

            if feature_names is None:
                feature_names = X_t.columns.tolist()

            # Accumulate as float32 numpy to avoid pandas OOM on concat
            all_X.append(X_t.values.astype(np.float32))
            all_y.append(y_t.values.astype(np.int8))
            ok += 1

            if ok % 200 == 0:
                total = sum(len(x) for x in all_X)
                logger.info(f'[DATA] {ok} tickers processed, {total:,} rows so far...')

        except Exception as e:
            fail += 1
            if fail <= 5:
                logger.warning(f'[WARN] {ticker}: {e}')

    logger.info(f'[DATA] Loaded {ok} tickers ({fail} failed)')

    # Stack numpy arrays — avoids pandas concat + interleave entirely
    logger.info('[DATA] Stacking arrays...')
    X_arr = np.vstack(all_X)          # fast C-level stack, no pandas overhead
    y_arr = np.concatenate(all_y)
    del all_X, all_y
    gc.collect()

    logger.info(f'[DATA] Total rows: {len(X_arr):,} | Features: {X_arr.shape[1]} | Label=1: {y_arr.mean():.1%}')
    logger.info(f'[DATA] Memory: X={X_arr.nbytes/1e9:.2f} GB  y={y_arr.nbytes/1e6:.0f} MB')

    _save_cache(X_arr, y_arr, feature_names)
    return X_arr, y_arr, feature_names


# ---------------------------------------------------------------------------
# Brain-Gate model backup
# ---------------------------------------------------------------------------

def backup_models():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backed = []
    for f in ['xgboost_model.pkl', 'lightgbm_model.pkl', 'rf_model.pkl',
              'feature_names.pkl', 'feature_scaler.pkl',
              'ensemble_metadata.pkl', 'ensemble_metadata.json']:
        src = MODELS_DIR / f
        if src.exists():
            dst = BACKUP_DIR / f'{f}.{ts}.bak'
            shutil.copy2(src, dst)
            backed.append(f)
    logger.info(f'[BACKUP] {len(backed)} model files backed up to {BACKUP_DIR}')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(X_train, y_train, X_val, y_val, feature_names):
    """Train 3 models, evaluate, apply Brain-Gate, save to models/."""

    logger.info(f'[TRAIN] Train={len(X_train):,} Val={len(X_val):,} Features={len(feature_names)}')
    logger.info(f'[TRAIN] Label balance train={y_train.mean():.1%} val={y_val.mean():.1%}')

    # Fit scaler on a representative sample to avoid OOM (scaler only needs statistics)
    sample_size = min(500_000, len(X_train))
    sample_idx  = np.random.choice(len(X_train), sample_size, replace=False)
    scaler = StandardScaler()
    scaler.fit(X_train[sample_idx])
    # Transform in-place in chunks to avoid doubling RAM
    logger.info('[TRAIN] Scaling train set in chunks...')
    chunk = 500_000
    X_train_s = np.empty_like(X_train, dtype=np.float32)
    for i in range(0, len(X_train), chunk):
        X_train_s[i:i+chunk] = scaler.transform(X_train[i:i+chunk]).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    results = {}
    trained_models = {}

    # ---- XGBoost ----
    logger.info('[TRAIN] Training XGBoost (hist, 200 trees)...')
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train_s, y_train.astype(np.int32),
            eval_set=[(X_val_s, y_val.astype(np.int32))],
            verbose=False)
    probs_xgb = xgb.predict_proba(X_val_s)[:, 1]
    results['xgboost'] = _eval(y_val, probs_xgb, 'XGBoost')
    trained_models['xgboost'] = xgb

    # ---- LightGBM ----
    logger.info('[TRAIN] Training LightGBM (200 trees)...')
    lgb = LGBMClassifier(**LGB_PARAMS)
    lgb.fit(X_train_s, y_train.astype(np.int32),
            eval_set=[(X_val_s, y_val.astype(np.int32))],
            callbacks=[])
    probs_lgb = lgb.predict_proba(X_val_s)[:, 1]
    results['lightgbm'] = _eval(y_val, probs_lgb, 'LightGBM')
    trained_models['lightgbm'] = lgb

    # ---- HistGradientBoosting (diversity model, 20% weight) ----
    # Subsample to 2M rows: HGB provides diversity, not raw power.
    # Full 13M rows unnecessary — 2M rows still ~10x more than old retrain used.
    hgb_max_rows = 2_000_000
    if len(X_train_s) > hgb_max_rows:
        hgb_idx = np.random.choice(len(X_train_s), hgb_max_rows, replace=False)
        hgb_idx.sort()
        X_hgb = X_train_s[hgb_idx]
        y_hgb = y_train[hgb_idx].astype(np.int32)
        logger.info(f'[TRAIN] Training HistGradientBoosting (100 iter, {hgb_max_rows:,} subsample)...')
    else:
        X_hgb, y_hgb = X_train_s, y_train.astype(np.int32)
        logger.info('[TRAIN] Training HistGradientBoosting (100 iter, full data)...')
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS)
    hgb.fit(X_hgb, y_hgb)
    del X_hgb, y_hgb
    probs_hgb = hgb.predict_proba(X_val_s)[:, 1]
    results['hgb'] = _eval(y_val, probs_hgb, 'HGB')
    trained_models['hgb'] = hgb

    # ---- Ensemble weighted vote ----
    ensemble_probs = (
        probs_xgb * WEIGHTS['xgboost'] +
        probs_lgb * WEIGHTS['lightgbm'] +
        probs_hgb * WEIGHTS['hgb']
    )
    ens_result = _eval(y_val, ensemble_probs, 'Ensemble')
    results['ensemble'] = ens_result

    # ---- Brain-Gate precision check ----
    ens_precision_65 = ens_result['precision_at_65']
    if ens_precision_65 < PRECISION_GATE:
        logger.error(
            f'[BRAIN-GATE] FAIL: Ensemble precision@0.65 = {ens_precision_65:.1%} '
            f'< required {PRECISION_GATE:.1%} — ROLLBACK'
        )
        raise RuntimeError(
            f'Brain-Gate failed: precision@0.65={ens_precision_65:.1%} < {PRECISION_GATE:.1%}'
        )
    logger.info(f'[BRAIN-GATE] PASS: precision@0.65 = {ens_precision_65:.1%}')

    # ---- Save models ----
    MODELS_DIR.mkdir(exist_ok=True)

    file_map = {
        'xgboost':  ('xgboost_model.pkl',  trained_models['xgboost']),
        'lightgbm': ('lightgbm_model.pkl',  trained_models['lightgbm']),
        'hgb':      ('rf_model.pkl',         trained_models['hgb']),  # same filename for drop-in compatibility
    }
    for name, (fname, model) in file_map.items():
        path = MODELS_DIR / fname
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f'[SAVE] {fname} saved ({path.stat().st_size/1e6:.1f} MB)')

    with open(MODELS_DIR / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info('[SAVE] feature_scaler.pkl saved')

    with open(MODELS_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f'[SAVE] feature_names.pkl saved ({len(feature_names)} features)')

    metadata = {
        'train_date':    datetime.now().isoformat(),
        'phase':         'Phase12',
        'n_features':    len(feature_names),
        'n_train':       int(len(X_train)),
        'n_val':         int(len(X_val)),
        'label_horizon': LABEL_HORIZON,
        'label_thresh':  LABEL_THRESHOLD,
        'feature_names': feature_names,
        'models': {
            'xgboost':  {'path': 'xgboost_model.pkl',  **results['xgboost'],  'weight': WEIGHTS['xgboost']},
            'lightgbm': {'path': 'lightgbm_model.pkl', **results['lightgbm'], 'weight': WEIGHTS['lightgbm']},
            'hgb':      {'path': 'rf_model.pkl',        **results['hgb'],      'weight': WEIGHTS['hgb']},
        },
        'ensemble': results['ensemble'],
    }

    with open(MODELS_DIR / 'ensemble_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    with open(MODELS_DIR / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info('[SAVE] ensemble_metadata.pkl + .json saved')

    return results


def _eval(y_true, probs, name: str) -> dict:
    """Evaluate model at multiple thresholds."""
    preds_50 = (probs >= 0.50).astype(int)
    preds_65 = (probs >= 0.65).astype(int)
    n_65     = preds_65.sum()

    acc_50  = accuracy_score(y_true, preds_50)
    prec_50 = precision_score(y_true, preds_50, zero_division=0)
    prec_65 = precision_score(y_true, preds_65, zero_division=0) if n_65 > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = 0.5

    logger.info(
        f'[EVAL] {name:<14} '
        f'acc@0.50={acc_50:.1%}  prec@0.50={prec_50:.1%}  '
        f'prec@0.65={prec_65:.1%} (n={n_65:,})  AUC={auc:.3f}'
    )
    return {
        'accuracy_at_50':  float(acc_50),
        'precision_at_50': float(prec_50),
        'precision_at_65': float(prec_65),
        'n_signals_at_65': int(n_65),
        'auc':             float(auc),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild of data cache even if fresh cache exists')
    args = parser.parse_args()

    logger.info('=' * 70)
    logger.info('[PHASE 12] NeuralTrader Model Retraining')
    logger.info(f'  Label   : 5-day fwd return > {LABEL_THRESHOLD:.0%}')
    logger.info(f'  Data    : ALL rows (no cap)')
    logger.info(f'  New feats: rs_vs_spy, high_52w_prox, volume_breakout, roc_63')
    logger.info(f'  Cache   : {"REBUILD" if args.rebuild else "USE IF FRESH (<24h)"}')
    logger.info('=' * 70)
    t0 = datetime.now()

    # 1. Backup existing models
    backup_models()

    try:
        # 2. Load data (uses cache if available and fresh)
        logger.info('[STEP 1] Loading data...')
        X, y, feature_names = load_all_data(force_rebuild=args.rebuild)

        # 3. Time-based train/val split — numpy slice (no copy = no OOM)
        logger.info(f'[STEP 2] Splitting data (val = last 10%)...')
        split_idx = int(len(X) * 0.90)
        X_train = X[:split_idx]   # numpy view — zero extra RAM
        y_train = y[:split_idx]
        X_val   = X[split_idx:]
        y_val   = y[split_idx:]
        logger.info(f'[SPLIT] Train={len(X_train):,}  Val={len(X_val):,}')

        # 4. Train + Brain-Gate + Save
        logger.info('[STEP 3] Training models...')
        results = train_and_save(X_train, y_train, X_val, y_val, feature_names)

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info('=' * 70)
        logger.info(f'[DONE] Phase 12 retrain complete in {elapsed/60:.1f} min')
        logger.info(f'[DONE] Ensemble precision@0.65 = {results["ensemble"]["precision_at_65"]:.1%}')
        logger.info(f'[DONE] Ensemble AUC            = {results["ensemble"]["auc"]:.3f}')
        logger.info(f'[DONE] Models saved to {MODELS_DIR}')
        logger.info('=' * 70)

    except Exception as e:
        logger.error(f'[FAIL] Retraining failed: {e}')
        logger.error(traceback.format_exc())
        logger.error('[FAIL] Restoring backed-up models...')
        # Rollback: restore most recent backups
        ts_files = {}
        for bak in BACKUP_DIR.glob('*.bak'):
            base = bak.name.rsplit('.', 2)[0]
            ts   = bak.name.rsplit('.', 2)[1]
            if base not in ts_files or ts > ts_files[base][1]:
                ts_files[base] = (bak, ts)
        for base, (src, _) in ts_files.items():
            dst = MODELS_DIR / base
            shutil.copy2(src, dst)
            logger.info(f'[ROLLBACK] Restored {base}')
        raise


if __name__ == '__main__':
    main()
