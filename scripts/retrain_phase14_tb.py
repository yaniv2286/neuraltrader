#!/usr/bin/env python3
"""
Phase 14 - Triple-Barrier Multi-Class Retraining
================================================
Replaces Phase 12's binary label with 3-class Triple-Barrier labels.

Config A (strict match to strategy.py):
  - TP: +40% (TAKE_PROFIT_PCT)
  - Stop: ATR 2.5x, clamped [8%-20%] (STOP_LOSS)
  - Timeout: 25 days (MAX_HOLD_DAYS)

Output: 3-class models (NEUTRAL=0, LONG_WIN=1, SHORT_WIN=2)
  - models/xgboost_model.pkl
  - models/lightgbm_model.pkl
  - models/rf_model.pkl (HistGradientBoosting)
  - models/feature_names.pkl
  - models/feature_scaler.pkl
  - models/ensemble_metadata.pkl

Brain-Gate: Precision@0.65 per class must exceed 50% or rollback.
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
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'retrain_phase14.log', mode='w', encoding='ascii'),
    ]
)
logger = logging.getLogger('Phase14')

# Config
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
BACKUP_DIR = MODELS_DIR / 'backup_phase13'
SPY_PATH = DATA_DIR / 'SPY.parquet'

# Triple-Barrier params (Config A - strict match)
TP_PCT = 0.40
STOP_ATR_MULT = 2.5
STOP_MIN = 0.08
STOP_MAX = 0.20
TIMEOUT_DAYS = 25
ATR_WINDOW = 20

MIN_ROWS = 252
PRECISION_GATE = 0.50  # per-class precision@0.65 threshold

# Model hyperparams
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, gamma=1.0,
    objective='multi:softprob', num_class=3,
    eval_metric='mlogloss',
    n_jobs=-1, tree_method='hist', verbosity=0,
)
LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    objective='multiclass', num_class=3,
    n_jobs=-1, verbosity=-1,
)
HGB_PARAMS = dict(
    max_iter=30, max_depth=3, learning_rate=0.15,
    min_samples_leaf=200, l2_regularization=3.0,
    max_bins=32,
)

WEIGHTS = {'xgboost': 0.40, 'lightgbm': 0.40, 'hgb': 0.20}


def _atr_pct(df: pd.DataFrame, idx: int) -> float:
    """ATR-scaled stop % at index (matches strategy.py)."""
    end = idx + 1
    start = max(0, end - (ATR_WINDOW + 2))
    win = df.iloc[start:end]
    if len(win) < 14:
        return STOP_MIN
    high = win['high'].values
    low = win['low'].values
    close = win['close'].values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = tr.mean()
    if close[-1] <= 0:
        return STOP_MIN
    atr_pct = (atr * STOP_ATR_MULT) / close[-1]
    return float(np.clip(atr_pct, STOP_MIN, STOP_MAX))


def build_triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    """Walk-forward TB labeling. Returns int8 series: +1=LONG_WIN, -1=SHORT_WIN, 0=NEUTRAL."""
    if 'adjClose' in df.columns:
        close = df['adjClose'].values
    else:
        close = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    labels = np.zeros(n, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)

    for t in range(n - TIMEOUT_DAYS):
        entry = close[t]
        if entry <= 0:
            continue
        stop_pct = _atr_pct(df, t)
        upper = entry * (1.0 + TP_PCT)
        lower = entry * (1.0 - stop_pct)
        label = 0
        for s in range(t + 1, t + 1 + TIMEOUT_DAYS):
            if highs[s] >= upper:
                label = 1
                break
            if lows[s] <= lower:
                label = -1
                break
        labels[t] = label
        valid[t] = True

    ser = pd.Series(labels, index=df.index, dtype=np.int8)
    return ser[valid]


def build_features(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Reuse Phase 12 feature pipeline (68 features)."""
    from core.feature_engineer import FeatureEngineer
    fe = FeatureEngineer(use_advanced_features=True, verbose=False)
    feats, _ = fe.create_features(df, target_type='direction')
    if feats.empty:
        return feats

    close = df['adjClose'] if 'adjClose' in df.columns else df['close']
    close = close.reindex(feats.index)

    ticker_ret20 = close.pct_change(20)
    if spy_close is not None and not spy_close.empty:
        spy_aligned = spy_close.reindex(feats.index, method='ffill')
        spy_ret20 = spy_aligned.pct_change(20)
        feats['rs_vs_spy'] = (ticker_ret20 - spy_ret20).fillna(0).clip(-2, 2)
    else:
        feats['rs_vs_spy'] = ticker_ret20.fillna(0).clip(-2, 2)

    high_52w = close.rolling(252, min_periods=60).max()
    feats['high_52w_prox'] = (close / high_52w.replace(0, np.nan)).fillna(0).clip(0, 1.5)

    if 'volume' in df.columns:
        vol = df['volume'].reindex(feats.index)
        vol_avg50 = vol.rolling(50, min_periods=20).mean()
        feats['volume_breakout'] = (vol / vol_avg50.replace(0, np.nan)).fillna(1.0).clip(0, 10)
    else:
        feats['volume_breakout'] = 1.0

    feats['roc_63'] = close.pct_change(63).fillna(0).clip(-2, 2)
    return feats.dropna()


def load_all_data():
    """Load all tickers, build TB labels, return (X, y, feature_names)."""
    parquet_files = sorted(DATA_DIR.glob('*.parquet'))
    logger.info(f'[DATA] Found {len(parquet_files)} parquet files')

    spy_close = None
    if SPY_PATH.exists():
        spy_df = pd.read_parquet(SPY_PATH)
        spy_close = spy_df['adjClose'] if 'adjClose' in spy_df.columns else spy_df['close']
        spy_close.index = pd.to_datetime(spy_close.index)
        logger.info('[SPY] Loaded for rs_vs_spy feature')

    all_X, all_y = [], []
    ok = fail = 0
    feature_names = None

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if len(df) < MIN_ROWS:
                continue
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            required = {'high', 'low', 'close'}
            if not required.issubset(df.columns):
                continue

            feats = build_features(df, spy_close)
            if feats.empty or len(feats) < 50:
                continue

            labels = build_triple_barrier_labels(df)
            common = feats.index.intersection(labels.index)
            if len(common) < 50:
                continue

            X_t = feats.loc[common]
            y_t = labels.loc[common]

            if feature_names is None:
                feature_names = list(X_t.columns)
            else:
                X_t = X_t.reindex(columns=feature_names, fill_value=0.0)

            all_X.append(X_t.values.astype(np.float32))
            all_y.append(y_t.values.astype(np.int8))
            ok += 1

            if ok % 200 == 0:
                total = sum(len(x) for x in all_X)
                logger.info(f'[DATA] {ok} tickers, {total:,} rows so far...')

        except Exception as e:
            fail += 1
            if fail <= 5:
                logger.warning(f'[{pf.stem}] failed: {e}')

    if ok == 0:
        raise RuntimeError('[FATAL] No tickers processed.')

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    logger.info(f'[DATA] Final: X={X.shape}  y={y.shape}  ok={ok}  fail={fail}')

    n_total = len(y)
    n_long = int((y == 1).sum())
    n_neutral = int((y == 0).sum())
    n_short = int((y == -1).sum())
    logger.info(f'[LABELS] LONG_WIN={n_long/n_total:.2%} ({n_long})  NEUTRAL={n_neutral/n_total:.2%} ({n_neutral})  SHORT_WIN={n_short/n_total:.2%} ({n_short})')

    return X, y, feature_names


def train_model(X_tr, y_tr, X_te, y_te, model_name, params):
    """Train one model, return (model, metrics)."""
    logger.info(f'[{model_name.upper()}] Training...')

    # Inverse-frequency sample weights
    class_counts = np.bincount(y_tr, minlength=3)
    class_weights = len(y_tr) / (3.0 * class_counts)
    sample_w = class_weights[y_tr]

    if model_name == 'xgboost':
        model = XGBClassifier(**params)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(**params)
    elif model_name == 'hgb':
        model = HistGradientBoostingClassifier(**params)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.fit(X_tr, y_tr, sample_weight=sample_w)

    proba = model.predict_proba(X_te)
    preds = proba.argmax(axis=1)
    acc = (preds == y_te).mean()

    # Precision@0.65 per class
    def prec_at(cls_idx, thr):
        sel = proba[:, cls_idx] >= thr
        if sel.sum() == 0:
            return None, 0
        correct = (y_te[sel] == cls_idx).sum()
        return float(correct / sel.sum()), int(sel.sum())

    prec_neu, n_neu = prec_at(0, 0.65)
    prec_lng, n_lng = prec_at(1, 0.65)
    prec_srt, n_srt = prec_at(2, 0.65)

    logger.info(f'[{model_name.upper()}] Acc={acc:.3f}  Prec@0.65: LONG={prec_lng} (n={n_lng})  SHORT={prec_srt} (n={n_srt})  NEU={prec_neu} (n={n_neu})')

    return model, {
        'accuracy': acc,
        'prec_long_065': prec_lng, 'n_long_065': n_lng,
        'prec_short_065': prec_srt, 'n_short_065': n_srt,
        'prec_neutral_065': prec_neu, 'n_neutral_065': n_neu,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-rebuild', action='store_true', help='Ignore cache, rebuild from parquets')
    args = ap.parse_args()

    t0 = datetime.now()
    logger.info('=== PHASE 14 TRIPLE-BARRIER RETRAINING ===')
    logger.info(f'Config A: TP={TP_PCT:.0%}  Stop=ATR{STOP_ATR_MULT}x [{STOP_MIN:.0%}-{STOP_MAX:.0%}]  Timeout={TIMEOUT_DAYS}d')

    # Backup old models
    if MODELS_DIR.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for f in ['xgboost_model.pkl', 'lightgbm_model.pkl', 'rf_model.pkl', 'feature_scaler.pkl', 'feature_names.pkl', 'ensemble_metadata.pkl']:
            src = MODELS_DIR / f
            if src.exists():
                shutil.copy2(src, BACKUP_DIR / f)
        logger.info(f'[BACKUP] Old models saved to {BACKUP_DIR}')

    # Load data
    X, y, feature_names = load_all_data()

    # Remap -1 -> 2 for sklearn
    y_xgb = np.where(y == -1, 2, y).astype(np.int8)

    # Time-ordered split (80/20)
    split_idx = int(len(y_xgb) * 0.8)
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_tr, y_te = y_xgb[:split_idx], y_xgb[split_idx:]
    logger.info(f'[SPLIT] train={len(X_tr):,}  test={len(X_te):,}')

    # Scale
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # Train ensemble
    models = {}
    metrics = {}
    for name, params in [('xgboost', XGB_PARAMS), ('lightgbm', LGB_PARAMS), ('hgb', HGB_PARAMS)]:
        models[name], metrics[name] = train_model(X_tr_sc, y_tr, X_te_sc, y_te, name, params)
        gc.collect()

    # Brain-Gate check
    gate_pass = True
    for name, m in metrics.items():
        if m['prec_long_065'] is None or m['prec_long_065'] < PRECISION_GATE:
            logger.error(f'[BRAIN-GATE] {name} LONG precision {m["prec_long_065"]} < {PRECISION_GATE} — FAIL')
            gate_pass = False
        if m['prec_short_065'] is None or m['prec_short_065'] < PRECISION_GATE:
            logger.error(f'[BRAIN-GATE] {name} SHORT precision {m["prec_short_065"]} < {PRECISION_GATE} — FAIL')
            gate_pass = False

    if not gate_pass:
        logger.error('[BRAIN-GATE] FAILED — rolling back to Phase 13 models')
        for f in BACKUP_DIR.glob('*.pkl'):
            shutil.copy2(f, MODELS_DIR / f.name)
        logger.info('[ROLLBACK] Phase 13 models restored')
        sys.exit(1)

    logger.info('[BRAIN-GATE] PASSED — saving new models')

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(models['xgboost'], open(MODELS_DIR / 'xgboost_model.pkl', 'wb'))
    pickle.dump(models['lightgbm'], open(MODELS_DIR / 'lightgbm_model.pkl', 'wb'))
    pickle.dump(models['hgb'], open(MODELS_DIR / 'rf_model.pkl', 'wb'))
    pickle.dump(scaler, open(MODELS_DIR / 'feature_scaler.pkl', 'wb'))
    pickle.dump(feature_names, open(MODELS_DIR / 'feature_names.pkl', 'wb'))

    metadata = {
        'phase': 14,
        'label_type': 'triple_barrier_3class',
        'config': f'TP={TP_PCT} Stop=ATR{STOP_ATR_MULT}x[{STOP_MIN}-{STOP_MAX}] Timeout={TIMEOUT_DAYS}d',
        'n_features': len(feature_names),
        'n_train': len(X_tr),
        'n_test': len(X_te),
        'weights': WEIGHTS,
        'metrics': metrics,
        'trained': datetime.now().isoformat(),
    }
    pickle.dump(metadata, open(MODELS_DIR / 'ensemble_metadata.pkl', 'wb'))
    (MODELS_DIR / 'ensemble_metadata.json').write_text(json.dumps(metadata, indent=2))

    logger.info(f'[DONE] Phase 14 models saved to {MODELS_DIR}')
    logger.info(f'[TIME] Total: {(datetime.now() - t0).total_seconds():.1f}s')


if __name__ == '__main__':
    main()
