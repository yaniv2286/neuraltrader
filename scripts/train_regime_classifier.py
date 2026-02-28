#!/usr/bin/env python3
"""
NeuralTrader Phase 12 — AI Regime Classifier
=============================================

Replaces the hardcoded 'SPY > 100-day SMA' rule with a 3-class ML model:
  0 = BEAR   (SPY below SMA200, VIX elevated)
  1 = BULL   (SPY above SMA200, VIX normal)
  2 = CRISIS (VIX > 2-sigma Bollinger upper band — extreme fear)

Entry rules by regime:
  BULL   -> confidence threshold = 0.65 (normal)
  BEAR   -> confidence threshold = 0.72 (much stricter — fewer entries)
  CRISIS -> no new entries at all

Output:
  models/regime_classifier.pkl
  models/regime_classifier_meta.json

The regime classifier is called once per day in the backtest/paper pipeline.
"""

import sys
import json
import pickle
import logging
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / 'logs' / 'regime_classifier.log',
            mode='w', encoding='ascii'
        ),
    ]
)
logger = logging.getLogger('RegimeClassifier')

MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR   = PROJECT_ROOT / 'data' / 'raw'
SPY_PATH   = DATA_DIR / 'SPY.parquet'
VXX_PATH   = DATA_DIR / 'VXX.parquet'

# Regime thresholds used at inference time — stored in metadata
REGIME_THRESHOLDS = {
    0: None,    # CRISIS  — no new entries
    1: 0.72,    # BEAR    — strict threshold
    2: 0.65,    # BULL    — standard threshold
}
REGIME_LABELS = {0: 'CRISIS', 1: 'BEAR', 2: 'BULL'}


# ---------------------------------------------------------------------------
# Feature builder for regime classification
# ---------------------------------------------------------------------------

def build_regime_features(spy: pd.DataFrame, vxx: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily regime features from SPY + VXX OHLCV data.
    All features are purely backward-looking (no lookahead).
    """
    close_spy = spy['adjClose'] if 'adjClose' in spy.columns else spy['close']
    close_spy = close_spy.sort_index()

    vol_spy = None
    if 'volume' in spy.columns:
        vol_spy = spy['volume'].sort_index()

    features = pd.DataFrame(index=close_spy.index)

    # SPY trend features
    for w in [20, 50, 100, 200]:
        sma = close_spy.rolling(w, min_periods=w).mean()
        features[f'spy_sma{w}_ratio'] = (close_spy / sma).fillna(1.0)

    # SPY momentum
    for w in [5, 10, 20, 60]:
        features[f'spy_roc_{w}'] = close_spy.pct_change(w).fillna(0)

    # SPY 20-day volatility (annualised)
    spy_ret = close_spy.pct_change()
    features['spy_vol_20'] = spy_ret.rolling(20).std().fillna(0) * np.sqrt(252)

    # SPY 20-day vol regime (current vol / 60-day vol)
    vol_60 = spy_ret.rolling(60).std()
    features['spy_vol_regime'] = (features['spy_vol_20'] / (vol_60 * np.sqrt(252))).fillna(1.0)

    # SPY drawdown from 52-week high
    high_52w = close_spy.rolling(252, min_periods=60).max()
    features['spy_dd_52w'] = ((close_spy - high_52w) / high_52w).fillna(0)

    # SPY above/below key SMAs (binary)
    for w in [50, 100, 200]:
        sma = close_spy.rolling(w, min_periods=w).mean()
        features[f'spy_above_sma{w}'] = (close_spy > sma).astype(float)

    # SPY consecutive up/down days
    spy_up = (spy_ret > 0).astype(int)
    features['spy_streak'] = spy_up.rolling(10).sum() / 10.0

    # VXX features
    if vxx is not None and not vxx.empty:
        close_vxx = vxx['adjClose'] if 'adjClose' in vxx.columns else vxx['close']
        close_vxx = close_vxx.reindex(close_spy.index, method='ffill')

        features['vxx_level'] = close_vxx.ffill()
        features['vxx_roc_5'] = close_vxx.pct_change(5).fillna(0)
        features['vxx_roc_20'] = close_vxx.pct_change(20).fillna(0)

        # VXX Bollinger Band position (crisis signal)
        vxx_sma20 = close_vxx.rolling(20).mean()
        vxx_std20 = close_vxx.rolling(20).std()
        bb_upper  = vxx_sma20 + 2 * vxx_std20
        features['vxx_bb_pct']  = ((close_vxx - vxx_sma20) / (vxx_std20 + 1e-9)).fillna(0)
        features['vxx_above_bb'] = (close_vxx > bb_upper).astype(float)
    else:
        logger.warning('[WARN] VXX not available — regime classifier will use SPY-only features')
        for col in ['vxx_level', 'vxx_roc_5', 'vxx_roc_20', 'vxx_bb_pct', 'vxx_above_bb']:
            features[col] = 0.0

    features = features.dropna()
    return features


# ---------------------------------------------------------------------------
# Label builder — 3-class regime
# ---------------------------------------------------------------------------

def build_regime_labels(spy: pd.DataFrame, vxx: pd.DataFrame, features_index) -> pd.Series:
    """
    Label each day:
      2 = BULL   : SPY > SMA200 AND not in crisis
      1 = BEAR   : SPY < SMA200 AND not in crisis
      0 = CRISIS : VXX > 2-sigma BB  OR  (no VXX) SPY 20d vol > 2x its 252d avg vol
    """
    close_spy = spy['adjClose'] if 'adjClose' in spy.columns else spy['close']
    close_spy = close_spy.sort_index().reindex(features_index)

    sma200   = close_spy.rolling(200, min_periods=100).mean()
    spy_bull = (close_spy > sma200)

    if vxx is not None and not vxx.empty:
        close_vxx = vxx['adjClose'] if 'adjClose' in vxx.columns else vxx['close']
        close_vxx = close_vxx.reindex(features_index, method='ffill')
        vxx_sma20 = close_vxx.rolling(20).mean()
        vxx_std20 = close_vxx.rolling(20).std()
        bb_upper  = vxx_sma20 + 2 * vxx_std20
        is_crisis = (close_vxx > bb_upper)
        logger.info(f'[LABEL] VXX-based crisis: {is_crisis.sum()} days')
    else:
        # Fallback: crisis = SPY 20d realised vol > 2x its own 252d average
        spy_ret   = close_spy.pct_change()
        vol_20    = spy_ret.rolling(20).std()
        vol_252   = spy_ret.rolling(252, min_periods=60).std()
        is_crisis = (vol_20 > vol_252 * 2.0).fillna(False)
        logger.info(f'[LABEL] SPY-vol crisis fallback: {is_crisis.sum()} days')

    labels = pd.Series(2, index=features_index)  # default BULL
    labels[~spy_bull] = 1                          # BEAR
    labels[is_crisis] = 0                          # CRISIS (overrides BEAR)

    return labels.dropna()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info('=' * 60)
    logger.info('[REGIME] Training 3-class Regime Classifier')
    logger.info('  0=CRISIS  1=BEAR  2=BULL')
    logger.info('=' * 60)

    # Load SPY
    if not SPY_PATH.exists():
        raise FileNotFoundError(f'SPY.parquet not found: {SPY_PATH}')
    spy = pd.read_parquet(SPY_PATH)
    if 'date' in spy.columns:
        spy = spy.set_index('date')
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    logger.info(f'[DATA] SPY: {len(spy)} rows ({spy.index[0].date()} -> {spy.index[-1].date()})')

    # Load VXX (optional)
    vxx = None
    if VXX_PATH.exists():
        vxx = pd.read_parquet(VXX_PATH)
        if 'date' in vxx.columns:
            vxx = vxx.set_index('date')
        vxx.index = pd.to_datetime(vxx.index)
        vxx = vxx.sort_index()
        logger.info(f'[DATA] VXX: {len(vxx)} rows ({vxx.index[0].date()} -> {vxx.index[-1].date()})')
    else:
        logger.warning('[WARN] VXX.parquet not found — using SPY-only features')

    # Build features and labels
    logger.info('[STEP 1] Building regime features...')
    X = build_regime_features(spy, vxx)
    logger.info(f'[STEP 1] Features: {X.shape} | Columns: {X.shape[1]}')

    logger.info('[STEP 2] Building regime labels...')
    y = build_regime_labels(spy, vxx, X.index)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    logger.info(f'[STEP 2] Labels: {dict(y.value_counts().sort_index())}')
    for code, name in REGIME_LABELS.items():
        n = (y == code).sum()
        logger.info(f'  {name:<8} ({code}): {n:,} days ({n/len(y):.1%})')

    # Time-based split: last 3 years = validation
    split_date = pd.Timestamp('2022-01-01')
    train_mask = X.index < split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[~train_mask], y[~train_mask]
    logger.info(f'[SPLIT] Train={len(X_train):,} (<2022)  Val={len(X_val):,} (2022+)')

    # Scale
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # Train GradientBoosting (fast, good on tabular, no random forest variance issues)
    logger.info('[STEP 3] Training GradientBoostingClassifier...')
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    val_preds = clf.predict(X_val_s)
    acc = accuracy_score(y_val, val_preds)
    logger.info(f'[EVAL] Validation accuracy: {acc:.1%}')
    logger.info('[EVAL] Classification report:')
    # Only include labels that actually appear in val set
    present_labels = sorted(set(y_val.unique()) | set(val_preds))
    present_names  = [REGIME_LABELS[i] for i in present_labels]
    report = classification_report(
        y_val, val_preds,
        labels=present_labels,
        target_names=present_names,
    )
    for line in report.splitlines():
        logger.info(f'  {line}')

    # Validate: accuracy must be reasonable (>55%)
    if acc < 0.55:
        raise RuntimeError(f'Regime classifier accuracy {acc:.1%} < 55% — aborting save')

    # Save
    MODELS_DIR.mkdir(exist_ok=True)
    clf_path    = MODELS_DIR / 'regime_classifier.pkl'
    scaler_path = MODELS_DIR / 'regime_scaler.pkl'
    meta_path   = MODELS_DIR / 'regime_classifier_meta.json'

    with open(clf_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    feature_names = X.columns.tolist()
    meta = {
        'train_date':        datetime.now().isoformat(),
        'phase':             'Phase12',
        'feature_names':     feature_names,
        'n_features':        len(feature_names),
        'n_train':           int(len(X_train)),
        'n_val':             int(len(X_val)),
        'val_accuracy':      float(acc),
        'regime_labels':     REGIME_LABELS,
        'regime_thresholds': REGIME_THRESHOLDS,
        'label_distribution': {
            str(k): int((y == k).sum()) for k in sorted(REGIME_LABELS)
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f'[SAVE] regime_classifier.pkl  ({clf_path.stat().st_size/1e3:.0f} KB)')
    logger.info(f'[SAVE] regime_scaler.pkl')
    logger.info(f'[SAVE] regime_classifier_meta.json')
    logger.info('=' * 60)
    logger.info(f'[DONE] Regime Classifier trained | accuracy={acc:.1%}')
    logger.info('=' * 60)
    return meta


if __name__ == '__main__':
    main()
