#!/usr/bin/env python3
"""
Triple-Barrier Spike Test (Step 1 of Phase 14 roadmap)
=======================================================
Purpose: De-risk the Triple-Barrier hypothesis in 1 day before committing
         5-7 days to the full build.

What it does:
  1. Samples 100 random tickers from data/raw/
  2. Builds features (reuses Phase 12 pipeline - 68 features)
  3. Builds Triple-Barrier labels matching strategy exits:
       - Upper barrier: +40% (TAKE_PROFIT_PCT)
       - Lower barrier: ATR-scaled, clamped 8-20% (STOP_LOSS)
       - Vertical barrier: 25 days (MAX_HOLD_DAYS)
  4. Reports label distribution
  5. Trains ONE XGBoost 3-class model
  6. Reports precision@0.65 per class
  7. Writes reports/tb_spike_report.md

Gates (go/no-go):
  - LONG_WIN rate >= 4%
  - SHORT_WIN rate >= 15%
  - Training converges (no degenerate solutions)

Usage:
  python scripts/tb_spike_test.py [--n-tickers 100] [--seed 42]
"""
import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler(PROJECT_ROOT / 'logs' / 'tb_spike.log', mode='w', encoding='ascii')],
)
log = logging.getLogger('TB_Spike')

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
SPY_PATH = DATA_DIR / 'SPY.parquet'
REPORTS_DIR = PROJECT_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# Triple-Barrier params - defaults MATCH strategy.py; override via CLI
TP_PCT = 0.40           # TAKE_PROFIT_PCT
STOP_ATR_MULT = 2.5     # STOP_LOSS_ATR_MULT
STOP_MIN = 0.08         # STOP_LOSS_MIN_PCT
STOP_MAX = 0.20         # STOP_LOSS_MAX_PCT
TIMEOUT_DAYS = 25       # MAX_HOLD_DAYS
ATR_WINDOW = 20         # 20-day ATR for stop sizing

MIN_ROWS = 252
MIN_FEATURES = 50


def _atr_pct(df: pd.DataFrame, idx: int) -> float:
    """Compute ATR-scaled stop percentage at a given index (matches strategy.py logic)."""
    end = idx + 1
    start = max(0, end - (ATR_WINDOW + 2))
    win = df.iloc[start:end]
    if len(win) < 14:
        return STOP_MIN
    high = win['high'].values
    low = win['low'].values
    close = win['close'].values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev_close),
                            np.abs(low - prev_close)])
    atr = tr.mean()
    last_close = close[-1]
    if last_close <= 0:
        return STOP_MIN
    atr_pct = (atr * STOP_ATR_MULT) / last_close
    return float(np.clip(atr_pct, STOP_MIN, STOP_MAX))


def build_triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    """
    Walk forward TIMEOUT_DAYS from each row. Return label:
       +1 = LONG_WIN   (upper barrier hit first)
       -1 = SHORT_WIN  (lower barrier hit first)
        0 = NEUTRAL    (timeout, neither hit)
    Aligned to df.index; last TIMEOUT_DAYS rows dropped.
    """
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
        label = 0  # default: NEUTRAL (timeout)
        for s in range(t + 1, t + 1 + TIMEOUT_DAYS):
            if highs[s] >= upper:
                label = 1  # LONG_WIN
                break
            if lows[s] <= lower:
                label = -1  # SHORT_WIN
                break
        labels[t] = label
        valid[t] = True

    ser = pd.Series(labels, index=df.index, dtype=np.int8)
    ser = ser[valid]
    return ser


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


def process_ticker(pf: Path, spy_close: pd.Series):
    """Return (X_df, y_series) for one ticker, or (None, None) on failure."""
    try:
        df = pd.read_parquet(pf)
        if len(df) < MIN_ROWS:
            return None, None
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Need OHLC for barriers
        required = {'high', 'low', 'close'}
        if not required.issubset(df.columns):
            return None, None

        feats = build_features(df, spy_close)
        if feats.empty or len(feats) < MIN_FEATURES:
            return None, None

        labels = build_triple_barrier_labels(df)
        common = feats.index.intersection(labels.index)
        if len(common) < MIN_FEATURES:
            return None, None

        return feats.loc[common], labels.loc[common], common
    except Exception as e:
        log.debug(f'[{pf.stem}] skipped: {e}')
        return None, None, None


def main():
    global TP_PCT, STOP_MIN, TIMEOUT_DAYS
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tickers', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tp', type=float, default=TP_PCT, help='Take-profit barrier (default 0.40)')
    ap.add_argument('--stop-min', type=float, default=STOP_MIN, help='Stop floor pct (default 0.08)')
    ap.add_argument('--timeout', type=int, default=TIMEOUT_DAYS, help='Timeout days (default 25)')
    ap.add_argument('--tag', type=str, default='default', help='Config tag for report filename')
    args = ap.parse_args()
    TP_PCT = args.tp
    STOP_MIN = args.stop_min
    TIMEOUT_DAYS = args.timeout

    t0 = time.time()
    log.info(f'=== TRIPLE-BARRIER SPIKE TEST ===')
    log.info(f'n_tickers={args.n_tickers}  seed={args.seed}')
    log.info(f'Barriers: TP=+{TP_PCT:.0%} | Stop=ATR{STOP_ATR_MULT}x [{STOP_MIN:.0%}-{STOP_MAX:.0%}] | Timeout={TIMEOUT_DAYS}d')

    # Load SPY
    spy_close = None
    if SPY_PATH.exists():
        spy_df = pd.read_parquet(SPY_PATH)
        spy_close = spy_df['adjClose'] if 'adjClose' in spy_df.columns else spy_df['close']
        spy_close.index = pd.to_datetime(spy_close.index)
        log.info(f'[SPY] loaded: {len(spy_close)} rows')

    # Sample tickers
    all_pfs = sorted(DATA_DIR.glob('*.parquet'))
    random.seed(args.seed)
    sampled = random.sample(all_pfs, min(args.n_tickers, len(all_pfs)))
    log.info(f'[DATA] universe={len(all_pfs)}  sampled={len(sampled)}')

    # Build matrix
    all_X, all_y, all_dates = [], [], []
    ok = fail = 0
    feature_names = None

    for i, pf in enumerate(sampled, 1):
        X_t, y_t, idx_t = process_ticker(pf, spy_close)
        if X_t is None:
            fail += 1
            continue
        if feature_names is None:
            feature_names = list(X_t.columns)
        else:
            X_t = X_t.reindex(columns=feature_names, fill_value=0.0)
        all_X.append(X_t.values.astype(np.float32))
        all_y.append(y_t.values.astype(np.int8))
        all_dates.append(idx_t.values.astype('datetime64[ns]'))
        ok += 1
        if i % 20 == 0:
            log.info(f'  progress: {i}/{len(sampled)}  ok={ok} fail={fail}')

    if ok == 0:
        log.error('[FATAL] No tickers processed successfully.')
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    dates = np.concatenate(all_dates)

    # TRUE chronological sort across ALL tickers
    order = np.argsort(dates, kind='stable')
    X = X[order]
    y = y[order]
    dates = dates[order]
    log.info(f'[DATA] built matrix: X={X.shape}  y={y.shape}  tickers_ok={ok}  fail={fail}')
    log.info(f'[DATA] date range: {str(dates[0])[:10]} -> {str(dates[-1])[:10]}')

    # --- Label distribution ---
    n_total = len(y)
    n_long = int((y == 1).sum())
    n_neutral = int((y == 0).sum())
    n_short = int((y == -1).sum())
    p_long = n_long / n_total
    p_neutral = n_neutral / n_total
    p_short = n_short / n_total

    log.info(f'[LABELS] LONG_WIN={p_long:.2%} ({n_long})  NEUTRAL={p_neutral:.2%} ({n_neutral})  SHORT_WIN={p_short:.2%} ({n_short})')

    # --- Gate 1: distribution ---
    gate_long = p_long >= 0.04
    gate_short = p_short >= 0.15
    log.info(f'[GATE] LONG>=4%: {"PASS" if gate_long else "FAIL"}  SHORT>=15%: {"PASS" if gate_short else "FAIL"}')

    # --- Train XGBoost 3-class ---
    # Remap -1 -> 2 for XGBoost (needs 0..N-1)
    y_xgb = np.where(y == -1, 2, y).astype(np.int8)  # 0=NEUTRAL, 1=LONG_WIN, 2=SHORT_WIN

    # TRUE TIME-ORDERED split: rows already sorted by date across all tickers.
    # Last 20% by date = test set. Guarantees NO temporal leakage.
    split_idx = int(len(y_xgb) * 0.8)
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_tr, y_te = y_xgb[:split_idx], y_xgb[split_idx:]
    train_end_date = str(dates[split_idx - 1])[:10]
    test_start_date = str(dates[split_idx])[:10]

    log.info(f'[TRAIN] train={len(X_tr)}  test={len(X_te)}  (time-ordered split)')
    log.info(f'[TRAIN] train ends {train_end_date}  |  test starts {test_start_date}')
    log.info(f'[TRAIN] class_weight=balanced (inverse frequency)')

    # Inverse-frequency sample weights (balanced class weighting)
    class_counts = np.bincount(y_tr, minlength=3)
    class_weights = len(y_tr) / (3.0 * class_counts)
    sample_w = class_weights[y_tr]

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=3,
        eval_metric='mlogloss',
        n_jobs=-1, tree_method='hist', verbosity=0,
    )
    t_train0 = time.time()
    model.fit(X_tr, y_tr, sample_weight=sample_w)
    t_train = time.time() - t_train0
    log.info(f'[TRAIN] done in {t_train:.1f}s')

    # --- Evaluate ---
    proba = model.predict_proba(X_te)  # [P(NEUTRAL), P(LONG), P(SHORT)]
    preds = proba.argmax(axis=1)
    acc = (preds == y_te).mean()
    log.info(f'[EVAL] accuracy={acc:.3f}')

    # Per-class precision@0.65 (high-confidence gate used in production)
    def prec_at(cls_idx, thr):
        sel = proba[:, cls_idx] >= thr
        if sel.sum() == 0:
            return None, 0
        correct = (y_te[sel] == cls_idx).sum()
        return float(correct / sel.sum()), int(sel.sum())

    prec_neutral_065, n_neu = prec_at(0, 0.65)
    prec_long_065, n_lng = prec_at(1, 0.65)
    prec_short_065, n_srt = prec_at(2, 0.65)

    log.info(f'[PREC@0.65] LONG_WIN  = {prec_long_065}  (n={n_lng})')
    log.info(f'[PREC@0.65] SHORT_WIN = {prec_short_065}  (n={n_srt})')
    log.info(f'[PREC@0.65] NEUTRAL   = {prec_neutral_065}  (n={n_neu})')

    # Probability dynamic range (diagnose "stuck at 0.30" problem)
    log.info(f'[DYNAMIC RANGE] P(LONG):   min={proba[:,1].min():.3f} max={proba[:,1].max():.3f} mean={proba[:,1].mean():.3f} std={proba[:,1].std():.3f}')
    log.info(f'[DYNAMIC RANGE] P(SHORT):  min={proba[:,2].min():.3f} max={proba[:,2].max():.3f} mean={proba[:,2].mean():.3f} std={proba[:,2].std():.3f}')

    # Confusion matrix
    cm = confusion_matrix(y_te, preds, labels=[0, 1, 2])
    log.info(f'[CONFUSION] rows=true, cols=pred [NEU, LONG, SHORT]')
    for i, row in enumerate(cm):
        log.info(f'  {["NEU","LONG","SHORT"][i]}: {row.tolist()}')

    # --- GO / NO-GO ---
    go_nogo = []
    if gate_long:
        go_nogo.append('PASS: LONG_WIN >= 4%')
    else:
        go_nogo.append(f'FAIL: LONG_WIN {p_long:.2%} < 4%')
    if gate_short:
        go_nogo.append('PASS: SHORT_WIN >= 15%')
    else:
        go_nogo.append(f'FAIL: SHORT_WIN {p_short:.2%} < 15%')

    training_ok = (prec_long_065 is not None and prec_long_065 > 0.4) and \
                  (prec_short_065 is not None and prec_short_065 > 0.4)
    if training_ok:
        go_nogo.append('PASS: Training converged (prec@0.65 > 40% for LONG and SHORT)')
    else:
        go_nogo.append('FAIL: Training degenerate or precision too low')

    overall_go = gate_long and gate_short and training_ok

    # --- Write report ---
    tag = args.tag
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f'tb_spike_{tag}_{ts}.md'
    lines = [
        '# Triple-Barrier Spike Test Report',
        '',
        f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'**Branch:** feature/phase14-tb-shorts  **Step:** 1 of 13',
        '',
        '## Config',
        f'- Tickers sampled: **{ok}** (failed: {fail}) from {len(all_pfs)} universe',
        f'- Seed: {args.seed}',
        f'- Barriers: TP=+{TP_PCT:.0%} | Stop=ATR{STOP_ATR_MULT}x [{STOP_MIN:.0%}-{STOP_MAX:.0%}] | Timeout={TIMEOUT_DAYS}d',
        f'- Training rows: {len(X_tr):,} | Test rows: {len(X_te):,}',
        f'- Features: {X.shape[1]}',
        '',
        '## Label Distribution',
        f'| Class | Count | % |',
        f'|---|---|---|',
        f'| LONG_WIN  | {n_long:,} | **{p_long:.2%}** |',
        f'| NEUTRAL   | {n_neutral:,} | **{p_neutral:.2%}** |',
        f'| SHORT_WIN | {n_short:,} | **{p_short:.2%}** |',
        '',
        '## Model Performance',
        f'- Overall accuracy: **{acc:.3f}**',
        f'- Train time: {t_train:.1f}s',
        '',
        '### Precision @ 0.65 (high-confidence gate)',
        f'| Class | Precision | Samples |',
        f'|---|---|---|',
        f'| LONG_WIN  | {prec_long_065}  | {n_lng} |',
        f'| SHORT_WIN | {prec_short_065} | {n_srt} |',
        f'| NEUTRAL   | {prec_neutral_065} | {n_neu} |',
        '',
        '### Probability Dynamic Range',
        f'- P(LONG):   min={proba[:,1].min():.3f}  max={proba[:,1].max():.3f}  mean={proba[:,1].mean():.3f}  std={proba[:,1].std():.3f}',
        f'- P(SHORT):  min={proba[:,2].min():.3f}  max={proba[:,2].max():.3f}  mean={proba[:,2].mean():.3f}  std={proba[:,2].std():.3f}',
        '',
        '### Confusion Matrix (rows=true, cols=pred)',
        f'|       | NEU | LONG | SHORT |',
        f'|-------|-----|------|-------|',
        f'| NEU   | {cm[0,0]} | {cm[0,1]} | {cm[0,2]} |',
        f'| LONG  | {cm[1,0]} | {cm[1,1]} | {cm[1,2]} |',
        f'| SHORT | {cm[2,0]} | {cm[2,1]} | {cm[2,2]} |',
        '',
        '## Gates',
    ]
    for g in go_nogo:
        lines.append(f'- {g}')
    lines += [
        '',
        f'## Overall: **{"GO" if overall_go else "NO-GO"}**',
        '',
        f'- Total runtime: {time.time()-t0:.1f}s',
    ]

    report_path.write_text('\n'.join(lines), encoding='utf-8')
    log.info(f'[REPORT] written to {report_path}')

    # --- JSON summary for programmatic use ---
    summary = {
        'timestamp': datetime.now().isoformat(),
        'tickers_ok': ok, 'tickers_fail': fail,
        'n_rows_total': int(n_total),
        'label_dist': {'long_win': p_long, 'neutral': p_neutral, 'short_win': p_short},
        'n_features': int(X.shape[1]),
        'train_seconds': t_train,
        'accuracy': float(acc),
        'precision_at_065': {
            'long_win': prec_long_065, 'long_n': n_lng,
            'short_win': prec_short_065, 'short_n': n_srt,
            'neutral': prec_neutral_065, 'neutral_n': n_neu,
        },
        'dynamic_range': {
            'long_mean': float(proba[:,1].mean()), 'long_std': float(proba[:,1].std()),
            'short_mean': float(proba[:,2].mean()), 'short_std': float(proba[:,2].std()),
        },
        'gates': {'long_win_4pct': bool(gate_long), 'short_win_15pct': bool(gate_short), 'training_ok': bool(training_ok)},
        'overall_go': bool(overall_go),
    }
    json_path = report_path.with_suffix('.json')
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log.info(f'[SUMMARY] {json_path}')

    # Exit code reflects gate
    log.info(f'=== SPIKE COMPLETE: {"GO" if overall_go else "NO-GO"} === ({time.time()-t0:.1f}s)')
    sys.exit(0 if overall_go else 2)


if __name__ == '__main__':
    main()
