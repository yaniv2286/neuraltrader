#!/usr/bin/env python3
"""
NeuralTrader - Phase C: Low-Turnover Harness
============================================

The walk-forward audit proved two things:
  * the model has a REAL out-of-sample signal (AUC 0.66-0.70, lift 1.6-1.8x,
    monotone deciles across all three folds),
  * the 5-day churn execution layer destroys it (Sharpe -0.26 / -0.43 OOS,
    ~207x annual turnover, -$521K of stop-losses vs +$439K of take-profits).

This harness asks the only question that matters now:
    does the same signal survive 15 bps costs when expressed as a
    low-turnover monthly top-N portfolio?

Design
------
* Scores come from the SAME walk-forward cache (data/cache/walk_forward) -
    identical features to serving, adjusted OHLC, no recomputation.
* Models come from --models-dir (a models/wf_foldN directory), so each test
    window uses a model that never saw those dates.
* Portfolio: at each rebalance date, hold the top --top-n tickers by ensemble
    LONG probability, equal weight. Optional hysteresis --keep-rank-buffer:
    an existing holding is kept while it still ranks inside top (N+buffer),
    cutting turnover further.
* Eligibility per rebalance date mirrors universe_filter.py: price >= $10,
    20d dollar volume >= $5M, last bar <= 3 business days stale, not OTC.
* Optional SPY>SMA100 regime gate (--regime sma): below it, hold cash.
* Optional --stop-pct hard stop checked daily (position -> cash until next
    rebalance).
* Costs: --slippage-bps per side + IBKR commission ($0.0035/sh, $0.35 min).
* Held position whose price data goes stale is force-exited at its last
    price (DELISTED) - mirrors the Phase A fix in the live engine.

Usage
-----
    python scripts/phase_c_lowturnover.py --models-dir models/wf_fold3 \
        --start 2024-01-01 --end 2026-10-16 --slippage-bps 15
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

from core.universe_filter import (  # noqa: E402
    is_otc_ticker, MIN_PRICE, MIN_DOLLAR_VOLUME, MAX_STALE_BDAYS, DOLLAR_VOL_WINDOW)
from scripts.walk_forward_validation import _gather_rows  # noqa: E402

DATA_DIR   = PROJECT_ROOT / 'data' / 'raw'
CACHE_DIR  = PROJECT_ROOT / 'data' / 'cache' / 'walk_forward'
REPORTS    = PROJECT_ROOT / 'reports'
LOG_DIR    = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOG_DIR / 'phase_c.log', mode='a', encoding='ascii')])
logger = logging.getLogger('PhaseC')

TRADING_DAYS_PER_MONTH = 21
COMMISSION_PER_SHARE   = 0.0035
COMMISSION_MIN         = 0.35


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cache():
    X = np.load(CACHE_DIR / 'X.npy', mmap_mode='r')
    dates = np.load(CACHE_DIR / 'dates.npy').astype('datetime64[D]')
    tix = np.load(CACHE_DIR / 'ticker_idx.npy')
    meta = json.loads((CACHE_DIR / 'meta.json').read_text())
    return X, dates, tix, meta


def load_price_matrix(tickers, start, warmup_days=140):
    """Adjusted close + adjusted volume matrices (dates x tickers), float32."""
    warmup_start = pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    close_parts, vol_parts = {}, {}
    t0 = datetime.now()
    for i, tk in enumerate(tickers):
        pf = DATA_DIR / f'{tk}.parquet'
        if not pf.exists():
            continue
        df = pd.read_parquet(pf, columns=['date', 'adjClose', 'adjVolume'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[df.index >= warmup_start]
        if len(df):
            close_parts[tk] = df['adjClose']
            vol_parts[tk] = df['adjVolume']
        if (i + 1) % 500 == 0:
            logger.info(f'[PX] {i+1}/{len(tickers)} tickers loaded '
                        f'({(datetime.now()-t0).total_seconds():.0f}s)')
    close = pd.DataFrame(close_parts).sort_index().astype(np.float32)
    vol = pd.DataFrame(vol_parts).sort_index().reindex(close.index).astype(np.float32)
    logger.info(f'[PX] price matrix {close.shape} | {(datetime.now()-t0).total_seconds():.0f}s')
    return close, vol


def load_ensemble(models_dir):
    md = Path(models_dir)
    with open(md / 'xgboost_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open(md / 'lightgbm_model.pkl', 'rb') as f:
        lgb = pickle.load(f)
    with open(md / 'rf_model.pkl', 'rb') as f:        # HGB stored under rf filename (production format)
        hgb = pickle.load(f)
    with open(md / 'feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    meta = json.loads((md / 'ensemble_metadata.json').read_text())
    weights = meta.get('weights', {'xgboost': 0.4, 'lightgbm': 0.4, 'hgb': 0.2})
    return {'xgboost': xgb, 'lightgbm': lgb, 'hgb': hgb}, scaler, weights


def score_window(X, dates, tix, meta, start, end, models, scaler, weights, tag=''):
    """Ensemble LONG-class probability for every cache row inside [start, end]."""
    cache_file = CACHE_DIR / f'scores_{tag}_{start}_{end}.parquet'
    if cache_file.exists():
        out = pd.read_parquet(cache_file)
        logger.info(f'[SCORE] cache hit {cache_file.name} ({len(out):,} rows)')
        return out
    d0, d1 = np.datetime64(start, 'D'), np.datetime64(end, 'D')
    idx = np.sort(np.where((dates >= d0) & (dates <= d1))[0])  # _gather_rows returns sorted order
    logger.info(f'[SCORE] {len(idx):,} cache rows in {start}..{end}')
    Xw = _gather_rows(X, idx)
    Xw = scaler.transform(Xw).astype(np.float32)
    tot = sum(weights.values())
    proba = None
    for name, mdl in models.items():
        p = mdl.predict_proba(Xw)[:, 1] * (weights.get(name, 0) / tot)
        proba = p if proba is None else proba + p
    out = pd.DataFrame({'date': dates[idx], 'tix': tix[idx], 'score': proba})
    out.to_parquet(cache_file)
    logger.info(f'[SCORE] done | mean={proba.mean():.3f} p90={np.quantile(proba,0.9):.3f} | cached -> {cache_file.name}')
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(scores: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame,
             tickers: list, args) -> dict:
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    days = close.index[(close.index >= start) & (close.index <= end)]
    if len(days) == 0:
        raise RuntimeError('no trading days in window')

    # score pivot: date x ticker -> ensemble LONG proba
    score_piv = scores.pivot_table(index='date', columns='tix', values='score')
    score_piv = score_piv.reindex(index=days, columns=range(len(tickers)))

    # 20d dollar volume matrix for eligibility
    dollar_vol = (close * vol).rolling(DOLLAR_VOL_WINDOW).mean().reindex(days)

    # business-day staleness: days since each ticker's last observed price
    last_seen = pd.DataFrame(index=days, columns=close.columns, dtype=np.int32)
    last_val = pd.Series(pd.NaT, index=close.columns)
    for d in days:
        seen = close.loc[d].notna()
        last_val[seen] = d
        last_seen.loc[d] = (d - last_val).dt.days
    stale = last_seen > (MAX_STALE_BDAYS + 4)   # bdays vs calendar days slack

    # SPY regime gate (leakage-free: SMA uses only past data)
    spy = None
    if args.regime == 'sma':
        spy_file = DATA_DIR / 'SPY.parquet'
        if spy_file.exists():
            sp = pd.read_parquet(spy_file, columns=['date', 'adjClose'])
            sp['date'] = pd.to_datetime(sp['date'])
            sp = sp.set_index('date').sort_index()['adjClose']
            spy = (sp > sp.rolling(100).mean()).reindex(days).fillna(False)

    rebalance_set = set(days[::args.rebalance_days])
    tk_cols = close.columns.tolist()                       # ticker name per column
    tk_to_col = {t: c for c, t in enumerate(tk_cols)}
    valid_cols = np.array([i for i, t in enumerate(tk_cols)
                           if t in set(meta_tickers(tickers)) and not is_otc_ticker(t)])
    px = close.reindex(days).values.astype(np.float64)     # nan-safe price matrix
    dv = dollar_vol.values.astype(np.float64)
    st = stale.values
    sc = score_piv.values.astype(np.float64)               # nan where no score

    cash = float(args.cash)
    pos = {}                                               # col -> (shares, entry_px)
    equity_curve, trades, n_reb = [], [], 0
    turnover_notional = 0.0
    stop_px = args.stop_pct

    for di, d in enumerate(days):
        row_px = px[di]

        # force-exit holdings whose price data went stale (delisted/acquired)
        for c in list(pos):
            if st[di, c] or np.isnan(row_px[c]):
                last_ok = np.where(~np.isnan(px[:di + 1, c]))[0]
                exit_px = px[last_ok[-1], c] if len(last_ok) else pos[c][1]
                proceeds = pos[c][0] * exit_px * (1 - args.slippage_bps / 1e4)
                proceeds -= max(COMMISSION_MIN, pos[c][0] * COMMISSION_PER_SHARE)
                cash += proceeds
                trades.append((str(d.date()), tk_cols[c], 'SELL_STALE', exit_px))
                del pos[c]

        # daily hard stop (close-basis)
        if stop_px:
            for c in list(pos):
                if not np.isnan(row_px[c]) and row_px[c] <= pos[c][1] * (1 - stop_px):
                    proceeds = pos[c][0] * row_px[c] * (1 - args.slippage_bps / 1e4)
                    proceeds -= max(COMMISSION_MIN, pos[c][0] * COMMISSION_PER_SHARE)
                    cash += proceeds
                    trades.append((str(d.date()), tk_cols[c], 'SELL_STOP', row_px[c]))
                    del pos[c]

        equity = cash + sum(sh * (row_px[c] if not np.isnan(row_px[c]) else ep)
                            for c, (sh, ep) in pos.items())
        equity_curve.append((d, equity))

        # monthly rebalance
        if d in rebalance_set:
            n_reb += 1
            if spy is not None and not spy.loc[d]:
                target_cols = []                            # regime off -> all cash
            else:
                elig = [c for c in valid_cols
                        if not np.isnan(row_px[c]) and not st[di, c]
                        and row_px[c] >= MIN_PRICE
                        and not np.isnan(dv[di, c]) and dv[di, c] >= MIN_DOLLAR_VOLUME
                        and not np.isnan(sc[di, c]) and sc[di, c] >= args.min_conf]
                ranked = sorted(elig, key=lambda c: -sc[di, c])
                keep = [c for c in pos if c in ranked[:args.top_n + args.keep_rank_buffer]]
                target_cols = keep + [c for c in ranked[:args.top_n] if c not in keep]
                target_cols = target_cols[:args.top_n]

            # sell everything not in target
            for c in list(pos):
                if c not in target_cols:
                    sh, _ = pos[c]
                    px_ = row_px[c]
                    if not np.isnan(px_):
                        val = sh * px_
                        cash += val * (1 - args.slippage_bps / 1e4) - max(COMMISSION_MIN, sh * COMMISSION_PER_SHARE)
                        turnover_notional += val
                        trades.append((str(d.date()), tk_cols[c], 'SELL_REBAL', px_))
                    del pos[c]

            # buy/resize to equal weight
            if target_cols:
                per = equity / len(target_cols)
                for c in target_cols:
                    px_ = row_px[c]
                    if np.isnan(px_):
                        continue
                    sh_cur = pos.get(c, (0.0, 0.0))[0]
                    sh_new = int(per / px_)
                    dsh = sh_new - sh_cur
                    if dsh > 0:
                        cost = dsh * px_ * (1 + args.slippage_bps / 1e4) + max(COMMISSION_MIN, dsh * COMMISSION_PER_SHARE)
                        if cost <= cash:
                            cash -= cost
                            pos[c] = (sh_new, px_ if sh_cur == 0 else pos[c][1])
                            turnover_notional += dsh * px_
                            trades.append((str(d.date()), tk_cols[c], 'BUY', px_))
                    elif dsh < 0:
                        val = -dsh * px_
                        cash += val * (1 - args.slippage_bps / 1e4) - max(COMMISSION_MIN, -dsh * COMMISSION_PER_SHARE)
                        turnover_notional += val
                        pos[c] = (sh_new, pos[c][1])
                        trades.append((str(d.date()), tk_cols[c], 'SELL_TRIM', px_))

    eq = pd.Series(dict(equity_curve)).sort_index()
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    turnover_ann = turnover_notional / eq.mean() / years

    tr = pd.DataFrame(trades, columns=['date', 'ticker', 'action', 'price'])
    tr.to_csv(REPORTS / f'phase_c_trades_{args.start}_{args.end}.csv', index=False)
    eq.to_csv(REPORTS / f'phase_c_equity_{args.start}_{args.end}.csv')

    return {'start': str(eq.index[0].date()), 'end': str(eq.index[-1].date()),
            'years': round(years, 2), 'final_value': round(float(eq.iloc[-1]), 2),
            'total_return': round(float(total_ret), 4), 'cagr': round(float(cagr), 4),
            'max_drawdown': round(float(-dd), 4), 'sharpe': round(float(sharpe), 3),
            'n_trades': len(tr), 'n_rebalances': n_reb,
            'ann_turnover_x': round(float(turnover_ann), 1)}


def meta_tickers(tickers):
    return set(tickers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--cash', type=float, default=100_000.0)
    ap.add_argument('--top-n', type=int, default=20)
    ap.add_argument('--rebalance-days', type=int, default=TRADING_DAYS_PER_MONTH,
                    help='21=monthly, 5=weekly')
    ap.add_argument('--min-conf', type=float, default=0.0,
                    help='minimum ensemble LONG proba to be eligible (e.g. 0.55)')
    ap.add_argument('--keep-rank-buffer', type=int, default=0,
                    help='hysteresis: keep holdings ranked within top N+buffer')
    ap.add_argument('--stop-pct', type=float, default=None,
                    help='optional daily hard stop, e.g. 0.10')
    ap.add_argument('--regime', choices=['none', 'sma'], default='sma')
    ap.add_argument('--slippage-bps', type=float, default=15.0)
    args = ap.parse_args()

    X, dates, tix, meta = load_cache()
    tickers = meta['tickers']
    logger.info(f'[INIT] cache {X.shape} | {len(tickers)} tickers | models={args.models_dir}')

    models, scaler, weights = load_ensemble(args.models_dir)
    scores = score_window(X, dates, tix, meta, args.start, args.end, models, scaler, weights,
                          tag=Path(args.models_dir).name)

    # map cache ticker_idx -> ticker name so scores align to price columns
    scores['ticker'] = [tickers[i] for i in scores['tix'].values]

    close, vol = load_price_matrix(tickers, args.start)
    # score pivot on ticker names
    scores_named = scores.copy()
    scores_named['tix'] = scores_named['ticker'].map({t: i for i, t in enumerate(close.columns)})
    scores_named = scores_named.dropna(subset=['tix'])
    scores_named['tix'] = scores_named['tix'].astype(int)

    res = simulate(scores_named, close, vol, list(close.columns), args)

    tag = f"{args.start}_{args.end}_{Path(args.models_dir).name}"
    (REPORTS / f'phase_c_metrics_{tag}.json').write_text(json.dumps({'args': vars(args), 'result': res}, indent=2))
    bar = '=' * 78
    print(f'\n{bar}\n  PHASE C - LOW-TURNOVER OOS RESULT ({Path(args.models_dir).name})\n{bar}')
    print(f"  period          {res['start']} to {res['end']}  ({res['years']}y)")
    print(f"  final_value     ${res['final_value']:,.0f}")
    print(f"  total_return    {res['total_return']:+.2%}")
    print(f"  cagr            {res['cagr']:+.2%}")
    print(f"  max_drawdown    {res['max_drawdown']:.2%}")
    print(f"  sharpe          {res['sharpe']:+.3f}")
    print(f"  trades          {res['n_trades']}   rebalances {res['n_rebalances']}")
    print(f"  ann_turnover    {res['ann_turnover_x']}x")
    print(bar)


if __name__ == '__main__':
    main()
