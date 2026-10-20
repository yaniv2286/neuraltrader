#!/usr/bin/env python3
"""
NeuralTrader - Benchmark Strategies
====================================

The bar the AI must clear OUT-OF-SAMPLE before it deserves capital:

  1. SPY buy-and-hold
  2. Cross-sectional 12-1 momentum: top-N by 12-month return skipping the most
     recent month, equal-weight, rebalanced monthly, same universe hygiene as
     the live system (no OTC, >= $10, >= $5M ADV, must have a bar on rebalance day)

Costs: slippage per side (bps) + IBKR tiered commission on the momentum sleeve.

Usage:
    python scripts/run_benchmarks.py --start 2018-01-01 --end 2020-12-31
    python scripts/run_benchmarks.py --start 2018-01-01 --end 2026-10-16 --top 20 --slippage-bps 15
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.universe_filter import is_otc_ticker, MIN_PRICE, MIN_DOLLAR_VOLUME  # noqa: E402

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
REPORTS = PROJECT_ROOT / 'reports'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Benchmarks')

COMMISSION_PER_SHARE = 0.0035
COMMISSION_MIN = 0.35


def load_closes(start: str, end: str):
    """Wide adjClose + dollar-volume frames for the whole universe (plus 13 months of warm-up)."""
    warm = pd.Timestamp(start) - pd.DateOffset(months=14)
    closes, dvols = {}, {}
    files = sorted(RAW_DIR.glob('*.parquet'))
    for f in files:
        t = f.stem.upper()
        if is_otc_ticker(t):
            continue
        try:
            df = pd.read_parquet(f, columns=['date', 'adjClose', 'adjVolume'])
        except Exception:
            try:
                df = pd.read_parquet(f, columns=['date', 'close', 'volume']).rename(
                    columns={'close': 'adjClose', 'volume': 'adjVolume'})
            except Exception:
                continue
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= warm) & (df['date'] <= end)].set_index('date')
        if len(df) < 260:
            continue
        closes[t] = df['adjClose']
        dvols[t] = df['adjClose'] * df['adjVolume']
    closes = pd.DataFrame(closes).sort_index()
    dvols = pd.DataFrame(dvols).sort_index()
    logger.info(f'[DATA] {closes.shape[1]} tickers x {closes.shape[0]} days loaded')
    return closes, dvols


def metrics(equity: pd.Series, label: str) -> dict:
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    return {'strategy': label, 'start': str(eq.index[0].date()), 'end': str(eq.index[-1].date()),
            'years': round(years, 2), 'total_return': f'{total:+.2%}', 'cagr': f'{cagr:+.2%}',
            'max_drawdown': f'{dd:.2%}', 'sharpe': round(float(sharpe), 2)}


def spy_buy_and_hold(closes: pd.DataFrame, start: str, end: str, cash: float):
    spy = closes['SPY'].loc[start:end].dropna()
    return cash * spy / spy.iloc[0]


def momentum_12_1(closes: pd.DataFrame, dvols: pd.DataFrame, start: str, end: str,
                  cash: float, top_n: int, slippage: float):
    """Equal-weight top-N 12-1 momentum, rebalanced on the first trading day of each month."""
    cal = closes.loc[start:end].index
    month_starts = cal[cal.to_series().groupby([cal.year, cal.month]).cumcount() == 0]
    equity = pd.Series(index=cal, dtype=float)
    holdings = {}   # ticker -> shares
    cash_bal = cash
    turnover_trades = 0
    for i, date in enumerate(cal):
        px = closes.loc[date]
        if date in month_starts:
            hist = closes.loc[:date]
            if len(hist) < 260:
                equity[date] = cash_bal + sum(s * px[t] for t, s in holdings.items() if not np.isnan(px[t]))
                continue
            # 12-1 momentum: return from t-252 to t-21
            p_252 = hist.iloc[-252]
            p_21 = hist.iloc[-21]
            mom = (p_21 / p_252 - 1)
            adv = dvols.loc[:date].iloc[-20:].mean()
            has_bar = px.notna()
            eligible = mom[(has_bar) & (px >= MIN_PRICE) & (adv >= MIN_DOLLAR_VOLUME) & mom.notna()]
            picks = eligible.sort_values(ascending=False).head(top_n).index.tolist()
            # liquidate everything not in picks
            for t in list(holdings):
                if t not in picks and not np.isnan(px[t]):
                    s = holdings.pop(t)
                    cash_bal += s * px[t] * (1 - slippage) - max(COMMISSION_MIN, COMMISSION_PER_SHARE * s)
                    turnover_trades += 1
            # delisted (no price) names: assume cash at last known price
            for t in list(holdings):
                if np.isnan(px[t]):
                    last = closes[t].loc[:date].dropna()
                    s = holdings.pop(t)
                    cash_bal += s * (last.iloc[-1] if len(last) else 0)
            # equal-weight rebalance into picks
            port_val = cash_bal + sum(s * px[t] for t, s in holdings.items())
            target = port_val / max(len(picks), 1)
            for t in picks:
                cur = holdings.get(t, 0) * px[t]
                diff = target - cur
                shares = int(diff / px[t])
                if shares > 0:
                    cost = shares * px[t] * (1 + slippage) + max(COMMISSION_MIN, COMMISSION_PER_SHARE * shares)
                    if cost <= cash_bal:
                        holdings[t] = holdings.get(t, 0) + shares
                        cash_bal -= cost
                        turnover_trades += 1
                elif shares < 0:
                    s = -shares
                    holdings[t] -= s
                    cash_bal += s * px[t] * (1 - slippage) - max(COMMISSION_MIN, COMMISSION_PER_SHARE * s)
                    turnover_trades += 1
        val = cash_bal
        for t, s in holdings.items():
            p = px[t]
            if np.isnan(p):
                last = closes[t].loc[:date].dropna()
                p = last.iloc[-1] if len(last) else 0
            val += s * p
        equity[date] = val
    logger.info(f'[MOM] {turnover_trades} trades over {len(month_starts)} rebalances')
    return equity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--cash', type=float, default=100_000)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--slippage-bps', type=float, default=15.0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    closes, dvols = load_closes(a.start, a.end)
    results = []
    spy_eq = spy_buy_and_hold(closes, a.start, a.end, a.cash)
    results.append(metrics(spy_eq, 'SPY buy-and-hold'))
    mom_eq = momentum_12_1(closes, dvols, a.start, a.end, a.cash, a.top, a.slippage_bps / 1e4)
    results.append(metrics(mom_eq, f'12-1 momentum top-{a.top} monthly'))

    bar = '=' * 78
    print(f'\n{bar}\n  BENCHMARKS  {a.start} -> {a.end}  (slippage {a.slippage_bps:.0f} bps/side)\n{bar}')
    print(f"  {'strategy':34} {'total':>9} {'cagr':>8} {'maxDD':>8} {'sharpe':>7}")
    for r in results:
        print(f"  {r['strategy']:34} {r['total_return']:>9} {r['cagr']:>8} {r['max_drawdown']:>8} {r['sharpe']:>7}")
    print(bar)

    REPORTS.mkdir(exist_ok=True)
    tag = f'_{a.tag}' if a.tag else ''
    out = REPORTS / f'benchmarks_{a.start}_{a.end}{tag}.json'
    out.write_text(json.dumps(results, indent=2))
    pd.DataFrame({'spy': spy_eq, 'momentum': mom_eq}).to_csv(REPORTS / f'benchmark_equity_{a.start}_{a.end}{tag}.csv')
    logger.info(f'[REPORT] {out}')


if __name__ == '__main__':
    main()
