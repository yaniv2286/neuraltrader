#!/usr/bin/env python3
"""
NEURALTRADER - SIMULATION MODE (Paper Trading without IBKR)
============================================================

Runs the unified pipeline in forward-simulation mode using parquet data.
Identical risk laws, models and entry/exit logic as backtest, but:
  - Starts from a configurable recent date (default: last 252 trading days)
  - Prints a live daily P&L summary to stdout
  - Writes simulation_metrics.json + simulation_trades.csv to reports/
  - No IBKR connection required

This is the intermediate validation step between backtest and live IBKR paper.

Usage:
    python scripts/run_simulation.py                         # last 252 days
    python scripts/run_simulation.py --start 2024-01-01     # custom start
    python scripts/run_simulation.py --cash 250000          # custom capital
    python scripts/run_simulation.py --tickers AAPL MSFT    # ticker subset

Modes wired into main_orchestrator_ist.py:
    python main_orchestrator_ist.py --mode simulation
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_backtest import UnifiedBacktest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'simulation.log', mode='a', encoding='ascii'),
    ]
)
logger = logging.getLogger('simulation')

REPORTS_DIR = PROJECT_ROOT / 'reports'


def run_simulation(
    start_date: str = None,
    end_date: str = None,
    initial_cash: float = 100_000.0,
    ticker_filter: list = None,
    lookback_days: int = 252,
) -> dict:
    """
    Run forward simulation using the unified backtest engine.
    Defaults to last `lookback_days` trading days if no start_date given.
    Saves to reports/simulation_metrics.json and reports/simulation_trades.csv.
    """
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=int(lookback_days * 1.4))
        start_date = start_dt.strftime('%Y-%m-%d')

    logger.info('[SIM] ======================================================')
    logger.info('[SIM] NEURALTRADER SIMULATION MODE (no IBKR required)')
    logger.info(f'[SIM] Period : {start_date} -> {end_date or "today"}')
    logger.info(f'[SIM] Capital: ${initial_cash:,.0f}')
    logger.info('[SIM] Pipeline: EnsemblePredictor -> RiskMachine -> UnifiedBacktest')
    logger.info('[SIM] ======================================================')

    bt = UnifiedBacktest(
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        ticker_filter=ticker_filter,
    )
    bt.run()

    # Use report() which computes metrics and saves files, then rename to simulation_ prefix
    metrics = bt.report()

    REPORTS_DIR.mkdir(exist_ok=True)

    # Rename backtest outputs to simulation_ prefix
    for src, dst in [
        (REPORTS_DIR / 'backtest_metrics.json', REPORTS_DIR / 'simulation_metrics.json'),
        (REPORTS_DIR / 'equity_curve.csv',       REPORTS_DIR / 'simulation_equity.csv'),
        (REPORTS_DIR / 'trade_log.csv',           REPORTS_DIR / 'simulation_trades.csv'),
    ]:
        if src.exists():
            src.rename(dst)

    logger.info(f'[SIM] Results saved to {REPORTS_DIR} (simulation_* prefix)')
    return metrics


def main():
    parser = argparse.ArgumentParser(description='NeuralTrader Simulation Mode')
    parser.add_argument('--start',   default=None,      help='Start date YYYY-MM-DD (default: last 252 trading days)')
    parser.add_argument('--end',     default=None,      help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--cash',    default=100000.0,  type=float, help='Initial capital')
    parser.add_argument('--tickers', nargs='*',         default=None, help='Ticker subset')
    parser.add_argument('--lookback',default=252,       type=int, help='Lookback days if no start date given')
    args = parser.parse_args()

    run_simulation(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        ticker_filter=args.tickers,
        lookback_days=args.lookback,
    )


if __name__ == '__main__':
    main()
