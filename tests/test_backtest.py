#!/usr/bin/env python3

from scripts.optimize_strategy import StrategyOptimizer
import pandas as pd

# Test the backtest method directly
optimizer = StrategyOptimizer()
params = {'AI_THRESHOLD': 0.35, 'VIX_FILTER': 30, 'STOP_LOSS': 0.08}

# Load a small sample of data
market_data = optimizer.load_data()
ai_signals = optimizer.generate_ai_signals(market_data)

# Test with just one ticker
sample_market_data = {k: v for k, v in list(market_data.items())[:5]}
sample_ai_signals = {k: v for k, v in list(ai_signals.items())[:5]}

print(f'Testing with {len(sample_market_data)} tickers')
result = optimizer.run_backtest(params, sample_market_data, sample_ai_signals, None)
print(f'Result type: {type(result)}')
print(f'Result keys: {result.keys() if isinstance(result, dict) else "Not a dict"}')

if isinstance(result, dict):
    print(f'CAGR: {result.get("cagr_pct", "N/A")}%')
    print(f'Max DD: {result.get("max_drawdown_pct", "N/A")}%')
    print(f'Sharpe: {result.get("sharpe_ratio", "N/A")}')
    print(f'Trades: {result.get("total_trades", "N/A")}')
else:
    print(f'Result: {result}')
