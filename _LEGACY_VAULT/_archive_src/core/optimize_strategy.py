"""
Strategy Optimizer - Find optimal risk management parameters.
Tests multiple configurations to maximize CAGR while keeping DD under control.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.core.ai_ensemble_strategy import AIEnsembleStrategy
from src.core.data_store import get_data_store


def run_optimization():
    """Test multiple risk management configurations."""
    print("=" * 70)
    print("🔧 STRATEGY OPTIMIZATION")
    print("=" * 70)
    print("Testing different risk management configurations...")
    
    store = get_data_store()
    tickers = store.available_tickers
    
    strategy = AIEnsembleStrategy()
    
    # Train once (expensive)
    print("\n📊 Training AI Ensemble (one-time)...")
    train_results = strategy.train_ensemble(tickers=tickers, train_end='2022-12-31')
    
    # Generate signals once
    signals = strategy.generate_signals(tickers=tickers, start_date='2023-01-01', end_date='2024-12-31')
    
    if signals.empty:
        print("❌ No signals!")
        return
    
    # Configurations to test
    configs = [
        # (position_pct, stop_loss_pct, take_profit_pct, max_hold_days, top_pct, name)
        (0.25, 0.05, 0.12, 15, 0.50, "Current (Baseline)"),
        (0.30, 0.04, 0.10, 10, 0.50, "Aggressive Position"),
        (0.35, 0.05, 0.15, 15, 0.50, "Very Aggressive"),
        (0.40, 0.06, 0.20, 20, 0.50, "Max Aggressive"),
        (0.25, 0.03, 0.08, 10, 0.30, "Tight Stops"),
        (0.30, 0.05, 0.15, 15, 0.30, "Selective + Aggressive"),
        (0.35, 0.04, 0.12, 12, 0.40, "Balanced Aggressive"),
        (0.20, 0.05, 0.10, 10, 0.70, "More Trades"),
        (0.30, 0.06, 0.18, 20, 0.40, "Wide Stops + Long Hold"),
        (0.40, 0.05, 0.15, 15, 0.30, "Max Position + Selective"),
    ]
    
    results = []
    
    print(f"\n🧪 Testing {len(configs)} configurations...\n")
    
    for pos_pct, sl_pct, tp_pct, hold_days, top_pct, name in configs:
        # Filter signals
        filtered = strategy.filter_top_signals(signals, max_per_day=5, top_pct=top_pct)
        
        if filtered.empty:
            continue
        
        # Run backtest
        result = strategy.backtest(
            filtered,
            initial_capital=100000,
            position_pct=pos_pct,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            max_hold_days=hold_days
        )
        
        if 'error' in result:
            continue
        
        results.append({
            'name': name,
            'position_pct': pos_pct * 100,
            'stop_loss_pct': sl_pct * 100,
            'take_profit_pct': tp_pct * 100,
            'max_hold_days': hold_days,
            'top_pct': top_pct * 100,
            'cagr': result['cagr_pct'],
            'max_dd': result['max_drawdown_pct'],
            'win_rate': result['win_rate_pct'],
            'profit_factor': result['profit_factor'],
            'trades': result['total_trades'],
            'total_return': result['total_return_pct']
        })
        
        status = "✅" if result['cagr_pct'] >= 20 and result['max_drawdown_pct'] >= -20 else "❌"
        print(f"{status} {name}: CAGR={result['cagr_pct']:.1f}%, DD={result['max_drawdown_pct']:.1f}%, WR={result['win_rate_pct']:.1f}%")
    
    # Sort by CAGR
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cagr', ascending=False)
    
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION RESULTS (sorted by CAGR)")
    print("=" * 70)
    
    print(f"\n{'Rank':<5} {'Config':<25} {'CAGR':<10} {'Max DD':<10} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 70)
    
    for i, row in results_df.head(10).iterrows():
        valid = row['cagr'] >= 20 and row['max_dd'] >= -20
        marker = "🏆" if i == results_df.index[0] else ("✅" if valid else "⚠️")
        print(f"{marker:<5} {row['name']:<25} {row['cagr']:>7.1f}%  {row['max_dd']:>7.1f}%  {row['win_rate']:>7.1f}%  {row['trades']:>6}")
    
    # Best configuration
    best = results_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
    print(f"   Position Size: {best['position_pct']:.0f}%")
    print(f"   Stop Loss: {best['stop_loss_pct']:.0f}%")
    print(f"   Take Profit: {best['take_profit_pct']:.0f}%")
    print(f"   Max Hold Days: {best['max_hold_days']:.0f}")
    print(f"   Signal Filter: Top {best['top_pct']:.0f}%")
    print(f"\n   Results:")
    print(f"   CAGR: {best['cagr']:.2f}%")
    print(f"   Max Drawdown: {best['max_dd']:.2f}%")
    print(f"   Win Rate: {best['win_rate']:.1f}%")
    print(f"   Profit Factor: {best['profit_factor']:.2f}")
    print(f"   Total Trades: {best['trades']:.0f}")
    
    return results_df


if __name__ == "__main__":
    results = run_optimization()
