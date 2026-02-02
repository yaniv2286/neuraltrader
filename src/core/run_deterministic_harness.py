"""
Deterministic Harness Runner - Phase 4
=======================================

Main entry point for running the deterministic backtest harness.

Execution Order (DO NOT REORDER):
1. Load SPY benchmark features (fix pipeline first)
2. Validate feature manifest (no silent skip)
3. Run strategies in parallel via registry
4. Generate 6-sheet Audit Excel

Compares at minimum:
- VT_SweetSpot_v1 (baseline - IMMUTABLE)
- VT_SweetSpot_v1_experimental (variant)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.data_store import get_data_store
from src.core.spy_benchmark import get_spy_benchmark_loader, reset_spy_benchmark_loader
from src.core.strategy_registry import (
    get_strategy_registry, 
    reset_strategy_registry,
    register_experimental_variants
)
from src.core.vt_sweetspot_baseline import (
    VTSweetSpotBaseline,
    create_baseline_strategy,
    create_variant_strategy
)
from src.core.deterministic_backtest_engine import (
    DeterministicBacktestEngine,
    BacktestResult,
    CostModel
)
from src.core.audit_excel_writer import write_audit_excel


class DeterministicHarnessError(Exception):
    """Raised when harness encounters a critical error."""
    pass


def run_deterministic_harness(
    start_date: str = '2015-01-01',
    end_date: str = '2024-12-31',
    initial_capital: float = 100000,
    output_dir: str = 'reports'
) -> Dict[str, Any]:
    """
    Run the complete deterministic backtest harness.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        output_dir: Directory for output files
        
    Returns:
        Dictionary with run results and file paths
    """
    print("\n" + "=" * 70)
    print("🏛️ DETERMINISTIC HARNESS - PHASE 4")
    print("=" * 70)
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Initial Capital: ${initial_capital:,.0f}")
    print("=" * 70)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Reset global instances for clean run
    reset_spy_benchmark_loader()
    reset_strategy_registry()
    
    # ========================================
    # STEP 1: Load SPY Benchmark Features
    # ========================================
    print("\n" + "=" * 70)
    print("📊 STEP 1: Loading SPY Benchmark Features")
    print("=" * 70)
    
    spy_loader = get_spy_benchmark_loader()
    spy_features = spy_loader.load_spy_features(start_date, end_date)
    print(f"   ✅ SPY features loaded: {len(spy_features)} days")
    print(f"   ✅ Feature hash: {spy_loader.get_feature_hash()}")
    
    # ========================================
    # STEP 2: Register Strategies
    # ========================================
    print("\n" + "=" * 70)
    print("📊 STEP 2: Registering Strategies")
    print("=" * 70)
    
    registry = get_strategy_registry()
    
    # Baseline is auto-registered
    baseline_config = registry.get_baseline()
    print(f"   ✅ Baseline: {baseline_config.strategy_id}")
    
    # Register experimental variants
    register_experimental_variants(registry)
    
    all_strategies = registry.get_strategy_ids()
    print(f"   ✅ Total strategies: {len(all_strategies)}")
    for sid in all_strategies:
        config = registry.get_strategy(sid)
        print(f"      - {sid} {'(BASELINE)' if config.is_baseline else ''}")
    
    # Validate baseline unchanged
    registry.validate_baseline_unchanged()
    print("   ✅ Baseline validation: PASSED")
    
    # ========================================
    # STEP 3: Load Ticker Data
    # ========================================
    print("\n" + "=" * 70)
    print("📊 STEP 3: Loading Ticker Data")
    print("=" * 70)
    
    data_store = get_data_store()
    tickers = data_store.available_tickers
    print(f"   📋 Universe: {len(tickers)} tickers")
    
    # Pre-load all ticker data
    ticker_data = {}
    loaded = 0
    failed = 0
    
    for ticker in tickers:
        try:
            df = data_store.get_ticker_data(ticker, start_date, end_date)
            if df is not None and len(df) >= 252:  # At least 1 year
                ticker_data[ticker] = df
                loaded += 1
        except Exception as e:
            failed += 1
    
    print(f"   ✅ Loaded: {loaded} tickers")
    print(f"   ❌ Failed: {failed} tickers")
    
    if loaded == 0:
        raise DeterministicHarnessError("ABORT: No tickers loaded successfully")
    
    # ========================================
    # STEP 4: Run Backtests
    # ========================================
    print("\n" + "=" * 70)
    print("📊 STEP 4: Running Backtests")
    print("=" * 70)
    
    cost_model = CostModel()
    engine = DeterministicBacktestEngine(
        initial_capital=initial_capital,
        cost_model=cost_model
    )
    
    results: List[BacktestResult] = []
    
    # Run baseline first
    print("\n--- Running Baseline Strategy ---")
    baseline_strategy = create_baseline_strategy()
    baseline_signals, baseline_meta = baseline_strategy.generate_signals(
        tickers=list(ticker_data.keys()),
        start_date=start_date,
        end_date=end_date
    )
    
    if baseline_signals.empty:
        print("   ⚠️ Baseline generated 0 signals")
    else:
        baseline_result = engine.run_backtest(
            strategy=baseline_strategy,
            signals_df=baseline_signals,
            ticker_data=ticker_data,
            start_date=start_date,
            end_date=end_date
        )
        baseline_result.ticker_summary = baseline_meta.get('ticker_summary', {})
        results.append(baseline_result)
        
        print(f"   ✅ Baseline CAGR: {baseline_result.cagr_pct:.2f}%")
        print(f"   ✅ Baseline Max DD: {baseline_result.max_drawdown_pct:.2f}%")
    
    # Run variants
    for strategy_id in all_strategies:
        if strategy_id == 'VT_SweetSpot_v1':
            continue  # Already ran baseline
        
        print(f"\n--- Running Variant: {strategy_id} ---")
        
        try:
            variant_strategy = create_variant_strategy(strategy_id)
            variant_signals, variant_meta = variant_strategy.generate_signals(
                tickers=list(ticker_data.keys()),
                start_date=start_date,
                end_date=end_date
            )
            
            if variant_signals.empty:
                print(f"   ⚠️ {strategy_id} generated 0 signals")
                continue
            
            variant_result = engine.run_backtest(
                strategy=variant_strategy,
                signals_df=variant_signals,
                ticker_data=ticker_data,
                start_date=start_date,
                end_date=end_date
            )
            variant_result.ticker_summary = variant_meta.get('ticker_summary', {})
            results.append(variant_result)
            
            print(f"   ✅ {strategy_id} CAGR: {variant_result.cagr_pct:.2f}%")
            print(f"   ✅ {strategy_id} Max DD: {variant_result.max_drawdown_pct:.2f}%")
            
        except Exception as e:
            print(f"   ❌ {strategy_id} failed: {e}")
    
    # ========================================
    # STEP 5: Generate Audit Excel
    # ========================================
    print("\n" + "=" * 70)
    print("📊 STEP 5: Generating Audit Excel")
    print("=" * 70)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'audit_report_{run_id}.xlsx')
    
    run_metadata = {
        'run_id': run_id,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'tickers_total': len(tickers),
        'tickers_loaded': loaded
    }
    
    if results:
        excel_path = write_audit_excel(
            results=results,
            run_metadata=run_metadata,
            cost_model=cost_model,
            output_path=output_path
        )
    else:
        print("   ⚠️ No results to write - all strategies failed")
        excel_path = None
    
    # ========================================
    # STEP 6: Summary
    # ========================================
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if results:
        print("\n   Strategy Comparison:")
        print("   " + "-" * 60)
        print(f"   {'Strategy':<35} {'CAGR':>10} {'Max DD':>10} {'Win Rate':>10}")
        print("   " + "-" * 60)
        
        baseline_cagr = None
        baseline_dd = None
        
        for result in results:
            marker = " (BASELINE)" if result.config.is_baseline else ""
            print(f"   {result.strategy_id:<35} {result.cagr_pct:>9.2f}% {result.max_drawdown_pct:>9.2f}% {result.win_rate_pct:>9.1f}%")
            
            if result.config.is_baseline:
                baseline_cagr = result.cagr_pct
                baseline_dd = result.max_drawdown_pct
        
        print("   " + "-" * 60)
        
        # Acceptance check
        print("\n   Acceptance Check (vs Baseline):")
        for result in results:
            if result.config.is_baseline:
                continue
            
            accepted = True
            reasons = []
            
            if baseline_cagr is not None and result.cagr_pct < baseline_cagr:
                accepted = False
                reasons.append(f"CAGR {result.cagr_pct:.2f}% < baseline {baseline_cagr:.2f}%")
            
            if baseline_dd is not None and result.max_drawdown_pct < baseline_dd:
                accepted = False
                reasons.append(f"Max DD {result.max_drawdown_pct:.2f}% < baseline {baseline_dd:.2f}%")
            
            status = "✅ ACCEPTED" if accepted else "❌ REJECTED"
            print(f"   {result.strategy_id}: {status}")
            if reasons:
                for r in reasons:
                    print(f"      - {r}")
    
    print("\n" + "=" * 70)
    print(f"📁 Output: {excel_path or 'No output generated'}")
    print("=" * 70)
    
    return {
        'run_id': run_id,
        'results': results,
        'excel_path': excel_path,
        'metadata': run_metadata
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Deterministic Backtest Harness')
    parser.add_argument('--start', default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--output', default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    result = run_deterministic_harness(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        output_dir=args.output
    )
    
    return result


if __name__ == "__main__":
    main()
