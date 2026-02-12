#!/usr/bin/env python3

import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_strategy import StrategyOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UniversalAlphaScorecard')

def main():
    """Generate Universal Alpha Scorecard"""
    
    logger.info("=" * 80)
    logger.info("UNIVERSAL ALPHA SCORECARD")
    logger.info("=" * 80)
    
    # Initialize optimizer
    optimizer = StrategyOptimizer()
    
    # Load data and generate AI signals
    logger.info("Loading market data and generating AI signals...")
    market_data = optimizer.load_data()
    ai_signals = optimizer.generate_ai_signals(market_data)
    
    # Load regime data
    try:
        import requests
        tiingo_key = os.getenv('TIINGO_API_KEY')
        if tiingo_key:
            # Simple regime filter (using SPY as proxy)
            spy_data = optimizer._fetch_ticker_data('SPY', '1970-01-01', '2026-01-30')
            if not spy_data.empty:
                regime_data = spy_data['close']
            else:
                regime_data = None
        else:
            regime_data = None
    except:
        regime_data = None
    
    # Test parameters
    test_params = [
        {'AI_THRESHOLD': 0.35, 'VIX_FILTER': 30, 'STOP_LOSS': 0.08},
        {'AI_THRESHOLD': 0.38, 'VIX_FILTER': 35, 'STOP_LOSS': 0.10},
        {'AI_THRESHOLD': 0.41, 'VIX_FILTER': 40, 'STOP_LOSS': 0.12},
    ]
    
    results = []
    
    for i, params in enumerate(test_params, 1):
        logger.info(f"[{i}/{len(test_params)}] Testing: AI={params['AI_THRESHOLD']}, VIX={params['VIX_FILTER']}, SL={params['STOP_LOSS']}")
        
        # Run backtest
        result = optimizer.run_backtest(params, market_data, ai_signals, regime_data)
        
        results.append(result)
        
        # Display results
        logger.info(f"  📊 CAGR: {result['cagr_pct']:.2f}%")
        logger.info(f"  📉 Max DD: {result['max_drawdown_pct']:.2f}%")
        logger.info(f"  📈 Sharpe: {result['sharpe_ratio']:.3f}")
        logger.info(f"  💼 Trades: {result['total_trades']:,}")
        logger.info(f"  ⏱️  Years: {result['years']:.1f}")
        logger.info("")
    
    # Create summary table
    logger.info("=" * 80)
    logger.info("UNIVERSAL ALPHA SCORECARD SUMMARY")
    logger.info("=" * 80)
    
    print(f"{'Config':<8} {'AI_TH':<7} {'VIX':<5} {'SL':<6} {'CAGR':<8} {'Max DD':<9} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        params = result['params']
        print(f"{i:<8} {params['AI_THRESHOLD']:<7} {params['VIX_FILTER']:<5} {params['STOP_LOSS']:<6} "
              f"{result['cagr_pct']:<8.2f} {result['max_drawdown_pct']:<9.2f} "
              f"{result['sharpe_ratio']:<8.3f} {result['total_trades']:<8}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['cagr_pct'])
    logger.info("")
    logger.info("🏆 BEST CONFIGURATION:")
    logger.info(f"   AI Threshold: {best_result['params']['AI_THRESHOLD']}")
    logger.info(f"   VIX Filter: {best_result['params']['VIX_FILTER']}")
    logger.info(f"   Stop Loss: {best_result['params']['STOP_LOSS']}")
    logger.info(f"   CAGR: {best_result['cagr_pct']:.2f}%")
    logger.info(f"   Max DD: {best_result['max_drawdown_pct']:.2f}%")
    logger.info(f"   Sharpe: {best_result['sharpe_ratio']:.3f}")
    logger.info(f"   Trades: {best_result['total_trades']:,}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_file = PROJECT_ROOT / 'reports' / f'universal_alpha_scorecard_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file}")
    
    logger.info("=" * 80)
    logger.info("UNIVERSAL ALPHA SCORECARD COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    import os
    main()
