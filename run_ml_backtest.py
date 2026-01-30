"""
Run ML-powered backtest using XGBoost predictions and comprehensive indicators.
This is the main entry point for Phase 5 strategy testing.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.core import BacktestEngine
from src.core.ml_predictor import MLPredictor, train_and_predict
from src.core.data_store import get_data_store

def run_ml_backtest():
    """Run backtest with ML predictions."""
    print("=" * 70)
    print("🤖 ML-POWERED BACKTEST (XGBoost + 60+ Indicators)")
    print("=" * 70)
    print("\nObjective: 25%+ ARR, <20% Max Drawdown, >50% Win Rate")
    
    # Get data store and tickers
    store = get_data_store()
    store.log_price_policy()
    
    # Get tickers (exclude blacklisted)
    engine = BacktestEngine()
    blacklist = set(engine.config.get('blacklist', []))
    all_tickers = [t for t in store.available_tickers if t not in blacklist]
    
    # Use subset for faster testing (can increase later)
    tickers = all_tickers[:50]  # First 50 non-blacklisted tickers
    
    print(f"\n📊 Universe: {len(tickers)} tickers (excluding {len(blacklist)} blacklisted)")
    
    # Train models and generate predictions
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING MODELS")
    print("=" * 70)
    
    predictor, predictions = train_and_predict(
        tickers=tickers,
        train_start='2010-01-01',
        train_end='2022-12-31',  # Train on 2010-2022
        predict_start='2020-01-01',
        predict_end='2024-12-31'  # Backtest on 2020-2024
    )
    
    print(f"\n📈 Model Summary:")
    summary = predictor.get_model_summary()
    print(f"   Models trained: {summary['total_models']}")
    print(f"   Features used: {summary['feature_count']}")
    print(f"   Predictions generated: {len(predictions)}")
    
    # Run backtest with predictions
    print("\n" + "=" * 70)
    print("PHASE 2: RUNNING BACKTEST WITH ML PREDICTIONS")
    print("=" * 70)
    
    # Only backtest tickers with predictions
    backtest_tickers = list(predictions.keys())
    
    result = engine.run(
        tickers=backtest_tickers,
        start_date='2020-01-01',
        end_date='2024-12-31',
        model_predictions=predictions
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Return: {result.total_return_pct:.2f}%")
    print(f"   CAGR: {result.cagr_pct:.2f}%")
    print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"   Win Rate: {result.win_rate_pct:.1f}%")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    print(f"   Total Trades: {result.total_trades}")
    
    print(f"\n📋 Validation (Target: 25% ARR, <20% DD, >50% Win):")
    print(f"   Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    if not result.passed:
        for reason in result.failure_reasons:
            print(f"   ⚠️ {reason}")
    
    # Write Excel output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'reports/ml_backtest_{timestamp}.xlsx'
    engine.write_excel(result, output_path)
    
    print(f"\n📁 Excel Output: {output_path}")
    
    # Top performers
    if result.stock_summary:
        print(f"\n🏆 Top 10 Performers:")
        sorted_tickers = sorted(
            result.stock_summary.items(),
            key=lambda x: x[1].get('total_return', 0),
            reverse=True
        )[:10]
        for ticker, metrics in sorted_tickers:
            print(f"   {ticker}: {metrics['trades']} trades, "
                  f"{metrics['win_rate']:.0f}% win, "
                  f"{metrics['total_return']:.1f}% return")
    
    print("\n" + "=" * 70)
    print("✅ ML Backtest Complete")
    print("=" * 70)
    
    return result, predictor

if __name__ == "__main__":
    result, predictor = run_ml_backtest()
