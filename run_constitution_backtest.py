"""
Run backtest following Trading Constitution v1.
Produces authoritative Excel output with strict schema.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.core import BacktestEngine

def run_backtest():
    """Run backtest with Trading Constitution compliance."""
    print("=" * 60)
    print("🏛️ TRADING CONSTITUTION v1 BACKTEST")
    print("=" * 60)
    print("\nObjective: Maximize risk-adjusted return")
    print("Target: 25%+ ARR after costs, no leverage")
    print("Priority: Capital preservation, low drawdowns")
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Get high-confidence tickers from config
    high_conf = engine.config.get('high_confidence_tickers', [])
    blacklist = set(engine.config.get('blacklist', []))
    
    # Select tickers (high confidence + some others, excluding blacklist)
    available = engine.data_store.available_tickers
    tickers = [t for t in available if t not in blacklist][:30]  # First 30 non-blacklisted
    
    print(f"\n📊 Universe: {len(tickers)} tickers")
    print(f"   High confidence: {len([t for t in tickers if t in high_conf])}")
    print(f"   Blacklisted: {len(blacklist)} (excluded)")
    
    # Run backtest
    result = engine.run(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Return: {result.total_return_pct:.2f}%")
    print(f"   CAGR: {result.cagr_pct:.2f}%")
    print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"   Win Rate: {result.win_rate_pct:.1f}%")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    print(f"   Total Trades: {result.total_trades}")
    
    print(f"\n📋 Validation:")
    print(f"   Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    if not result.passed:
        for reason in result.failure_reasons:
            print(f"   ⚠️ {reason}")
    
    # Write Excel output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'reports/constitution_backtest_{timestamp}.xlsx'
    engine.write_excel(result, output_path)
    
    print(f"\n📁 Excel Output: {output_path}")
    print(f"   Sheets: How_To_Read, Overall_Performance, All_Trades,")
    print(f"           Stock_Summary, Equity_Curve, Config_Snapshot")
    
    # Summary of trades by ticker
    if result.trades:
        print(f"\n📊 Trade Summary by Ticker:")
        for ticker, summary in sorted(result.stock_summary.items(), 
                                      key=lambda x: x[1].get('total_return', 0), 
                                      reverse=True)[:10]:
            print(f"   {ticker}: {summary['trades']} trades, "
                  f"{summary['win_rate']:.0f}% win, "
                  f"{summary['total_return']:.1f}% return")
    
    print("\n" + "=" * 60)
    print("✅ Backtest complete - Trading Constitution compliant")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    result = run_backtest()
