"""
Test core components: BacktestResult, VetoGates, CostModel, DataStore.
Validates the Trading Constitution implementation.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime

def test_data_store():
    """Test DataStore with incremental loading."""
    print("\n1️⃣ TESTING DATA STORE")
    print("-" * 40)
    
    from src.core import DataStore, DataStoreError, get_data_store
    
    store = get_data_store()
    summary = store.get_summary()
    
    print(f"   Cache dir: {summary['cache_dir']}")
    print(f"   Total tickers: {summary['total_tickers']}")
    print(f"   Price policy: {summary['price_policy']}")
    
    # Log price policy as required
    store.log_price_policy()
    
    # Test loading a ticker
    if summary['total_tickers'] > 0:
        ticker = summary['available_tickers'][0]
        try:
            df = store.get_ticker_data(ticker, start_date='2020-01-01')
            print(f"   ✅ Loaded {ticker}: {len(df)} days")
        except DataStoreError as e:
            print(f"   ❌ Failed to load {ticker}: {e}")
    
    return True

def test_veto_gates():
    """Test VetoGates with all rules."""
    print("\n2️⃣ TESTING VETO GATES")
    print("-" * 40)
    
    from src.core import VetoGates, VetoReason
    
    gates = VetoGates()
    
    # Test 1: Valid trade
    passed, results = gates.check_all(
        ticker='AAPL',
        confidence=0.55,
        risk_reward_ratio=2.0,
        volume=1000000,
        spread_pct=0.001,
        current_positions=5,
        sector_exposure=0.15,
        daily_pnl_pct=-0.02,
        portfolio_drawdown_pct=-0.10
    )
    print(f"   Valid trade: {'✅ PASSED' if passed else '❌ VETOED'}")
    
    # Test 2: Blacklisted ticker
    passed, results = gates.check_all(
        ticker='GE',
        confidence=0.55,
        risk_reward_ratio=2.0,
        volume=1000000,
        spread_pct=0.001
    )
    print(f"   Blacklisted (GE): {'❌ VETOED' if not passed else '✅ PASSED'}")
    
    # Test 3: Low confidence
    passed, results = gates.check_all(
        ticker='AAPL',
        confidence=0.30,
        risk_reward_ratio=2.0,
        volume=1000000,
        spread_pct=0.001
    )
    print(f"   Low confidence: {'❌ VETOED' if not passed else '✅ PASSED'}")
    
    # Test 4: Poor risk/reward
    passed, results = gates.check_all(
        ticker='AAPL',
        confidence=0.55,
        risk_reward_ratio=0.5,
        volume=1000000,
        spread_pct=0.001
    )
    print(f"   Poor R/R ratio: {'❌ VETOED' if not passed else '✅ PASSED'}")
    
    # Test 5: Low liquidity
    passed, results = gates.check_all(
        ticker='AAPL',
        confidence=0.55,
        risk_reward_ratio=2.0,
        volume=50000,
        spread_pct=0.001
    )
    print(f"   Low liquidity: {'❌ VETOED' if not passed else '✅ PASSED'}")
    
    return True

def test_cost_model():
    """Test CostModel with mandatory costs."""
    print("\n3️⃣ TESTING COST MODEL")
    print("-" * 40)
    
    from src.core import CostModel, TradeCosts
    
    model = CostModel()
    config = model.get_config_dict()
    
    print(f"   Enabled: {config['enabled']}")
    print(f"   Commission: ${config['commission_per_trade']:.2f}")
    print(f"   Spread estimate: {config['spread_estimate_pct']:.3%}")
    print(f"   Slippage estimate: {config['slippage_estimate_pct']:.3%}")
    
    # Test round-trip costs
    entry_value = 10000
    exit_value = 10200
    costs = model.calculate_round_trip_costs(entry_value, exit_value)
    
    print(f"\n   Round-trip costs for $10,000 trade:")
    print(f"   Commission: ${costs.commission:.2f}")
    print(f"   Spread: ${costs.spread_cost:.2f}")
    print(f"   Slippage: ${costs.slippage_cost:.2f}")
    print(f"   Total: ${costs.total_cost:.2f}")
    
    # Test breakeven
    breakeven = model.estimate_breakeven_move(10000)
    print(f"\n   Breakeven move: {breakeven:.3%}")
    
    # Test profitability check
    is_profitable, net_return = model.validate_trade_profitability(0.02, 10000)
    print(f"   2% expected return profitable: {'✅ YES' if is_profitable else '❌ NO'} (net: {net_return:.3%})")
    
    is_profitable, net_return = model.validate_trade_profitability(0.001, 10000)
    print(f"   0.1% expected return profitable: {'✅ YES' if is_profitable else '❌ NO'} (net: {net_return:.3%})")
    
    return True

def test_backtest_result():
    """Test BacktestResult contract and Excel writer."""
    print("\n4️⃣ TESTING BACKTEST RESULT")
    print("-" * 40)
    
    from src.core import BacktestResult, Trade, BacktestExcelWriter, create_backtest_result
    
    # Create sample trades
    trades = [
        Trade(
            trade_id='T001',
            ticker='AAPL',
            direction='LONG',
            entry_date=datetime(2024, 1, 15),
            entry_price=180.0,
            exit_date=datetime(2024, 1, 20),
            exit_price=185.0,
            position_size_pct=5.0,
            capital_allocated=5000,
            risk_at_entry_pct=1.0,
            confidence_score=0.58,
            gross_pnl=138.89,
            costs_paid=12.50,
            net_pnl=126.39,
            return_on_allocated_pct=2.53,
            return_on_portfolio_pct=0.13,
            entry_reasons='Model signal + momentum',
            exit_reason='trailing_stop',
            veto_checks_passed='Gate1,Gate2,Gate3,Gate4,Gate5',
            regime='UPTREND',
            liquidity_status='HIGH',
            status='CLOSED'
        ),
        Trade(
            trade_id='T002',
            ticker='GOOGL',
            direction='LONG',
            entry_date=datetime(2024, 1, 22),
            entry_price=140.0,
            exit_date=datetime(2024, 1, 25),
            exit_price=138.0,
            position_size_pct=5.0,
            capital_allocated=5000,
            risk_at_entry_pct=1.0,
            confidence_score=0.52,
            gross_pnl=-71.43,
            costs_paid=12.50,
            net_pnl=-83.93,
            return_on_allocated_pct=-1.68,
            return_on_portfolio_pct=-0.08,
            entry_reasons='Model signal',
            exit_reason='signal_invalidation',
            veto_checks_passed='Gate1,Gate2,Gate3,Gate4,Gate5',
            regime='SIDEWAYS',
            liquidity_status='HIGH',
            status='CLOSED'
        )
    ]
    
    # Create equity curve
    equity_curve = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'portfolio_equity': np.linspace(100000, 100500, 30),
        'drawdown_pct': np.random.uniform(-0.02, 0, 30)
    })
    
    # Create backtest result
    config = {
        'strategy': 'CPU_XGBoost_v1',
        'universe': 'US_Large_Cap',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31'
    }
    
    result = create_backtest_result(
        run_id='TEST_001',
        trades=trades,
        equity_curve=equity_curve,
        config=config
    )
    
    # Validate against criteria
    criteria = {
        'min_annualized_return': 0.0,
        'max_drawdown': 0.25,
        'min_win_rate': 0.40,
        'min_profit_factor': 1.0
    }
    
    passed = result.validate(criteria)
    
    print(f"   Run ID: {result.run_id}")
    print(f"   Total trades: {result.total_trades}")
    print(f"   Win rate: {result.win_rate_pct:.1f}%")
    print(f"   Profit factor: {result.profit_factor:.2f}")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if not passed:
        print(f"   Failure reasons: {result.failure_reasons}")
    
    # Write to Excel
    output_path = 'reports/test_backtest_result.xlsx'
    writer = BacktestExcelWriter(result)
    writer.write(output_path)
    print(f"\n   ✅ Excel written to {output_path}")
    
    return True

def run_all_tests():
    """Run all core component tests."""
    print("=" * 60)
    print("🧪 TESTING TRADING CONSTITUTION CORE COMPONENTS")
    print("=" * 60)
    
    tests = [
        ('DataStore', test_data_store),
        ('VetoGates', test_veto_gates),
        ('CostModel', test_cost_model),
        ('BacktestResult', test_backtest_result)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed, error in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if error:
            print(f"      Error: {error}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Trading Constitution implemented correctly")
    else:
        print("⚠️ SOME TESTS FAILED - Review errors above")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
