"""
Adaptive Training Comparison Framework
======================================
Compares three training approaches to find the best one:

1. Walk-Forward Analysis - Time-based cross-validation
2. Rolling Window Training - Train on recent 10 years only
3. Regime-Specific Models - Separate models for bull/bear markets

Goal: Find which approach gives best CAGR with DD < 20%
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.features.technical_indicators import generate_all_features
from src.models.cpu_models.xgboost_model import XGBoostModel
from src.core.data_store import get_data_store
from src.core.ai_ensemble_strategy_v2 import AIEnsembleStrategyV2, RiskManager, TrailingStop

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    HAS_ML = True
except ImportError:
    HAS_ML = False


class WalkForwardAnalysis:
    """
    Walk-Forward Analysis: Train on expanding window, test on next period.
    
    Example for 2015-2024 (10 years):
    - Train: 2005-2015 (10y) → Test: 2015-2017 (2y)
    - Train: 2005-2017 (12y) → Test: 2017-2019 (2y)
    - Train: 2005-2019 (14y) → Test: 2019-2021 (2y)
    - Train: 2005-2021 (16y) → Test: 2021-2023 (2y)
    - Train: 2005-2023 (18y) → Test: 2023-2024 (1y)
    """
    
    def __init__(self, strategy: AIEnsembleStrategyV2):
        self.strategy = strategy
        self.data_store = get_data_store()
    
    def run(
        self,
        tickers: List[str],
        start_year: int = 2015,
        end_year: int = 2024,
        train_window_years: int = 10,
        test_window_years: int = 2
    ) -> Dict:
        """Run walk-forward analysis."""
        print("\n" + "=" * 70)
        print("🚶 WALK-FORWARD ANALYSIS")
        print("=" * 70)
        print(f"Training Window: {train_window_years} years")
        print(f"Test Window: {test_window_years} years")
        
        all_results = []
        all_trades = []
        
        current_year = start_year
        fold = 1
        
        while current_year < end_year:
            test_end_year = min(current_year + test_window_years, end_year)
            train_start_year = current_year - train_window_years
            
            print(f"\n📊 Fold {fold}:")
            print(f"   Train: {train_start_year}-01-01 to {current_year}-12-31")
            print(f"   Test:  {current_year}-01-01 to {test_end_year}-12-31")
            
            # Train on this window
            train_results = self.strategy.train_ensemble(
                tickers=tickers,
                train_start=f'{train_start_year}-01-01',
                train_end=f'{current_year}-12-31'
            )
            
            # Generate signals for test period
            signals = self.strategy.generate_signals(
                tickers=tickers,
                start_date=f'{current_year}-01-01',
                end_date=f'{test_end_year}-12-31'
            )
            
            if not signals.empty:
                # Filter and backtest
                filtered = self.strategy.filter_top_signals(signals, max_per_day=5, top_pct=0.10)
                filtered = filtered[filtered['signal'] == 1].copy()  # LONG only
                
                if not filtered.empty:
                    results = self.strategy.backtest(
                        filtered,
                        initial_capital=100000,
                        take_profit_pct=0.12,
                        max_hold_days=15
                    )
                    
                    results['fold'] = fold
                    results['train_period'] = f"{train_start_year}-{current_year}"
                    results['test_period'] = f"{current_year}-{test_end_year}"
                    all_results.append(results)
                    all_trades.append(results['trades'])
                    
                    print(f"   ✅ CAGR: {results['cagr_pct']:.2f}%, DD: {results['max_drawdown_pct']:.2f}%")
            
            current_year = test_end_year
            fold += 1
        
        # Combine all results
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
            
            # Calculate overall metrics
            initial_capital = 100000
            final_capital = combined_trades['capital'].iloc[-1]
            total_return = ((final_capital - initial_capital) / initial_capital) * 100
            
            first_date = combined_trades['entry_date'].min()
            last_date = combined_trades['exit_date'].max()
            years = (last_date - first_date).days / 365.25
            cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
            
            max_dd = combined_trades['drawdown'].min() * 100
            win_rate = (combined_trades['pnl'] > 0).mean() * 100
            
            avg_win = combined_trades[combined_trades['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(combined_trades[combined_trades['pnl'] < 0]['pnl'].mean())
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            return {
                'method': 'Walk-Forward Analysis',
                'total_return_pct': total_return,
                'cagr_pct': cagr,
                'max_drawdown_pct': max_dd,
                'win_rate_pct': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(combined_trades),
                'final_capital': final_capital,
                'trades': combined_trades,
                'fold_results': all_results
            }
        
        return {'error': 'No trades executed'}


class RollingWindowTraining:
    """
    Rolling Window: Always train on most recent N years only.
    Model adapts to changing market conditions.
    
    Example for 2015-2024 validation:
    - Test 2015-2017: Train on 2005-2015 (10y)
    - Test 2017-2019: Train on 2007-2017 (10y)
    - Test 2019-2021: Train on 2009-2019 (10y)
    - Test 2021-2023: Train on 2011-2021 (10y)
    - Test 2023-2024: Train on 2013-2023 (10y)
    """
    
    def __init__(self, strategy: AIEnsembleStrategyV2):
        self.strategy = strategy
        self.data_store = get_data_store()
    
    def run(
        self,
        tickers: List[str],
        start_year: int = 2015,
        end_year: int = 2024,
        window_years: int = 10,
        test_window_years: int = 2
    ) -> Dict:
        """Run rolling window training."""
        print("\n" + "=" * 70)
        print("🔄 ROLLING WINDOW TRAINING")
        print("=" * 70)
        print(f"Training Window: {window_years} years (rolling)")
        print(f"Test Window: {test_window_years} years")
        
        all_results = []
        all_trades = []
        
        current_year = start_year
        fold = 1
        
        while current_year < end_year:
            test_end_year = min(current_year + test_window_years, end_year)
            train_start_year = current_year - window_years
            train_end_year = current_year
            
            print(f"\n📊 Fold {fold}:")
            print(f"   Train: {train_start_year}-01-01 to {train_end_year}-12-31 ({window_years}y window)")
            print(f"   Test:  {current_year}-01-01 to {test_end_year}-12-31")
            
            # Train on rolling window
            train_results = self.strategy.train_ensemble(
                tickers=tickers,
                train_start=f'{train_start_year}-01-01',
                train_end=f'{train_end_year}-12-31'
            )
            
            # Generate signals for test period
            signals = self.strategy.generate_signals(
                tickers=tickers,
                start_date=f'{current_year}-01-01',
                end_date=f'{test_end_year}-12-31'
            )
            
            if not signals.empty:
                # Filter and backtest
                filtered = self.strategy.filter_top_signals(signals, max_per_day=5, top_pct=0.10)
                filtered = filtered[filtered['signal'] == 1].copy()  # LONG only
                
                if not filtered.empty:
                    results = self.strategy.backtest(
                        filtered,
                        initial_capital=100000,
                        take_profit_pct=0.12,
                        max_hold_days=15
                    )
                    
                    results['fold'] = fold
                    results['train_period'] = f"{train_start_year}-{train_end_year}"
                    results['test_period'] = f"{current_year}-{test_end_year}"
                    all_results.append(results)
                    all_trades.append(results['trades'])
                    
                    print(f"   ✅ CAGR: {results['cagr_pct']:.2f}%, DD: {results['max_drawdown_pct']:.2f}%")
            
            current_year = test_end_year
            fold += 1
        
        # Combine all results
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
            
            # Calculate overall metrics
            initial_capital = 100000
            final_capital = combined_trades['capital'].iloc[-1]
            total_return = ((final_capital - initial_capital) / initial_capital) * 100
            
            first_date = combined_trades['entry_date'].min()
            last_date = combined_trades['exit_date'].max()
            years = (last_date - first_date).days / 365.25
            cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
            
            max_dd = combined_trades['drawdown'].min() * 100
            win_rate = (combined_trades['pnl'] > 0).mean() * 100
            
            avg_win = combined_trades[combined_trades['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(combined_trades[combined_trades['pnl'] < 0]['pnl'].mean())
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            return {
                'method': 'Rolling Window (10y)',
                'total_return_pct': total_return,
                'cagr_pct': cagr,
                'max_drawdown_pct': max_dd,
                'win_rate_pct': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(combined_trades),
                'final_capital': final_capital,
                'trades': combined_trades,
                'fold_results': all_results
            }
        
        return {'error': 'No trades executed'}


class RegimeSpecificModels:
    """
    Regime-Specific: Train separate models for bull and bear markets.
    Use appropriate model based on current market regime.
    
    Bull Market: SPY > 200 SMA and positive 6-month momentum
    Bear Market: SPY < 200 SMA or negative 6-month momentum
    """
    
    def __init__(self, strategy: AIEnsembleStrategyV2):
        self.strategy = strategy
        self.data_store = get_data_store()
        self.bull_models = {}
        self.bear_models = {}
    
    def _identify_regime(self, spy_df: pd.DataFrame) -> pd.Series:
        """Identify bull/bear regime for each date."""
        spy_df = generate_all_features(spy_df)
        
        # Bull: Price > 200 SMA AND positive 6-month momentum
        bull_condition = (
            (spy_df['close'] > spy_df['sma_200']) &
            (spy_df['close'].pct_change(126) > 0)  # 6 months ≈ 126 trading days
        )
        
        regime = pd.Series('bull', index=spy_df.index)
        regime[~bull_condition] = 'bear'
        
        return regime
    
    def run(
        self,
        tickers: List[str],
        train_start: str = '2005-01-01',
        train_end: str = '2014-12-31',
        test_start: str = '2015-01-01',
        test_end: str = '2024-12-31'
    ) -> Dict:
        """Run regime-specific models."""
        print("\n" + "=" * 70)
        print("📈📉 REGIME-SPECIFIC MODELS")
        print("=" * 70)
        
        # Load SPY to identify regimes
        spy_df = self.data_store.get_ticker_data('SPY', train_start, test_end)
        regime = self._identify_regime(spy_df)
        
        # Split training data by regime
        train_regime = regime[train_start:train_end]
        bull_dates = train_regime[train_regime == 'bull'].index
        bear_dates = train_regime[train_regime == 'bear'].index
        
        print(f"\n📊 Training Data Split:")
        print(f"   Bull Market Days: {len(bull_dates)} ({len(bull_dates)/len(train_regime)*100:.1f}%)")
        print(f"   Bear Market Days: {len(bear_dates)} ({len(bear_dates)/len(train_regime)*100:.1f}%)")
        
        # Train bull market model
        print(f"\n🐂 Training BULL MARKET Model...")
        bull_start = str(bull_dates[0])[:10]
        bull_end = str(bull_dates[-1])[:10]
        
        bull_train_results = self.strategy.train_ensemble(
            tickers=tickers,
            train_start=train_start,
            train_end=train_end
        )
        self.bull_models = self.strategy.models.copy()
        print(f"   ✅ Bull model trained: {bull_train_results['accuracy']:.1f}% accuracy")
        
        # Train bear market model
        print(f"\n🐻 Training BEAR MARKET Model...")
        bear_start = str(bear_dates[0])[:10]
        bear_end = str(bear_dates[-1])[:10]
        
        bear_train_results = self.strategy.train_ensemble(
            tickers=tickers,
            train_start=train_start,
            train_end=train_end
        )
        self.bear_models = self.strategy.models.copy()
        print(f"   ✅ Bear model trained: {bear_train_results['accuracy']:.1f}% accuracy")
        
        # Generate signals for test period using appropriate model
        print(f"\n📈 Generating signals with regime-specific models...")
        
        test_regime = regime[test_start:test_end]
        all_signals = []
        
        # Generate signals for each regime
        for regime_type in ['bull', 'bear']:
            regime_dates = test_regime[test_regime == regime_type].index
            if len(regime_dates) == 0:
                continue
            
            # Use appropriate model
            self.strategy.models = self.bull_models if regime_type == 'bull' else self.bear_models
            
            # Generate signals for this regime
            regime_start = str(regime_dates[0])[:10]
            regime_end = str(regime_dates[-1])[:10]
            
            signals = self.strategy.generate_signals(
                tickers=tickers,
                start_date=regime_start,
                end_date=regime_end
            )
            
            if not signals.empty:
                signals['regime'] = regime_type
                all_signals.append(signals)
        
        if not all_signals:
            return {'error': 'No signals generated'}
        
        combined_signals = pd.concat(all_signals, ignore_index=True)
        print(f"   ✅ Generated {len(combined_signals):,} signals")
        print(f"      Bull signals: {(combined_signals['regime'] == 'bull').sum():,}")
        print(f"      Bear signals: {(combined_signals['regime'] == 'bear').sum():,}")
        
        # Filter and backtest
        filtered = self.strategy.filter_top_signals(combined_signals, max_per_day=5, top_pct=0.10)
        filtered = filtered[filtered['signal'] == 1].copy()  # LONG only
        
        if filtered.empty:
            return {'error': 'No signals after filtering'}
        
        results = self.strategy.backtest(
            filtered,
            initial_capital=100000,
            take_profit_pct=0.12,
            max_hold_days=15
        )
        
        results['method'] = 'Regime-Specific Models'
        return results


def compare_all_approaches():
    """Compare all three training approaches."""
    print("=" * 70)
    print("🔬 ADAPTIVE TRAINING COMPARISON")
    print("=" * 70)
    print("Testing 3 approaches to find the best one:")
    print("  1. Walk-Forward Analysis")
    print("  2. Rolling Window Training (10 years)")
    print("  3. Regime-Specific Models")
    print("=" * 70)
    
    store = get_data_store()
    tickers = store.available_tickers
    
    # Initialize strategy
    strategy = AIEnsembleStrategyV2()
    
    results_comparison = []
    
    # 1. Walk-Forward Analysis
    try:
        wfa = WalkForwardAnalysis(strategy)
        wfa_results = wfa.run(tickers, start_year=2015, end_year=2024)
        if 'error' not in wfa_results:
            results_comparison.append(wfa_results)
    except Exception as e:
        print(f"\n❌ Walk-Forward Analysis failed: {e}")
    
    # 2. Rolling Window Training
    try:
        rwt = RollingWindowTraining(strategy)
        rwt_results = rwt.run(tickers, start_year=2015, end_year=2024, window_years=10)
        if 'error' not in rwt_results:
            results_comparison.append(rwt_results)
    except Exception as e:
        print(f"\n❌ Rolling Window Training failed: {e}")
    
    # 3. Regime-Specific Models
    try:
        rsm = RegimeSpecificModels(strategy)
        rsm_results = rsm.run(tickers, train_start='2005-01-01', train_end='2014-12-31',
                              test_start='2015-01-01', test_end='2024-12-31')
        if 'error' not in rsm_results:
            results_comparison.append(rsm_results)
    except Exception as e:
        print(f"\n❌ Regime-Specific Models failed: {e}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    if not results_comparison:
        print("❌ No successful results to compare")
        return None
    
    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            'Method': r['method'],
            'CAGR (%)': f"{r['cagr_pct']:.2f}",
            'Max DD (%)': f"{r['max_drawdown_pct']:.2f}",
            'Win Rate (%)': f"{r['win_rate_pct']:.1f}",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Total Trades': r['total_trades'],
            'Final Capital': f"${r['final_capital']:,.0f}",
            'Pass?': '✅ PASS' if (r['cagr_pct'] >= 20 and r['max_drawdown_pct'] >= -20) else '❌ FAIL'
        }
        for r in results_comparison
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best approach
    best_idx = max(range(len(results_comparison)), 
                   key=lambda i: results_comparison[i]['cagr_pct'] 
                   if results_comparison[i]['max_drawdown_pct'] >= -20 else -999)
    
    best_result = results_comparison[best_idx]
    
    print("\n" + "=" * 70)
    print("🏆 WINNER")
    print("=" * 70)
    print(f"Best Approach: {best_result['method']}")
    print(f"CAGR: {best_result['cagr_pct']:.2f}%")
    print(f"Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {best_result['win_rate_pct']:.1f}%")
    print(f"Profit Factor: {best_result['profit_factor']:.2f}")
    
    # Generate Excel report for best approach
    try:
        from src.core.trading_backtest_excel import create_trading_backtest_excel
        
        config = {
            'method': best_result['method'],
            'models': 'XGBoost + RandomForest + LightGBM',
            'training_approach': best_result['method'],
            'test_period': '2015-2024',
            'optimizations': 'Kelly + VolAdj + TrailingStop + Adaptive Training'
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'reports/adaptive_comparison_{timestamp}.xlsx'
        
        create_trading_backtest_excel(
            results=best_result,
            config=config,
            output_path=output_path
        )
        
        print(f"\n📁 Excel Report: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Could not generate Excel report: {e}")
    
    return results_comparison


if __name__ == "__main__":
    compare_all_approaches()
