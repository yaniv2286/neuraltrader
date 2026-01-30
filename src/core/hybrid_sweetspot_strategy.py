"""
Hybrid Strategy: AI Ensemble V2 + VT Sweet Spot Filters
========================================================

Combines:
1. AI Ensemble V2 (XGBoost + RF + LGBM with all optimizations)
2. VT Sweet Spot momentum filters (Power Stock + Dual Stochastic)

Philosophy:
- AI provides directional predictions (50-53% accuracy)
- Sweet Spot filters to top 0.5-1% highest quality setups
- Result: Fewer trades, much higher win rate, better CAGR

Architecture:
- Baseline: AI Ensemble V2 only (existing)
- Hybrid: AI Ensemble V2 + Sweet Spot filters (new)
- Parallel execution for fair comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.core.ai_ensemble_strategy_v2 import AIEnsembleStrategyV2
from src.core.sweetspot_indicators import add_sweetspot_indicators, is_in_sweetspot
from src.core.data_store import get_data_store


class SweetSpotFilter:
    """
    VT Sweet Spot Filter - applies momentum confirmation rules.
    
    Sweet Spot Definition (Canonical):
    A market condition where:
    1. Price in confirmed uptrend (above SMA 25/50/100/200)
    2. Volume confirms participation (≥30-day average)
    3. Daily stochastic %K ≥80 AND %D ≥80
    4. Weekly stochastic %K ≥80 AND %D ≥80
    
    This is NOT a parameter to optimize.
    This is the REFERENCE REGIME.
    """
    
    def __init__(self):
        self.data_store = get_data_store()
    
    def calculate_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate stochastic %K and %D."""
        # Calculate %K
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * (data['close'] - low_min) / (high_max - low_min)
        
        # Smooth %K if needed
        if smooth > 1:
            stoch_k = stoch_k.rolling(window=smooth).mean()
        
        # Calculate %D (moving average of %K)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def calculate_weekly_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = 19,
        d_period: int = 4,
        smooth: int = 4
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate weekly stochastic from daily data."""
        # Ensure date is datetime and set as index
        if 'date' in data.columns:
            data_indexed = data.set_index('date')
        else:
            data_indexed = data.copy()
        
        # Resample to weekly
        weekly = data_indexed.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate stochastic on weekly bars
        stoch_k, stoch_d = self.calculate_stochastic(
            weekly, k_period, d_period, smooth
        )
        
        # Map back to daily (forward fill)
        stoch_k_daily = stoch_k.reindex(data_indexed.index, method='ffill')
        stoch_d_daily = stoch_d.reindex(data_indexed.index, method='ffill')
        
        # Reset to match original data structure
        stoch_k_daily.index = data.index
        stoch_d_daily.index = data.index
        
        return stoch_k_daily, stoch_d_daily
    
    def is_power_stock(self, row: pd.Series) -> bool:
        """
        Check if stock is a Power Stock.
        NO PARTIAL CREDIT - price must be above ALL SMAs.
        """
        return (
            row['close'] > row['sma_25'] and
            row['close'] > row['sma_50'] and
            row['close'] > row['sma_100'] and
            row['close'] > row['sma_200']
        )
    
    def has_volume_fuel(self, row: pd.Series) -> bool:
        """Check if volume confirms participation."""
        return row['volume'] >= row['volume_avg_30']
    
    def in_sweet_spot(
        self,
        daily_k: float,
        daily_d: float,
        weekly_k: float,
        weekly_d: float
    ) -> bool:
        """
        Check if BOTH daily and weekly stochastic are in sweet spot.
        Sweet Spot = %K ≥80 AND %D ≥80 (both red and yellow lines).
        """
        daily_in = (daily_k >= 80 and daily_d >= 80)
        weekly_in = (weekly_k >= 80 and weekly_d >= 80)
        return daily_in and weekly_in
    
    def should_exit_sweet_spot(
        self,
        daily_k: float,
        daily_d: float
    ) -> bool:
        """
        Check if daily stochastic has exited sweet spot.
        Exit when EITHER %K OR %D drops below 80.
        """
        return daily_k < 80 or daily_d < 80
    
    def apply_filter(
        self,
        signals: pd.DataFrame,
        ticker_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply Sweet Spot filter to AI signals.
        
        Args:
            signals: AI-generated signals with confidence scores
            ticker_data: Full OHLCV data (must have OHLCV columns)
            
        Returns:
            Filtered signals (only Sweet Spot qualified)
        """
        if signals.empty or ticker_data.empty:
            return pd.DataFrame()
        
        # Add Sweet Spot indicators
        try:
            data = add_sweetspot_indicators(ticker_data)
        except Exception as e:
            print(f"      ⚠️ Error adding indicators: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        
        # Debug: Check if indicators were added
        required_indicators = ['sma_25', 'sma_50', 'sma_100', 'sma_200', 
                              'volume_avg_30', 'stoch_daily_k', 'stoch_daily_d',
                              'stoch_weekly_k', 'stoch_weekly_d']
        missing_indicators = [col for col in required_indicators if col not in data.columns]
        if missing_indicators:
            print(f"      ⚠️ Missing indicators after add_sweetspot_indicators: {missing_indicators}")
            print(f"      Available columns: {data.columns.tolist()}")
            return pd.DataFrame()
        
        # Ensure date is datetime for merge
        if 'date' in signals.columns:
            signals = signals.copy()
            signals['date'] = pd.to_datetime(signals['date'])
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        # Merge signals with ticker data
        if 'date' in data.columns and 'date' in signals.columns:
            merged = signals.merge(data, on='date', how='left', suffixes=('', '_data'))
        else:
            print(f"      ⚠️ Date column missing - signals has date: {'date' in signals.columns}, data has date: {'date' in data.columns}")
            return pd.DataFrame()
        
        if merged.empty:
            return pd.DataFrame()
        
        # Apply Sweet Spot filter
        filtered = []
        debug_count = 0
        for idx, row in merged.iterrows():
            # Debug first 3 signals to see why they fail
            debug = (debug_count < 3)
            if debug:
                print(f"      🔍 Checking signal {debug_count + 1}:")
                debug_count += 1
            
            if is_in_sweetspot(row, debug=debug):
                filtered.append(row)
        
        if not filtered:
            return pd.DataFrame()
        
        filtered_df = pd.DataFrame(filtered)
        
        # Keep only original signal columns
        signal_cols = signals.columns.tolist()
        return filtered_df[signal_cols]


class HybridSweetSpotStrategy:
    """
    Hybrid Strategy: AI Ensemble V2 + Sweet Spot Filters
    
    Runs two strategies in parallel:
    1. AI_Ensemble_V2 (baseline - existing optimizations)
    2. AI_Ensemble_V2_SweetSpot (hybrid - AI + momentum filters)
    """
    
    def __init__(self):
        self.ai_strategy = AIEnsembleStrategyV2()
        self.sweetspot_filter = SweetSpotFilter()
        self.data_store = get_data_store()
    
    def run_comparison(
        self,
        tickers: List[str],
        train_start: str = '2005-01-01',
        train_end: str = '2014-12-31',
        test_start: str = '2015-01-01',
        test_end: str = '2024-12-31',
        use_80_20_split: bool = True
    ) -> Dict:
        """
        Run parallel comparison of AI-only vs AI+SweetSpot hybrid.
        
        Returns:
            Dictionary with results for both strategies
        """
        print("\n" + "=" * 70)
        print("🔬 HYBRID STRATEGY COMPARISON")
        print("=" * 70)
        print("Strategy 1: AI Ensemble V2 (baseline)")
        print("Strategy 2: AI Ensemble V2 + Sweet Spot (hybrid)")
        print("=" * 70)
        
        # Train AI ensemble once (used by both strategies)
        print("\n🤖 Training AI Ensemble...")
        train_results = self.ai_strategy.train_ensemble(
            tickers=tickers,
            train_start=train_start,
            train_end=train_end
        )
        
        # Generate AI signals
        print("\n📈 Generating AI signals...")
        ai_signals = self.ai_strategy.generate_signals(
            tickers=tickers,
            start_date=test_start,
            end_date=test_end
        )
        
        if ai_signals.empty:
            print("❌ No AI signals generated")
            return {'error': 'No signals'}
        
        print(f"   ✅ Generated {len(ai_signals):,} AI signals")
        
        # Filter to top signals
        ai_signals_filtered = self.ai_strategy.filter_top_signals(
            ai_signals,
            max_per_day=5,
            top_pct=0.10
        )
        
        # LONG only
        ai_signals_long = ai_signals_filtered[ai_signals_filtered['signal'] == 1].copy()
        
        print(f"   ✅ Filtered to {len(ai_signals_long):,} LONG signals (top 10%)")
        
        # Strategy 1: AI Ensemble V2 only (baseline)
        print("\n" + "=" * 70)
        print("📊 STRATEGY 1: AI Ensemble V2 (Baseline)")
        print("=" * 70)
        
        baseline_results = self.ai_strategy.backtest(
            ai_signals_long,
            initial_capital=100000,
            take_profit_pct=0.12,
            max_hold_days=15
        )
        baseline_results['strategy_id'] = 'AI_Ensemble_V2'
        
        print(f"\n   CAGR: {baseline_results['cagr_pct']:.2f}%")
        print(f"   Max DD: {baseline_results['max_drawdown_pct']:.2f}%")
        print(f"   Win Rate: {baseline_results['win_rate_pct']:.1f}%")
        print(f"   Total Trades: {baseline_results['total_trades']}")
        
        # Strategy 2: AI + Sweet Spot (hybrid)
        print("\n" + "=" * 70)
        print("📊 STRATEGY 2: AI Ensemble V2 + Sweet Spot (Hybrid)")
        print("=" * 70)
        
        # Apply Sweet Spot filter to AI signals
        print("\n🔍 Applying Sweet Spot filters...")
        
        hybrid_signals = []
        total_signals = len(ai_signals_long)
        processed = 0
        
        for ticker in ai_signals_long['ticker'].unique():
            ticker_signals = ai_signals_long[ai_signals_long['ticker'] == ticker].copy()
            
            # Load raw ticker data
            try:
                ticker_data = self.data_store.get_ticker_data(ticker)
                if ticker_data is None or ticker_data.empty:
                    continue
                
                # Debug: Check columns
                if processed == 0:
                    print(f"   🔍 Debug - Data columns for {ticker}: {ticker_data.columns.tolist()}")
                
                # Ensure date column exists
                if 'date' not in ticker_data.columns:
                    if ticker_data.index.name == 'date' or isinstance(ticker_data.index, pd.DatetimeIndex):
                        ticker_data = ticker_data.reset_index()
                
                # Filter to test period
                if 'date' in ticker_data.columns:
                    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                    ticker_data = ticker_data[
                        (ticker_data['date'] >= test_start) & 
                        (ticker_data['date'] <= test_end)
                    ]
                
                # Apply Sweet Spot filter (will add indicators internally)
                filtered = self.sweetspot_filter.apply_filter(ticker_signals, ticker_data)
                
                if not filtered.empty:
                    hybrid_signals.append(filtered)
                    
                processed += len(ticker_signals)
                if processed % 100 == 0:
                    print(f"   📊 Processed {processed}/{total_signals} signals...")
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {ticker}: {e}")
                continue
        
        if not hybrid_signals:
            print("   ❌ No signals passed Sweet Spot filter")
            hybrid_results = {
                'strategy_id': 'AI_Ensemble_V2_SweetSpot',
                'error': 'No signals after Sweet Spot filter',
                'total_trades': 0
            }
        else:
            hybrid_signals_df = pd.concat(hybrid_signals, ignore_index=True)
            print(f"   ✅ {len(hybrid_signals_df):,} signals passed Sweet Spot filter")
            print(f"   📉 Filter rate: {(1 - len(hybrid_signals_df)/len(ai_signals_long))*100:.1f}% filtered out")
            
            # Backtest hybrid strategy
            hybrid_results = self.ai_strategy.backtest(
                hybrid_signals_df,
                initial_capital=100000,
                take_profit_pct=0.12,
                max_hold_days=15
            )
            hybrid_results['strategy_id'] = 'AI_Ensemble_V2_SweetSpot'
            
            print(f"\n   CAGR: {hybrid_results['cagr_pct']:.2f}%")
            print(f"   Max DD: {hybrid_results['max_drawdown_pct']:.2f}%")
            print(f"   Win Rate: {hybrid_results['win_rate_pct']:.1f}%")
            print(f"   Total Trades: {hybrid_results['total_trades']}")
        
        # Comparison
        print("\n" + "=" * 70)
        print("📊 COMPARISON RESULTS")
        print("=" * 70)
        
        if 'error' not in hybrid_results:
            cagr_improvement = hybrid_results['cagr_pct'] - baseline_results['cagr_pct']
            dd_improvement = hybrid_results['max_drawdown_pct'] - baseline_results['max_drawdown_pct']
            wr_improvement = hybrid_results['win_rate_pct'] - baseline_results['win_rate_pct']
            
            print(f"\nCAGR Improvement: {cagr_improvement:+.2f}% ({cagr_improvement/baseline_results['cagr_pct']*100:+.1f}%)")
            print(f"Max DD Improvement: {dd_improvement:+.2f}% (less negative = better)")
            print(f"Win Rate Improvement: {wr_improvement:+.1f}%")
            print(f"Trade Reduction: {baseline_results['total_trades'] - hybrid_results['total_trades']} trades")
            
            # Acceptance criteria
            accepted = (
                hybrid_results['cagr_pct'] >= baseline_results['cagr_pct'] and
                hybrid_results['max_drawdown_pct'] >= baseline_results['max_drawdown_pct']
            )
            
            print(f"\n{'✅ HYBRID ACCEPTED' if accepted else '❌ HYBRID REJECTED'}")
            
            if accepted:
                print("   Sweet Spot filter adds value!")
            else:
                print("   Sweet Spot filter does not improve performance")
        
        return {
            'baseline': baseline_results,
            'hybrid': hybrid_results,
            'train_results': train_results
        }


def run_hybrid_comparison(use_80_20_split: bool = True):
    """Run hybrid strategy comparison."""
    store = get_data_store()
    tickers = store.available_tickers
    
    strategy = HybridSweetSpotStrategy()
    
    if use_80_20_split:
        results = strategy.run_comparison(
            tickers=tickers,
            train_start='1970-01-01',
            train_end='2014-12-31',
            test_start='2015-01-01',
            test_end='2024-12-31',
            use_80_20_split=True
        )
    else:
        results = strategy.run_comparison(
            tickers=tickers,
            use_80_20_split=False
        )
    
    # Generate Excel report
    if 'error' not in results.get('hybrid', {}):
        try:
            from src.core.trading_backtest_excel import create_trading_backtest_excel
            
            # Create comparison report with both strategies
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Baseline report
            baseline_path = f'reports/hybrid_baseline_{timestamp}.xlsx'
            create_trading_backtest_excel(
                results=results['baseline'],
                config={'strategy_id': 'AI_Ensemble_V2'},
                output_path=baseline_path
            )
            print(f"\n📁 Baseline Report: {baseline_path}")
            
            # Hybrid report
            hybrid_path = f'reports/hybrid_sweetspot_{timestamp}.xlsx'
            create_trading_backtest_excel(
                results=results['hybrid'],
                config={'strategy_id': 'AI_Ensemble_V2_SweetSpot'},
                output_path=hybrid_path
            )
            print(f"📁 Hybrid Report: {hybrid_path}")
            
        except Exception as e:
            print(f"\n⚠️ Could not generate Excel reports: {e}")
    
    return results


if __name__ == "__main__":
    results = run_hybrid_comparison(use_80_20_split=True)
