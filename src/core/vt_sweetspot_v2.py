"""
VT Sweet Spot V2 - Optimized Position Sizing
=============================================

Simplified optimization focusing on:
1. Larger fixed position sizes (15% vs 10%)
2. Better trailing stops (tighter, earlier breakeven)
3. Confidence-based sizing multiplier
4. Max 8 positions (vs 10) for better concentration

Goal: Beat baseline 1.82% CAGR with better position management
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.core.sweetspot_indicators import add_sweetspot_indicators, is_in_sweetspot, should_exit_sweetspot
from src.core.data_store import get_data_store


class VTSweetSpotV2:
    """Sweet Spot with optimized position sizing and stops."""
    
    def __init__(self):
        self.data_store = get_data_store()
    
    def calculate_stoch_strength(self, row: pd.Series) -> float:
        """Calculate stochastic strength (0.0 to 1.0)."""
        daily_k = row['stoch_daily_k']
        weekly_k = row['stoch_weekly_k']
        
        # How far above 80
        daily_strength = min((daily_k - 80) / 20, 1.0)
        weekly_strength = min((weekly_k - 80) / 20, 1.0)
        
        return (daily_strength + weekly_strength) / 2
    
    def generate_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Generate Sweet Spot signals with confidence."""
        from src.core.signal_cache import get_signal_cache
        
        # Try to load from cache first
        if use_cache:
            signal_cache = get_signal_cache()
            cached_signals = signal_cache.load_signals(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                strategy_type='sweetspot_v2'
            )
            
            if cached_signals is not None:
                print(f"   ⚡ Using cached signals - GENERATION SKIPPED!")
                return cached_signals
        
        print(f"\n📊 Generating VT Sweet Spot V2 signals...")
        print(f"   Period: {start_date} to {end_date}")
        
        all_signals = []
        processed = 0
        
        for ticker in tickers:
            try:
                data = self.data_store.get_ticker_data(ticker)
                if data is None or data.empty:
                    continue
                
                if 'date' not in data.columns:
                    if isinstance(data.index, pd.DatetimeIndex):
                        data = data.reset_index()
                
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
                
                if data.empty:
                    continue
                
                data = add_sweetspot_indicators(data)
                
                for idx, row in data.iterrows():
                    if is_in_sweetspot(row):
                        confidence = self.calculate_stoch_strength(row)
                        
                        all_signals.append({
                            'date': row['date'],
                            'ticker': ticker,
                            'signal': 1,
                            'entry_price': row['close'],
                            'confidence': confidence
                        })
                
                processed += 1
                if processed % 20 == 0:
                    print(f"   📊 Processed {processed}/{len(tickers)} tickers...")
                    
            except Exception as e:
                continue
        
        if not all_signals:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(all_signals)
        print(f"   ✅ Generated {len(signals_df):,} signals")
        
        # Save to cache
        if use_cache:
            signal_cache = get_signal_cache()
            signal_cache.save_signals(
                signals=signals_df,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                strategy_type='sweetspot_v2'
            )
        
        return signals_df
    
    def backtest(
        self,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        base_position_pct: float = 0.15,  # 15% base (vs 10%)
        max_positions: int = 8  # Max 8 positions (vs 10)
    ) -> Dict:
        """Backtest with optimized position sizing."""
        print(f"\n💰 Running Sweet Spot V2 backtest...")
        print(f"   Base Position: {base_position_pct*100:.0f}%")
        print(f"   Max Positions: {max_positions}")
        print(f"   Features: Confidence Sizing + Tighter Stops")
        
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = []
        
        all_dates = sorted(signals['date'].unique())
        
        for current_date in all_dates:
            # Check exits
            tickers_to_exit = []
            for ticker, pos in list(positions.items()):
                try:
                    ticker_data = self.data_store.get_ticker_data(ticker)
                    if ticker_data is None or ticker_data.empty:
                        tickers_to_exit.append(ticker)
                        continue
                    
                    if 'date' not in ticker_data.columns:
                        ticker_data = ticker_data.reset_index()
                    
                    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                    current_row = ticker_data[ticker_data['date'] == current_date]
                    
                    if current_row.empty:
                        continue
                    
                    current_row = current_row.iloc[0]
                    current_price = current_row['close']
                    
                    # Update peak
                    if current_price > pos['peak_price']:
                        pos['peak_price'] = current_price
                    
                    # Add indicators
                    ticker_data_with_indicators = add_sweetspot_indicators(ticker_data)
                    current_row_with_indicators = ticker_data_with_indicators[
                        ticker_data_with_indicators['date'] == current_date
                    ]
                    
                    if current_row_with_indicators.empty:
                        continue
                    
                    current_row_with_indicators = current_row_with_indicators.iloc[0]
                    
                    # Exit conditions
                    exit_reason = None
                    
                    # 1. Sweet Spot exit
                    if should_exit_sweetspot(current_row_with_indicators):
                        exit_reason = 'sweetspot_exit'
                    else:
                        # 2. Improved trailing stop
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                        peak_pnl_pct = (pos['peak_price'] - pos['entry_price']) / pos['entry_price']
                        
                        # Tighter stops
                        if peak_pnl_pct >= 0.04:  # After 4% gain
                            stop_price = pos['peak_price'] * 0.98  # Trail 2%
                        elif peak_pnl_pct >= 0.015:  # After 1.5% gain
                            stop_price = pos['entry_price'] * 1.001  # Breakeven
                        else:
                            stop_price = pos['entry_price'] * 0.98  # Initial 2% stop
                        
                        if current_price <= stop_price:
                            exit_reason = 'trailing_stop'
                    
                    if exit_reason:
                        exit_value = pos['shares'] * current_price
                        capital += exit_value
                        
                        pnl = exit_value - (pos['shares'] * pos['entry_price'])
                        pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': pos['entry_date'],
                            'exit_date': current_date,
                            'entry_price': pos['entry_price'],
                            'exit_price': current_price,
                            'shares': pos['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason
                        })
                        
                        tickers_to_exit.append(ticker)
                        
                except Exception as e:
                    tickers_to_exit.append(ticker)
            
            for ticker in tickers_to_exit:
                del positions[ticker]
            
            # Check new entries
            today_signals = signals[signals['date'] == current_date]
            
            for _, signal in today_signals.iterrows():
                ticker = signal['ticker']
                
                if ticker in positions:
                    continue
                
                if len(positions) >= max_positions:
                    break
                
                # Confidence-based sizing: 0.5x to 1.5x base size
                confidence_multiplier = 0.5 + signal['confidence']
                position_size_pct = base_position_pct * confidence_multiplier
                
                # Cap at 20%
                position_size_pct = min(position_size_pct, 0.20)
                
                position_value = capital * position_size_pct
                shares = position_value / signal['entry_price']
                
                if position_value > capital:
                    continue
                
                capital -= position_value
                
                positions[ticker] = {
                    'entry_date': current_date,
                    'entry_price': signal['entry_price'],
                    'shares': shares,
                    'peak_price': signal['entry_price']
                }
            
            # Calculate equity
            position_value = sum(
                pos['shares'] * pos['entry_price'] for pos in positions.values()
            )
            total_equity = capital + position_value
            
            equity_curve.append({
                'date': current_date,
                'equity': total_equity,
                'cash': capital,
                'positions': len(positions)
            })
        
        # Close remaining
        for ticker, pos in positions.items():
            try:
                ticker_data = self.data_store.get_ticker_data(ticker)
                if ticker_data is None or ticker_data.empty:
                    continue
                
                if 'date' not in ticker_data.columns:
                    ticker_data = ticker_data.reset_index()
                
                ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                final_price = ticker_data.iloc[-1]['close']
                
                exit_value = pos['shares'] * final_price
                capital += exit_value
                
                pnl = exit_value - (pos['shares'] * pos['entry_price'])
                pnl_pct = (final_price / pos['entry_price'] - 1) * 100
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': ticker_data.iloc[-1]['date'],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'end_of_period'
                })
            except:
                continue
        
        # Calculate metrics
        if not trades:
            return {
                'strategy_id': 'VT_SweetSpot_V2',
                'total_trades': 0,
                'error': 'No trades'
            }
        
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        final_capital = capital
        total_return = (final_capital / initial_capital - 1) * 100
        
        years = (equity_df['date'].max() - equity_df['date'].min()).days / 365.25
        cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        results = {
            'strategy_id': 'VT_SweetSpot_V2',
            'strategy_name': 'VT Sweet Spot V2 (Optimized Sizing)',
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        print(f"\n   ✅ Backtest complete!")
        print(f"   CAGR: {cagr:.2f}%")
        print(f"   Max DD: {max_drawdown:.2f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total Trades: {len(trades_df)}")
        
        return results


if __name__ == "__main__":
    store = get_data_store()
    tickers = store.available_tickers
    
    print("\n" + "=" * 70)
    print("📊 VT SWEET SPOT V2 - OPTIMIZED POSITION SIZING")
    print("=" * 70)
    
    strategy = VTSweetSpotV2()
    signals = strategy.generate_signals(tickers, '2015-01-01', '2024-12-31')
    
    if not signals.empty:
        results = strategy.backtest(signals)
