"""
VT Sweet Spot Strategy - Pure Momentum (No AI/ML)
==================================================

Pure rule-based momentum strategy based on:
1. Power Stock (price > all 4 SMAs)
2. Volume Fuel (volume >= 30-day avg)
3. Daily Stochastic (both %K and %D >= 80)
4. Weekly Stochastic (both %K and %D >= 80)

Exit when:
- Daily stochastic drops below 80 (either %K or %D)
- Trailing stop hit (3%)

This is the baseline momentum strategy WITHOUT any AI/ML.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.core.sweetspot_indicators import add_sweetspot_indicators, is_in_sweetspot, should_exit_sweetspot
from src.core.data_store import get_data_store


class VTSweetSpotStrategy:
    """
    Pure VT Sweet Spot momentum strategy.
    No AI, no ML - just rule-based momentum confirmation.
    """
    
    def __init__(self):
        self.data_store = get_data_store()
    
    def generate_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate Sweet Spot signals for all tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with signals (date, ticker, signal, entry_price)
        """
        print(f"\n📊 Generating VT Sweet Spot signals...")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Tickers: {len(tickers)}")
        
        all_signals = []
        processed = 0
        
        for ticker in tickers:
            try:
                # Load ticker data
                data = self.data_store.get_ticker_data(ticker)
                if data is None or data.empty:
                    continue
                
                # Ensure date column
                if 'date' not in data.columns:
                    if isinstance(data.index, pd.DatetimeIndex):
                        data = data.reset_index()
                
                # Filter to date range
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
                
                if data.empty:
                    continue
                
                # Add Sweet Spot indicators
                data = add_sweetspot_indicators(data)
                
                # Generate signals
                for idx, row in data.iterrows():
                    if is_in_sweetspot(row):
                        all_signals.append({
                            'date': row['date'],
                            'ticker': ticker,
                            'signal': 1,  # LONG only
                            'entry_price': row['close'],
                            'confidence': 1.0  # All Sweet Spot signals have equal confidence
                        })
                
                processed += 1
                if processed % 20 == 0:
                    print(f"   📊 Processed {processed}/{len(tickers)} tickers...")
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {ticker}: {e}")
                continue
        
        if not all_signals:
            print("   ❌ No Sweet Spot signals generated")
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(all_signals)
        print(f"   ✅ Generated {len(signals_df):,} Sweet Spot signals")
        
        return signals_df
    
    def backtest(
        self,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        position_size_pct: float = 0.10,
        trailing_stop_pct: float = 0.03,
        max_positions: int = 10
    ) -> Dict:
        """
        Backtest Sweet Spot strategy.
        
        Args:
            signals: DataFrame with signals
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            trailing_stop_pct: Trailing stop %
            max_positions: Max concurrent positions
        
        Returns:
            Dictionary with backtest results
        """
        print(f"\n💰 Running VT Sweet Spot backtest...")
        print(f"   Initial Capital: ${initial_capital:,.0f}")
        print(f"   Position Size: {position_size_pct*100:.0f}%")
        print(f"   Trailing Stop: {trailing_stop_pct*100:.1f}%")
        
        capital = initial_capital
        positions = {}  # ticker -> {entry_date, entry_price, shares, peak_price}
        trades = []
        equity_curve = []
        
        # Get all unique dates
        all_dates = sorted(signals['date'].unique())
        
        for current_date in all_dates:
            # Check exits first
            tickers_to_exit = []
            for ticker, pos in positions.items():
                try:
                    # Get current price
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
                    
                    # Update peak price
                    if current_price > pos['peak_price']:
                        pos['peak_price'] = current_price
                    
                    # Add indicators for exit check
                    ticker_data_with_indicators = add_sweetspot_indicators(ticker_data)
                    current_row_with_indicators = ticker_data_with_indicators[
                        ticker_data_with_indicators['date'] == current_date
                    ]
                    
                    if current_row_with_indicators.empty:
                        continue
                    
                    current_row_with_indicators = current_row_with_indicators.iloc[0]
                    
                    # Check exit conditions
                    exit_reason = None
                    
                    # 1. Sweet Spot exit (daily stoch < 80)
                    if should_exit_sweetspot(current_row_with_indicators):
                        exit_reason = 'sweetspot_exit'
                    
                    # 2. Trailing stop
                    elif current_price < pos['peak_price'] * (1 - trailing_stop_pct):
                        exit_reason = 'trailing_stop'
                    
                    if exit_reason:
                        # Exit position
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
                    print(f"   ⚠️ Error checking exit for {ticker}: {e}")
                    tickers_to_exit.append(ticker)
            
            # Remove exited positions
            for ticker in tickers_to_exit:
                del positions[ticker]
            
            # Check new entries
            today_signals = signals[signals['date'] == current_date]
            
            for _, signal in today_signals.iterrows():
                ticker = signal['ticker']
                
                # Skip if already in position
                if ticker in positions:
                    continue
                
                # Skip if max positions reached
                if len(positions) >= max_positions:
                    break
                
                # Enter position
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
            
            # Calculate current equity
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
        
        # Close remaining positions at end
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
                'strategy_id': 'VT_SweetSpot_Pure',
                'total_trades': 0,
                'error': 'No trades executed'
            }
        
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        final_capital = capital
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Calculate CAGR
        years = (equity_df['date'].max() - equity_df['date'].min()).days / 365.25
        cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        results = {
            'strategy_id': 'VT_SweetSpot_Pure',
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


def run_sweetspot_only(
    start_date: str = '2015-01-01',
    end_date: str = '2024-12-31'
):
    """Run standalone VT Sweet Spot strategy."""
    store = get_data_store()
    tickers = store.available_tickers
    
    print("\n" + "=" * 70)
    print("📊 VT SWEET SPOT STRATEGY (PURE MOMENTUM - NO AI/ML)")
    print("=" * 70)
    
    strategy = VTSweetSpotStrategy()
    
    # Generate signals
    signals = strategy.generate_signals(tickers, start_date, end_date)
    
    if signals.empty:
        print("❌ No signals generated")
        return None
    
    # Backtest
    results = strategy.backtest(
        signals,
        initial_capital=100000,
        position_size_pct=0.10,
        trailing_stop_pct=0.03,
        max_positions=10
    )
    
    return results


if __name__ == "__main__":
    results = run_sweetspot_only()
