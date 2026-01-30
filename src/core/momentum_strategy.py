"""
Momentum-Based Trading Strategy
Uses proven technical indicators instead of ML predictions.
Goal: CAGR > 20%, Max DD < 20%
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.features.technical_indicators import generate_all_features
from src.core.data_store import get_data_store


class MomentumStrategy:
    """
    Momentum strategy using proven technical signals.
    
    Entry conditions (ALL must be true):
    1. Price above 200-day SMA (uptrend)
    2. RSI between 30-70 (not overbought/oversold)
    3. MACD histogram positive (momentum)
    4. Price broke above 20-day high (breakout)
    
    Exit conditions:
    1. Trailing stop of 8%
    2. Price below 50-day SMA
    """
    
    def __init__(self):
        self.data_store = get_data_store()
        
    def generate_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate momentum signals for all tickers."""
        print(f"📈 Generating momentum signals for {len(tickers)} tickers...")
        
        all_signals = []
        
        for ticker in tickers:
            try:
                df = self.data_store.get_ticker_data(ticker, start_date, end_date)
                if df is None or len(df) < 250:
                    continue
                
                df = generate_all_features(df)
                if len(df) < 200:
                    continue
                
                # Calculate additional signals
                df['high_20'] = df['high'].rolling(20).max()
                df['breakout'] = df['close'] > df['high_20'].shift(1)
                df['macd_hist_positive'] = df['macd_histogram'] > 0
                df['above_sma200'] = df['close'] > df['sma_200']
                df['above_sma50'] = df['close'] > df['sma_50']
                df['rsi_ok'] = (df['rsi'] > 30) & (df['rsi'] < 70)
                
                # Entry signal: ALL conditions must be true
                df['entry_signal'] = (
                    df['above_sma200'] &
                    df['rsi_ok'] &
                    df['macd_hist_positive'] &
                    df['breakout']
                )
                
                # Calculate signal strength (for ranking)
                df['signal_strength'] = (
                    (df['close'] / df['sma_200'] - 1) * 100 +  # Distance from SMA200
                    (50 - abs(df['rsi'] - 50)) / 50 +  # RSI centered score
                    df['macd_histogram'].clip(-1, 1)  # MACD strength
                )
                
                # Only keep entry signals
                signals = df[df['entry_signal']].copy()
                
                if len(signals) > 0:
                    signals['ticker'] = ticker
                    signals['date'] = signals.index
                    all_signals.append(signals[['date', 'ticker', 'close', 'signal_strength', 'rsi', 'volume']])
                
            except Exception as e:
                continue
        
        if not all_signals:
            return pd.DataFrame()
        
        combined = pd.concat(all_signals, ignore_index=True)
        print(f"   ✅ Generated {len(combined):,} momentum signals")
        
        return combined
    
    def filter_best_signals(
        self,
        signals: pd.DataFrame,
        max_per_day: int = 2
    ) -> pd.DataFrame:
        """Keep only best signals per day."""
        print(f"\n🎯 Filtering to top {max_per_day} signals per day...")
        
        signals['date_only'] = pd.to_datetime(signals['date']).dt.date
        signals['rank'] = signals.groupby('date_only')['signal_strength'].rank(ascending=False)
        filtered = signals[signals['rank'] <= max_per_day].copy()
        
        print(f"   ✅ Filtered to {len(filtered):,} signals")
        
        return filtered
    
    def backtest(
        self,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        position_pct: float = 0.25,  # 25% per position
        trailing_stop_pct: float = 0.08,  # 8% trailing stop
        max_hold_days: int = 20  # Max holding period
    ) -> Dict:
        """Backtest momentum strategy with trailing stops."""
        print(f"\n💰 Running momentum backtest...")
        
        capital = initial_capital
        peak_capital = initial_capital
        trades = []
        
        signals = signals.sort_values('date')
        
        # Load price data for exit calculations
        ticker_data = {}
        for ticker in signals['ticker'].unique():
            try:
                df = self.data_store.get_ticker_data(ticker, '2020-01-01', '2024-12-31')
                if df is not None:
                    ticker_data[ticker] = df
            except:
                pass
        
        for _, signal in signals.iterrows():
            entry_date = pd.to_datetime(signal['date'])
            ticker = signal['ticker']
            entry_price = signal['close']
            
            if ticker not in ticker_data:
                continue
            
            df = ticker_data[ticker]
            
            # Find exit point
            future_data = df[df.index > entry_date].head(max_hold_days)
            
            if len(future_data) == 0:
                continue
            
            # Track trailing stop
            highest_price = entry_price
            exit_price = None
            exit_date = None
            exit_reason = None
            
            for idx, row in future_data.iterrows():
                current_price = row['close']
                
                # Update trailing stop
                if current_price > highest_price:
                    highest_price = current_price
                
                stop_price = highest_price * (1 - trailing_stop_pct)
                
                # Check stop loss
                if current_price <= stop_price:
                    exit_price = stop_price
                    exit_date = idx
                    exit_reason = 'trailing_stop'
                    break
                
                # Check SMA50 exit
                if 'sma_50' in row and current_price < row.get('sma_50', current_price):
                    exit_price = current_price
                    exit_date = idx
                    exit_reason = 'sma50_exit'
                    break
            
            # If no exit triggered, exit at end of period
            if exit_price is None:
                exit_price = future_data.iloc[-1]['close']
                exit_date = future_data.index[-1]
                exit_reason = 'max_hold'
            
            # Calculate trade result
            trade_return = (exit_price - entry_price) / entry_price
            position_size = capital * position_pct
            shares = int(position_size / entry_price)
            
            if shares <= 0:
                continue
            
            pnl = shares * (exit_price - entry_price)
            capital += pnl
            
            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (capital - peak_capital) / peak_capital
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'ticker': ticker,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'return_pct': trade_return * 100,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'capital': capital,
                'drawdown': drawdown
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        
        first_date = trades_df['entry_date'].min()
        last_date = trades_df['exit_date'].max()
        years = (last_date - first_date).days / 365.25
        cagr = ((capital / initial_capital) ** (1 / max(years, 0.1)) - 1) * 100
        
        max_dd = trades_df['drawdown'].min() * 100
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (trades_df['pnl'] < 0).any() else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'max_drawdown_pct': max_dd,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_return_per_trade': trades_df['return_pct'].mean(),
            'final_capital': capital,
            'trades': trades_df
        }


def run_momentum_backtest():
    """Run momentum strategy backtest."""
    print("=" * 70)
    print("📈 MOMENTUM TRADING STRATEGY")
    print("=" * 70)
    print("Target: CAGR > 20%, Max DD < 20%, Win Rate > 55%")
    
    store = get_data_store()
    tickers = store.available_tickers
    print(f"\n📊 Universe: {len(tickers)} tickers")
    
    strategy = MomentumStrategy()
    
    # Generate signals for 2020-2024
    signals = strategy.generate_signals(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    if signals.empty:
        print("❌ No signals generated!")
        return None
    
    # Filter to best signals
    filtered = strategy.filter_best_signals(signals, max_per_day=2)
    
    # Run backtest with CONSERVATIVE risk management
    results = strategy.backtest(
        filtered,
        initial_capital=100000,
        position_pct=0.10,  # 10% per position (smaller for lower DD)
        trailing_stop_pct=0.05,  # 5% trailing stop (tighter)
        max_hold_days=10  # Hold up to 10 days (shorter)
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 MOMENTUM BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Return: {results['total_return_pct']:.2f}%")
    print(f"   CAGR: {results['cagr_pct']:.2f}%")
    print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Avg Return/Trade: {results['avg_return_per_trade']:.2f}%")
    
    passed = (
        results['cagr_pct'] >= 20 and
        results['max_drawdown_pct'] >= -20 and
        results['win_rate_pct'] >= 55
    )
    
    print(f"\n📋 Validation:")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if results['cagr_pct'] < 20:
        print(f"   ⚠️ CAGR {results['cagr_pct']:.1f}% < 20%")
    if results['max_drawdown_pct'] < -20:
        print(f"   ⚠️ Max DD {results['max_drawdown_pct']:.1f}% > 20%")
    if results['win_rate_pct'] < 55:
        print(f"   ⚠️ Win Rate {results['win_rate_pct']:.1f}% < 55%")
    
    # Show top trades
    if 'trades' in results:
        print(f"\n🏆 Top 10 Trades:")
        top_trades = results['trades'].nlargest(10, 'pnl')
        for _, t in top_trades.iterrows():
            print(f"   {t['ticker']}: {t['return_pct']:.1f}% ({t['exit_reason']})")
    
    return results


if __name__ == "__main__":
    results = run_momentum_backtest()
