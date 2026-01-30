"""
VT Sweet Spot Strategy - OPTIMIZED with Advanced Risk Management
==================================================================

Applies AI Ensemble V2's risk management to Sweet Spot signals:
1. Kelly Criterion position sizing
2. Volatility-adjusted sizing (ATR-based)
3. Confidence-based sizing (stochastic strength)
4. Advanced trailing stops (breakeven + trailing)
5. Drawdown-based position reduction
6. Daily loss limits
7. Max ticker exposure limits

Goal: Improve Sweet Spot CAGR from 1.82% to 10%+ while maintaining low drawdown
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.core.sweetspot_indicators import add_sweetspot_indicators, is_in_sweetspot, should_exit_sweetspot
from src.core.data_store import get_data_store


class SweetSpotRiskManager:
    """
    Advanced Risk Management for Sweet Spot Strategy
    (Adapted from AI Ensemble V2)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        kelly_fraction: float = 1.0,  # Full Kelly (more aggressive)
        target_volatility: float = 0.025,  # 2.5% daily volatility target
        max_ticker_exposure_pct: float = 0.25,  # 25% max per ticker
        daily_loss_limit_pct: float = 0.03,  # 3% daily loss limit
        drawdown_threshold_pct: float = 0.15  # Reduce size at 15% DD (less restrictive)
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.target_volatility = target_volatility
        self.max_ticker_exposure_pct = max_ticker_exposure_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.drawdown_threshold_pct = drawdown_threshold_pct
        
        # Tracking
        self.ticker_trade_counts = defaultdict(int)
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0
        self.total_losses = 0
        self.daily_pnl = defaultdict(float)
        self.peak_capital = initial_capital
    
    def calculate_kelly_size(self) -> float:
        """Calculate Kelly Criterion position size."""
        if self.total_trades < 10:
            return 0.12  # Start more aggressive (12% vs 5%)
        
        win_rate = self.win_count / self.total_trades
        
        if self.win_count == 0:
            return 0.08
        if self.loss_count == 0:
            return 0.15
        
        avg_win = self.total_wins / self.win_count
        avg_loss = self.total_losses / self.loss_count
        
        if avg_loss == 0:
            return 0.15
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly = max(0.05, min(0.25, kelly * self.kelly_fraction))  # Higher caps
        
        return kelly
    
    def calculate_volatility_adjusted_size(
        self,
        base_size_pct: float,
        ticker_atr_pct: float
    ) -> float:
        """Adjust position size based on ticker volatility."""
        if ticker_atr_pct <= 0:
            return base_size_pct
        
        vol_ratio = self.target_volatility / ticker_atr_pct
        adjusted = base_size_pct * max(0.5, min(2.0, vol_ratio))
        
        return adjusted
    
    def calculate_confidence_size(
        self,
        base_size_pct: float,
        stoch_strength: float
    ) -> float:
        """
        Adjust size based on stochastic strength.
        stoch_strength: 0.0 to 1.0 (how far above 80 the stochastics are)
        """
        # Scale from 0.5x to 1.5x based on strength
        multiplier = 0.5 + stoch_strength
        return base_size_pct * multiplier
    
    def calculate_drawdown_multiplier(self) -> float:
        """Reduce position size during drawdown."""
        if self.capital >= self.peak_capital:
            return 1.0
        
        drawdown_pct = (self.peak_capital - self.capital) / self.peak_capital
        
        if drawdown_pct < self.drawdown_threshold_pct:
            return 1.0
        elif drawdown_pct < 0.15:
            return 0.75
        elif drawdown_pct < 0.20:
            return 0.50
        else:
            return 0.25
    
    def calculate_position_size(
        self,
        ticker_atr_pct: float,
        stoch_strength: float
    ) -> float:
        """Calculate final position size with all adjustments."""
        # Base Kelly size
        kelly_size = self.calculate_kelly_size()
        
        # Volatility adjustment
        vol_adjusted = self.calculate_volatility_adjusted_size(kelly_size, ticker_atr_pct)
        
        # Confidence adjustment
        conf_adjusted = self.calculate_confidence_size(vol_adjusted, stoch_strength)
        
        # Drawdown adjustment
        dd_multiplier = self.calculate_drawdown_multiplier()
        final_size = conf_adjusted * dd_multiplier
        
        # Cap at 20%
        final_size = min(final_size, 0.20)
        
        return final_size
    
    def can_trade_ticker(self, ticker: str) -> bool:
        """Check if ticker hasn't exceeded max exposure limit."""
        if self.total_trades == 0:
            return True
        
        ticker_pct = self.ticker_trade_counts[ticker] / self.total_trades
        return ticker_pct < self.max_ticker_exposure_pct
    
    def can_trade_today(self, date) -> bool:
        """Check if daily loss limit hasn't been hit."""
        daily_loss = self.daily_pnl[date]
        daily_loss_pct = daily_loss / self.capital
        return daily_loss_pct > -self.daily_loss_limit_pct
    
    def record_trade(self, ticker: str, date, pnl: float):
        """Record trade for tracking."""
        self.ticker_trade_counts[ticker] += 1
        self.total_trades += 1
        self.daily_pnl[date] += pnl
        self.capital += pnl
        
        if pnl > 0:
            self.win_count += 1
            self.total_wins += pnl
        else:
            self.loss_count += 1
            self.total_losses += abs(pnl)
        
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital


class AdvancedTrailingStop:
    """Advanced trailing stop with breakeven and trailing modes."""
    
    def __init__(
        self,
        initial_stop_pct: float = 0.02,
        breakeven_trigger_pct: float = 0.015,
        trail_trigger_pct: float = 0.03,
        trail_distance_pct: float = 0.02
    ):
        self.initial_stop_pct = initial_stop_pct
        self.breakeven_trigger_pct = breakeven_trigger_pct
        self.trail_trigger_pct = trail_trigger_pct
        self.trail_distance_pct = trail_distance_pct
    
    def get_stop_price(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float
    ) -> float:
        """Calculate current stop price."""
        pnl_pct = (current_price - entry_price) / entry_price
        peak_pnl_pct = (peak_price - entry_price) / entry_price
        
        # Trail mode: after trail_trigger_pct gain
        if peak_pnl_pct >= self.trail_trigger_pct:
            stop_price = peak_price * (1 - self.trail_distance_pct)
        # Breakeven mode: after breakeven_trigger_pct gain
        elif peak_pnl_pct >= self.breakeven_trigger_pct:
            stop_price = entry_price * 1.001
        # Initial stop
        else:
            stop_price = entry_price * (1 - self.initial_stop_pct)
        
        return stop_price
    
    def is_stopped_out(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float
    ) -> bool:
        """Check if position should be stopped out."""
        stop_price = self.get_stop_price(entry_price, current_price, peak_price)
        return current_price <= stop_price


class VTSweetSpotOptimized:
    """
    VT Sweet Spot Strategy with Advanced Risk Management.
    Same signals, but with AI Ensemble V2's risk management.
    """
    
    def __init__(self):
        self.data_store = get_data_store()
        self.risk_manager = None
        self.trailing_stop = AdvancedTrailingStop()
    
    def calculate_stoch_strength(self, row: pd.Series) -> float:
        """
        Calculate stochastic strength (0.0 to 1.0).
        Higher when stochastics are well above 80.
        """
        daily_k = row['stoch_daily_k']
        daily_d = row['stoch_daily_d']
        weekly_k = row['stoch_weekly_k']
        weekly_d = row['stoch_weekly_d']
        
        # Average of how far above 80 each indicator is
        daily_strength = min((daily_k - 80) / 20, 1.0)
        weekly_strength = min((weekly_k - 80) / 20, 1.0)
        
        avg_strength = (daily_strength + weekly_strength) / 2
        return max(0.0, min(1.0, avg_strength))
    
    def generate_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate Sweet Spot signals with confidence scores."""
        print(f"\n📊 Generating VT Sweet Spot signals (OPTIMIZED)...")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Tickers: {len(tickers)}")
        
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
                
                # Add Sweet Spot indicators
                data = add_sweetspot_indicators(data)
                
                # Calculate ATR for volatility adjustment
                data['atr'] = data['high'].rolling(14).max() - data['low'].rolling(14).min()
                data['atr_pct'] = data['atr'] / data['close']
                
                # Generate signals
                for idx, row in data.iterrows():
                    if is_in_sweetspot(row):
                        stoch_strength = self.calculate_stoch_strength(row)
                        
                        all_signals.append({
                            'date': row['date'],
                            'ticker': ticker,
                            'signal': 1,
                            'entry_price': row['close'],
                            'confidence': stoch_strength,
                            'atr_pct': row['atr_pct']
                        })
                
                processed += 1
                if processed % 20 == 0:
                    print(f"   📊 Processed {processed}/{len(tickers)} tickers...")
                    
            except Exception as e:
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
        initial_capital: float = 100000
    ) -> Dict:
        """Backtest with advanced risk management."""
        print(f"\n💰 Running OPTIMIZED Sweet Spot backtest...")
        print(f"   Initial Capital: ${initial_capital:,.0f}")
        print(f"   Features: Kelly + Vol-Adj + Confidence + Trailing + DD-Reduction")
        
        self.risk_manager = SweetSpotRiskManager(initial_capital=initial_capital)
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
                    
                    # 1. Sweet Spot exit
                    if should_exit_sweetspot(current_row_with_indicators):
                        exit_reason = 'sweetspot_exit'
                    # 2. Advanced trailing stop
                    elif self.trailing_stop.is_stopped_out(
                        pos['entry_price'],
                        current_price,
                        pos['peak_price']
                    ):
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
                        
                        self.risk_manager.record_trade(ticker, current_date, pnl)
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
                
                if len(positions) >= 10:
                    break
                
                # Risk management checks
                if not self.risk_manager.can_trade_ticker(ticker):
                    continue
                
                if not self.risk_manager.can_trade_today(current_date):
                    continue
                
                # Calculate position size
                position_size_pct = self.risk_manager.calculate_position_size(
                    ticker_atr_pct=signal['atr_pct'],
                    stoch_strength=signal['confidence']
                )
                
                position_value = capital * position_size_pct
                shares = position_value / signal['entry_price']
                
                if position_value > capital:
                    continue
                
                capital -= position_value
                
                positions[ticker] = {
                    'entry_date': current_date,
                    'entry_price': signal['entry_price'],
                    'shares': shares,
                    'peak_price': signal['entry_price'],
                    'size_pct': position_size_pct
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
        
        # Close remaining positions
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
                'strategy_id': 'VT_SweetSpot_Optimized',
                'total_trades': 0,
                'error': 'No trades executed'
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
            'strategy_id': 'VT_SweetSpot_Optimized',
            'strategy_name': 'VT Sweet Spot Optimized (Advanced Risk Mgmt)',
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
        
        print(f"\n   📊 Risk Manager Stats:")
        print(f"      Kelly Fraction: {self.risk_manager.calculate_kelly_size()*100:.1f}%")
        print(f"      Unique Tickers: {len(self.risk_manager.ticker_trade_counts)}")
        
        print(f"\n   ✅ Backtest complete!")
        print(f"   CAGR: {cagr:.2f}%")
        print(f"   Max DD: {max_drawdown:.2f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total Trades: {len(trades_df)}")
        
        return results


if __name__ == "__main__":
    from src.core.data_store import get_data_store
    
    store = get_data_store()
    tickers = store.available_tickers
    
    print("\n" + "=" * 70)
    print("📊 VT SWEET SPOT OPTIMIZED (Advanced Risk Management)")
    print("=" * 70)
    
    strategy = VTSweetSpotOptimized()
    
    signals = strategy.generate_signals(tickers, '2015-01-01', '2024-12-31')
    
    if not signals.empty:
        results = strategy.backtest(signals, initial_capital=100000)
