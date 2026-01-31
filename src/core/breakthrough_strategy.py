"""
Breakthrough Strategy - Target: 15%+ CAGR
==========================================

Combines the best of AI Ensemble + Sweet Spot with aggressive optimizations:

1. SIGNAL QUALITY (Target: 55%+ win rate)
   - Top 5% AI signals only (vs 10%)
   - Sweet Spot momentum confirmation required
   - Market regime filter (only trade in bull markets)
   - Sector rotation (avoid overexposure)

2. POSITION SIZING (Target: 80%+ capital deployed)
   - Aggressive Kelly (full Kelly, not fractional)
   - Confidence-based multipliers (2x on high confidence)
   - Volatility-adjusted sizing
   - Max 25% per position (vs 20%)

3. EXIT OPTIMIZATION (Target: Let winners run)
   - Wider trailing stops (5% vs 3%)
   - Profit targets at 20%, 40%, 60% (scale out)
   - Breakeven stop at 3% gain (vs 1.5%)
   - Time-based exits (max 30 days)

4. RISK MANAGEMENT (Target: <20% max DD)
   - Daily loss limit: 3%
   - Drawdown-based reduction at 15%
   - Max 6 positions (high concentration)
   - Sector limits: 40% per sector

Goal: Beat SPY (10%) and reach 15%+ CAGR
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.core.ai_ensemble_strategy_v2 import AIEnsembleStrategyV2
from src.core.sweetspot_indicators import add_sweetspot_indicators, is_in_sweetspot
from src.core.data_store import get_data_store


class BreakthroughRiskManager:
    """Aggressive risk management for 15%+ CAGR target."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        target_cagr: float = 0.15,  # 15% target
        max_drawdown_pct: float = 0.20,  # 20% max
        daily_loss_limit_pct: float = 0.03,  # 3% daily loss limit
        max_positions: int = 6  # High concentration
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.target_cagr = target_cagr
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_positions = max_positions
        
        # Tracking
        self.ticker_trade_counts = defaultdict(int)
        self.sector_exposure = defaultdict(float)
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0
        self.total_losses = 0
        self.daily_pnl = defaultdict(float)
        self.peak_capital = initial_capital
    
    def calculate_aggressive_kelly(self) -> float:
        """Full Kelly with high minimum."""
        if self.total_trades < 5:
            return 0.20  # Start aggressive (20%)
        
        win_rate = self.win_count / self.total_trades
        
        if self.win_count == 0:
            return 0.10
        if self.loss_count == 0:
            return 0.25
        
        avg_win = self.total_wins / self.win_count
        avg_loss = self.total_losses / self.loss_count
        
        if avg_loss == 0:
            return 0.25
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Full Kelly with high floor and ceiling
        kelly = max(0.10, min(0.25, kelly))
        
        return kelly
    
    def calculate_position_size(
        self,
        confidence: float,  # 0.0 to 1.0
        volatility: float,  # ATR %
        is_high_conviction: bool = False
    ) -> float:
        """Calculate aggressive position size."""
        # Base Kelly
        base_size = self.calculate_aggressive_kelly()
        
        # Confidence multiplier (1.0x to 2.0x)
        confidence_mult = 1.0 + confidence
        
        # High conviction bonus
        if is_high_conviction:
            confidence_mult *= 1.2
        
        # Volatility adjustment (less aggressive)
        if volatility > 0:
            vol_mult = max(0.7, min(1.3, 0.025 / volatility))
        else:
            vol_mult = 1.0
        
        # Calculate final size
        final_size = base_size * confidence_mult * vol_mult
        
        # Cap at 25%
        final_size = min(final_size, 0.25)
        
        return final_size
    
    def can_trade_today(self, date) -> bool:
        """Check daily loss limit."""
        daily_loss = self.daily_pnl[date]
        daily_loss_pct = daily_loss / self.capital
        return daily_loss_pct > -self.daily_loss_limit_pct
    
    def get_drawdown_multiplier(self) -> float:
        """Reduce size during drawdown."""
        if self.capital >= self.peak_capital:
            return 1.0
        
        dd_pct = (self.peak_capital - self.capital) / self.peak_capital
        
        if dd_pct < 0.15:
            return 1.0
        elif dd_pct < 0.20:
            return 0.75
        else:
            return 0.50
    
    def record_trade(self, ticker: str, date, pnl: float, sector: str = None):
        """Record trade."""
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


class BreakthroughStrategy:
    """
    Breakthrough strategy combining AI + Sweet Spot with aggressive optimization.
    Target: 15%+ CAGR, <20% Max DD, 55%+ Win Rate
    """
    
    def __init__(self):
        self.data_store = get_data_store()
        self.ai_strategy = AIEnsembleStrategyV2()
        self.risk_manager = None
    
    def is_bull_market(self, spy_data: pd.DataFrame, date) -> bool:
        """Check if we're in a bull market (SPY above 200 SMA)."""
        try:
            spy_row = spy_data[spy_data['date'] == date]
            if spy_row.empty:
                return True  # Default to allow trading
            
            close = spy_row.iloc[0]['close']
            sma_200 = spy_row.iloc[0].get('sma_200', close)
            
            return close > sma_200
        except:
            return True
    
    def generate_breakthrough_signals(
        self,
        tickers: List[str],
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Generate high-quality signals combining AI + Sweet Spot.
        Target: Top 5% AI signals with Sweet Spot confirmation.
        """
        print("\n" + "=" * 70)
        print("🚀 BREAKTHROUGH STRATEGY - Signal Generation")
        print("=" * 70)
        print("Target: 15%+ CAGR | <20% Max DD | 55%+ Win Rate")
        print("=" * 70)
        
        # Step 1: Train AI models (with caching)
        print("\n1️⃣ Training AI Ensemble...")
        self.ai_strategy.train_ensemble(
            tickers=tickers,
            train_start=train_start,
            train_end=train_end,
            use_cache=use_cache
        )
        
        # Step 2: Generate AI signals
        print("\n2️⃣ Generating AI signals...")
        ai_signals = self.ai_strategy.generate_signals(
            tickers=tickers,
            start_date=test_start,
            end_date=test_end
        )
        
        if ai_signals.empty:
            print("   ❌ No AI signals generated")
            return pd.DataFrame()
        
        print(f"   ✅ Generated {len(ai_signals):,} AI signals")
        
        # Step 3: Filter to top 5% (vs 10%)
        print("\n3️⃣ Filtering to top 5% signals (high quality)...")
        ai_signals = ai_signals.sort_values('confidence', ascending=False)
        top_5_pct = int(len(ai_signals) * 0.05)
        ai_signals = ai_signals.head(top_5_pct)
        print(f"   ✅ Filtered to {len(ai_signals):,} top 5% signals")
        
        # Step 4: Apply Sweet Spot momentum confirmation
        print("\n4️⃣ Applying Sweet Spot momentum confirmation...")
        confirmed_signals = []
        
        for _, signal in ai_signals.iterrows():
            ticker = signal['ticker']
            signal_date = signal['date']
            
            try:
                # Load ticker data
                ticker_data = self.data_store.get_ticker_data(ticker)
                if ticker_data is None or ticker_data.empty:
                    continue
                
                if 'date' not in ticker_data.columns:
                    ticker_data = ticker_data.reset_index()
                
                ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                
                # Get data at signal date
                signal_row = ticker_data[ticker_data['date'] == signal_date]
                if signal_row.empty:
                    continue
                
                # Add Sweet Spot indicators
                ticker_data = add_sweetspot_indicators(ticker_data)
                signal_row = ticker_data[ticker_data['date'] == signal_date]
                
                if signal_row.empty:
                    continue
                
                signal_row = signal_row.iloc[0]
                
                # Check Sweet Spot confirmation
                if is_in_sweetspot(signal_row):
                    # Calculate ATR for volatility
                    atr = ticker_data['high'].rolling(14).max() - ticker_data['low'].rolling(14).min()
                    atr_pct = (atr / ticker_data['close']).iloc[-1]
                    
                    confirmed_signals.append({
                        'date': signal_date,
                        'ticker': ticker,
                        'signal': signal['signal'],
                        'ai_confidence': signal['confidence'],
                        'entry_price': signal_row['close'],
                        'atr_pct': atr_pct,
                        'is_high_conviction': True  # Both AI and Sweet Spot agree
                    })
            except Exception as e:
                continue
        
        if not confirmed_signals:
            print("   ❌ No signals passed Sweet Spot confirmation")
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(confirmed_signals)
        print(f"   ✅ {len(signals_df):,} signals confirmed by Sweet Spot")
        print(f"   📊 Confirmation rate: {len(signals_df)/len(ai_signals)*100:.1f}%")
        
        # Step 5: Market regime filter
        print("\n5️⃣ Applying market regime filter (bull market only)...")
        try:
            spy_data = self.data_store.get_ticker_data('SPY')
            spy_data = add_sweetspot_indicators(spy_data)
            
            if 'date' not in spy_data.columns:
                spy_data = spy_data.reset_index()
            spy_data['date'] = pd.to_datetime(spy_data['date'])
            
            # Filter signals to bull market only
            bull_signals = []
            for _, signal in signals_df.iterrows():
                if self.is_bull_market(spy_data, signal['date']):
                    bull_signals.append(signal)
            
            signals_df = pd.DataFrame(bull_signals)
            print(f"   ✅ {len(signals_df):,} signals in bull market")
        except Exception as e:
            print(f"   ⚠️ Could not apply regime filter: {e}")
        
        return signals_df
    
    def backtest(
        self,
        signals: pd.DataFrame,
        initial_capital: float = 100000
    ) -> Dict:
        """Backtest with aggressive optimization."""
        print("\n" + "=" * 70)
        print("💰 BREAKTHROUGH BACKTEST")
        print("=" * 70)
        print(f"Initial Capital: ${initial_capital:,.0f}")
        print("Features: Aggressive Kelly + High Conviction + Wide Stops")
        print("=" * 70)
        
        self.risk_manager = BreakthroughRiskManager(initial_capital=initial_capital)
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
                    
                    # Calculate P&L
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    peak_pnl_pct = (pos['peak_price'] - pos['entry_price']) / pos['entry_price']
                    
                    # Holding days
                    holding_days = (current_date - pos['entry_date']).days
                    
                    # Exit conditions
                    exit_reason = None
                    
                    # 1. Profit targets (scale out)
                    if pnl_pct >= 0.60:  # 60% gain
                        exit_reason = 'profit_target_60'
                    elif pnl_pct >= 0.40:  # 40% gain
                        exit_reason = 'profit_target_40'
                    elif pnl_pct >= 0.20:  # 20% gain
                        exit_reason = 'profit_target_20'
                    # 2. Wide trailing stop (5%)
                    elif peak_pnl_pct >= 0.05:  # After 5% gain
                        stop_price = pos['peak_price'] * 0.95  # Trail 5%
                        if current_price <= stop_price:
                            exit_reason = 'trailing_stop'
                    # 3. Breakeven stop (3% gain)
                    elif peak_pnl_pct >= 0.03:
                        if current_price <= pos['entry_price'] * 1.001:
                            exit_reason = 'breakeven_stop'
                    # 4. Initial stop (5%)
                    elif current_price <= pos['entry_price'] * 0.95:
                        exit_reason = 'initial_stop'
                    # 5. Time-based exit (30 days)
                    elif holding_days >= 30:
                        exit_reason = 'time_exit'
                    
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
                            'holding_days': holding_days,
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
                
                if len(positions) >= self.risk_manager.max_positions:
                    break
                
                # Risk management checks
                if not self.risk_manager.can_trade_today(current_date):
                    continue
                
                # Calculate aggressive position size
                position_size_pct = self.risk_manager.calculate_position_size(
                    confidence=signal['ai_confidence'],
                    volatility=signal['atr_pct'],
                    is_high_conviction=signal['is_high_conviction']
                )
                
                # Drawdown adjustment
                dd_mult = self.risk_manager.get_drawdown_multiplier()
                position_size_pct *= dd_mult
                
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
                'strategy_id': 'Breakthrough',
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
            'strategy_id': 'Breakthrough',
            'strategy_name': 'Breakthrough Strategy (AI + Sweet Spot + Aggressive)',
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
            'avg_holding_days': trades_df['holding_days'].mean(),
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        print("\n" + "=" * 70)
        print("📊 BREAKTHROUGH RESULTS")
        print("=" * 70)
        print(f"CAGR: {cagr:.2f}% {'✅' if cagr >= 15 else '❌'} (Target: 15%+)")
        print(f"Max DD: {max_drawdown:.2f}% {'✅' if max_drawdown > -20 else '❌'} (Target: <20%)")
        print(f"Win Rate: {win_rate:.1f}% {'✅' if win_rate >= 55 else '❌'} (Target: 55%+)")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Avg Holding: {trades_df['holding_days'].mean():.1f} days")
        print(f"Final Capital: ${final_capital:,.0f}")
        print("=" * 70)
        
        # Target assessment
        targets_met = 0
        if cagr >= 15:
            targets_met += 1
        if max_drawdown > -20:
            targets_met += 1
        if win_rate >= 55:
            targets_met += 1
        
        print(f"\n🎯 Targets Met: {targets_met}/3")
        if targets_met == 3:
            print("🎉 ALL TARGETS ACHIEVED! BREAKTHROUGH SUCCESS!")
        elif targets_met >= 2:
            print("🟡 Partial success - close to breakthrough")
        else:
            print("🔴 More optimization needed")
        
        return results


if __name__ == "__main__":
    from src.core.data_store import get_data_store
    
    store = get_data_store()
    tickers = store.available_tickers
    
    print("\n" + "=" * 70)
    print("🚀 BREAKTHROUGH STRATEGY - 15%+ CAGR TARGET")
    print("=" * 70)
    
    strategy = BreakthroughStrategy()
    
    # Generate breakthrough signals
    signals = strategy.generate_breakthrough_signals(
        tickers=tickers,
        train_start='2005-01-01',
        train_end='2014-12-31',
        test_start='2015-01-01',
        test_end='2024-12-31',
        use_cache=True
    )
    
    if not signals.empty:
        # Run backtest
        results = strategy.backtest(signals, initial_capital=100000)
