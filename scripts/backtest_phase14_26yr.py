#!/usr/bin/env python3
"""
Phase 14 - 26-Year Backtest with Triple-Barrier + Short Selling
===============================================================
Comprehensive backtest of the new 3-class TB models on full historical data.
Tests both long and short positions with proper risk management.

Metrics to track:
- CAGR (Compound Annual Growth Rate)
- Max Drawdown
- Sharpe Ratio
- Win Rate (overall, long, short)
- Profit Factor
- Average Trade P&L
- Number of trades (long vs short)
"""
import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_models import EnsemblePredictor
from core.portfolio_manager import PortfolioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'backtest_phase14.log', mode='w', encoding='ascii'),
    ]
)
logger = logging.getLogger('BacktestPhase14')

# Backtest configuration
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'reports'
START_DATE = '1998-01-01'  # Start with 2 years of data for features
END_DATE = '2026-04-27'    # Through today (28+ years of data)
INITIAL_CAPITAL = 100000   # $100K starting capital
MAX_POSITIONS = 15          # Max concurrent positions
POSITION_SIZE_USD = 5000   # $5K per position

# Strategy parameters (matching Phase 14 training)
STOP_LOSS_ATR_MULT = 2.5
STOP_LOSS_MIN_PCT = 0.08
STOP_LOSS_MAX_PCT = 0.20
TAKE_PROFIT_PCT = 0.40
MAX_HOLD_DAYS = 25
CONFIDENCE_THRESHOLD = 0.65


class BacktestEngine:
    """Backtest engine for Phase 14 TB models"""
    
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.portfolio = PortfolioManager(PROJECT_ROOT)
        self.results = []
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.positions_history = []
        
        # Trade statistics
        self.trades = []
        self.current_positions = {}
        
        # Load all tickers
        self.tickers = self._load_ticker_list()
        self.data_cache = {}
        
        logger.info(f'[INIT] Loaded {len(self.tickers)} tickers for backtest')
    
    def _load_ticker_list(self):
        """Get list of all available tickers"""
        parquet_files = sorted(DATA_DIR.glob('*.parquet'))
        tickers = [f.stem for f in parquet_files]
        return tickers
    
    def _load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Load and prepare ticker data"""
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        
        try:
            df = pd.read_parquet(DATA_DIR / f'{ticker}.parquet')
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Filter date range
            start = pd.to_datetime(START_DATE)
            end = pd.to_datetime(END_DATE)
            df = df[(df.index >= start) & (df.index <= end)]
            
            if len(df) < 252:  # Need at least 1 year
                return None
            
            self.data_cache[ticker] = df
            return df
            
        except Exception as e:
            logger.warning(f'[DATA] Failed to load {ticker}: {e}')
            return None
    
    def _generate_signals_for_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """Generate AI signals for all tickers on given date"""
        signals = []
        
        for ticker in self.tickers:
            df = self._load_ticker_data(ticker)
            if df is None:
                continue
            
            # Check if we have data for this date
            if date not in df.index:
                continue
            
            # Get historical data up to this date
            hist_df = df[df.index <= date]
            if len(hist_df) < 50:  # Need enough history for features
                continue
            
            try:
                # Generate features (same as training)
                from core.feature_engineer import FeatureEngineer
                fe = FeatureEngineer(use_advanced_features=True, verbose=False)
                feats, _ = fe.create_features(hist_df, target_type='direction')
                
                if feats.empty:
                    continue
                
                # Get latest features
                latest_features = feats.iloc[-1:]
                
                # Get AI prediction
                signal, confidence, details = self.ensemble.predict(latest_features, threshold=CONFIDENCE_THRESHOLD)
                
                if signal != 'HOLD':  # Only include actionable signals
                    current_price = float(df.loc[date, 'adjClose'] if 'adjClose' in df.columns else df.loc[date, 'close'])
                    
                    signals.append({
                        'ticker': ticker,
                        'signal': signal,
                        'confidence': confidence,
                        'price': current_price,
                        'details': details
                    })
                    
            except Exception as e:
                logger.debug(f'[SIGNAL] Failed for {ticker} on {date}: {e}')
                continue
        
        if signals:
            signals_df = pd.DataFrame(signals)
            # Sort by confidence
            signals_df = signals_df.sort_values('confidence', ascending=False)
            return signals_df
        
        return pd.DataFrame()
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate ATR for stop loss"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        prev_close = np.concatenate([[close[0]], close[:-1]])
        
        tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
        atr = tr[-window:].mean()
        return float(atr)
    
    def _calculate_position_size(self, price: float, atr: float) -> int:
        """Calculate position size based on risk"""
        # Risk 2% of position size
        risk_pct = max(STOP_LOSS_MIN_PCT, min(STOP_LOSS_MAX_PCT, (atr * STOP_LOSS_ATR_MULT) / price))
        position_value = POSITION_SIZE_USD
        shares = int(position_value / price)
        return shares
    
    def _check_exit_conditions(self, position: dict, current_price: float, current_date: pd.Timestamp) -> tuple:
        """Check if position should be exited"""
        ticker = position['ticker']
        entry_price = position['entry_price']
        action = position['action']
        entry_date = position['entry_date']
        
        # Calculate days held
        days_held = (current_date - entry_date).days
        
        # Calculate P&L
        if action == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        elif action == 'SELL_SHORT':
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            return False, 'Unknown action'
        
        # Check exit conditions
        exit_reason = None
        
        # Take profit
        if action == 'BUY' and pnl_pct >= TAKE_PROFIT_PCT * 100:
            exit_reason = 'Take Profit'
        elif action == 'SELL_SHORT' and pnl_pct >= TAKE_PROFIT_PCT * 100:
            exit_reason = 'Take Profit'
        
        # Stop loss (would need ATR from entry date, simplified here)
        if pnl_pct <= -STOP_LOSS_MIN_PCT * 100:
            exit_reason = 'Stop Loss'
        
        # Timeout
        if days_held >= MAX_HOLD_DAYS:
            exit_reason = 'Timeout'
        
        return (exit_reason is not None), exit_reason
    
    def run_backtest(self):
        """Run the full 26-year backtest"""
        logger.info(f'[BACKTEST] Starting {START_DATE} to {END_DATE}')
        
        # Generate date series (monthly rebalancing)
        dates = pd.date_range(START_DATE, END_DATE, freq='M')
        
        for date in dates:
            logger.info(f'[BACKTEST] Processing {date.strftime("%Y-%m")}')
            
            # Step 1: Check exits for current positions
            positions_to_close = []
            for ticker, pos in self.current_positions.items():
                df = self._load_ticker_data(ticker)
                if df is None or date not in df.index:
                    continue
                
                current_price = float(df.loc[date, 'adjClose'] if 'adjClose' in df.columns else df.loc[date, 'close'])
                
                should_exit, exit_reason = self._check_exit_conditions(pos, current_price, date)
                
                if should_exit:
                    positions_to_close.append((ticker, pos, current_price, exit_reason))
            
            # Close positions
            for ticker, pos, exit_price, exit_reason in positions_to_close:
                # Calculate final P&L
                if pos['action'] == 'BUY':
                    pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                else:  # SELL_SHORT
                    pnl_pct = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100
                
                pnl_usd = pnl_pct * pos['quantity'] * pos['entry_price'] / 100
                
                # Record trade
                trade = {
                    'ticker': ticker,
                    'action': pos['action'],
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'quantity': pos['quantity'],
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'days_held': (date - pos['entry_date']).days
                }
                self.trades.append(trade)
                
                # Remove from positions
                del self.current_positions[ticker]
                
                logger.info(f"[TRADE] Closed {ticker} {pos['action']}: PnL {pnl_pct:.2f}% (${pnl_usd:.2f}) - {exit_reason}")
            
            # Step 2: Generate new signals
            signals_df = self._generate_signals_for_date(date)
            
            if signals_df.empty:
                # Update equity curve
                self._update_equity_curve(date)
                continue
            
            # Step 3: Open new positions (up to max)
            available_slots = MAX_POSITIONS - len(self.current_positions)
            if available_slots > 0:
                new_signals = signals_df.head(available_slots)
                
                for _, signal in new_signals.iterrows():
                    ticker = signal['ticker']
                    action = signal['signal']
                    price = signal['price']
                    
                    # Calculate position size
                    df = self._load_ticker_data(ticker)
                    if df is None:
                        continue
                    
                    atr = self._calculate_atr(df[df.index <= date])
                    quantity = self._calculate_position_size(price, atr)
                    
                    # Adjust for short positions
                    if action == 'SELL_SHORT':
                        quantity = -quantity
                    
                    # Open position
                    self.current_positions[ticker] = {
                        'ticker': ticker,
                        'action': action,
                        'entry_date': date,
                        'entry_price': price,
                        'quantity': quantity
                    }
                    
                    logger.info(f"[POSITION] Opened {ticker} {action}: {quantity} shares @ ${price:.2f}")
            
            # Step 4: Update equity curve
            self._update_equity_curve(date)
        
        # Close any remaining positions at end
        self._close_remaining_positions()
        
        # Calculate final metrics
        self._calculate_metrics()
        
        logger.info('[BACKTEST] Completed')
    
    def _update_equity_curve(self, date: pd.Timestamp):
        """Update equity curve with current positions value"""
        total_value = INITIAL_CAPITAL
        
        # Add realized P&L from closed trades
        for trade in self.trades:
            if trade['exit_date'] <= date:
                total_value += trade['pnl_usd']
        
        # Add unrealized P&L from open positions
        for ticker, pos in self.current_positions.items():
            df = self._load_ticker_data(ticker)
            if df is None or date not in df.index:
                continue
            
            current_price = float(df.loc[date, 'adjClose'] if 'adjClose' in df.columns else df.loc[date, 'close'])
            
            if pos['action'] == 'BUY':
                unrealized_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            else:  # SELL_SHORT
                unrealized_pct = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
            
            unrealized_usd = unrealized_pct * pos['quantity'] * pos['entry_price'] / 100
            total_value += unrealized_usd
        
        self.equity_curve.append({
            'date': date,
            'equity': total_value,
            'n_positions': len(self.current_positions)
        })
    
    def _close_remaining_positions(self):
        """Close all open positions at end of backtest"""
        end_date = pd.to_datetime(END_DATE)
        
        for ticker, pos in list(self.current_positions.items()):
            df = self._load_ticker_data(ticker)
            if df is None:
                continue
            
            # Get last available price
            available_dates = df.index[df.index <= end_date]
            if len(available_dates) == 0:
                continue
            
            last_date = available_dates[-1]
            last_price = float(df.loc[last_date, 'adjClose'] if 'adjClose' in df.columns else df.loc[last_date, 'close'])
            
            # Calculate P&L
            if pos['action'] == 'BUY':
                pnl_pct = ((last_price - pos['entry_price']) / pos['entry_price']) * 100
            else:  # SELL_SHORT
                pnl_pct = ((pos['entry_price'] - last_price) / pos['entry_price']) * 100
            
            pnl_usd = pnl_pct * pos['quantity'] * pos['entry_price'] / 100
            
            # Record trade
            trade = {
                'ticker': ticker,
                'action': pos['action'],
                'entry_date': pos['entry_date'],
                'exit_date': last_date,
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'quantity': pos['quantity'],
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'exit_reason': 'End of Backtest',
                'days_held': (last_date - pos['entry_date']).days
            }
            self.trades.append(trade)
            
            del self.current_positions[ticker]
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            logger.warning('[METRICS] No trades to analyze')
            return
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades * 100
        
        # Long vs Short performance
        long_trades = trades_df[trades_df['action'] == 'BUY']
        short_trades = trades_df[trades_df['action'] == 'SELL_SHORT']
        
        long_win_rate = len(long_trades[long_trades['pnl_pct'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl_pct'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl_usd'].sum()
        avg_trade_pnl = trades_df['pnl_usd'].mean()
        avg_win = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean()
        avg_loss = trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].mean()
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Time-based metrics
        avg_days_held = trades_df['days_held'].mean()
        
        # Equity curve metrics
        if len(equity_df) > 1:
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df = equity_df.sort_values('date')
            
            # CAGR
            start_equity = equity_df.iloc[0]['equity']
            end_equity = equity_df.iloc[-1]['equity']
            years = (equity_df.iloc[-1]['date'] - equity_df.iloc[0]['date']).days / 365.25
            cagr = ((end_equity / start_equity) ** (1/years) - 1) * 100
            
            # Max drawdown
            equity_df['peak'] = equity_df['equity'].expanding().max()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            max_drawdown = equity_df['drawdown'].min()
            
            # Sharpe ratio (simplified, assuming 0% risk-free rate)
            equity_df['returns'] = equity_df['equity'].pct_change()
            sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if equity_df['returns'].std() > 0 else 0
        else:
            cagr = max_drawdown = sharpe_ratio = 0
        
        # Store results
        self.results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_days_held': avg_days_held,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'n_long_trades': len(long_trades),
            'n_short_trades': len(short_trades),
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': INITIAL_CAPITAL + total_pnl
        }
        
        # Log results
        logger.info('=== BACKTEST RESULTS ===')
        logger.info(f'Total Trades: {total_trades} (Long: {len(long_trades)}, Short: {len(short_trades)})')
        logger.info(f'Win Rate: {win_rate:.1f}% (Long: {long_win_rate:.1f}%, Short: {short_win_rate:.1f}%)')
        logger.info(f'Total P&L: ${total_pnl:,.2f}')
        logger.info(f'CAGR: {cagr:.2f}%')
        logger.info(f'Max Drawdown: {max_drawdown:.2f}%')
        logger.info(f'Sharpe Ratio: {sharpe_ratio:.2f}')
        logger.info(f'Profit Factor: {profit_factor:.2f}')
        logger.info(f'Avg Trade: ${avg_trade_pnl:.2f}')
        logger.info(f'Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}')
    
    def save_results(self):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = RESULTS_DIR / f'backtest_phase14_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = RESULTS_DIR / f'backtest_phase14_trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = RESULTS_DIR / f'backtest_phase14_equity_{timestamp}.csv'
            equity_df.to_csv(equity_file, index=False)
        
        logger.info(f'[SAVE] Results saved to {summary_file}')


def main():
    """Run the backtest"""
    logger.info('=== Phase 14 26-Year Backtest ===')
    
    engine = BacktestEngine()
    engine.run_backtest()
    engine.save_results()
    
    logger.info('[DONE] Backtest completed successfully')


if __name__ == '__main__':
    main()
