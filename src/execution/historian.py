"""
NeuralTrader Trade Historian - Endgame Architecture
==================================================

Manages comprehensive trade logging with performance-optimized buffering.
Follows NeuralTrader Constitution v5.2 strictly.

Features:
- Memory-safe buffered CSV writing (50-trade batches)
- Complete trade lifecycle tracking
- Performance metrics calculation
- 26-year simulation support
- No silent failures - comprehensive error handling

Usage:
    from src.execution.historian import TradeHistorian
    
    historian = TradeHistorian()
    historian.log_trade(date, ticker, action, price, quantity, ai_score, exit_reason, pnl_realized)
    historian.finalize()  # Flush remaining buffer and calculate metrics
"""

import os
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TradeHistorian:
    """
    Trade Historian - Manages full backtest trade logging with performance optimization
    """
    
    def __init__(self, buffer_size: int = 50):
        """
        Initialize Trade Historian
        
        Args:
            buffer_size: Number of trades to buffer before writing to CSV
        """
        self.buffer_size = buffer_size
        self.trade_buffer: List[Dict] = []
        self.total_trades = 0
        self.starting_capital = 100000.0
        self.current_capital = 100000.0
        self.peak_capital = 100000.0
        self.all_trades: List[Dict] = []
        
        # Performance tracking
        self.gains = []
        self.losses = []
        self.equity_curve: List[Tuple[str, float]] = []
        
        # Setup output directory
        self.reports_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.csv_file = os.path.join(self.reports_dir, 'full_backtest_trades.csv')
        
        # Initialize CSV with headers
        self._initialize_csv()
        
        logger.info(f"[HISTORIAN] Trade Historian initialized - Buffer size: {buffer_size}")
        logger.info(f"[HISTORIAN] Output file: {self.csv_file}")
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers"""
        try:
            file_exists = os.path.exists(self.csv_file)
            
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    headers = [
                        'Date', 'Ticker', 'Action', 'Price', 'Quantity', 
                        'AI_Score', 'Exit_Reason', 'PnL_Realized'
                    ]
                    writer.writerow(headers)
                    logger.info("[HISTORIAN] CSV file initialized with headers")
                else:
                    logger.info("[HISTORIAN] Existing CSV file found - appending")
                    
        except Exception as e:
            logger.error(f"[HISTORIAN] Failed to initialize CSV: {e}")
            raise
    
    def log_trade(self, date: str, ticker: str, action: str, price: float, 
                  quantity: int, ai_score: float = None, exit_reason: str = None, 
                  pnl_realized: float = None) -> None:
        """
        Log a trade to the buffer
        
        Args:
            date: Trade date (YYYY-MM-DD)
            ticker: Stock ticker symbol
            action: 'BUY' or 'SELL'
            price: Trade price
            quantity: Number of shares
            ai_score: AI confidence score (optional)
            exit_reason: Reason for exit (optional)
            pnl_realized: Realized P&L for the trade (optional)
        """
        try:
            trade = {
                'Date': date,
                'Ticker': ticker,
                'Action': action,
                'Price': price,
                'Quantity': quantity,
                'AI_Score': ai_score or '',
                'Exit_Reason': exit_reason or '',
                'PnL_Realized': pnl_realized or 0.0
            }
            
            # Add to buffer
            self.trade_buffer.append(trade)
            self.all_trades.append(trade)
            self.total_trades += 1
            
            # Update capital and track performance
            if pnl_realized is not None:
                self.current_capital += pnl_realized
                
                if pnl_realized > 0:
                    self.gains.append(pnl_realized)
                elif pnl_realized < 0:
                    self.losses.append(abs(pnl_realized))
            
            # Track peak capital
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            # Add to equity curve
            self.equity_curve.append((date, self.current_capital))
            
            # Write buffer if full
            if len(self.trade_buffer) >= self.buffer_size:
                self._flush_buffer()
            
            logger.debug(f"[HISTORIAN] Trade logged: {action} {quantity} {ticker} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"[HISTORIAN] Failed to log trade: {e}")
            raise
    
    def _flush_buffer(self) -> None:
        """Flush trade buffer to CSV file"""
        if not self.trade_buffer:
            return
        
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                for trade in self.trade_buffer:
                    writer.writerow([
                        trade['Date'], trade['Ticker'], trade['Action'],
                        trade['Price'], trade['Quantity'], trade['AI_Score'],
                        trade['Exit_Reason'], trade['PnL_Realized']
                    ])
            
            logger.info(f"[HISTORIAN] Flushed {len(self.trade_buffer)} trades to CSV")
            self.trade_buffer.clear()
            
        except Exception as e:
            logger.error(f"[HISTORIAN] Failed to flush buffer: {e}")
            raise
    
    def finalize(self) -> Dict:
        """
        Finalize trading session - flush buffer and calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Flush remaining trades
            self._flush_buffer()
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            
            logger.info(f"[HISTORIAN] Finalized - Total trades: {self.total_trades}")
            logger.info(f"[HISTORIAN] Final capital: ${self.current_capital:,.2f}")
            logger.info(f"[HISTORIAN] CAGR: {metrics['cagr']:.2%}")
            logger.info(f"[HISTORIAN] Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"[HISTORIAN] Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"[HISTORIAN] Profit Factor: {metrics['profit_factor']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"[HISTORIAN] Failed to finalize: {e}")
            raise
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            years = len(self.equity_curve) / 252  # Trading days per year
            final_value = self.current_capital
            start_value = self.starting_capital
            
            # CAGR
            if years > 0 and final_value > 0:
                cagr = ((final_value / start_value) ** (1 / years)) - 1
            else:
                cagr = 0.0
            
            # Maximum Drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Win Rate
            winning_trades = len([g for g in self.gains if g > 0])
            total_closed_trades = len(self.gains) + len(self.losses)
            win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0.0
            
            # Profit Factor
            total_gains = sum(self.gains)
            total_losses = sum(self.losses)
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Additional metrics
            total_trades = self.total_trades
            total_return = (final_value - start_value) / start_value
            
            metrics = {
                'total_trades': total_trades,
                'starting_capital': start_value,
                'final_capital': final_value,
                'total_return': total_return,
                'cagr': cagr,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_gains': total_gains,
                'total_losses': total_losses,
                'years_simulated': years,
                'csv_file': self.csv_file,
                'csv_size_mb': os.path.getsize(self.csv_file) / (1024 * 1024) if os.path.exists(self.csv_file) else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"[HISTORIAN] Failed to calculate metrics: {e}")
            raise
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            if not self.equity_curve:
                return 0.0
            
            # Extract equity values
            equity_values = [equity for date, equity in self.equity_curve]
            
            # Find peak values and drawdowns
            peak = equity_values[0]
            max_drawdown = 0.0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"[HISTORIAN] Failed to calculate max drawdown: {e}")
            return 0.0
    
    def get_equity_curve(self) -> List[Tuple[str, float]]:
        """Get the equity curve data"""
        return self.equity_curve.copy()
    
    def get_trade_count(self) -> int:
        """Get total number of trades"""
        return self.total_trades
