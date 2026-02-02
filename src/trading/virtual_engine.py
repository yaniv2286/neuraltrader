"""
NeuralTrader Virtual Portfolio Engine - Shadow Trading Simulator
================================================================

Manages virtual portfolio for shadow trading simulation with Yahoo Finance.
Tracks cash, positions, and trades without real money execution.

Features:
- Virtual portfolio management with JSON persistence
- Trade execution at Daily Adjusted Close + 0.1% slippage
- Cash and position value tracking
- Risk management integration
- Portfolio performance analytics
- Shadow trading simulation

Usage:
    from src.trading.virtual_engine import VirtualEngine
    
    ve = VirtualEngine()
    result = ve.execute_trade('AAPL', 'buy', 100, 150.0)
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.notifier import EmailNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VirtualEngine:
    """
    Virtual Portfolio Engine for Shadow Trading Simulation
    Manages virtual portfolio with trade execution and tracking
    """
    
    def __init__(self, portfolio_file: str = "data/portfolio.json", initial_cash: float = 100000.0):
        """
        Initialize Virtual Engine
        
        Args:
            portfolio_file: Path to portfolio JSON file
            initial_cash: Initial cash balance (default: $100,000)
        """
        self.portfolio_file = portfolio_file
        self.initial_cash = initial_cash
        self.slippage_rate = 0.001  # 0.1% slippage
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(portfolio_file), exist_ok=True)
        
        # Load or create portfolio
        self.portfolio = self._load_portfolio()
        
        logger.info(f"Virtual Engine initialized")
        logger.info(f"Portfolio file: {portfolio_file}")
        logger.info(f"Initial cash: ${initial_cash:,.2f}")
        logger.info(f"Current cash: ${self.portfolio['cash']:,.2f}")
        logger.info(f"Slippage rate: {self.slippage_rate:.1%}")
    
    def _load_portfolio(self) -> Dict:
        """Load portfolio from JSON file or create new one"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    portfolio = json.load(f)
                logger.info(f"Loaded existing portfolio: {len(portfolio.get('positions', {}))} positions")
                return portfolio
            else:
                # Create new portfolio with specified structure
                portfolio = {
                    'cash': self.initial_cash,
                    'positions': {},
                    'history': [],
                    'performance': {
                        'total_return': 0.0,
                        'total_return_pct': 0.0,
                        'unrealized_pnl': 0.0,
                        'realized_pnl': 0.0
                    },
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                self._save_portfolio(portfolio)
                logger.info("Created new portfolio")
                return portfolio
                
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            # Create new portfolio as fallback
            return {
                'cash': self.initial_cash,
                'positions': {},
                'history': [],
                'performance': {
                    'total_return': 0.0,
                    'total_return_pct': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                },
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
    
    def _save_portfolio(self, portfolio: Dict = None):
        """Save portfolio to JSON file"""
        try:
            if portfolio is None:
                portfolio = self.portfolio
            
            portfolio['updated_at'] = datetime.now().isoformat()
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio, f, indent=2)
            
            logger.debug(f"Portfolio saved to {self.portfolio_file}")
            
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for ticker using Yahoo Finance
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Current price or None if error
        """
        try:
            # Fetch latest data
            data = yf.download(ticker, period="1d", progress=False)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
            
            # Get adjusted close price for Daily Adjusted Close
            if 'Adj Close' in data.columns:
                price = data['Adj Close'].iloc[-1]
            elif 'Close' in data.columns:
                price = data['Close'].iloc[-1]
            else:
                logger.warning(f"No price data found for {ticker}")
                return None
            
            if pd.notna(price) and price > 0:
                return float(price)
            else:
                logger.warning(f"Invalid price for {ticker}: {price}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
            return None
    
    def execute_trade(self, ticker: str, side: str, quantity: int, 
                     signal_price: float, notes: str = "") -> Dict:
        """
        Execute virtual trade with slippage
        
        Args:
            ticker: Ticker symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            signal_price: Signal price (for reference)
            notes: Additional notes
            
        Returns:
            Dictionary with trade result
        """
        try:
            # Get current market price
            current_price = self.get_current_price(ticker)
            if current_price is None:
                return {
                    'success': False,
                    'error': f'Could not get price for {ticker}',
                    'ticker': ticker,
                    'side': side,
                    'quantity': quantity
                }
            
            # Apply slippage (0.1%)
            if side.lower() == 'buy':
                execution_price = current_price * (1 + self.slippage_rate)
                cost = execution_price * quantity
            else:  # sell
                execution_price = current_price * (1 - self.slippage_rate)
                cost = execution_price * quantity
            
            # Check if we have enough cash for buys
            if side.lower() == 'buy':
                if cost > self.portfolio['cash']:
                    return {
                        'success': False,
                        'error': f'Insufficient cash: need ${cost:.2f}, have ${self.portfolio["cash"]:.2f}',
                        'ticker': ticker,
                        'side': side,
                        'quantity': quantity,
                        'cost': cost
                    }
            
            # Check if we have enough shares for sells
            if side.lower() == 'sell':
                current_position = self.portfolio['positions'].get(ticker, {'quantity': 0})
                if current_position['quantity'] < quantity:
                    return {
                        'success': False,
                        'error': f'Insufficient shares: need {quantity}, have {current_position["quantity"]}',
                        'ticker': ticker,
                        'side': side,
                        'quantity': quantity
                    }
            
            # Create trade record
            trade = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'side': side.lower(),
                'quantity': quantity,
                'signal_price': signal_price,
                'execution_price': execution_price,
                'cost': cost,
                'notes': notes
            }
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'side': side.lower(),
                'quantity': quantity,
                'signal_price': signal_price,
                'execution_price': execution_price,
                'cost': cost,
                'cash_before': self.portfolio['cash'] + cost if side.lower() == 'buy' else self.portfolio['cash'] - cost,
                'cash_after': self.portfolio['cash'],
                'notes': notes
            }
            
            self.portfolio['history'].append(history_entry)
            
            # Update portfolio
            if side.lower() == 'buy':
                # Deduct cash
                self.portfolio['cash'] -= cost
                
                # Update or create position
                if ticker in self.portfolio['positions']:
                    current_pos = self.portfolio['positions'][ticker]
                    new_quantity = current_pos['quantity'] + quantity
                    new_avg_cost = ((current_pos['quantity'] * current_pos['avg_cost']) + 
                                  (quantity * execution_price)) / new_quantity
                    
                    self.portfolio['positions'][ticker] = {
                        'quantity': new_quantity,
                        'avg_cost': new_avg_cost,
                        'last_price': execution_price,
                        'updated_at': datetime.now().isoformat()
                    }
                else:
                    self.portfolio['positions'][ticker] = {
                        'quantity': quantity,
                        'avg_cost': execution_price,
                        'last_price': execution_price,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                
                trade['cash_before'] = self.portfolio['cash'] + cost
                trade['cash_after'] = self.portfolio['cash']
                
            else:  # sell
                # Add cash
                self.portfolio['cash'] += cost
                
                # Update position
                current_pos = self.portfolio['positions'][ticker]
                new_quantity = current_pos['quantity'] - quantity
                realized_pnl = (execution_price - current_pos['avg_cost']) * quantity
                
                if new_quantity > 0:
                    self.portfolio['positions'][ticker] = {
                        'quantity': new_quantity,
                        'avg_cost': current_pos['avg_cost'],
                        'last_price': execution_price,
                        'updated_at': datetime.now().isoformat()
                    }
                else:
                    # Position closed
                    del self.portfolio['positions'][ticker]
                
                # Update realized PnL
                self.portfolio['performance']['realized_pnl'] += realized_pnl
                history_entry['realized_pnl'] = realized_pnl
                trade['realized_pnl'] = realized_pnl
                trade['cash_before'] = self.portfolio['cash'] - cost
                trade['cash_after'] = self.portfolio['cash']
            
            # Add trade to history
            self.portfolio['trades'].append(trade)
            
            # Save portfolio
            self._save_portfolio()
            
            logger.info(f"✅ Trade executed: {side.upper()} {quantity} {ticker} @ ${execution_price:.2f}")
            logger.info(f"   Cost: ${cost:.2f}, Cash: ${self.portfolio['cash']:.2f}")
            
            return {
                'success': True,
                'trade': trade,
                'ticker': ticker,
                'side': side,
                'quantity': quantity,
                'execution_price': execution_price,
                'cost': cost,
                'cash_remaining': self.portfolio['cash']
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'side': side,
                'quantity': quantity
            }
    
    def update_portfolio_values(self) -> Dict:
        """
        Update portfolio values with current market prices
        
        Returns:
            Dictionary with updated portfolio values
        """
        try:
            logger.info("Updating portfolio values...")
            
            total_value = self.portfolio['cash']
            position_values = {}
            
            # Update each position
            for ticker, position in self.portfolio['positions'].items():
                current_price = self.get_current_price(ticker)
                
                if current_price is not None:
                    position_value = current_price * position['quantity']
                    unrealized_pnl = (current_price - position['avg_cost']) * position['quantity']
                    
                    position_values[ticker] = {
                        'quantity': position['quantity'],
                        'avg_cost': position['avg_cost'],
                        'current_price': current_price,
                        'position_value': position_value,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': (unrealized_pnl / (position['avg_cost'] * position['quantity'])) * 100 if position['quantity'] > 0 else 0
                    }
                    
                    total_value += position_value
                    
                    # Update position in portfolio
                    self.portfolio['positions'][ticker]['last_price'] = current_price
                    self.portfolio['positions'][ticker]['updated_at'] = datetime.now().isoformat()
                else:
                    logger.warning(f"Could not get price for {ticker}")
                    # Use last known price
                    last_price = position.get('last_price', position['avg_cost'])
                    position_value = last_price * position['quantity']
                    position_values[ticker] = {
                        'quantity': position['quantity'],
                        'avg_cost': position['avg_cost'],
                        'current_price': last_price,
                        'position_value': position_value,
                        'unrealized_pnl': 0,
                        'unrealized_pnl_pct': 0
                    }
                    total_value += position_value
            
            # Calculate performance metrics
            total_return = total_value - self.portfolio['initial_cash']
            total_return_pct = (total_return / self.portfolio['initial_cash']) * 100
            
            total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in position_values.values())
            
            # Update portfolio performance
            self.portfolio['performance'] = {
                'total_value': total_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': self.portfolio['performance'].get('realized_pnl', 0.0)
            }
            
            # Save updated portfolio
            self._save_portfolio()
            
            result = {
                'total_value': total_value,
                'cash': self.portfolio['cash'],
                'positions_value': total_value - self.portfolio['cash'],
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': self.portfolio['performance']['realized_pnl'],
                'position_count': len(self.portfolio['positions']),
                'position_details': position_values
            }
            
            logger.info(f"Portfolio updated: ${total_value:,.2f} total value")
            logger.info(f"  Cash: ${self.portfolio['cash']:,.2f}")
            logger.info(f"  Positions: ${total_value - self.portfolio['cash']:,.2f}")
            logger.info(f"  Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating portfolio values: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary
        
        Returns:
            Dictionary with portfolio summary
        """
        try:
            # Update values first
            values = self.update_portfolio_values()
            
            # Get recent trades
            recent_trades = self.portfolio['trades'][-10:] if self.portfolio['trades'] else []
            
            # Calculate additional metrics
            total_trades = len(self.portfolio['trades'])
            buy_trades = len([t for t in self.portfolio['trades'] if t['side'] == 'buy'])
            sell_trades = len([t for t in self.portfolio['trades'] if t['side'] == 'sell'])
            
            summary = {
                'portfolio_info': {
                    'created_at': self.portfolio['created_at'],
                    'updated_at': self.portfolio['updated_at'],
                    'initial_cash': self.portfolio['initial_cash'],
                    'current_cash': self.portfolio['cash']
                },
                'performance': values,
                'positions': self.portfolio['positions'],
                'trading_stats': {
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'recent_trades': recent_trades
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_position(self, ticker: str) -> Optional[Dict]:
        """
        Get position details for a specific ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Position details or None if not found
        """
        try:
            if ticker in self.portfolio['positions']:
                position = self.portfolio['positions'][ticker].copy()
                
                # Get current price
                current_price = self.get_current_price(ticker)
                if current_price is not None:
                    position['current_price'] = current_price
                    position['position_value'] = current_price * position['quantity']
                    position['unrealized_pnl'] = (current_price - position['avg_cost']) * position['quantity']
                    position['unrealized_pnl_pct'] = (position['unrealized_pnl'] / (position['avg_cost'] * position['quantity'])) * 100
                
                return position
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting position for {ticker}: {e}")
            return None
    
    def reset_portfolio(self, initial_cash: float = None):
        """
        Reset portfolio to initial state
        
        Args:
            initial_cash: New initial cash (default: current initial_cash)
        """
        try:
            if initial_cash is None:
                initial_cash = self.portfolio['initial_cash']
            
            self.portfolio = {
                'cash': initial_cash,
                'initial_cash': initial_cash,
                'positions': {},
                'trades': [],
                'performance': {
                    'total_return': 0.0,
                    'total_return_pct': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                },
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            self._save_portfolio()
            logger.info(f"Portfolio reset with ${initial_cash:,.2f} initial cash")
            
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")

# Usage example
if __name__ == "__main__":
    # Test the virtual engine
    try:
        ve = VirtualEngine()
        
        # Test portfolio summary
        summary = ve.get_portfolio_summary()
        logger.info(f"Portfolio summary: {summary['performance']['total_value']:,.2f}")
        
        # Test trade execution (example)
        # result = ve.execute_trade('AAPL', 'buy', 10, 150.0, "Test buy")
        # logger.info(f"Trade result: {result}")
        
    except Exception as e:
        logger.error(f"Error in virtual engine test: {e}")
