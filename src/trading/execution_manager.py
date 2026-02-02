"""
NeuralTrader Execution Manager - The Bridge
==========================================

Handles order execution with limit orders, retry logic, and shadow ledger.
Provides robust trade execution with comprehensive logging and error handling.

Features:
- Limit orders with 0.1% price buffer
- 3-attempt retry logic (5s, 30s, 60s)
- Shadow ledger logging to CSV
- Comprehensive error handling
- Trade confirmation and validation

Usage:
    from src.trading.execution_manager import ExecutionManager
    
    em = ExecutionManager()
    result = em.execute_order('AAPL', 'buy', 100, 150.0)
"""

import os
import time
import csv
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExecutionResult:
    """Execution result constants"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRY_EXCEEDED = "RETRY_EXCEEDED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class ExecutionManager:
    """
    Execution Manager - The Bridge
    Handles order execution with safety and logging
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, shadow_ledger_path: str = None):
        """Initialize Execution Manager with Alpaca API"""
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper_url = 'https://paper-api.alpaca.markets'
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.paper_url
        )
        
        # Shadow ledger path
        self.shadow_ledger_path = shadow_ledger_path or 'reports/live_trade_log.csv'
        self._ensure_shadow_ledger()
        
        # Execution parameters
        self.price_buffer = 0.001  # 0.1% price buffer
        self.retry_delays = [5, 30, 60]  # Retry delays in seconds
        
        logger.info("Execution Manager initialized")
        logger.info(f"Shadow ledger: {self.shadow_ledger_path}")
        logger.info(f"Price buffer: {self.price_buffer:.1%}")
    
    def _ensure_shadow_ledger(self):
        """Ensure shadow ledger CSV file exists with proper headers"""
        try:
            os.makedirs(os.path.dirname(self.shadow_ledger_path), exist_ok=True)
            
            # Create file if it doesn't exist
            if not os.path.exists(self.shadow_ledger_path):
                with open(self.shadow_ledger_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'order_id', 'symbol', 'side', 'quantity',
                        'limit_price', 'market_price', 'status', 'attempt',
                        'error_message', 'execution_time', 'notes'
                    ])
                
                logger.info(f"Created shadow ledger: {self.shadow_ledger_path}")
        
        except Exception as e:
            logger.error(f"Error creating shadow ledger: {e}")
    
    def execute_order(self, symbol: str, side: str, quantity: int, 
                     market_price: float, notes: str = "") -> Dict:
        """
        Execute order with limit order and retry logic
        
        Args:
            symbol: Ticker symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            market_price: Current market price
            notes: Additional notes for logging
            
        Returns:
            Dictionary with execution result
        """
        start_time = time.time()
        
        # Calculate limit price with buffer
        if side.lower() == 'buy':
            limit_price = market_price * (1 + self.price_buffer)
        else:  # sell
            limit_price = market_price * (1 - self.price_buffer)
        
        logger.info(f"Executing {side} order: {quantity} shares of {symbol} @ ${limit_price:.2f} "
                   f"(Market: ${market_price:.2f})")
        
        # Log to shadow ledger before execution
        self._log_to_shadow_ledger({
            'timestamp': datetime.now().isoformat(),
            'order_id': 'PENDING',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
            'market_price': market_price,
            'status': 'ATTEMPTING',
            'attempt': 1,
            'error_message': '',
            'execution_time': 0,
            'notes': notes
        })
        
        # Execute with retry logic
        for attempt, delay in enumerate(self.retry_delays, 1):
            try:
                result = self._attempt_order(symbol, side, quantity, limit_price, attempt)
                
                execution_time = time.time() - start_time
                
                if result['status'] == ExecutionResult.SUCCESS:
                    logger.info(f"✅ Order SUCCESS: {result['order_id']} - {quantity} {symbol} @ ${limit_price:.2f}")
                    
                    # Log success to shadow ledger
                    self._log_to_shadow_ledger({
                        'timestamp': datetime.now().isoformat(),
                        'order_id': result['order_id'],
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'market_price': market_price,
                        'status': result['status'],
                        'attempt': attempt,
                        'error_message': result.get('error', ''),
                        'execution_time': execution_time,
                        'notes': f"SUCCESS: {notes}"
                    })
                    
                    return result
                
                else:
                    logger.warning(f"❌ Order FAILED (attempt {attempt}): {result.get('error', 'Unknown error')}")
                    
                    # Log failure to shadow ledger
                    self._log_to_shadow_ledger({
                        'timestamp': datetime.now().isoformat(),
                        'order_id': result.get('order_id', 'FAILED'),
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'market_price': market_price,
                        'status': result['status'],
                        'attempt': attempt,
                        'error_message': result.get('error', ''),
                        'execution_time': execution_time,
                        'notes': f"FAILED: {notes}"
                    })
                    
                    # Don't retry on certain errors
                    if result.get('error', '').lower() in ['insufficient funds', 'symbol not found']:
                        break
                
            except Exception as e:
                logger.error(f"❌ Order EXCEPTION (attempt {attempt}): {str(e)}")
                
                # Log exception to shadow ledger
                self._log_to_shadow_ledger({
                    'timestamp': datetime.now().isoformat(),
                    'order_id': 'EXCEPTION',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'market_price': market_price,
                    'status': 'EXCEPTION',
                    'attempt': attempt,
                    'error_message': str(e),
                    'execution_time': time.time() - start_time,
                    'notes': f"EXCEPTION: {notes}"
                })
            
            # Wait before retry (except on last attempt)
            if attempt < len(self.retry_delays):
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # All retries failed
        logger.error(f"🚨 Order FAILED after {len(self.retry_delays)} attempts: {side} {quantity} {symbol}")
        
        return {
            'status': ExecutionResult.RETRY_EXCEEDED,
            'order_id': None,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
            'market_price': market_price,
            'error': f'Failed after {len(self.retry_delays)} attempts',
            'execution_time': time.time() - start_time
        }
    
    def _attempt_order(self, symbol: str, side: str, quantity: int, 
                      limit_price: float, attempt: int) -> Dict:
        """
        Attempt to place a single order
        
        Args:
            symbol: Ticker symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            limit_price: Limit price
            attempt: Attempt number
            
        Returns:
            Dictionary with attempt result
        """
        try:
            # Submit limit order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=limit_price
            )
            
            logger.info(f"Order submitted: {order.id} (attempt {attempt})")
            
            # Wait a moment for order to be processed
            time.sleep(1)
            
            # Check order status
            order_status = self.api.get_order(order.id)
            
            if order_status.status == 'filled':
                # Get filled price
                filled_price = float(order_status.filled_avg_price)
                
                return {
                    'status': ExecutionResult.SUCCESS,
                    'order_id': order.id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'filled_price': filled_price,
                    'execution_time': datetime.now().isoformat()
                }
            
            elif order_status.status in ['rejected', 'canceled']:
                return {
                    'status': ExecutionResult.REJECTED,
                    'order_id': order.id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'error': f'Order {order_status.status}',
                    'execution_time': datetime.now().isoformat()
                }
            
            else:
                # Order is open/pending, cancel it for retry
                try:
                    self.api.cancel_order(order.id)
                    logger.info(f"Cancelled pending order {order.id} for retry")
                except:
                    pass  # Order might already be filled/cancelled
                
                return {
                    'status': ExecutionResult.FAILED,
                    'order_id': order.id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'error': 'Order not filled, cancelled for retry',
                    'execution_time': datetime.now().isoformat()
                }
        
        except tradeapi.rest.APIError as e:
            return {
                'status': ExecutionResult.FAILED,
                'order_id': None,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'limit_price': limit_price,
                'error': f'API Error: {str(e)}',
                'execution_time': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': ExecutionResult.FAILED,
                'order_id': None,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'limit_price': limit_price,
                'error': f'Exception: {str(e)}',
                'execution_time': datetime.now().isoformat()
            }
    
    def _log_to_shadow_ledger(self, trade_record: Dict):
        """Log trade record to shadow ledger CSV"""
        try:
            with open(self.shadow_ledger_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_record['timestamp'],
                    trade_record['order_id'],
                    trade_record['symbol'],
                    trade_record['side'],
                    trade_record['quantity'],
                    trade_record['limit_price'],
                    trade_record['market_price'],
                    trade_record['status'],
                    trade_record['attempt'],
                    trade_record['error_message'],
                    trade_record['execution_time'],
                    trade_record['notes']
                ])
        
        except Exception as e:
            logger.error(f"Error logging to shadow ledger: {e}")
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of a specific order
        
        Args:
            order_id: Order ID from Alpaca
            
        Returns:
            Dictionary with order status
        """
        try:
            order = self.api.get_order(order_id)
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.qty,
                'status': order.status,
                'filled_qty': order.filled_qty,
                'filled_avg_price': order.filled_avg_price,
                'created_at': order.created_at,
                'updated_at': order.updated_at
            }
        
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_shadow_ledger_summary(self) -> Dict:
        """
        Get summary of shadow ledger trades
        
        Returns:
            Dictionary with trade summary
        """
        try:
            if not os.path.exists(self.shadow_ledger_path):
                return {'total_trades': 0, 'successful_trades': 0, 'failed_trades': 0}
            
            df = pd.read_csv(self.shadow_ledger_path)
            
            if df.empty:
                return {'total_trades': 0, 'successful_trades': 0, 'failed_trades': 0}
            
            total_trades = len(df)
            successful_trades = len(df[df['status'] == ExecutionResult.SUCCESS])
            failed_trades = len(df[df['status'] != ExecutionResult.SUCCESS])
            
            # Recent trades (last 10)
            recent_trades = df.tail(10).to_dict('records')
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': failed_trades,
                'success_rate': (successful_trades / total_trades * 100) if total_trades > 0 else 0,
                'recent_trades': recent_trades
            }
        
        except Exception as e:
            logger.error(f"Error getting shadow ledger summary: {e}")
            return {'error': str(e)}
    
    def log_execution_summary(self):
        """Log execution summary from shadow ledger"""
        summary = self.get_shadow_ledger_summary()
        
        logger.info("=== Execution Summary ===")
        logger.info(f"Total Trades: {summary.get('total_trades', 0)}")
        logger.info(f"Successful: {summary.get('successful_trades', 0)}")
        logger.info(f"Failed: {summary.get('failed_trades', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        if 'recent_trades' in summary:
            logger.info("Recent Trades:")
            for trade in summary['recent_trades'][-5:]:  # Last 5 trades
                logger.info(f"  {trade['timestamp']}: {trade['side']} {trade['quantity']} {trade['symbol']} "
                           f"@ ${trade['limit_price']:.2f} - {trade['status']}")

# Usage example
if __name__ == "__main__":
    # Test the execution manager
    em = ExecutionManager()
    
    # Log current summary
    em.log_execution_summary()
    
    # Test a small order (commented out for safety)
    # result = em.execute_order('AAPL', 'buy', 1, 150.0, "Test order")
    # logger.info(f"Test result: {result}")
