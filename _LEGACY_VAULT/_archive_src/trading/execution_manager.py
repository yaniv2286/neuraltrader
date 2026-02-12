"""
NeuralTrader Execution Manager - The Bridge (IBKR Version)
==========================================================

Handles order execution with Interactive Brokers using ib_insync.
Provides robust trade execution with comprehensive logging and error handling.

Features:
- IBKR TWS/IB Gateway integration
- Limit orders with retry logic
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
from ib_insync import IB, Stock, Order, util

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
    Execution Manager - The Bridge (IBKR Version)
    Handles order execution with Interactive Brokers
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Initialize Execution Manager with IBKR connection
        
        Args:
            host: IBKR host (default: localhost)
            port: IBKR port (7497 for paper, 7496 for live)
            client_id: Client ID for connection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Shadow ledger path
        self.shadow_ledger_path = 'reports/live_trade_log.csv'
        self._ensure_shadow_ledger()
        
        # Execution parameters
        self.price_buffer = 0.001  # 0.1% price buffer
        self.retry_delays = [5, 30, 60]  # Retry delays in seconds
        
        # Initialize IB connection
        self.ib = None
        self._connect_ibkr()
        
        logger.info(f"Execution Manager initialized (IBKR)")
        logger.info(f"Connection: {host}:{port} (Paper Trading)")
        logger.info(f"Shadow ledger: {self.shadow_ledger_path}")
        logger.info(f"Price buffer: {self.price_buffer:.1%}")
    
    def _connect_ibkr(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib = IB()
            self.ib.connect(host=self.host, port=self.port, clientId=self.client_id)
            
            if self.ib.isConnected():
                logger.info("✅ Connected to Interactive Brokers")
            else:
                logger.error("❌ Failed to connect to Interactive Brokers")
                raise ConnectionError("Could not connect to IBKR")
                
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            raise
    
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
        
        # Ensure IBKR connection
        if not self.ib.isConnected():
            self._connect_ibkr()
        
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
        Attempt to place a single order with IBKR
        
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
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Create order
            order = Order()
            order.action = side.upper()
            order.totalQuantity = quantity
            order.orderType = 'LMT'
            order.lmtPrice = limit_price
            order.transmit = True
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Order submitted: {trade.order.orderId} (attempt {attempt})")
            
            # Wait for order to be processed
            self.ib.sleep(1)
            
            # Check order status
            order_status = self.ib.reqOrderStatus(trade.order.orderId)
            
            if order_status.status == 'Filled':
                # Get filled price
                filled_price = order_status.avgFillPrice
                
                return {
                    'status': ExecutionResult.SUCCESS,
                    'order_id': str(trade.order.orderId),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'filled_price': filled_price,
                    'execution_time': datetime.now().isoformat()
                }
            
            elif order_status.status in ['Cancelled', 'Rejected']:
                return {
                    'status': ExecutionResult.REJECTED,
                    'order_id': str(trade.order.orderId),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'error': f'Order {order_status.status}',
                    'execution_time': datetime.now().isoformat()
                }
            
            else:
                # Order is still pending, cancel it for retry
                try:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled pending order {trade.order.orderId} for retry")
                except:
                    pass  # Order might already be filled/cancelled
                
                return {
                    'status': ExecutionResult.FAILED,
                    'order_id': str(trade.order.orderId),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'error': 'Order not filled, cancelled for retry',
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
            order_id: Order ID from IBKR
            
        Returns:
            Dictionary with order status
        """
        try:
            # Convert to int if needed
            order_id_int = int(order_id) if order_id.isdigit() else order_id
            
            order_status = self.ib.reqOrderStatus(order_id_int)
            
            return {
                'order_id': str(order_status.orderId),
                'symbol': order_status.contract.symbol if order_status.contract else 'Unknown',
                'side': order_status.action,
                'quantity': order_status.totalQuantity,
                'status': order_status.status,
                'filled_qty': order_status.filled,
                'filled_avg_price': order_status.avgFillPrice,
                'remaining_qty': order_status.remaining,
                'client_id': order_status.clientId
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
            # Convert to int if needed
            order_id_int = int(order_id) if order_id.isdigit() else order_id
            
            # Create order object for cancellation
            order = Order()
            order.orderId = order_id_int
            
            # Cancel order
            self.ib.cancelOrder(order)
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
    
    def get_account_info(self) -> Dict:
        """
        Get current account information from IBKR
        
        Returns:
            Dictionary with account information
        """
        try:
            if not self.ib.isConnected():
                self._connect_ibkr()
            
            # Get account summary
            account_summary = self.ib.accountSummary()
            
            account_info = {}
            for item in account_summary:
                account_info[item.tag] = item.value
            
            # Convert to proper types
            return {
                'account_id': account_info.get('AccountId', ''),
                'equity': float(account_info.get('NetLiquidation', 0)),
                'cash': float(account_info.get('CashBalance', 0)),
                'portfolio_value': float(account_info.get('NetLiquidation', 0)),
                'buying_power': float(account_info.get('BuyingPower', 0)),
                'maint_margin_req': float(account_info.get('MaintMarginReq', 0)),
                'available_funds': float(account_info.get('AvailableFunds', 0))
            }
        
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions from IBKR
        
        Returns:
            List of position dictionaries
        """
        try:
            if not self.ib.isConnected():
                self._connect_ibkr()
            
            positions = self.ib.positions()
            
            return [
                {
                    'symbol': pos.contract.symbol,
                    'sec_type': pos.contract.secType,
                    'exchange': pos.contract.exchange,
                    'currency': pos.contract.currency,
                    'position': float(pos.position),
                    'market_price': float(pos.marketPrice),
                    'market_value': float(pos.marketValue),
                    'average_cost': float(pos.averageCost),
                    'unrealized_pnl': float(pos.unrealizedPNL),
                    'realized_pnl': float(pos.realizedPNL),
                    'account': pos.account
                }
                for pos in positions
                if pos.position != 0  # Only include positions with non-zero quantity
            ]
        
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def disconnect(self):
        """Disconnect from IBKR"""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from Interactive Brokers")
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()
    
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
    try:
        em = ExecutionManager()
        
        # Log current summary
        em.log_execution_summary()
        
        # Get account info
        account = em.get_account_info()
        logger.info(f"Account info: {account}")
        
        # Get positions
        positions = em.get_positions()
        logger.info(f"Positions: {len(positions)}")
        
        # Test a small order (commented out for safety)
        # result = em.execute_order('AAPL', 'buy', 1, 150.0, "Test order")
        # logger.info(f"Test result: {result}")
        
    except Exception as e:
        logger.error(f"Error in execution manager test: {e}")
    finally:
        if 'em' in locals():
            em.disconnect()
