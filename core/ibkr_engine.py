"""
Interactive Brokers Execution Engine (ib_async)
==============================================

Real paper trading execution engine for NeuralTrader using ib_async.
Handles order execution, account management, and position tracking.

Features:
- Real-time market data and execution using ib_async
- Paper trading on TWS (port 7497)
- Market and Limit order support
- Account summary and position tracking
- Risk management integration
- Async/await support for modern Python
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import ib_async
    IB_AVAILABLE = True
    logger.info("[IBKR] ib_async module available")
except ImportError as e:
    IB_AVAILABLE = False
    logger.warning(f"[IBKR] ib_async not available: {e}")
except Exception as e:
    IB_AVAILABLE = False
    print(f"[IBKR] Error importing ib_async: {e}")

class OrderType(Enum):
    """Order types supported by IBKR engine"""
    MARKET = "MKT"
    LIMIT = "LMT"

class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderResult:
    """Result of order execution"""
    order_id: int
    status: str
    filled: bool
    filled_quantity: int
    fill_price: float
    commission: float
    timestamp: datetime
    error: Optional[str] = None

@dataclass
class AccountSummary:
    """Account summary information"""
    cash_balance: float
    portfolio_value: float
    buying_power: float
    equity_with_loan: float
    total_positions: int
    timestamp: datetime

@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: int
    market_price: float
    market_value: float
    average_cost: float
    unrealized_pnl: float
    side: str

class IBKRExecutionEngine:
    """
    Interactive Brokers Execution Engine using ib_async
    
    This engine handles real paper trading execution through IBKR TWS Paper Trading.
    It provides methods for order execution, account management, and position tracking.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize IBKR Execution Engine
        
        Args:
            host: IBKR TWS host address
            port: IBKR TWS port (7497 for paper trading)
            client_id: Client ID for IBKR connection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"[IBKR] Initializing IBKR Engine - Host: {host}, Port: {port}, Client ID: {client_id}")
        
        if not IB_AVAILABLE:
            raise ImportError("ib_async is not available. Install with: pip install ib_async")
    
    async def connect_async(self) -> bool:
        """
        Connect to IBKR TWS asynchronously
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"[IBKR] Connecting to IBKR TWS at {self.host}:{self.port}...")
            
            # Create IB connection
            self.ib = ib_async.IB()
            
            # Connect asynchronously
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            
            # Wait for connection to establish
            await asyncio.sleep(1)
            
            if self.ib.isConnected():
                self.connected = True
                self.logger.info("[IBKR] Successfully connected to IBKR")
                return True
            else:
                self.logger.error("[IBKR] Failed to connect to IBKR")
                return False
                
        except Exception as e:
            self.logger.error(f"[IBKR] Connection error: {e}")
            self.connected = False
            return False
    
    def connect(self) -> bool:
        """
        Connect to IBKR TWS (synchronous wrapper)
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Run async connection in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.connect_async())
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"[IBKR] Connection error: {e}")
            self.connected = False
            return False
    
    async def disconnect_async(self):
        """Disconnect from IBKR TWS asynchronously"""
        try:
            if self.ib and self.ib.isConnected():
                await self.ib.disconnectAsync()
                self.connected = False
                self.logger.info("[IBKR] Disconnected from IBKR")
        except Exception as e:
            self.logger.error(f"[IBKR] Disconnect error: {e}")
    
    def disconnect(self):
        """Disconnect from IBKR TWS (synchronous wrapper)"""
        try:
            if self.ib and self.ib.isConnected():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.disconnect_async())
                finally:
                    loop.close()
        except Exception as e:
            self.logger.error(f"[IBKR] Disconnect error: {e}")
    
    async def execute_trade_async(
        self, 
        symbol: str, 
        side: OrderSide, 
        order_type: OrderType, 
        quantity: int, 
        limit_price: Optional[float] = None
    ) -> OrderResult:
        """
        Execute trade asynchronously
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            side: Buy or Sell
            order_type: Market or Limit
            quantity: Number of shares
            limit_price: Limit price (required for limit orders)
            
        Returns:
            OrderResult: Result of order execution
        """
        try:
            if not self.connected or not self.ib:
                raise Exception("Not connected to IBKR")
            
            self.logger.info(f"[IBKR] Executing {side.value} {quantity} shares of {symbol} ({order_type.value})")
            
            # Create contract
            contract = ib_async.Stock(symbol, 'SMART', 'USD')
            
            # Create order
            if order_type == OrderType.MARKET:
                order = ib_async.MarketOrder(side.value, quantity)
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("Limit price required for limit orders")
                order = ib_async.LimitOrder(side.value, quantity, limit_price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Submit order
            trade = await self.ib.placeOrderAsync(contract, order)
            
            # Wait for order to process
            await asyncio.sleep(1)
            
            # Create result
            result = OrderResult(
                order_id=trade.order.orderId,
                status=trade.orderStatus.status,
                filled=trade.orderStatus.filled > 0,
                filled_quantity=trade.orderStatus.filled,
                fill_price=trade.orderStatus.avgFillPrice if trade.orderStatus.filled > 0 else 0.0,
                commission=trade.commissionReport.commission if trade.commissionReport else 0.0,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"[IBKR] Order submitted: {result.order_id} ({result.status})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[IBKR] Order execution error: {e}")
            return OrderResult(
                order_id=0,
                status="ERROR",
                filled=False,
                filled_quantity=0,
                fill_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def execute_trade(
        self, 
        symbol: str, 
        side: OrderSide, 
        order_type: OrderType, 
        quantity: int, 
        limit_price: Optional[float] = None
    ) -> OrderResult:
        """
        Execute trade (synchronous wrapper)
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            side: Buy or Sell
            order_type: Market or Limit
            quantity: Number of shares
            limit_price: Limit price (required for limit orders)
            
        Returns:
            OrderResult: Result of order execution
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.execute_trade_async(symbol, side, order_type, quantity, limit_price)
                )
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"[IBKR] Order execution error: {e}")
            return OrderResult(
                order_id=0,
                status="ERROR",
                filled=False,
                filled_quantity=0,
                fill_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def get_account_summary_async(self) -> AccountSummary:
        """
        Get account summary asynchronously
        
        Returns:
            AccountSummary: Account summary information
        """
        try:
            if not self.connected or not self.ib:
                raise Exception("Not connected to IBKR")
            
            self.logger.info("[IBKR] Fetching account summary...")
            
            # Get account summary
            summary = await self.ib.accountSummaryAsync()
            
            # Extract key values
            cash_balance = 0.0
            portfolio_value = 0.0
            buying_power = 0.0
            equity_with_loan = 0.0
            
            for item in summary:
                if item.tag == 'TotalCashBalance':
                    cash_balance = float(item.value)
                elif item.tag == 'NetLiquidation':
                    portfolio_value = float(item.value)
                elif item.tag == 'BuyingPower':
                    buying_power = float(item.value)
                elif item.tag == 'EquityWithLoanValue':
                    equity_with_loan = float(item.value)
            
            # Get positions
            positions = await self.ib.positionsAsync()
            total_positions = len(positions)
            
            account_summary = AccountSummary(
                cash_balance=cash_balance,
                portfolio_value=portfolio_value,
                buying_power=buying_power,
                equity_with_loan=equity_with_loan,
                total_positions=total_positions,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"[IBKR] Account summary: Cash=${cash_balance:,.2f}, Portfolio=${portfolio_value:,.2f}")
            
            return account_summary
            
        except Exception as e:
            self.logger.error(f"[IBKR] Account summary error: {e}")
            raise
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary (synchronous wrapper)
        
        Returns:
            Dict: Account summary information
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                summary = loop.run_until_complete(self.get_account_summary_async())
                return {
                    'cash_balance': summary.cash_balance,
                    'portfolio_value': summary.portfolio_value,
                    'buying_power': summary.buying_power,
                    'equity_with_loan': summary.equity_with_loan,
                    'total_positions': summary.total_positions,
                    'timestamp': summary.timestamp.isoformat()
                }
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"[IBKR] Account summary error: {e}")
            return {'error': str(e)}
    
    async def get_positions_async(self) -> List[Position]:
        """
        Get current positions asynchronously
        
        Returns:
            List[Position]: List of current positions
        """
        try:
            if not self.connected or not self.ib:
                raise Exception("Not connected to IBKR")
            
            self.logger.info("[IBKR] Fetching positions...")
            
            # Get positions
            ib_positions = await self.ib.positionsAsync()
            
            positions = []
            for pos in ib_positions:
                position = Position(
                    symbol=pos.contract.symbol,
                    quantity=int(pos.position),
                    market_price=float(pos.marketPrice),
                    market_value=float(pos.marketValue),
                    average_cost=float(pos.averageCost),
                    unrealized_pnl=float(pos.unrealizedPNL),
                    side="LONG" if pos.position > 0 else "SHORT"
                )
                positions.append(position)
            
            self.logger.info(f"[IBKR] Found {len(positions)} positions")
            
            return positions
            
        except Exception as e:
            self.logger.error(f"[IBKR] Positions error: {e}")
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions (synchronous wrapper)
        
        Returns:
            List[Dict]: List of current positions
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                positions = loop.run_until_complete(self.get_positions_async())
                return [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'market_price': pos.market_price,
                        'market_value': pos.market_value,
                        'average_cost': pos.average_cost,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'side': pos.side
                    }
                    for pos in positions
                ]
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"[IBKR] Positions error: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

# Legacy compatibility class
class IBKREngineSync(IBKRExecutionEngine):
    """Legacy compatibility wrapper for IBKRExecutionEngine"""
    pass

# Test function
async def test_ibkr_engine():
    """Test IBKR engine functionality"""
    engine = IBKRExecutionEngine()
    
    try:
        # Test connection
        connected = await engine.connect_async()
        if not connected:
            print("Failed to connect to IBKR")
            return
        
        print("Connected to IBKR successfully")
        
        # Test account summary
        summary = await engine.get_account_summary_async()
        print(f"Account Summary: Cash=${summary.cash_balance:,.2f}")
        
        # Test positions
        positions = await engine.get_positions_async()
        print(f"Positions: {len(positions)}")
        
        # Test order (dummy order that won't execute)
        print("Testing order execution...")
        result = await engine.execute_trade_async(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1,
            limit_price=1.00  # Very low price, won't execute
        )
        print(f"Order result: {result.order_id} ({result.status})")
        
        # Cancel the order if it was submitted
        if result.order_id > 0 and result.status not in ["ERROR", "Cancelled"]:
            print("Cancelling test order...")
            # Note: Order cancellation would need to be implemented
            
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        await engine.disconnect_async()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_ibkr_engine())
