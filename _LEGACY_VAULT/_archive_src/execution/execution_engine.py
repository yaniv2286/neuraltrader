"""
Execution Engine - Order Management & Execution
=============================================

Handles order execution, position management, and risk controls.
Integrates with slot manager for position allocation.

Key Features:
- Order management with limit orders
- Slippage and cost modeling
- Risk management and position limits
- Real-time P&L tracking
- Trade execution logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import json

from ..core.slot_manager import SlotManager, SlotStatus, Position
from ..core.regime_classifier import MarketRegime

logger = logging.getLogger("NeuralTrader.ExecutionEngine")

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Order data structure."""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class Trade:
    """Trade data structure."""
    trade_id: str
    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    executed_at: datetime

@dataclass
class CostModel:
    """Trading cost model."""
    commission_per_trade: float = 1.0
    spread_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    
    def calculate_cost(self, quantity: float, price: float, order_type: OrderType) -> Tuple[float, float]:
        """
        Calculate commission and slippage costs.
        
        Args:
            quantity: Number of shares
            price: Price per share
            order_type: Type of order
            
        Returns:
            Tuple of (commission, slippage)
        """
        # Commission
        commission = self.commission_per_trade
        
        # Slippage (worse for market orders)
        if order_type == OrderType.MARKET:
            slippage = quantity * price * self.slippage_pct * 2  # Double slippage for market orders
        else:
            slippage = quantity * price * self.slippage_pct
        
        return commission, slippage

class ExecutionEngine:
    """
    Order execution engine with risk management.
    
    Features:
    - Limit order execution
    - Realistic cost modeling
    - Risk management
    - Trade logging
    """
    
    def __init__(self, slot_manager: SlotManager, cost_model: Optional[CostModel] = None):
        self.slot_manager = slot_manager
        self.cost_model = cost_model or CostModel()
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.order_counter = 0
        self.trade_counter = 0
        
        # Execution state
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.pending_orders: List[str] = []
        
        # Risk management
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_position_size = 0.10  # 10% max position size
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Logging
        self.execution_log: List[Dict] = []
        self.log_path = Path("data/processed/execution_log.csv")
        
    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
    
    def generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"TRD_{datetime.now().strftime('%Y%m%d')}_{self.trade_counter:06d}"
    
    def update_market_data(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Update market data for execution.
        
        Args:
            market_data: Dictionary of ticker to DataFrame
        """
        self.market_data = market_data
        
        # Update slot manager positions
        self.slot_manager.update_positions(market_data)
        
        # Reset daily P&L if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def check_risk_limits(self, order: Order) -> bool:
        """
        Check if order passes risk limits.
        
        Args:
            order: Order to check
            
        Returns:
            True if order passes risk checks
        """
        # Daily loss limit
        if self.daily_pnl < -self.daily_loss_limit * self.slot_manager.total_capital:
            logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False
        
        # Position size limit
        if order.side == OrderSide.BUY:
            current_positions = len([s for s in self.slot_manager.slots.values() if s.status == SlotStatus.FILLED])
            if current_positions >= 10:
                logger.warning("Maximum number of positions reached")
                return False
        
        # Sector limit
        if order.side == OrderSide.BUY:
            if not self.slot_manager.can_allocate_to_sector(order.ticker):
                logger.warning(f"Sector limit exceeded for {order.ticker}")
                return False
        
        return True
    
    def submit_order(self, ticker: str, side: OrderSide, quantity: float, 
                    order_type: OrderType = OrderType.LIMIT, 
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> str:
        """
        Submit a new order.
        
        Args:
            ticker: Ticker symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Order type
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order ID
        """
        order_id = self.generate_order_id()
        
        order = Order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # Risk check
        if not self.check_risk_limits(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order_id}")
            return order_id
        
        # Submit order
        order.status = OrderStatus.SUBMITTED
        self.orders[order_id] = order
        self.pending_orders.append(order_id)
        
        # Log order submission
        self.log_event("ORDER_SUBMITTED", {
            'order_id': order_id,
            'ticker': ticker,
            'side': side.value,
            'quantity': quantity,
            'order_type': order_type.value,
            'price': price
        })
        
        logger.info(f"Order submitted: {order_id} - {side.value} {quantity} {ticker}")
        
        return order_id
    
    def execute_market_order(self, order: Order) -> bool:
        """
        Execute a market order immediately.
        
        Args:
            order: Order to execute
            
        Returns:
            True if execution successful
        """
        if order.ticker not in self.market_data:
            logger.error(f"No market data for {order.ticker}")
            return False
        
        # Get current price
        current_price = self.market_data[order.ticker]['adjClose'].iloc[-1]
        
        # Calculate execution price with slippage
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + self.cost_model.slippage_pct)
        else:
            execution_price = current_price * (1 - self.cost_model.slippage_pct)
        
        # Calculate costs
        commission, slippage = self.cost_model.calculate_cost(
            order.quantity, execution_price, OrderType.MARKET
        )
        
        # Create trade
        trade = Trade(
            trade_id=self.generate_trade_id(),
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            executed_at=datetime.now()
        )
        
        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.filled_at = datetime.now()
        order.status = OrderStatus.FILLED
        order.commission = commission
        order.slippage = slippage
        
        # Store trade
        self.trades.append(trade)
        
        # Update slot manager
        if order.side == OrderSide.BUY:
            # Find empty slot and fill
            empty_slots = self.slot_manager.get_empty_slots()
            if empty_slots:
                slot_id = empty_slots[0]
                # Calculate ATR (simplified)
                atr = self.market_data[order.ticker]['high'].iloc[-1] - self.market_data[order.ticker]['low'].iloc[-1]
                volatility = self.market_data[order.ticker]['adjClose'].pct_change().rolling(20).std().iloc[-1]
                
                self.slot_manager.fill_slot(
                    slot_id, order.ticker, execution_price, 
                    atr, volatility, "BULL"  # Simplified regime
                )
        else:
            # Close position
            for slot_id, slot in self.slot_manager.slots.items():
                if slot.status == SlotStatus.FILLED and slot.position and slot.position.ticker == order.ticker:
                    self.slot_manager.close_slot(slot_id, execution_price)
                    break
        
        # Update daily P&L
        trade_pnl = (execution_price - order.price) * order.quantity if order.price else 0.0
        self.daily_pnl += trade_pnl - commission - slippage
        
        # Log execution
        self.log_event("ORDER_FILLED", {
            'order_id': order.order_id,
            'trade_id': trade.trade_id,
            'ticker': order.ticker,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'slippage': slippage
        })
        
        logger.info(f"Order filled: {order.order_id} - {order.quantity} @ ${execution_price:.2f}")
        
        return True
    
    def execute_limit_orders(self) -> None:
        """Execute pending limit orders based on current market data."""
        executed_orders = []
        
        for order_id in self.pending_orders:
            order = self.orders[order_id]
            
            if order.status != OrderStatus.SUBMITTED:
                continue
            
            if order.ticker not in self.market_data:
                continue
            
            # Get current market data
            current_price = self.market_data[order.ticker]['adjClose'].iloc[-1]
            
            # Check if limit order should execute
            should_execute = False
            
            if order.side == OrderSide.BUY and order.price:
                should_execute = current_price <= order.price
            elif order.side == OrderSide.SELL and order.price:
                should_execute = current_price >= order.price
            
            if should_execute:
                if self.execute_market_order(order):
                    executed_orders.append(order_id)
        
        # Remove executed orders from pending
        for order_id in executed_orders:
            self.pending_orders.remove(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order cancelled successfully
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.warning(f"Cannot cancel order {order_id}: status {order.status}")
            return False
        
        order.status = OrderStatus.CANCELLED
        
        # Remove from pending
        if order_id in self.pending_orders:
            self.pending_orders.remove(order_id)
        
        # Log cancellation
        self.log_event("ORDER_CANCELLED", {
            'order_id': order_id,
            'ticker': order.ticker,
            'side': order.side.value
        })
        
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        summary = self.slot_manager.get_portfolio_summary()
        return summary['allocated_capital'] + summary['cash_available'] + summary['unrealized_pnl']
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L."""
        return self.daily_pnl
    
    def log_event(self, event_type: str, data: Dict) -> None:
        """Log execution event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }
        self.execution_log.append(event)
        
        # Save to file periodically
        if len(self.execution_log) % 10 == 0:
            self.save_execution_log()
    
    def save_execution_log(self) -> None:
        """Save execution log to CSV."""
        if self.execution_log:
            df = pd.DataFrame(self.execution_log)
            df.to_csv(self.log_path, index=False)
    
    def get_execution_summary(self) -> Dict:
        """Get execution summary statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'win_rate': 0.0
            }
        
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        
        # Calculate win rate (simplified)
        buy_trades = [t for t in self.trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in self.trades if t.side == OrderSide.SELL]
        
        win_rate = 0.0
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Pair trades (simplified)
            profitable_trades = 0
            for buy in buy_trades[:len(sell_trades)]:
                sell = sell_trades[buy_trades.index(buy)]
                if sell.price > buy.price:
                    profitable_trades += 1
            win_rate = profitable_trades / len(buy_trades)
        
        return {
            'total_trades': len(self.trades),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'win_rate': win_rate,
            'daily_pnl': self.daily_pnl,
            'portfolio_value': self.get_portfolio_value()
        }

# Usage Example
if __name__ == "__main__":
    # Initialize components
    slot_manager = SlotManager(total_capital=100000)
    execution_engine = ExecutionEngine(slot_manager)
    
    # Mock market data
    market_data = {
        'AAPL': pd.DataFrame({
            'adjClose': [150.0],
            'high': [152.0],
            'low': [148.0],
            'volume': [1000000]
        }, index=[pd.Timestamp.now()])
    }
    
    # Update market data
    execution_engine.update_market_data(market_data)
    
    # Submit buy order
    order_id = execution_engine.submit_order(
        ticker='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        price=149.5
    )
    
    print(f"Submitted order: {order_id}")
    
    # Execute limit orders
    execution_engine.execute_limit_orders()
    
    # Get execution summary
    summary = execution_engine.get_execution_summary()
    print(f"Execution summary: {summary}")
