"""
Interactive Brokers Execution Engine (ib_insync)
================================================

Synchronous IBKR engine for NeuralTrader paper trading.
Built on ib_insync which manages its own internal event loop —
no asyncio wrappers needed, no event-loop conflicts on Windows.

Reference: D:/GitHub/VolatilityHunter/src/brokerage_interface.py (IBKRInterface)

Port mapping:
  7497 — TWS Paper Trading  (default)
  7496 — TWS Live Trading
  4002 — IB Gateway Paper
  4001 — IB Gateway Live
"""

import os
import socket
import random
import logging
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check — ib_insync only, no ib_async
# ---------------------------------------------------------------------------
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    IB_AVAILABLE = True
    logger.info("[IBKR] ib_insync available")
except ImportError:
    IB_AVAILABLE = False
    logger.warning("[IBKR] ib_insync not installed. Run: pip install ib_insync")
except Exception as e:
    IB_AVAILABLE = False
    logger.warning(f"[IBKR] ib_insync import error: {e}")


# ---------------------------------------------------------------------------
# Data classes (kept for backward compatibility)
# ---------------------------------------------------------------------------

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT  = "LMT"

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"

@dataclass
class OrderResult:
    order_id:        int
    status:          str
    filled:          bool
    filled_quantity: int
    fill_price:      float
    commission:      float
    timestamp:       datetime
    error:           Optional[str] = None

@dataclass
class AccountSummary:
    cash_balance:     float
    portfolio_value:  float
    buying_power:     float
    equity_with_loan: float
    total_positions:  int
    timestamp:        datetime
    account_id:       str = ""

@dataclass
class Position:
    symbol:         str
    quantity:       int
    market_price:   float
    market_value:   float
    average_cost:   float
    unrealized_pnl: float
    side:           str


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class IBKRExecutionEngine:
    """
    NeuralTrader IBKR Execution Engine — ib_insync, synchronous API.

    ib_insync manages its own event loop internally.
    All public methods are synchronous and safe to call from the scheduler.
    """

    def __init__(
        self,
        host:      str = "127.0.0.1",
        port:      int = 7497,
        client_id: int = None,
    ):
        self.host      = host
        self.port      = port
        self.client_id = client_id or random.randint(100, 999)
        self.ib:       Optional[IB] = None
        self.connected = False
        self.logger    = logging.getLogger(__name__)
        self.logger.info(
            f"[IBKR] Engine init | host={host} port={port} clientId={self.client_id}"
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _port_open(self) -> bool:
        """Quick TCP probe to avoid hanging on connect()."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(5)
        result = probe.connect_ex((self.host, self.port)) == 0
        probe.close()
        return result

    def connect(self) -> bool:
        """
        Connect to IBKR TWS / IB Gateway.
        Uses ib_insync's synchronous connect() — no asyncio event loop needed.
        """
        if not IB_AVAILABLE:
            self.logger.error("[IBKR] ib_insync not installed — cannot connect")
            return False

        try:
            if not self._port_open():
                self.logger.error(
                    f"[IBKR] Port {self.port} not reachable. "
                    "Is TWS / IB Gateway running?"
                )
                return False

            self.ib = IB()
            self.ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                timeout=15,
                readonly=False,
            )

            if not self.ib.isConnected():
                self.logger.error("[IBKR] connect() returned but isConnected() is False")
                return False

            self.connected = True

            # Log available cash for confirmation
            try:
                for v in self.ib.accountValues():
                    if v.tag == 'AvailableFunds' and v.currency == 'USD':
                        self.logger.info(
                            f"[IBKR] Connected | clientId={self.client_id} "
                            f"| AvailableFunds=${float(v.value):,.2f}"
                        )
                        break
            except Exception:
                self.logger.info(f"[IBKR] Connected | clientId={self.client_id}")

            return True

        except Exception as e:
            self.logger.error(f"[IBKR] connect() failed: {e}")
            self.logger.error(traceback.format_exc())
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect cleanly."""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                self.logger.info("[IBKR] Disconnected")
        except Exception as e:
            self.logger.error(f"[IBKR] disconnect() error: {e}")
        finally:
            self.ib        = None
            self.connected = False

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Returns a dict with keys:
          cash_balance, portfolio_value, buying_power, equity_with_loan,
          total_positions, account_id, timestamp
        On error returns {'error': <message>}.
        """
        if not self.connected or not self.ib:
            return {'error': 'Not connected to IBKR'}

        try:
            self.logger.info("[IBKR] Fetching account summary...")

            cash_balance     = 0.0
            portfolio_value  = 0.0
            buying_power     = 0.0
            equity_with_loan = 0.0
            account_id       = ""

            for v in self.ib.accountValues():
                if v.currency != 'USD':
                    continue
                if v.tag == 'AvailableFunds':
                    cash_balance = float(v.value)
                elif v.tag == 'NetLiquidation':
                    equity_with_loan = float(v.value)
                    portfolio_value  = float(v.value)
                elif v.tag == 'BuyingPower':
                    buying_power = float(v.value)
                elif v.tag == 'GrossPositionValue':
                    portfolio_value = float(v.value)
                if v.account:
                    account_id = v.account

            positions = self.ib.positions()

            self.logger.info(
                f"[IBKR] Account | Cash=${cash_balance:,.2f} "
                f"| NetLiq=${equity_with_loan:,.2f} "
                f"| Positions={len(positions)}"
            )

            return {
                'cash_balance':     cash_balance,
                'portfolio_value':  portfolio_value,
                'buying_power':     buying_power,
                'equity_with_loan': equity_with_loan,
                'total_positions':  len(positions),
                'account_id':       account_id,
                'timestamp':        datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"[IBKR] get_account_summary() error: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Returns list of dicts with keys:
          symbol, quantity, market_price, market_value,
          average_cost, unrealized_pnl, side
        Returns [] on error.
        """
        if not self.connected or not self.ib:
            self.logger.warning("[IBKR] get_positions() called but not connected")
            return []

        try:
            self.logger.info("[IBKR] Fetching positions...")
            result = []
            for pos in self.ib.positions():
                if pos.contract.secType != 'STK':
                    continue
                qty = pos.position
                avg = float(pos.avgCost)
                mv  = float(pos.marketValue) if hasattr(pos, 'marketValue') else avg * abs(qty)
                mp  = mv / qty if qty != 0 else avg
                pnl = mv - avg * abs(qty)

                result.append({
                    'symbol':        pos.contract.symbol,
                    'quantity':      int(qty),
                    'market_price':  round(mp, 4),
                    'market_value':  round(mv, 2),
                    'average_cost':  round(avg, 4),
                    'unrealized_pnl': round(pnl, 2),
                    'side':          'long' if qty > 0 else 'short',
                })

            self.logger.info(f"[IBKR] {len(result)} stock positions fetched")
            return result

        except Exception as e:
            self.logger.error(f"[IBKR] get_positions() error: {e}")
            self.logger.error(traceback.format_exc())
            return []

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol:   str,
        quantity: int,
        side:     str,
    ) -> Dict[str, Any]:
        """
        Place a market order.
        side: 'BUY' or 'SELL' (case-insensitive).
        Returns dict with keys: success, order_id, symbol, quantity, side, status.
        """
        if not self.connected or not self.ib:
            return {'success': False, 'reason': 'Not connected to IBKR'}

        if quantity <= 0:
            return {'success': False, 'reason': f'Invalid quantity: {quantity}'}

        action = side.upper()
        if action not in ('BUY', 'SELL'):
            return {'success': False, 'reason': f'Invalid side: {side}'}

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            order    = MarketOrder(action, quantity)
            trade    = self.ib.placeOrder(contract, order)

            self.logger.info(
                f"[IBKR] Market order placed | {action} {quantity} {symbol} "
                f"| orderId={trade.order.orderId}"
            )
            return {
                'success':  True,
                'order_id': str(trade.order.orderId),
                'symbol':   symbol,
                'quantity': quantity,
                'side':     action,
                'type':     'market',
                'status':   trade.orderStatus.status,
            }

        except Exception as e:
            self.logger.error(f"[IBKR] place_market_order() error: {e}")
            self.logger.error(traceback.format_exc())
            return {'success': False, 'reason': str(e)}

    def place_limit_order(
        self,
        symbol:   str,
        quantity: int,
        side:     str,
        price:    float,
    ) -> Dict[str, Any]:
        """
        Place a limit order.
        Returns dict with keys: success, order_id, symbol, quantity, side, price, status.
        """
        if not self.connected or not self.ib:
            return {'success': False, 'reason': 'Not connected to IBKR'}

        if quantity <= 0:
            return {'success': False, 'reason': f'Invalid quantity: {quantity}'}

        action = side.upper()
        if action not in ('BUY', 'SELL'):
            return {'success': False, 'reason': f'Invalid side: {side}'}

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            order    = LimitOrder(action, quantity, price)
            trade    = self.ib.placeOrder(contract, order)

            self.logger.info(
                f"[IBKR] Limit order placed | {action} {quantity} {symbol} @ ${price:.2f} "
                f"| orderId={trade.order.orderId}"
            )
            return {
                'success':  True,
                'order_id': str(trade.order.orderId),
                'symbol':   symbol,
                'quantity': quantity,
                'side':     action,
                'type':     'limit',
                'price':    price,
                'status':   trade.orderStatus.status,
            }

        except Exception as e:
            self.logger.error(f"[IBKR] place_limit_order() error: {e}")
            self.logger.error(traceback.format_exc())
            return {'success': False, 'reason': str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order by order_id string."""
        if not self.connected or not self.ib:
            return {'success': False, 'reason': 'Not connected to IBKR'}

        try:
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    self.ib.cancelOrder(trade.order)
                    self.logger.info(f"[IBKR] Order cancelled | orderId={order_id}")
                    return {'success': True, 'order_id': order_id}

            return {'success': False, 'reason': f'Order {order_id} not found in open trades'}

        except Exception as e:
            self.logger.error(f"[IBKR] cancel_order() error: {e}")
            return {'success': False, 'reason': str(e)}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an open order."""
        if not self.connected or not self.ib:
            return {'success': False, 'reason': 'Not connected to IBKR'}

        try:
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    return {
                        'success':          True,
                        'order_id':         order_id,
                        'symbol':           trade.contract.symbol,
                        'quantity':         trade.order.totalQuantity,
                        'side':             trade.order.action.lower(),
                        'type':             trade.order.orderType.lower(),
                        'status':           trade.orderStatus.status,
                        'filled_qty':       trade.orderStatus.filled,
                        'filled_avg_price': trade.orderStatus.avgFillPrice or 0.0,
                    }

            return {'success': False, 'reason': f'Order {order_id} not found'}

        except Exception as e:
            self.logger.error(f"[IBKR] get_order_status() error: {e}")
            return {'success': False, 'reason': str(e)}

    # ------------------------------------------------------------------
    # High-level helpers used by NeuralTrader orchestrator
    # ------------------------------------------------------------------

    def execute_trade(
        self,
        symbol:      str,
        side:        Any,        # OrderSide enum or str 'BUY'/'SELL'
        order_type:  Any = None, # OrderType enum or str 'MKT'/'LMT'
        quantity:    int  = 1,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Unified trade execution — wraps place_market_order / place_limit_order.
        Accepts both enum and string arguments for backward compatibility.
        """
        action = side.value if isinstance(side, OrderSide) else str(side).upper()
        otype  = (order_type.value if isinstance(order_type, OrderType)
                  else str(order_type).upper() if order_type else 'MKT')

        if otype == 'LMT' and limit_price is not None:
            res = self.place_limit_order(symbol, quantity, action, limit_price)
        else:
            res = self.place_market_order(symbol, quantity, action)

        if res.get('success'):
            return OrderResult(
                order_id=int(res.get('order_id', 0)),
                status=res.get('status', 'Submitted'),
                filled=False,
                filled_quantity=0,
                fill_price=limit_price or 0.0,
                commission=0.0,
                timestamp=datetime.now(),
            )
        else:
            return OrderResult(
                order_id=0,
                status='ERROR',
                filled=False,
                filled_quantity=0,
                fill_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error=res.get('reason', 'Unknown error'),
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ---------------------------------------------------------------------------
# IBKREngineSync — alias kept for backward compatibility with orchestrator
# ---------------------------------------------------------------------------
IBKREngineSync = IBKRExecutionEngine


# ---------------------------------------------------------------------------
# Quick connection test (run standalone)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    engine = IBKRExecutionEngine(port=7497)
    if not engine.connect():
        print("[FAIL] Could not connect to IBKR TWS on port 7497")
        sys.exit(1)

    try:
        acct = engine.get_account_summary()
        print(f"[OK] Account | Cash=${acct.get('cash_balance', 0):,.2f} "
              f"| NetLiq=${acct.get('equity_with_loan', 0):,.2f}")

        positions = engine.get_positions()
        print(f"[OK] Positions: {len(positions)}")
        for p in positions:
            print(f"     {p['symbol']}: {p['quantity']} shares @ ${p['average_cost']:.2f}")

    finally:
        engine.disconnect()
        print("[OK] Disconnected")
