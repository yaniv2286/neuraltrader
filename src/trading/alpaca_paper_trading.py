"""
NeuralTrader Phase 6: Paper Trading Deployment
==============================================

This module implements paper trading functionality for NeuralTrader V7.11.
Connects to Alpaca API for live paper trading with real market data.

Features:
- Real-time market data via Alpaca
- Paper trading with $100,000 virtual capital
- Risk management with 0.9% per-trade risk
- Black Swan exit protection
- Live position monitoring
- Trade execution with market orders
- Portfolio tracking and reporting

Usage:
    python -m src.trading.alpaca_paper_trading

Requirements:
    pip install alpaca-trade-api
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuralTraderPaperTrading:
    """
    NeuralTrader Paper Trading System
    Implements V7.11 strategy with Alpaca paper trading
    """
    
    def __init__(self):
        """Initialize paper trading system"""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.paper_url = 'https://paper-api.alpaca.markets'
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.paper_url
        )
        
        # Trading parameters (V7.11 optimized)
        self.initial_capital = 100000.0
        self.risk_per_trade = 0.009  # 0.9% risk per trade
        self.max_positions = 10
        self.universe = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'UNH']
        
        # Risk management
        self.max_portfolio_drawdown = 0.05  # 5% weekly stop-loss
        self.black_swan_threshold = 0.15  # 15% VXX surge
        
        # Track positions and trades
        self.positions = {}
        self.trades = []
        self.portfolio_value = self.initial_capital
        
        logger.info("NeuralTrader Paper Trading initialized")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Risk per trade: {self.risk_per_trade:.1%}")
        logger.info(f"Trading universe: {self.universe}")
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'daytrade_count': int(account.daytrade_count)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_market_data(self, symbol: str, timeframe: TimeFrame = TimeFrame.Day, limit: int = 100) -> pd.DataFrame:
        """Get market data for a symbol"""
        try:
            # Get bar data
            bars = self.api.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            ).df
            
            # Convert to standard format
            df = bars.reset_index()
            df = df.rename(columns={
                'timestamp': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            return df
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, symbol: str, current_price: float) -> int:
        """Calculate position size based on risk management"""
        # Get ATR for volatility-based sizing (simplified)
        df = self.get_market_data(symbol)
        if df.empty or len(df) < 14:
            return 0
        
        # Simple ATR calculation (14-day)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Risk-based position sizing
        risk_amount = self.portfolio_value * self.risk_per_trade
        stop_distance = atr * 2.0  # 2x ATR stop
        position_value = risk_amount / stop_distance
        shares = int(position_value / current_price)
        
        # Maximum position size limits
        max_shares = int(self.portfolio_value * 0.25 / current_price)  # 25% max per position
        shares = min(shares, max_shares)
        
        return shares
    
    def check_black_swan(self) -> bool:
        """Check for Black Swan event (VXX surge > 15%)"""
        try:
            # Get VXX data
            vxx_df = self.get_market_data('VXX')
            if vxx_df.empty or len(vxx_df) < 5:
                return False
            
            # Calculate 5-day return
            recent_close = vxx_df['close'].iloc[-1]
            five_days_ago_close = vxx_df['close'].iloc[-5]
            vxx_return = (recent_close - five_days_ago_close) / five_days_ago_close
            
            if vxx_return > self.black_swan_threshold:
                logger.warning(f"BLACK SWAN EVENT: VXX surged {vxx_return:.1%} > {self.black_swan_threshold:.1%}")
                return True
            
        except Exception as e:
            logger.error(f"Error checking Black Swan: {e}")
        
        return False
    
    def execute_trade(self, symbol: str, side: str, quantity: int, reason: str = ""):
        """Execute a trade"""
        try:
            if quantity <= 0:
                return
            
            # Submit order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': self.api.get_latest_trade(symbol).price,
                'reason': reason,
                'order_id': order.id
            }
            
            self.trades.append(trade)
            logger.info(f"Trade executed: {side} {quantity} shares of {symbol} - {reason}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def manage_positions(self):
        """Manage existing positions (stop losses, Black Swan exits)"""
        try:
            # Get current positions
            positions = self.api.list_positions()
            
            # Check Black Swan
            black_swan = self.check_black_swan()
            
            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                current_price = float(position.current_price)
                
                # Black Swan exit (cut non-Super-Alpha positions by 50%)
                if black_swan and symbol not in ['AAPL', 'MSFT', 'NVDA']:  # Simplified Super-Alpha check
                    exit_qty = qty // 2
                    if exit_qty > 0:
                        self.execute_trade(symbol, 'sell', exit_qty, "Black Swan exit")
                
                # Stop loss logic (simplified - would need more sophisticated implementation)
                # This would require tracking entry prices and calculating stops
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def run_trading_session(self):
        """Run main trading session"""
        logger.info("Starting NeuralTrader Paper Trading Session")
        
        try:
            # Get account info
            account = self.get_account_info()
            self.portfolio_value = account.get('portfolio_value', self.initial_capital)
            
            logger.info(f"Portfolio value: ${self.portfolio_value:,.2f}")
            
            # Manage existing positions
            self.manage_positions()
            
            # Generate new signals (placeholder for actual ML model)
            # This would integrate with the trained XGBoost model
            for symbol in self.universe:
                try:
                    # Get current price
                    current_price = float(self.api.get_latest_trade(symbol).price)
                    
                    # Calculate position size
                    shares = self.calculate_position_size(symbol, current_price)
                    
                    # Placeholder for actual signal generation
                    # Would integrate with ML model here
                    signal = self.generate_signal(symbol)
                    
                    if signal == 'BUY' and shares > 0:
                        self.execute_trade(symbol, 'buy', shares, "ML signal")
                    elif signal == 'SELL':
                        current_positions = self.api.list_positions()
                        for pos in current_positions:
                            if pos.symbol == symbol:
                                qty = int(pos.qty)
                                if qty > 0:
                                    self.execute_trade(symbol, 'sell', qty, "ML signal")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Print summary
            self.print_session_summary()
            
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
    
    def generate_signal(self, symbol: str) -> str:
        """Generate trading signal (placeholder for ML model)"""
        # This is a placeholder - would integrate with actual XGBoost model
        # For now, return random signals for testing
        import random
        return random.choice(['BUY', 'SELL', 'HOLD'])
    
    def print_session_summary(self):
        """Print trading session summary"""
        account = self.get_account_info()
        
        logger.info("=== Trading Session Summary ===")
        logger.info(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        logger.info(f"Cash: ${account.get('cash', 0):,.2f}")
        logger.info(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        logger.info(f"Trades Executed: {len(self.trades)}")
        logger.info(f"Current Positions: {len(self.api.list_positions())}")
        
        # Print recent trades
        if self.trades:
            logger.info("Recent Trades:")
            for trade in self.trades[-5:]:  # Last 5 trades
                logger.info(f"  {trade['timestamp']}: {trade['side']} {trade['quantity']} {trade['symbol']} - {trade['reason']}")
    
    def run_continuous(self, interval_minutes: int = 5):
        """Run continuous trading with specified interval"""
        logger.info(f"Starting continuous trading with {interval_minutes} minute intervals")
        
        while True:
            try:
                self.run_trading_session()
                logger.info(f"Waiting {interval_minutes} minutes until next trading cycle...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous trading: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    """Main function to run paper trading"""
    try:
        trader = NeuralTraderPaperTrading()
        
        # Run single session for testing
        trader.run_trading_session()
        
        # For continuous trading, uncomment:
        # trader.run_continuous(interval_minutes=5)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
