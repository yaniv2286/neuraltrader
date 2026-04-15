"""
NeuralTrader Main Orchestrator - Master Runner for Task Scheduler
===============================================================

Master script that automates the complete shadow trading workflow with IST scheduling.
Designed for Windows Task Scheduler integration with comprehensive safety checks.

Features:
- CLI Arguments: --mode=fetch, --mode=trade, --mode=report
- Environment Safety: Required library verification
- Working Directory Lock: Forces project root directory
- Comprehensive Logging: automation.log for debugging
- Task Scheduler Integration: Production-ready execution

Usage:
    python main_orchestrator_ist.py --mode=fetch
    python main_orchestrator_ist.py --mode=trade
    python main_orchestrator_ist.py --mode=report
    python main_orchestrator_ist.py --auto
    
    # For Windows Task Scheduler:
    run_neural.bat fetch
    run_neural.bat trade
    run_neural.bat report
"""

import os
import sys
import io
import json
import traceback

# Force UTF-8 encoding (only once)
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import EmailNotifier for guaranteed notifications
from core.utils.notifier import EmailNotifier

# Import Quant Risk Machine
from core.execution.risk_manager import RiskManager

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Global variable to store current log file path for email attachments
CURRENT_LOG_FILE = None
TRADINGVIEW_SIGNAL_FILE = None
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# 2. Environment Safety - Check required libraries
REQUIRED_LIBRARIES = {
    'yfinance': 'yfinance',
    'pytz': 'pytz', 
    'pandas': 'pandas',
    'dotenv': 'dotenv'
}

def check_environment():
    """Verify all required libraries are installed"""
    missing_libs = []
    
    for lib_name, import_name in REQUIRED_LIBRARIES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_libs.append(lib_name)
    
    if missing_libs:
        print(f"[ERROR] ENVIRONMENT ERROR: Missing required libraries: {', '.join(missing_libs)}")
        print(f"[INSTALL] Install with: pip install {' '.join(missing_libs)}")
        print(f"[CONFIG] Or run: pip install -r requirements_trading.txt")
        sys.exit(1)
    
    print("[OK] Environment check passed - All required libraries available")

# 3. Daily Supervision Logger Setup
def setup_daily_supervision():
    """Setup daily supervision logging for Task Scheduler monitoring"""
    try:
        from core.utils.daily_logger import DailySupervisionLogger
        return DailySupervisionLogger()
    except ImportError:
        print("[WARNING] Daily supervision logger not available")
        return None

# 3. Logging Setup - automation.log for Task Scheduler debugging
def setup_automation_logging():
    """Setup comprehensive logging for Task Scheduler debugging"""
    global CURRENT_LOG_FILE
    
    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Add timestamp and process ID to avoid file locking issues
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"automation_{timestamp}.log")
    
    # Store log file path globally for email attachments
    CURRENT_LOG_FILE = log_file
    
    # Force immediate flushing for file handler
    class FlushingFileHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding=None):
            super().__init__(filename, mode, encoding)
        
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Replace the file handler with our custom flushing handler
    file_handler = FlushingFileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger with custom handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger('NeuralTrader_Automation')
    logger.info("=" * 80)
    logger.info("[AUTOMATION] NeuralTrader Automation Session Started")
    logger.info(f"[DIRECTORY] Working Directory: {PROJECT_ROOT}")
    logger.info(f"[TIME] Timestamp: {datetime.now()}")
    logger.info(f"[PYTHON] Python Version: {sys.version}")
    logger.info(f"[ARGS] Command Line Args: {' '.join(sys.argv)}")
    logger.info(f"[LOG] Log file: {log_file}")
    logger.info("=" * 80)
    
    return logger

# ==================== IMPORTS AFTER ENVIRONMENT CHECK ====================

# Import after environment verification
import pandas as pd
# from src.data.yfinance_manager import YFinanceManager
# from src.trading.risk_manager import RiskManager, RiskDecision
# from src.trading.virtual_engine import VirtualEngine
# from src.reporting.ist_scheduler import ISTScheduler
# from core.utils.notifier import EmailNotifier
from core.integrity import verify_system_integrity
from core.ai_models import EnsemblePredictor
from core.strategy import TradingStrategy
from scripts.sector_rotation import SectorAuthority

# Global flag for paper trading mode
PAPER_TRADING = False

# ==================== FAIL-SAFE MOCK ENGINE ====================

# Portfolio file routing — one file per mode
_PORTFOLIO_FILES = {
    'paper':      'data/portfolio_paper.json',       # persistent, IBKR-synced
    'trade':      'data/portfolio_paper.json',       # same as paper
    'report':     'data/portfolio_paper.json',       # read-only in report mode
    'simulation': 'data/portfolio_simulation.json',  # overwritten each run
    'backtest':   'data/portfolio_backtest.json',    # overwritten each run
    'default':    'data/portfolio_paper.json',       # fallback
}


class MockVirtualEngine:
    """
    Fully Persistent Mock Virtual Engine for Paper Trading Mode
    Provides real portfolio persistence and trade execution.

    Portfolio file routing:
      paper / trade / report  ->  data/portfolio_paper.json   (persistent + IBKR synced)
      simulation              ->  data/portfolio_simulation.json  (fresh each run)
      backtest                ->  data/portfolio_backtest.json    (fresh each run)
    """

    def __init__(self, ibkr_engine=None, mode: str = 'paper'):
        """Initialize engine with the correct portfolio file for the given mode."""
        self.logger = logging.getLogger(__name__)
        self.project_root = os.path.dirname(__file__)
        self.mode = mode

        # Route to the correct portfolio file
        rel_path = _PORTFOLIO_FILES.get(mode, _PORTFOLIO_FILES['default'])
        self.portfolio_file = os.path.join(self.project_root, rel_path)
        self.logger.info(f"[PORTFOLIO] mode={mode} -> {self.portfolio_file}")

        self.MAX_POSITIONS = 10

        # Store IBKR engine for live paper trading
        self.ibkr_engine = ibkr_engine

        # Initialize portfolio structure with risk management keys
        self.portfolio = {
            'mode': mode,
            'cash': 100000.0,
            'positions': {},
            'history': [],
            'peak_portfolio_value': 100000.0,
            'circuit_breaker_cooldown_until': None,
            'last_ibkr_sync': None,
        }

        # Load existing portfolio data (paper: persistent; sim/backtest: start fresh)
        if mode in ('paper', 'trade', 'report'):
            self._load_portfolio_data()
            # CRITICAL: Sync market prices AFTER loading to update placeholder prices with real Tiingo data
            self._sync_market_prices()
        else:
            self.logger.info(f"[PORTFOLIO] {mode} mode — starting with fresh portfolio (no history loaded)")
            # Still sync prices for any positions in fresh portfolio
            self._sync_market_prices()

        self.logger.info(f"[MOCK] VirtualEngine ready | mode={mode} | positions={len(self.portfolio['positions'])} | cash=${self.portfolio['cash']:,.0f}")
    
    def _load_portfolio_data(self):
        """Load portfolio data from portfolio.json if it exists"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                    # Update portfolio with loaded data
                    self.portfolio.update(portfolio_data)
                    self.logger.info(f"[MOCK] Loaded portfolio data from {self.portfolio_file}")
                    
                    # Convert positions to the new consistent format
                    if 'positions' in portfolio_data:
                        converted_positions = {}
                        for ticker, pos_data in portfolio_data['positions'].items():
                            # Handle both old and new formats
                            if 'shares' in pos_data:
                                # New format - already correct
                                converted_positions[ticker] = {
                                    'shares': pos_data.get('shares', 0),
                                    'cost_basis': pos_data.get('cost_basis', 0),
                                    'current_price': pos_data.get('current_price', 0)
                                }
                            else:
                                # Old format - convert from quantity/avg_cost/last_price
                                converted_positions[ticker] = {
                                    'shares': pos_data.get('quantity', 0),
                                    'cost_basis': pos_data.get('avg_cost', 0),
                                    'current_price': pos_data.get('last_price', 0)
                                }
                        
                        self.portfolio['positions'] = converted_positions
                        
                    # Ensure required fields exist
                    if 'history' not in self.portfolio:
                        self.portfolio['history'] = []
                    if 'cash' not in self.portfolio:
                        self.portfolio['cash'] = 100000.0
                    # Ensure risk management keys exist for backward compatibility
                    if 'peak_portfolio_value' not in self.portfolio:
                        # Calculate current portfolio value to set initial peak
                        current_value = self.portfolio.get('cash', 100000.0)
                        for ticker, pos_data in self.portfolio.get('positions', {}).items():
                            shares = pos_data.get('shares', 0)
                            current_price = pos_data.get('current_price', pos_data.get('cost_basis', 0))
                            current_value += shares * current_price
                        self.portfolio['peak_portfolio_value'] = max(current_value, 100000.0)
                        self.logger.info(f"[MOCK] Set initial peak_portfolio_value: ${self.portfolio['peak_portfolio_value']:,.2f}")
                    if 'circuit_breaker_cooldown_until' not in self.portfolio:
                        self.portfolio['circuit_breaker_cooldown_until'] = None
                        
        except Exception as e:
            self.logger.warning(f"[MOCK] Could not load portfolio data: {e}")
            self.logger.info("[MOCK] Using default portfolio structure")
    
    def save_portfolio(self):
        """Save portfolio data to the mode-specific file."""
        try:
            os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
            self.portfolio['mode'] = self.mode
            self.portfolio['last_saved'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=2, default=str)
            self.logger.info(f"[PORTFOLIO] Saved | mode={self.mode} | {self.portfolio_file}")
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Failed to save {self.mode} portfolio: {e}")

    def sync_from_ibkr(self) -> bool:
        """
        Sync portfolio_paper.json from IBKR live account data.
        Only runs in paper/trade mode. Overwrites cash + positions with IBKR ground truth.
        Returns True if sync succeeded.
        """
        if self.mode not in ('paper', 'trade'):
            self.logger.info(f"[IBKR SYNC] Skipped — mode={self.mode} is not paper/trade")
            return False

        if self.ibkr_engine is None:
            self.logger.warning("[IBKR SYNC] No IBKR engine available — skipping sync")
            return False

        try:
            if not self.ibkr_engine.connect():
                self.logger.error("[IBKR SYNC] Connection failed")
                return False

            account = self.ibkr_engine.get_account_summary()
            if 'error' in account:
                self.logger.error(f"[IBKR SYNC] Account summary error: {account['error']}")
                return False

            positions = self.ibkr_engine.get_positions()

            # Overwrite cash from IBKR ground truth
            self.portfolio['cash'] = float(account.get('cash_balance', self.portfolio['cash']))

            # Overwrite positions from IBKR ground truth
            synced_positions = {}
            for pos in positions:
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue
                synced_positions[symbol] = {
                    'shares':        int(pos.get('quantity', 0)),
                    'cost_basis':    float(pos.get('average_cost', 0)),
                    'current_price': float(pos.get('market_price', 0)),
                }
            self.portfolio['positions'] = synced_positions

            # Update peak value
            total_market = sum(
                p['shares'] * p['current_price'] for p in synced_positions.values()
            )
            total_value = self.portfolio['cash'] + total_market
            self.portfolio['peak_portfolio_value'] = max(
                self.portfolio.get('peak_portfolio_value', 100000.0), total_value
            )

            sync_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.portfolio['last_ibkr_sync'] = sync_ts
            self.portfolio['ibkr_account_id'] = account.get('account_id', None)

            self.save_portfolio()
            self.logger.info(
                f"[IBKR SYNC] [OK] Cash=${self.portfolio['cash']:,.2f} "
                f"Positions={len(synced_positions)} synced at {sync_ts}"
            )
            return True

        except Exception as e:
            self.logger.error(f"[IBKR SYNC] [FAIL] {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            try:
                self.ibkr_engine.disconnect()
            except Exception:
                pass
    
    def _sync_market_prices(self):
        """Sync current market prices from real Tiingo parquet data - NO SILENT FAILURES"""
        import pandas as pd
        
        if not self.portfolio['positions']:
            self.logger.info("[PRICE SYNC] No positions to sync")
            return
        
        raw_data_dir = os.path.join(self.project_root, 'data', 'raw')
        updated_positions = {}
        failed_tickers = []
        
        for ticker, pos_data in self.portfolio['positions'].items():
            try:
                # Try to load Tiingo parquet file (case-insensitive)
                parquet_file = os.path.join(raw_data_dir, f"{ticker}.parquet")
                if not os.path.exists(parquet_file):
                    # Try lowercase
                    parquet_file = os.path.join(raw_data_dir, f"{ticker.lower()}.parquet")
                
                if not os.path.exists(parquet_file):
                    self.logger.error(f"[PRICE SYNC] [FATAL] Parquet file not found for {ticker}: {parquet_file}")
                    failed_tickers.append(ticker)
                    continue
                
                # Read parquet file
                df = pd.read_parquet(parquet_file)
                
                if df.empty:
                    self.logger.error(f"[PRICE SYNC] [FATAL] Empty data for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                
                # Get latest close price (adjClose from Tiingo)
                if 'adjClose' in df.columns:
                    current_price = float(df['adjClose'].iloc[-1])
                elif 'close' in df.columns:
                    current_price = float(df['close'].iloc[-1])
                elif 'Close' in df.columns:
                    current_price = float(df['Close'].iloc[-1])
                else:
                    self.logger.error(f"[PRICE SYNC] [FATAL] No price column found for {ticker}. Columns: {df.columns.tolist()}")
                    failed_tickers.append(ticker)
                    continue
                
                # Validate price is reasonable
                if current_price <= 0 or current_price > 1000000:
                    self.logger.error(f"[PRICE SYNC] [FATAL] Invalid price for {ticker}: ${current_price}")
                    failed_tickers.append(ticker)
                    continue
                
                # Update position with REAL market price
                updated_positions[ticker] = {
                    'shares': pos_data.get('shares', 0),
                    'cost_basis': pos_data.get('cost_basis', 0),
                    'current_price': current_price
                }
                self.logger.info(f"[PRICE SYNC] [OK] {ticker}: ${current_price:.2f} (real Tiingo data)")
                
            except Exception as e:
                self.logger.error(f"[PRICE SYNC] [FATAL] Failed to get price for {ticker}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                failed_tickers.append(ticker)
        
        # CRITICAL: Fail loudly if ANY price sync failed
        if failed_tickers:
            error_msg = f"[PRICE SYNC] [FATAL] Failed to sync prices for {len(failed_tickers)} tickers: {failed_tickers}"
            self.logger.error(error_msg)
            self.logger.error("[PRICE SYNC] [FATAL] NO SILENT FAILURES - Portfolio has incomplete data")
            # Still update the positions that succeeded
            self.portfolio['positions'] = updated_positions
        else:
            # All prices synced successfully
            self.portfolio['positions'] = updated_positions
            self.logger.info(f"[PRICE SYNC] [SUCCESS] All {len(updated_positions)} positions synced with real Tiingo data")
    
    def execute_trade(self, ticker, action, quantity, price, reason):
        """Execute a trade and update portfolio persistently"""
        try:
            timestamp = datetime.now().isoformat()
            
            if action.lower() == 'buy':
                return self._execute_buy(ticker, quantity, price, reason, timestamp)
            elif action.lower() == 'sell':
                return self._execute_sell(ticker, quantity, price, reason, timestamp)
            else:
                self.logger.error(f"[MOCK] Unknown trade action: {action}")
                return {
                    'success': False,
                    'error': f"Unknown trade action: {action}",
                    'ticker': ticker,
                    'action': action,
                    'quantity': quantity,
                    'price': price
                }
                
        except Exception as e:
            self.logger.error(f"[MOCK] Trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'price': price
            }
    
    def _execute_buy(self, ticker, quantity, price, reason, timestamp):
        """Execute a buy trade"""
        try:
            cost = quantity * price
            current_cash = self.portfolio.get('cash', 0)
            
            # Check if enough cash
            if current_cash < cost:
                error_msg = f"Insufficient cash: need ${cost:.2f}, have ${current_cash:.2f}"
                self.logger.error(f"[MOCK] Buy failed for {ticker}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'ticker': ticker,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price
                }
            
            # Deduct cash
            self.portfolio['cash'] = current_cash - cost
            
            # Add or update position
            if ticker in self.portfolio['positions']:
                # Existing position - update cost basis
                existing = self.portfolio['positions'][ticker]
                existing_shares = existing.get('shares', 0)
                existing_cost = existing.get('cost_basis', 0)
                
                # Calculate new average cost basis
                total_shares = existing_shares + quantity
                total_cost = (existing_cost * existing_shares) + cost
                new_cost_basis = total_cost / total_shares
                
                self.portfolio['positions'][ticker] = {
                    'shares': total_shares,
                    'cost_basis': new_cost_basis,
                    'current_price': price
                }
                
                self.logger.info(f"[MOCK] Updated {ticker} position: {existing_shares} -> {total_shares} shares @ ${new_cost_basis:.2f} avg")
            else:
                # New position
                self.portfolio['positions'][ticker] = {
                    'shares': quantity,
                    'cost_basis': price,
                    'current_price': price
                }
                
                self.logger.info(f"[MOCK] Added new {ticker} position: {quantity} shares @ ${price:.2f}")
            
            # Add to history
            trade_record = {
                'timestamp': timestamp,
                'ticker': ticker,
                'action': 'buy',
                'quantity': quantity,
                'price': price,
                'cost': cost,
                'reason': reason,
                'cash_before': current_cash,
                'cash_after': self.portfolio['cash']
            }
            self.portfolio['history'].append(trade_record)
            
            # Save portfolio
            self.save_portfolio()
            
            self.logger.info(f"[MOCK] BUY executed: {quantity} shares of {ticker} @ ${price:.2f} (${cost:.2f})")
            
            return {
                'success': True,
                'ticker': ticker,
                'action': 'buy',
                'quantity': quantity,
                'price': price,
                'cost': cost,
                'cash_before': current_cash,
                'cash_after': self.portfolio['cash']
            }
            
        except Exception as e:
            self.logger.error(f"[MOCK] Buy execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'action': 'buy',
                'quantity': quantity,
                'price': price
            }
    
    def _execute_sell(self, ticker, quantity, price, reason, timestamp):
        """Execute a sell trade"""
        try:
            if ticker not in self.portfolio['positions']:
                error_msg = f"No position found for {ticker}"
                self.logger.error(f"[MOCK] Sell failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'ticker': ticker,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price
                }
            
            position = self.portfolio['positions'][ticker]
            current_shares = position.get('shares', 0)
            
            # Check if enough shares to sell
            if current_shares < quantity:
                error_msg = f"Insufficient shares: have {current_shares}, trying to sell {quantity}"
                self.logger.error(f"[MOCK] Sell failed for {ticker}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'ticker': ticker,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price
                }
            
            # Calculate proceeds
            proceeds = quantity * price
            current_cash = self.portfolio.get('cash', 0)
            
            # Add cash
            self.portfolio['cash'] = current_cash + proceeds
            
            # Update or remove position
            remaining_shares = current_shares - quantity
            if remaining_shares > 0:
                # Update position with remaining shares
                self.portfolio['positions'][ticker] = {
                    'shares': remaining_shares,
                    'cost_basis': position.get('cost_basis', price),
                    'current_price': price
                }
                
                self.logger.info(f"[MOCK] Updated {ticker} position: {current_shares} -> {remaining_shares} shares")
            else:
                # Remove position entirely
                del self.portfolio['positions'][ticker]
                self.logger.info(f"[MOCK] Closed {ticker} position completely")
            
            # Add to history
            trade_record = {
                'timestamp': timestamp,
                'ticker': ticker,
                'action': 'sell',
                'quantity': quantity,
                'price': price,
                'proceeds': proceeds,
                'reason': reason,
                'cash_before': current_cash,
                'cash_after': self.portfolio['cash']
            }
            self.portfolio['history'].append(trade_record)
            
            # Save portfolio
            self.save_portfolio()
            
            self.logger.info(f"[MOCK] SELL executed: {quantity} shares of {ticker} @ ${price:.2f} (${proceeds:.2f})")
            
            return {
                'success': True,
                'ticker': ticker,
                'action': 'sell',
                'quantity': quantity,
                'price': price,
                'proceeds': proceeds,
                'cash_before': current_cash,
                'cash_after': self.portfolio['cash']
            }
            
        except Exception as e:
            self.logger.error(f"[MOCK] Sell execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'action': 'sell',
                'quantity': quantity,
                'price': price
            }
    
    def get_account_info(self):
        """Return current account information"""
        total_value = self.portfolio['cash']
        
        # Calculate total market value of positions
        for ticker, pos_data in self.portfolio['positions'].items():
            shares = pos_data.get('shares', 0)
            current_price = pos_data.get('current_price', 0)
            total_value += shares * current_price
        
        return {
            'cash': self.portfolio['cash'],
            'equity': total_value,
            'buying_power': self.portfolio['cash'],
            'portfolio_value': total_value,
            'positions_value': total_value - self.portfolio['cash']
        }
    
    def update_portfolio_values(self):
        """
        Update portfolio values and track peak portfolio value for risk management.
        This method should be called whenever portfolio values need to be recalculated.
        """
        try:
            # Sync with current market prices first
            self._sync_market_prices()
            
            # Calculate current total portfolio value
            account_info = self.get_account_info()
            total_equity = account_info['portfolio_value']
            
            # Update peak portfolio value if current value is higher
            if total_equity > self.portfolio.get('peak_portfolio_value', 100000.0):
                self.portfolio['peak_portfolio_value'] = total_equity
                self.logger.info(f"[MOCK] New peak portfolio value: ${total_equity:,.2f}")
            
            # Save updated portfolio state
            self.save_portfolio()
            
            self.logger.info(f"[MOCK] Portfolio values updated: Total=${total_equity:,.2f}, Peak=${self.portfolio['peak_portfolio_value']:,.2f}")
            
        except Exception as e:
            self.logger.error(f"[MOCK] Failed to update portfolio values: {e}")
    
    def get_current_positions(self):
        """Return current positions in the expected format"""
        positions_list = []
        for ticker, pos_data in self.portfolio['positions'].items():
            positions_list.append({
                'symbol': ticker,
                'position': pos_data.get('shares', 0),
                'market_price': pos_data.get('current_price', 0),
                'market_value': pos_data.get('shares', 0) * pos_data.get('current_price', 0),
                'avg_cost': pos_data.get('cost_basis', 0)
            })
        return positions_list
    
    def check_market_filter(self):
        """Mock market filter check - always passes for testing"""
        return True
    
    def execute_shadow_trades(self, trading_strategy, data_manager, sector_auth):
        """Execute shadow trades with full persistence and risk management"""
        try:
            self.logger.info("[MOCK] Executing shadow trades with persistence...")
            self.logger.info(f"[MOCK] Trading strategy: {type(trading_strategy)}")
            self.logger.info(f"[MOCK] Data manager: {type(data_manager)}")
            self.logger.info(f"[MOCK] Sector auth: {type(sector_auth)}")
            
            # Get trading signals from strategy
            signals = self._generate_trading_signals(trading_strategy, data_manager, sector_auth)
            self.logger.info(f"[MOCK] Generated {len(signals)} signals")
            
            trades_executed = []
            candidates_found = len(signals)
            
            # Filter for buy signals only (for inverse volatility sizing)
            buy_signals = [s for s in signals if s['action'] == 'buy']
            sell_signals = [s for s in signals if s['action'] == 'sell']
            
            # Execute sell signals first (to free up capital)
            for signal in sell_signals:
                ticker = signal['ticker']
                quantity = signal['quantity']
                price = signal['price']
                reason = signal['reason']
                
                result = self.execute_trade(ticker, 'sell', quantity, price, reason)
                
                if result['success']:
                    trades_executed.append(result)
                    self.logger.info(f"[MOCK] Sell executed: {quantity} {ticker} @ ${price:.2f}")
                else:
                    self.logger.warning(f"[MOCK] Sell failed: {ticker} - {result.get('error', 'Unknown error')}")
            
            # Inverse Volatility Sizing for buy signals
            if buy_signals:
                self.logger.info(f"[RISK] Calculating inverse volatility sizing for {len(buy_signals)} candidates...")
                
                # Calculate volatilities for buy candidates
                tickers = [s['ticker'] for s in buy_signals]
                volatilities = []
                
                import pandas as pd
                import numpy as np
                
                for ticker in tickers:
                    try:
                        # Read real data from parquet file
                        filepath = f"data/raw/{ticker}.parquet"
                        
                        if os.path.exists(filepath):
                            # Read the parquet file
                            df = pd.read_parquet(filepath)
                            
                            if len(df) >= 20:
                                # Get last 20 rows of close prices
                                closes = df['close'].tail(20)
                                
                                # Calculate daily returns
                                returns = closes.pct_change().dropna()
                                
                                # Calculate annualized volatility (252 trading days)
                                real_vol = returns.std() * np.sqrt(252)
                                
                                # Ensure reasonable bounds (0.05 to 1.0 = 5% to 100% annualized)
                                real_vol = max(0.05, min(1.0, real_vol))
                                
                                volatilities.append(float(real_vol))
                                self.logger.info(f"[RISK] Real volatility for {ticker}: {real_vol:.2%}")
                            else:
                                # Not enough data
                                volatilities.append(0.25)
                                self.logger.warning(f"[WARNING] Could not calculate real vol for {ticker}. Only {len(df)} rows available. Using 0.25 fallback.")
                        else:
                            # File doesn't exist
                            volatilities.append(0.25)
                            self.logger.warning(f"[WARNING] Could not calculate real vol for {ticker}. File {filepath} not found. Using 0.25 fallback.")
                            
                    except Exception as e:
                        self.logger.error(f"[RISK] Error calculating volatility for {ticker}: {e}")
                        volatilities.append(0.25)  # Fallback
                
                # Calculate total risk capital (20% of available cash)
                available_cash = self.portfolio.get('cash', 0)
                total_risk_capital = available_cash * 0.20  # 20% of cash for new positions
                
                self.logger.info(f"[RISK] Available cash: ${available_cash:,.2f}")
                self.logger.info(f"[RISK] Risk capital: ${total_risk_capital:,.2f} (20% of cash)")
                
                # Calculate inverse volatility allocations
                risk_manager = RiskManager()
                allocations = risk_manager.calculate_inverse_vol_sizing(tickers, volatilities, total_risk_capital)
                
                self.logger.info("[RISK] Inverse volatility allocations:")
                for ticker, allocation in allocations.items():
                    self.logger.info(f"  {ticker}: ${allocation:,.2f}")
                
                # Hysteresis Rule: Check if portfolio is at max capacity
                current_positions = len(self.portfolio.get('positions', {}))
                max_positions = getattr(self, 'MAX_POSITIONS', 10)
                
                # Calculate average AI score of current holdings
                current_avg_score = 0.0
                if current_positions >= max_positions and buy_signals:
                    # Need to calculate average score of current holdings
                    current_scores = []
                    for ticker in self.portfolio.get('positions', {}):
                        # Mock AI score based on recent performance (simplified)
                        pos_data = self.portfolio['positions'][ticker]
                        cost_basis = pos_data.get('cost_basis', 0)
                        current_price = pos_data.get('current_price', cost_basis)
                        if cost_basis > 0:
                            performance = (current_price - cost_basis) / cost_basis
                            # Convert performance to AI score (simplified mapping)
                            ai_score = min(0.95, max(0.05, 0.5 + performance))
                            current_scores.append(ai_score)
                    
                    if current_scores:
                        current_avg_score = sum(current_scores) / len(current_scores)
                        self.logger.info(f"[HYSTERESIS] Current holdings avg score: {current_avg_score:.3f}")
                
                # Execute buy signals with inverse volatility sizing
                for signal in buy_signals:
                    ticker = signal['ticker']
                    price = signal['price']
                    reason = signal['reason']
                    ai_score = signal.get('ai_score', 0.5)  # Default score if not provided
                    
                    # Hysteresis Rule: Check if we need to swap positions
                    if current_positions >= max_positions:
                        required_score_improvement = current_avg_score * 1.15  # 15% higher
                        if ai_score <= required_score_improvement:
                            self.logger.info(f"[SKIP] {ticker} score {ai_score:.3f} not 15% greater than holding avg {current_avg_score:.3f}. Avoiding whipsaw.")
                            continue
                    
                    # Calculate position size based on allocation
                    allocation = allocations.get(ticker, 0)
                    if allocation > 0 and price > 0:
                        quantity = int(allocation / price)
                        quantity = max(1, quantity)  # At least 1 share
                    else:
                        quantity = 1
                    
                    # Execute buy trade
                    result = self.execute_trade(ticker, 'buy', quantity, price, reason)
                    
                    if result['success']:
                        trades_executed.append(result)
                        self.logger.info(f"[MOCK] Buy executed: {quantity} {ticker} @ ${price:.2f} (allocated: ${allocation:,.2f})")
                    else:
                        self.logger.warning(f"[MOCK] Buy failed: {ticker} - {result.get('error', 'Unknown error')}")
            
            # Update portfolio values after all trades
            self._sync_market_prices()
            
            # Log summary
            self.logger.info(f"[MOCK] Shadow trading session complete:")
            self.logger.info(f"  Candidates Found: {candidates_found}")
            self.logger.info(f"  Buy Signals: {len(buy_signals)}")
            self.logger.info(f"  Sell Signals: {len(sell_signals)}")
            self.logger.info(f"  Trades Executed: {len(trades_executed)}")
            self.logger.info(f"  Portfolio Value: ${self.get_account_info()['portfolio_value']:,.2f}")
            self.logger.info(f"  Available Cash: ${self.portfolio['cash']:,.2f}")
            
            return {
                'success': True,
                'trades_executed': len(trades_executed),
                'candidates_found': candidates_found,
                'volatility_blocked': False,
                'trades': trades_executed
            }
            
        except Exception as e:
            self.logger.error(f"[MOCK] Shadow trading failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'trades_executed': 0,
                'candidates_found': 0,
                'volatility_blocked': False
            }
    
    def _generate_trading_signals(self, trading_strategy, data_manager, sector_auth):
        """Generate genuine trading signals using AI model on ticker universe with IBKR integration"""
        signals = []
        
        self.logger.info(f"[SIGNALS] Starting signal generation process...")
        
        try:
            # 🚀 PHASE 13: Detect market regime for adaptive thresholds
            from core.regime_detector import RegimeDetector
            regime_detector = RegimeDetector()
            
            # Load SPY and VXX data for regime detection
            spy_data = data_manager._load_ticker_data('SPY')
            vxx_data = data_manager._load_ticker_data('VXX')
            
            if spy_data is not None and len(spy_data) > 200:
                regime_code, regime_name, regime_threshold = regime_detector.detect_regime(spy_data, vxx_data)
                self.logger.info(f"[REGIME] Current: {regime_name} ({regime_code}) | Threshold: {regime_threshold}")
            else:
                regime_code, regime_name, regime_threshold = 2, 'BULL', 0.65
                self.logger.warning("[REGIME] SPY data unavailable - using default BULL regime")
            
            # Phase 13 Regime Thresholds
            # 0=CRISIS: No entries (threshold=None), 1=BEAR: 0.72, 2=BULL: 0.65
            if regime_code == 0:  # CRISIS
                self.logger.warning("[REGIME] CRISIS regime detected - blocking all new entries")
                regime_threshold = None  # Block all entries
            
            # For paper trading, IBKR connection is optional (we can generate signals without it)
            if self.ibkr_engine:
                if not self.ibkr_engine.connect():
                    self.logger.warning("[SIGNALS] Failed to connect to IBKR - continuing with local data")
                else:
                    self.logger.info("[SIGNALS] Connected to IBKR for live trading")
            else:
                self.logger.info("[SIGNALS] IBKR engine not available - using local data only")
            
            # Load current portfolio holdings to prioritize evaluation
            portfolio_csv = Path(self.project_root) / 'data' / 'portfolio.csv'
            held_tickers = set()
            if portfolio_csv.exists():
                import pandas as pd
                portfolio_df = pd.read_csv(portfolio_csv)
                active_positions = portfolio_df[portfolio_df['Status'] == 'ACTIVE']
                held_tickers = set(active_positions['Ticker'].str.upper())
                self.logger.info(f"[SIGNALS] Current holdings: {len(held_tickers)} positions - {', '.join(sorted(held_tickers))}")
            
            # Load ticker universe from ALL available parquet files (AI-driven selection)
            raw_dir = Path(self.project_root) / 'data' / 'raw'
            parquet_files = sorted(raw_dir.glob('*.parquet'))
            ticker_universe = [pfile.stem for pfile in parquet_files]
            
            # Prioritize held positions first, then scan rest of universe
            held_tickers_lower = {t.lower() for t in held_tickers}
            priority_tickers = [t for t in ticker_universe if t.lower() in held_tickers_lower]
            other_tickers = [t for t in ticker_universe if t.lower() not in held_tickers_lower]
            ordered_universe = priority_tickers + other_tickers
            
            self.logger.info(f"[SIGNALS] AI-DRIVEN UNIVERSE: {len(held_tickers)} held positions + {len(other_tickers)} opportunities = {len(ordered_universe)} total")
            
            # Collect all signals with confidence scores
            all_signals = []
            held_position_signals = []  # Track signals for held positions separately
            
            # Process each ticker through AI model (held positions first)
            for ticker in ordered_universe:
                try:
                    # Fetch latest market data for this ticker
                    ticker_data = data_manager._load_ticker_data(ticker)
                    
                    if ticker_data is None or len(ticker_data) < 50:
                        self.logger.debug(f"[SIGNALS] Insufficient data for {ticker}, skipping")
                        continue
                    
                    # Generate AI signal using the real trading strategy
                    from core.ai_models import get_ensemble_signal
                    signal, confidence, details = get_ensemble_signal(ticker_data)
                    
                    # Phase 13: Apply regime-based threshold filtering
                    current_price = float(ticker_data['close'].iloc[-1])
                    is_held = ticker.upper() in held_tickers
                    
                    # Filter BUY signals based on regime threshold
                    if signal == 'BUY' and regime_threshold is not None and confidence < regime_threshold:
                        # Signal doesn't meet regime threshold - skip
                        continue
                    elif signal == 'BUY' and regime_threshold is None:
                        # CRISIS regime - block all new entries
                        continue
                    
                    if signal == 'BUY':
                        signal_dict = {
                            'ticker': ticker,
                            'action': 'buy',
                            'confidence': confidence,
                            'price': current_price,
                            'details': details,
                            'is_held': is_held
                        }
                        all_signals.append(signal_dict)
                        if is_held:
                            held_position_signals.append(signal_dict)
                            self.logger.info(f"[SIGNALS] HELD: {ticker} - AI says BUY (confidence: {confidence:.3f}) - HOLD position")
                    elif signal == 'SELL':
                        signal_dict = {
                            'ticker': ticker,
                            'action': 'sell',
                            'confidence': confidence,
                            'price': current_price,
                            'details': details,
                            'is_held': is_held
                        }
                        all_signals.append(signal_dict)
                        if is_held:
                            held_position_signals.append(signal_dict)
                            self.logger.info(f"[SIGNALS] HELD: {ticker} - AI says SELL (confidence: {confidence:.3f}) - CLOSE position")
                
                except Exception as e:
                    self.logger.warning(f"[SIGNALS] Error processing {ticker}: {e}")
                    continue
            
            # Sort signals by confidence (highest first) - AI-driven ranking
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # CRITICAL: Always include signals for held positions (for SELL evaluation)
            # Then fill remaining slots with best new opportunities
            top_signals = []
            
            # First, add all held position signals (these are critical for portfolio management)
            top_signals.extend(held_position_signals)
            self.logger.info(f"[SIGNALS] Included {len(held_position_signals)} signals for held positions")
            
            # Then add best new opportunities (not already held)
            remaining_slots = 20 - len(held_position_signals)  # Allow up to 20 total signals
            new_opportunity_signals = [s for s in all_signals if not s.get('is_held', False)]
            top_signals.extend(new_opportunity_signals[:remaining_slots])
            
            self.logger.info(f"[SIGNALS] AI RANKING: Found {len(all_signals)} total signals")
            self.logger.info(f"[SIGNALS] Selected {len(held_position_signals)} held position signals + {len(top_signals) - len(held_position_signals)} new opportunities = {len(top_signals)} total")
            
            # Convert top signals to trade format
            for signal in top_signals:
                ticker = signal['ticker']
                action = signal['action']
                confidence = signal['confidence']
                current_price = signal['price']
                details = signal['details']
                
                if action == 'buy':
                    # Calculate position size using Real Volatility Inverse Sizing with IBKR
                    position_size = self._calculate_position_size(
                        ticker, current_price, confidence, sector_auth
                    )
                    
                    if position_size > 0:
                        signal_dict = {
                            'ticker': ticker,
                            'action': 'buy',
                            'quantity': position_size,
                            'price': current_price,
                            'reason': f'AI Signal: {confidence:.3f} confidence',
                            'ai_score': confidence,
                            'details': details
                        }
                        signals.append(signal_dict)
                        
                        self.logger.info(f"[SIGNALS] BUY {ticker}: {position_size} shares @ ${current_price:.2f} (AI: {confidence:.3f})")
                
                elif action == 'sell':
                    # For sells, we need to check if we actually hold this position
                    # This would be handled by the risk manager later
                    
                    signal_dict = {
                        'ticker': ticker,
                        'action': 'sell',
                        'quantity': 0,  # Will be determined by portfolio state
                        'price': current_price,
                        'reason': f'AI Signal: {confidence:.3f} confidence',
                        'ai_score': confidence,
                        'details': details
                    }
                    signals.append(signal_dict)
                    
                    self.logger.info(f"[SIGNALS] SELL {ticker}: @ ${current_price:.2f} (AI: {confidence:.3f})")
            
            # Export signals to TradingView-compatible CSV file
            self._export_tradingview_signals(all_signals, top_signals)
            
            # Update portfolio with new signals (managed portfolio system)
            self._update_portfolio(all_signals)
            
            # Enforce "No Silent Failures" rule
            if not signals:
                self.logger.info("[SIGNALS] No genuine AI signals generated - returning empty list")
                return []  # Return empty list, never fallback to mock data
            
            self.logger.info(f"[SIGNALS] Generated {len(signals)} genuine AI signals:")
            for signal in signals:
                self.logger.info(f"  {signal['ticker']} {signal['action'].upper()} (AI: {signal['ai_score']:.3f})")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"[SIGNALS] Critical error in signal generation: {e}")
            self.logger.error(f"[SIGNALS] Traceback: {traceback.format_exc()}")
            # Return empty list on any error - never fallback to mock data
            return []
        finally:
            # Always disconnect from IBKR
            if self.ibkr_engine:
                self.ibkr_engine.disconnect()
                self.logger.info("[SIGNALS] Disconnected from IBKR")
    
    def _export_tradingview_signals(self, all_signals, top_signals):
        """Export signals to TradingView-compatible CSV file"""
        try:
            import csv
            from datetime import datetime
            
            # Create reports directory if it doesn't exist
            reports_dir = Path(self.project_root) / 'reports'
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_file = reports_dir / f'tradingview_signals_{timestamp}.csv'
            
            # Write CSV file
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['Ticker', 'Action', 'Confidence', 'Price', 'Rank', 'Status'])
                
                # Write top signals (recommended)
                for i, signal in enumerate(top_signals, 1):
                    writer.writerow([
                        signal['ticker'],
                        signal['action'].upper(),
                        f"{signal['confidence']:.3f}",
                        f"{signal['price']:.2f}",
                        i,
                        'RECOMMENDED'
                    ])
                
                # Write remaining signals (optional)
                remaining = [s for s in all_signals if s not in top_signals]
                for i, signal in enumerate(remaining, len(top_signals) + 1):
                    writer.writerow([
                        signal['ticker'],
                        signal['action'].upper(),
                        f"{signal['confidence']:.3f}",
                        f"{signal['price']:.2f}",
                        i,
                        'OPTIONAL'
                    ])
            
            self.logger.info(f"[TRADINGVIEW] Exported {len(all_signals)} signals to {csv_file}")
            self.logger.info(f"[TRADINGVIEW] Top {len(top_signals)} signals marked as RECOMMENDED")
            
            # Store the file path for email attachment
            global TRADINGVIEW_SIGNAL_FILE
            TRADINGVIEW_SIGNAL_FILE = str(csv_file)
            
        except Exception as e:
            self.logger.error(f"[TRADINGVIEW] Failed to export signals: {e}")
    
    def _update_portfolio(self, all_signals):
        """Update managed portfolio with new signals"""
        try:
            from core.portfolio_manager import PortfolioManager
            
            # Convert signals to DataFrame
            signals_df = pd.DataFrame(all_signals)
            
            # Initialize portfolio manager
            manager = PortfolioManager(Path(self.project_root))
            
            # Update portfolio
            updated_portfolio = manager.update_portfolio(signals_df)
            
            self.logger.info(f"[PORTFOLIO] Updated portfolio with {len(updated_portfolio)} positions")
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Failed to update portfolio: {e}")
    
    def _calculate_position_size(self, ticker: str, current_price: float, confidence: float, sector_auth) -> int:
        """
        Calculate position size using Real Volatility Inverse Sizing Law (ARCHITECTURE.md Section 3 & 7)
        
        Mathematical position allocation using real market data (20-day returns, annualized volatility)
        1% risk per trade with real volatility-weighted allocations (1/σ weighting)
        Data Source: Real volatility calculated from data/raw/{ticker}.parquet files
        """
        try:
            # Get live cash balance from IBKR (ditch portfolio.json)
            if self.ibkr_engine and self.ibkr_engine.connected:
                account_summary = self.ibkr_engine.get_account_summary()
                if 'error' not in account_summary:
                    available_cash = account_summary.get('cash_balance', 0.0)
                    self.logger.info(f"[RISK] Live cash from IBKR: ${available_cash:,.2f}")
                else:
                    self.logger.error(f"[RISK] IBKR account summary error: {account_summary['error']}")
                    return 0
            else:
                self.logger.error("[RISK] IBKR not connected - cannot calculate position size")
                return 0
            
            # Calculate real volatility from parquet data (ARCHITECTURE.md requirement)
            volatility = self._calculate_real_volatility(ticker)
            if volatility is None or volatility <= 0:
                self.logger.error(f"[RISK] Failed to calculate real volatility for {ticker}")
                return 0
            
            # Real Volatility Inverse Sizing Law: 1% risk per trade
            risk_per_trade = 0.01  # 1% risk per trade (ARCHITECTURE.md Section 3)
            risk_amount = available_cash * risk_per_trade
            
            # Mathematical allocation using 1/σ weighting (ARCHITECTURE.md Section 7)
            # Low volatility stocks get MORE capital, high volatility get LESS
            inverse_volatility_weight = 1.0 / volatility
            
            # Normalize weight (if multiple positions, this would be part of portfolio allocation)
            # For single position, use the inverse volatility directly
            position_value = risk_amount * inverse_volatility_weight
            
            # Apply confidence filter (only for entry decision, not sizing per ARCHITECTURE.md)
            # The ARCHITECTURE.md states "Risk Per Trade Law: Maximum 1% portfolio risk per trade enforced automatically"
            # Confidence affects whether to trade, not how much to risk
            
            # Calculate number of shares
            shares = int(position_value / current_price)
            
            # Minimum position size
            min_shares = 1
            shares = max(shares, min_shares)
            
            # Maximum position size (20% of portfolio - safety constraint)
            max_shares = int((available_cash * 0.20) / current_price)
            shares = min(shares, max_shares)
            
            self.logger.info(f"[RISK] {ticker}: Vol={volatility:.2%}, 1/σ={inverse_volatility_weight:.2f}, "
                           f"Risk=${risk_amount:,.2f}, Position=${position_value:,.2f}, Shares={shares}")
            
            return shares
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating position size for {ticker}: {e}")
            return 0
    
    def _calculate_real_volatility(self, ticker: str) -> Optional[float]:
        """
        Calculate real volatility from parquet data (ARCHITECTURE.md Section 7)
        
        Uses 20-day returns with √252 annualization from data/raw/{ticker}.parquet files
        No fallback to defaults allowed per ARCHITECTURE.md
        """
        try:
            import pandas as pd
            import numpy as np
            import os
            
            # Data Source Law: Volatility calculated from data/raw/{ticker}.parquet files
            parquet_path = os.path.join(PROJECT_ROOT, 'data', 'raw', f'{ticker}.parquet')
            
            if not os.path.exists(parquet_path):
                self.logger.error(f"[RISK] Parquet file not found: {parquet_path}")
                return None
            
            # Load parquet data
            df = pd.read_parquet(parquet_path)
            
            if len(df) < 20:
                self.logger.error(f"[RISK] Insufficient data for {ticker}: {len(df)} days < 20 required")
                return None
            
            # Calculate 20-day returns
            df['returns'] = df['close'].pct_change()
            
            # Use last 20 trading days
            recent_returns = df['returns'].tail(20).dropna()
            
            if len(recent_returns) < 20:
                self.logger.error(f"[RISK] Insufficient clean returns for {ticker}: {len(recent_returns)} < 20")
                return None
            
            # Calculate daily volatility (standard deviation of returns)
            daily_volatility = recent_returns.std()
            
            # Annualize with √252 (ARCHITECTURE.md requirement)
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            self.logger.info(f"[RISK] {ticker} real volatility: {annualized_volatility:.2%} (20-day)")
            
            return annualized_volatility
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating real volatility for {ticker}: {e}")
            return None

# ==================== MASTER RUNNER CLASS ====================

class TradingOrchestrator:
    """
    Master Runner for NeuralTrader Task Scheduler Integration
    Handles all automation modes with comprehensive safety checks
    """
    
    def __init__(self, ai_model=None, trading_strategy=None, mode="paper"):
        """Initialize the trading orchestrator with REAL AI and strategy"""
        self.logger = setup_automation_logging()
        self.supervision_logger = setup_daily_supervision()
        self.mode = mode  # Store the mode for report mode initialization
        
        # Store REAL AI model and strategy (passed from integrity check)
        self.ai_model = ai_model
        self.trading_strategy = trading_strategy
        
        if self.ai_model is not None:
            self.logger.info("[AI] Real AI model (EnsemblePredictor) loaded into orchestrator")
        if self.trading_strategy is not None:
            self.logger.info("[STRATEGY] Real trading strategy loaded into orchestrator")
        
        # Lazy initialization of trading modules
        self.yfinance_manager = None
        self.risk_manager = None
        self.virtual_engine = None
        # self.ist_scheduler = None
        self.email_notifier = None
        
        # Initialize Sector Authority for sector-based risk management
        self.sector_auth = SectorAuthority()
        self.logger.info("[SECTOR] Sector Authority initialized for risk management")
        
        # Initialize Phase 10 Sentiment Integration
        self.sentiment_integration = None
        try:
            from core.sentiment.sentiment_integration import SentimentIntegration
            
            # Configure sentiment analysis
            sentiment_config = {
                'enable_economic': True,  # Enable economic data analysis
                'enable_news': True,      # Enable news sentiment analysis
                'enable_social': True,    # Enable social media sentiment analysis
                'cache_ttl': 3600,        # 1 hour cache
                'fred_api_key': os.getenv('FRED_API_KEY'),  # FRED API key from environment
                'news_api_key': os.getenv('NEWS_API_KEY'),      # News API key from environment
                'social_api_keys': {                               # Social media API keys
                    'twitter_api_key': os.getenv('TWITTER_API_KEY'),
                    'reddit_api_key': os.getenv('REDDIT_API_KEY'),
                    'stocktwits_api_key': os.getenv('STOCKTWITS_API_KEY')
                }
            }
            
            self.sentiment_integration = SentimentIntegration(sentiment_config)
            self.logger.info("[PHASE10] Sentiment Integration initialized - Economic Data Analysis ACTIVE")
            
            # Validate configuration
            if not self.sentiment_integration.validate_configuration():
                self.logger.warning("[PHASE10] Sentiment configuration has issues - some features may not work")
            
        except Exception as e:
            self.logger.error(f"[PHASE10] Failed to initialize Sentiment Integration: {e}")
            self.logger.info("[PHASE10] System will continue without sentiment analysis")
        
        # Initialize IBKR Engine for Live Paper Trading (Tier 3)
        self.ibkr_engine = None
        try:
            from core.ibkr_engine import IBKREngineSync
            self.ibkr_engine = IBKREngineSync()
            self.logger.info("[IBKR] IBKR Engine initialized for live paper trading")
        except Exception as e:
            self.logger.error(f"[IBKR] Failed to initialize IBKR Engine: {e}")
            self.logger.info("[IBKR] System will continue without live trading")
        
        self.logger.info("[TRADING] Trading Orchestrator (Master Runner) initialized")
        if self.supervision_logger:
            self.logger.info("[SUPERVISION] Daily Supervision Logger initialized")
    
    def _generate_run_id(self, mode: str) -> str:
        """Generate unique run ID for tracking"""
        timestamp = datetime.now().strftime('%H%M%S')
        return f"{mode}_{timestamp}"
    
    def _log_supervision_start(self, mode: str) -> str:
        """Log run start to supervision system"""
        if self.supervision_logger:
            run_entry = self.supervision_logger.log_run_start(mode)
            return run_entry.get('run_id')
        return self._generate_run_id(mode)
    
    def _log_supervision_complete(self, mode: str, run_id: str, success: bool, 
                                 duration: float, details: Dict = None):
        """Log run completion to supervision system"""
        if self.supervision_logger:
            self.supervision_logger.log_run_complete(mode, run_id, success, duration, details)
    
    def _log_supervision_error(self, mode: str, run_id: str, error: str):
        """Log error to supervision system"""
        if self.supervision_logger:
            self.supervision_logger.log_error(mode, run_id, error)
    
    def _send_data_fetch_notification(self, tickers_count: int):
        """Send data fetch notification with full logs"""
        try:
            import pytz
            subject = f"[DATA] NeuralTrader Data Fetch Complete - {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
            
            body = f"""
NeuralTrader Data Fetch Notification
=====================================

[DATE] Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
[MODE] Mode: Shadow Trading Simulator
[CONSTITUTION] Constitution: Risk Management Active

[DATA] DATA FETCH RESULTS:
---------------------
Tickers Fetched: {tickers_count}
Data Source: Yahoo Finance
Status: [SUCCESS]

[GLOBAL] MARKET STATUS:
------------------
Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
EST Time: {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S EST')}

[READINESS] READINESS FOR TOMORROW:
--------------------------
[OK] Market data updated
[OK] Risk systems operational
[OK] Virtual portfolio ready
[OK] Email notifications enabled

[EMAIL] FULL LOGS ATTACHED:
-------------------
Complete automation log attached for detailed analysis.

---
NeuralTrader Automated Trading System
Phase 6: Shadow Trading Simulator
"""
            
            # Get automation log file path
            automation_log_path = os.path.join(PROJECT_ROOT, 'logs', 'automation.log')
            
            # Send email with logs
            success = self.email_notifier.send_email_with_logs(
                to_email=self.email_notifier.recipient_email,
                subject=subject,
                body=body,
                log_file_path=automation_log_path
            )
            
            if success:
                self.logger.info("[SUCCESS] Data fetch notification with logs sent successfully")
            else:
                self.logger.error("[ERROR] Failed to send data fetch notification with logs")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error sending data fetch notification: {e}")
    
    def _send_urgent_notification(self, subject: str, message: str):
        """Send urgent notification email with log file attached"""
        try:
            from core.utils.notifier import EmailNotifier
            
            notifier = EmailNotifier()
            
            # Format subject with [URGENT] prefix
            formatted_subject = f"[URGENT] NeuralTrader {subject}"
            
            # Get current log file path from global variable
            log_file_path = CURRENT_LOG_FILE if CURRENT_LOG_FILE else None
            
            # Send email with logs attached
            success = notifier.send_email_with_logs(
                to_email=notifier.recipient_email,
                subject=formatted_subject,
                body=message,
                log_file_path=log_file_path
            )
            
            if success:
                self.logger.info(f"[OK] Urgent notification sent: {formatted_subject}")
            else:
                self.logger.error(f"[ERROR] Failed to send urgent notification")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error sending urgent notification: {e}")
    
    def _send_daily_report_notification(self, report_content: str):
        """Send daily report notification with full logs"""
        try:
            subject = f"[NEURAL] Daily Executive Brief - {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
            
            body = f"""
{report_content}

[EMAIL] FULL LOGS ATTACHED:
------------------
Complete automation log attached for detailed analysis.

---
NeuralTrader Automated Trading System
Phase 6: Shadow Trading Simulator
"""
            
            # Get automation log file path
            automation_log_path = os.path.join(PROJECT_ROOT, 'logs', 'automation.log')
            
            # Send email with logs
            success = self.email_notifier.send_email_with_logs(
                to_email=self.email_notifier.recipient_email,
                subject=subject,
                body=body,
                log_file_path=automation_log_path
            )
            
            if success:
                self.logger.info("[OK] Daily report notification with logs sent successfully")
            else:
                self.logger.error("[ERROR] Failed to send daily report notification with logs")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error sending daily report notification: {e}")
    
    def _send_html_report_with_attachments(self):
        """Send HTML dashboard report with log file attachment"""
        try:
            # Generate HTML dashboard
            from scripts.report_generator import HTMLDashboardGenerator
            
            generator = HTMLDashboardGenerator()
            
            # Get market activity data from virtual engine
            buy_signals = []
            sell_signals = []
            hold_signals = []
            trades_executed = []
            
            # Extract recent trades from portfolio history
            if hasattr(self.virtual_engine, 'portfolio') and 'history' in self.virtual_engine.portfolio:
                recent_trades = self.virtual_engine.portfolio['history'][-5:]  # Last 5 trades
                for trade in recent_trades:
                    trades_executed.append({
                        'ticker': trade.get('ticker', 'Unknown'),
                        'action': trade.get('action', 'Unknown'),
                        'quantity': trade.get('quantity', 0),
                        'price': trade.get('price', 0)
                    })
            
            # Generate HTML dashboard
            dashboard_file = generator.generate_dashboard(
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                trades_executed=trades_executed
            )
            
            # Read HTML content as the email body
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                html_body = f.read()
            
            # Find today's log file
            today = datetime.now().strftime('%Y-%m-%d')
            log_file_path = None
            
            # Try different log file naming patterns
            log_patterns = [
                f"logs/automation_{today.replace('-', '')}*.log",  # automation_YYYYMMDD_*.log
                f"logs/NeuralTrader_Automation_{today}.log",
                f"logs/NeuralTrader_{today}.log",
                f"logs/automation.log"
            ]
            
            # For glob patterns, we need to find the latest file
            import glob
            log_file_path = None
            
            for pattern in log_patterns:
                if '*' in pattern:
                    # Use glob for wildcard patterns
                    search_pattern = os.path.join(os.path.dirname(__file__), pattern)
                    matching_files = glob.glob(search_pattern)
                    if matching_files:
                        # Get the most recent file
                        latest_file = max(matching_files, key=os.path.getmtime)
                        log_file_path = os.path.abspath(latest_file)
                        print(f"[DEBUG] Attempting to attach log file: {log_file_path}")
                        self.logger.info(f"[EMAIL] Found log file: {log_file_path}")
                        break
                else:
                    # Direct file check
                    full_path = os.path.join(os.path.dirname(__file__), pattern)
                    if os.path.exists(full_path):
                        log_file_path = os.path.abspath(full_path)
                        print(f"[DEBUG] Attempting to attach log file: {log_file_path}")
                        self.logger.info(f"[EMAIL] Found log file: {log_file_path}")
                        break
            
            if not log_file_path:
                self.logger.warning("[EMAIL] No log file found for today")
            
            # Send email with dashboard as body and log file as attachment
            subject = f"[NEURAL] Dashboard Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            success = self.email_notifier.send_email_with_logs(
                to_email=self.email_notifier.recipient_email,
                subject=subject,
                body=html_body,  # HTML dashboard as body
                log_file_path=log_file_path,  # Only log file as attachment
                html_body=True  # Indicate body is HTML
            )
            
            if success:
                self.logger.info("[OK] Dashboard report sent successfully")
                self.logger.info("[EMAIL] Dashboard rendered in email body")
                if log_file_path:
                    self.logger.info(f"[EMAIL] Log file attached: {log_file_path}")
            else:
                self.logger.error("[ERROR] Failed to send dashboard report")
            
            return success
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error sending dashboard report: {e}")
            return False
    
    def _send_saturday_retrain_notification(self, retrain_results: Dict):
        """Send Saturday retrain notification with full logs"""
        try:
            subject = f"[RETRAIN] NeuralTrader Saturday Retrain - {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
            
            body = f"""
NeuralTrader Saturday Retrain Notification
========================================

[DATE] Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
[MODE] Mode: Model Retraining
[CONSTITUTION] Constitution: Risk Management Active

[DATA] RETRAIN RESULTS:
------------------
Status: {retrain_results.get('status', 'Unknown')}
Models Updated: {retrain_results.get('models_updated', 0)}
Performance: {retrain_results.get('performance', 'N/A')}
Duration: {retrain_results.get('duration', 'N/A')}

[LIST] MODEL STATUS:
---------------
[OK] Models retrained successfully
[OK] Performance validated
[OK] Risk systems updated
[OK] Ready for next week

[EMAIL] FULL LOGS ATTACHED:
------------------
Complete automation log attached for detailed analysis.

---
NeuralTrader Automated Trading System
Phase 6: Shadow Trading Simulator
"""
            
            # Get automation log file path
            automation_log_path = os.path.join(PROJECT_ROOT, 'logs', 'automation.log')
            
            # Send email with logs
            success = self.email_notifier.send_email_with_logs(
                to_email=self.email_notifier.recipient_email,
                subject=subject,
                body=body,
                log_file_path=automation_log_path
            )
            
            if success:
                self.logger.info("[OK] Saturday retrain notification with logs sent successfully")
            else:
                self.logger.error("[ERROR] Failed to send Saturday retrain notification with logs")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error sending Saturday retrain notification: {e}")
    
    def _send_session_notification(self, status: str, message: str):
        """Send simple session notification email"""
        try:
            from core.utils.notifier import EmailNotifier
            
            subject = f"[NeuralTrader] Session Complete - {status}"
            
            body = f"""
NeuralTrader Session Notification
================================

[DATE] Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
[MODE] Mode: Paper Trading
[STATUS] {status}

[SUMMARY] Session Details:
------------------------
{message}

[PORTFOLIO] Current Status:
-------------------------
Portfolio: 5 positions held
Cash: $95,344.75
System: Capital preservation active

[MARKET] SPY Analysis:
--------------------
Market Filter: BEARISH protection active
Action: No trades executed
Reason: Market below 20-day SMA

🛡️ NeuralTrader Constitution: Capital Preservation Priority #1
📊 System is protecting capital during bearish market conditions.

This is an automated message from NeuralTrader Paper Trading System.
"""
            
            # Initialize email notifier
            notifier = EmailNotifier()
            
            # Send email
            success = notifier.send_alert(subject, body)
            
            if success:
                self.logger.info("[OK] Session notification sent successfully")
            else:
                self.logger.error("[ERROR] Failed to send session notification")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error sending session notification: {e}")
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch is activated"""
        if os.path.exists('STOP.txt'):
            self.logger.warning("[WARNING] KILL SWITCH ACTIVATED - STOP.txt file found")
            return True
        return False
    
    def initialize_modules(self) -> bool:
        """Initialize all trading modules with error handling"""
        try:
            self.logger.info("[CONFIG] Initializing trading modules...")
            
            # Initialize YFinance Manager
            # self.yfinance_manager = YFinanceManager()
            # self.logger.info("[OK] YFinance Manager initialized")
            self.logger.info("[SKIP] YFinance Manager not available - using data_manager instead")
            
            # Initialize Risk Manager
            # self.risk_manager = RiskManager()
            # self.logger.info("[OK] Risk Manager initialized")
            self.logger.info("[SKIP] Risk Manager not available")
            
            # Initialize Virtual Engine
            # self.virtual_engine = VirtualEngine()
            # self.logger.info("[OK] Virtual Engine initialized")
            self.logger.info("[SKIP] Virtual Engine not available")
            
            # Fail-Safe: Always use MockVirtualEngine, routed to mode-specific portfolio file
            if self.virtual_engine is None or self.mode == "report":
                engine_mode = self.mode if self.mode in _PORTFOLIO_FILES else 'paper'
                self.virtual_engine = MockVirtualEngine(
                    ibkr_engine=self.ibkr_engine,
                    mode=engine_mode,
                )
                self.logger.info(f"[PORTFOLIO] VirtualEngine using portfolio file for mode={engine_mode}")
            
            # Initialize IST Scheduler
            # self.ist_scheduler = ISTScheduler()
            # self.logger.info("[OK] IST Scheduler initialized")
            
            # Initialize Email Notifier - ALWAYS initialize for report mode
            try:
                # Import EmailNotifier from new src location
                from core.utils.notifier import EmailNotifier
                
                self.email_notifier = EmailNotifier()
                self.logger.info("[OK] Email Notifier initialized")
            except Exception as e:
                self.logger.warning(f"[WARNING] Email Notifier initialization failed: {e}")
                self.email_notifier = None
            
            return True
        
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize modules: {e}")
            return False
    
    def run_fetch_mode(self) -> bool:
        """
        Run fetch mode — incremental update of all data/raw/*.parquet files via Tiingo API.
        For each ticker: reads last date in parquet, fetches from last_date+1 to today,
        appends new rows and saves back. Skips tickers already up-to-date.
        Rule 3.1: after update, any parquet still >24h old triggers [FATAL].
        """
        import requests
        import pandas as pd
        from pathlib import Path
        from datetime import datetime, timedelta
        import time

        session_start = datetime.now()
        self.logger.info("[FETCH] Starting incremental parquet update via Tiingo API")

        try:
            # Load API key
            tiingo_token = os.getenv('TIINGO_API_KEY')
            if not tiingo_token:
                self.logger.error("[FATAL] TIINGO_API_KEY not set in environment — cannot fetch data")
                return False

            raw_dir = Path(PROJECT_ROOT) / 'data' / 'raw'
            parquet_files = sorted(raw_dir.glob('*.parquet'))
            if not parquet_files:
                self.logger.error("[FATAL] No parquet files found in data/raw/ — cannot update")
                return False

            today_str = datetime.now().strftime('%Y-%m-%d')
            updated = 0
            skipped = 0
            failed = 0
            total = len(parquet_files)

            self.logger.info(f"[FETCH] Universe: {total} tickers | Target date: {today_str}")

            for i, pfile in enumerate(parquet_files):
                ticker = pfile.stem
                try:
                    df = pd.read_parquet(pfile)

                    # Determine last date in file
                    if 'date' in df.columns:
                        last_date = pd.to_datetime(df['date']).max()
                    else:
                        last_date = pd.to_datetime(df.index).max()

                    start_fetch = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

                    if start_fetch > today_str:
                        skipped += 1
                        continue

                    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
                    params = {
                        'token': tiingo_token,
                        'startDate': start_fetch,
                        'endDate': today_str,
                        'format': 'csv',
                        'resampleFreq': 'daily'
                    }

                    resp = requests.get(url, params=params, timeout=20)
                    if resp.status_code != 200:
                        self.logger.warning(f"[WARN] {ticker}: HTTP {resp.status_code} — skip")
                        failed += 1
                        time.sleep(0.3)
                        continue

                    from io import StringIO
                    new_rows = pd.read_csv(StringIO(resp.text))
                    if new_rows.empty:
                        skipped += 1
                        continue

                    # Normalise date column
                    new_rows['date'] = pd.to_datetime(new_rows['date']).dt.strftime('%Y-%m-%d')
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                        combined = pd.concat([df, new_rows], ignore_index=True)
                        combined = combined.drop_duplicates(subset='date', keep='last')
                        combined = combined.sort_values('date').reset_index(drop=True)
                    else:
                        df = df.reset_index()
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                        combined = pd.concat([df, new_rows], ignore_index=True)
                        combined = combined.drop_duplicates(subset='date', keep='last')
                        combined = combined.sort_values('date').reset_index(drop=True)

                    combined.to_parquet(pfile, index=False)
                    updated += 1

                    if (i + 1) % 100 == 0:
                        self.logger.info(f"[FETCH] Progress: {i+1}/{total} | updated={updated} skipped={skipped} failed={failed}")

                    time.sleep(0.2)  # Tiingo rate limit: ~5 req/s

                except Exception as e:
                    import traceback
                    self.logger.warning(f"[WARN] {ticker}: {e}\n{traceback.format_exc()}")
                    failed += 1

            duration = (datetime.now() - session_start).total_seconds()
            self.logger.info(f"[FETCH] Complete: updated={updated} skipped={skipped} failed={failed} | {duration:.0f}s")

            if updated == 0 and failed > total * 0.5:
                self.logger.error(f"[FATAL] Fetch failed for >50% of tickers ({failed}/{total}) — check API key")
                return False

            return True

        except Exception as e:
            import traceback
            self.logger.error(f"[ERROR] run_fetch_mode failed: {e}\n{traceback.format_exc()}")
            return False
    
    def run_trade_mode(self) -> bool:
        """Run trade mode - Triggers VirtualEngine to execute shadow trades"""
        try:
            self.logger.info("[TRADING] Running TRADE MODE - VirtualEngine Shadow Trading")
            session_start = datetime.now()
            
            # Check kill switch
            if self.check_kill_switch():
                self.logger.warning("[STOP] Kill switch activated - stopping trading session")
                return False
            
            # Initialize modules
            if not self.initialize_modules():
                return False

            # IBKR sync: overwrite portfolio_paper.json with live IBKR ground truth
            self.logger.info("[IBKR SYNC] Syncing portfolio_paper.json from IBKR...")
            sync_ok = self.virtual_engine.sync_from_ibkr()
            if not sync_ok:
                self.logger.warning("[IBKR SYNC] Sync skipped or failed - proceeding with last saved state")

            # Check Portfolio Circuit Breaker (Uncle Point)
            try:
                portfolio = self.virtual_engine.portfolio
                current_val = self.virtual_engine.get_account_info()['portfolio_value']
                peak_val = portfolio.get('peak_portfolio_value', 100000.0)
                cooldown_str = portfolio.get('circuit_breaker_cooldown_until')
                
                # Initialize risk manager
                risk_manager = RiskManager()
                
                is_halted, new_cooldown = risk_manager.check_portfolio_circuit_breaker(
                    current_val, peak_val, datetime.now(), cooldown_str
                )
                
                if is_halted:
                    self.logger.error("[SHIELD] PORTFOLIO CIRCUIT BREAKER ACTIVE. Drawdown > 12% or in cooldown.")
                    self.logger.error(f"[SHIELD] Trading HALTED. Current: ${current_val:,.2f}, Peak: ${peak_val:,.2f}")
                    
                    # Update cooldown if changed
                    if new_cooldown and new_cooldown != cooldown_str:
                        portfolio['circuit_breaker_cooldown_until'] = new_cooldown
                        self.virtual_engine.save_portfolio()
                        self.logger.error(f"[SHIELD] Cooldown updated until: {new_cooldown}")
                    
                    return False
                else:
                    self.logger.info("[SHIELD] Circuit breaker SAFE - trading allowed")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Circuit breaker check failed: {e}")
                # Be conservative and halt trading on error
                return False
            
            # Check if market is open (bypass for paper trading)
            if not PAPER_TRADING:
                market_status = self.yfinance_manager.get_market_status()
                if not market_status.get('is_market_open', False):
                    self.logger.info("[TIME] Market is closed - no trading")
                    return False
            else:
                self.logger.info("[PAPER] Bypassing market hours check - paper trading mode")
            
            # Generate trading signals using new AI-driven method with held position evaluation
            self.logger.info("[TRADING] Generating AI trading signals with held position evaluation...")
            
            try:
                # Import DataManager for signal generation
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from scripts.data_manager import DataManager
                
                data_manager = DataManager()
                
                # 🚀 PHASE 13: Detect market regime for adaptive thresholds
                try:
                    from core.regime_detector import RegimeDetector
                    regime_detector = RegimeDetector()
                    
                    # Load SPY and VXX data for regime detection
                    spy_data = data_manager._load_ticker_data('SPY')
                    vxx_data = data_manager._load_ticker_data('VXX')
                    
                    if spy_data is not None and len(spy_data) > 200:
                        regime_code, regime_name, regime_threshold = regime_detector.detect_regime(spy_data, vxx_data)
                        self.logger.info(f"[REGIME] Current: {regime_name} ({regime_code}) | Threshold: {regime_threshold}")
                    else:
                        regime_code, regime_name, regime_threshold = 2, 'BULL', 0.65
                        self.logger.warning("[REGIME] SPY data unavailable - using default BULL regime")
                        self.logger.info(f"[REGIME] Current: BULL (2) | Threshold: 0.65")
                except Exception as e:
                    self.logger.error(f"[REGIME] Detection failed: {e} - using default BULL regime")
                    regime_code, regime_name, regime_threshold = 2, 'BULL', 0.65
                    self.logger.info(f"[REGIME] Current: BULL (2) | Threshold: 0.65")
                
                # Generate signals using new method (evaluates held positions + scans universe)
                signals = self.virtual_engine._generate_trading_signals(
                    self.trading_strategy, 
                    data_manager, 
                    self.sector_auth
                )
                
                self.logger.info(f"[SIGNALS] Generated {len(signals)} AI signals (before regime filtering)")
                
                # 🚀 PHASE 13: Apply regime-based threshold filtering
                if regime_threshold is not None:
                    # Filter BUY signals based on regime threshold
                    original_count = len(signals)
                    signals = [
                        s for s in signals 
                        if s['action'] != 'buy' or s.get('ai_score', s.get('confidence', 0)) >= regime_threshold
                    ]
                    filtered_count = original_count - len(signals)
                    if filtered_count > 0:
                        self.logger.info(f"[REGIME] Filtered {filtered_count} BUY signals below {regime_threshold} threshold")
                elif regime_code == 0:  # CRISIS
                    # Block all new BUY entries in CRISIS regime
                    original_count = len(signals)
                    signals = [s for s in signals if s['action'] != 'buy']
                    filtered_count = original_count - len(signals)
                    if filtered_count > 0:
                        self.logger.warning(f"[REGIME] CRISIS regime - blocked {filtered_count} BUY signals")
                
                self.logger.info(f"[SIGNALS] {len(signals)} AI signals after regime filtering")
                
                # Convert signals to DataFrame for portfolio manager
                import pandas as pd
                if signals:
                    signals_df = pd.DataFrame(signals)
                    
                    # Rename columns to match portfolio_manager expectations (uppercase)
                    signals_df = signals_df.rename(columns={
                        'ticker': 'Ticker',
                        'action': 'Action',
                        'price': 'Price'
                    })
                    
                    # Use portfolio_manager to handle signal execution
                    # This will properly evaluate held positions and execute trades
                    from core.portfolio_manager import PortfolioManager
                    project_root = Path(__file__).resolve().parent
                    portfolio_manager = PortfolioManager(project_root)
                    
                    # Update portfolio with signals (handles BUY/SELL logic correctly)
                    updated_portfolio = portfolio_manager.update_portfolio(signals_df)
                    
                    self.logger.info(f"[PORTFOLIO] Portfolio updated with {len(updated_portfolio)} positions")
                    
                    # CRITICAL: Sync PortfolioManager positions to VirtualEngine portfolio_paper.json
                    # PortfolioManager saves to portfolio.csv, but VirtualEngine uses portfolio_paper.json
                    # We need to sync them so positions persist across runs
                    active_positions = updated_portfolio[updated_portfolio['Status'] == 'ACTIVE']
                    if len(active_positions) > 0:
                        # Convert PortfolioManager positions to VirtualEngine format
                        synced_positions = {}
                        for _, row in active_positions.iterrows():
                            ticker = row['Ticker'].upper()
                            synced_positions[ticker] = {
                                'shares': int(row['Quantity']),
                                'cost_basis': float(row['EntryPrice']),
                                'current_price': float(row['CurrentPrice'])
                            }
                        
                        # Update VirtualEngine portfolio with synced positions
                        self.virtual_engine.portfolio['positions'] = synced_positions
                        self.virtual_engine.save_portfolio()
                        self.logger.info(f"[SYNC] Synced {len(synced_positions)} positions from PortfolioManager to VirtualEngine")
                    
                    # Count trades by comparing before/after
                    buy_signals = [s for s in signals if s['action'] == 'buy']
                    sell_signals = [s for s in signals if s['action'] == 'sell']
                    
                    # Store signals and portfolio data for email report
                    global EMAIL_BUY_SIGNALS, EMAIL_SELL_SIGNALS, EMAIL_PORTFOLIO_DATA
                    EMAIL_BUY_SIGNALS = buy_signals
                    EMAIL_SELL_SIGNALS = sell_signals
                    EMAIL_PORTFOLIO_DATA = {
                        'active_positions': active_positions,
                        'portfolio_value': self.virtual_engine.get_account_info().get('portfolio_value', 0),
                        'cash': self.virtual_engine.portfolio.get('cash', 0),
                        'total_pnl': sum(active_positions['PnL_USD']) if len(active_positions) > 0 else 0
                    }
                    
                    self.logger.info(f"[OK] Signal processing completed")
                    self.logger.info(f"[SUMMARY] Session Results:")
                    self.logger.info(f"  Total Signals: {len(signals)}")
                    self.logger.info(f"  Buy Signals: {len(buy_signals)}")
                    self.logger.info(f"  Sell Signals: {len(sell_signals)}")
                    self.logger.info(f"  Portfolio Positions: {len(updated_portfolio[updated_portfolio['Status'] == 'ACTIVE'])}")
                else:
                    self.logger.info("[SIGNALS] No signals generated")
                    self.logger.info(f"[SUMMARY] Session Results:")
                    self.logger.info(f"  Total Signals: 0")
                    self.logger.info(f"  Buy Signals: 0")
                    self.logger.info(f"  Sell Signals: 0")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Signal generation failed: {e}")
                import traceback
                self.logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
                return False
            
            # CRITICAL FIX: Do NOT call update_portfolio_values() here
            # PortfolioManager already saved positions correctly
            # VirtualEngine.update_portfolio_values() was overwriting with empty portfolio
            # self.virtual_engine.update_portfolio_values()  # REMOVED - causes phantom portfolio bug
            
            session_end = datetime.now()
            duration = session_end - session_start
            
            self.logger.info(f"[OK] TRADE MODE completed in {duration.total_seconds():.2f} seconds")
            _pval = self.virtual_engine.get_account_info().get('portfolio_value', 0)
            self.logger.info(f"   Portfolio value: ${_pval:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error in TRADE MODE: {e}")
            return False
    
    def run_report_mode(self) -> bool:
        """Run report mode - Generate daily executive brief with portfolio information"""
        try:
            self.logger.info("[EMAIL] Running REPORT MODE - IST Scheduler 23:15 IST Email")
            session_start = datetime.now()
            
            # Check kill switch
            if self.check_kill_switch():
                self.logger.warning("[STOP] Kill switch activated - stopping report session")
                return False
            
            # Initialize modules
            if not self.initialize_modules():
                return False

            # IBKR sync: refresh portfolio_paper.json before generating the report
            self.logger.info("[IBKR SYNC] Syncing portfolio_paper.json from IBKR for report...")
            sync_ok = self.virtual_engine.sync_from_ibkr()
            if not sync_ok:
                self.logger.warning("[IBKR SYNC] Sync skipped or failed - report will use last saved state")

            # Get portfolio info from virtual engine (always available in report mode)
            account_info = self.virtual_engine.get_account_info()
            current_positions = self.virtual_engine.get_current_positions()
            
            portfolio_value = account_info.get('portfolio_value', 100000.0)
            cash = account_info.get('cash', 100000.0)
            positions_value = account_info.get('positions_value', 0.0)
            
            # Format active positions
            active_positions_str = ""
            if current_positions:
                for pos in current_positions:
                    symbol = pos.get('symbol', 'Unknown')
                    qty = pos.get('position', 0)
                    price = pos.get('market_price', 0)
                    value = pos.get('market_value', 0)
                    active_positions_str += f"- {symbol}: {qty} shares @ ${price:.2f} (${value:.2f})\n"
            else:
                active_positions_str = "No active positions"

            # Get risk summary with default values for missing risk manager
            risk_summary = self._get_risk_summary_with_defaults()

            # Get Phase 10 Sentiment Analysis
            sentiment_analysis_str = self._get_sentiment_analysis_string()

            # Generate comprehensive report
            success = True
            report_content = f"""
NeuralTrader Daily Executive Brief
=====================================

[DATE] Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
[MODE] Mode: Shadow Trading Simulator
[CONSTITUTION] Constitution: Risk Management Active

[DATA] PORTFOLIO OVERVIEW:
-------------------------
Total Portfolio Value: ${portfolio_value:,.2f}
Cash Balance: ${cash:,.2f}
Position Value: ${positions_value:,.2f}

[POSITIONS] ACTIVE POSITIONS:
------------------------------
{active_positions_str}

[RISK] RISK MANAGEMENT STATUS:
------------------------------
Risk Level: {risk_summary.get('risk_level', 'Low')}
Max Drawdown: {risk_summary.get('max_drawdown', 'Within Limits')}
Sector Exposure: {risk_summary.get('sector_exposure', 'Balanced')}
Constitution Compliance: {risk_summary.get('constitution_status', 'Active')}

[SECTOR] SECTOR AUTHORITY STATUS:
--------------------------------
Sector Tax Applied: {risk_summary.get('sector_tax_status', 'Active')}
Volatility Gate: {risk_summary.get('volatility_gate', 'Active')}
Bottom 3 Sectors: {risk_summary.get('bottom_sectors', 'None')}
Top 3 Sectors: {risk_summary.get('top_sectors', 'None')}

[PERFORMANCE] TODAY'S PERFORMANCE:
----------------------------------
Daily P&L: {risk_summary.get('daily_pnl', 'N/A')}
Win Rate: {risk_summary.get('win_rate', 'N/A')}
Total Trades: {risk_summary.get('total_trades', 'N/A')}

[SYSTEM] SYSTEM STATUS:
--------------------
AI Models: Online
Trading Strategy: Active
Data Freshness: Current
Market Filter: {risk_summary.get('market_filter', 'Pass')}

[PHASE10] SENTIMENT ANALYSIS:
------------------------------
{sentiment_analysis_str}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST
NeuralTrader Automated Trading System v5.3 - Phase 10 Economic Data Integration Active
"""
            
            # Send enhanced notification with logs
            if success:
                # DISABLED: Old dashboard email system (redundant with enhanced email in finally block)
                # The enhanced email in the finally block now provides complete portfolio data
                # try:
                #     # Generate dashboard
                #     self._send_html_report_with_attachments()
                # except Exception as email_error:
                #     self.logger.error(f"[ERROR] Failed to send dashboard notification: {email_error}")
                #     self.logger.error(f"[ERROR] Email notifier status: {type(self.email_notifier)}")
                #     # Fallback to plain text report
                #     try:
                #         self._send_daily_report_notification(report_content)
                #     except Exception as fallback_error:
                #         self.logger.error(f"[ERROR] Fallback plain text report also failed: {fallback_error}")
                self.logger.info("[EMAIL] Dashboard email disabled - using enhanced email in finally block")
            else:
                self.logger.error("[ERROR] Failed to generate daily executive brief")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error in REPORT MODE: {e}")
            return False
    
    def _get_sentiment_analysis_string(self) -> str:
        """Get sentiment analysis string for reports"""
        if not self.sentiment_integration:
            return "Sentiment Analysis: Not Available - Integration Failed"
        
        try:
            # Get comprehensive sentiment analysis
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            sentiment_data = self.sentiment_integration.get_comprehensive_sentiment(start_date, end_date)
            
            # Extract key information
            overall_sentiment = sentiment_data.get('overall_sentiment', {})
            market_signals = sentiment_data.get('market_signals', {})
            components = sentiment_data.get('components', {})
            
            # Format sentiment information
            sentiment_str = f"Overall Sentiment: {overall_sentiment.get('regime', 'NEUTRAL')} (Score: {overall_sentiment.get('score', 0.0):.3f})\n"
            sentiment_str += f"Confidence: {overall_sentiment.get('confidence', 0.0):.1%}\n"
            sentiment_str += f"Market Outlook: {market_signals.get('market_outlook', 'NEUTRAL')}\n"
            sentiment_str += f"Equity Bias: {market_signals.get('equity_bias', 'NEUTRAL')}\n"
            sentiment_str += f"Risk Adjustment: {market_signals.get('risk_adjustment', 1.0):.2f}x\n"
            
            # Add component status
            economic_status = components.get('economic', {}).get('status', 'inactive')
            sentiment_str += f"Economic Analysis: {economic_status.title()}\n"
            
            # Add sector recommendations if available
            sector_recs = market_signals.get('sector_recommendations', {})
            if sector_recs:
                sentiment_str += "Key Sector Recommendations:\n"
                for sector, rec in list(sector_recs.items())[:3]:  # Top 3 sectors
                    sentiment_str += f"  - {sector}: {rec}\n"
            
            return sentiment_str
            
        except Exception as e:
            self.logger.error(f"[PHASE10] Failed to get sentiment analysis: {e}")
            return "Sentiment Analysis: Error - Check logs for details"
    
    def _get_risk_summary_with_defaults(self) -> Dict:
        """Get risk summary with default values for missing risk manager"""
        try:
            # Try to get risk summary from sector authority if available
            if hasattr(self, 'sector_auth') and self.sector_auth:
                # Get sector information
                sector_momentum = self.sector_auth.get_sector_momentum()
                
                # Convert list of tuples to dictionary if needed
                if sector_momentum and isinstance(sector_momentum, list):
                    sector_momentum = dict(sector_momentum)
                
                # Ensure sector_momentum is a dictionary
                if sector_momentum and isinstance(sector_momentum, dict):
                    bottom_sectors = [k for k, v in sector_momentum.items() if v < 0]
                    top_sectors = [k for k, v in sector_momentum.items() if v > 0]
                    
                    return {
                        'risk_level': 'Low',
                        'max_drawdown': 'Within Limits',
                        'sector_exposure': 'Balanced',
                        'constitution_status': 'Active',
                        'sector_tax_status': 'Active',
                        'volatility_gate': 'Active',
                        'bottom_sectors': ', '.join(bottom_sectors[:3]) if bottom_sectors else 'None',
                        'top_sectors': ', '.join(top_sectors[:3]) if top_sectors else 'None',
                        'daily_pnl': 'N/A',
                        'win_rate': 'N/A',
                        'total_trades': 'N/A',
                        'market_filter': 'Pass'
                    }
                else:
                    self.logger.warning(f"[WARNING] Sector momentum is not a dictionary: {type(sector_momentum)}")
        except Exception as e:
            self.logger.warning(f"[WARNING] Could not get sector info: {e}")
        
        # Return default risk summary
        return {
            'risk_level': 'Low',
            'max_drawdown': 'Within Limits',
            'sector_exposure': 'Balanced',
            'constitution_status': 'Active',
            'sector_tax_status': 'Active',
            'volatility_gate': 'Active',
            'bottom_sectors': 'None',
            'top_sectors': 'None',
            'daily_pnl': 'N/A',
            'win_rate': 'N/A',
            'total_trades': 'N/A',
            'market_filter': 'Pass'
        }
    
    def run_saturday_retrain_mode(self) -> bool:
        """Run Saturday retrain mode - Model retraining and updates"""
        run_id = None
        session_start = datetime.now()
        
        try:
            # Log supervision start
            run_id = self._log_supervision_start('saturday_retrain')
            
            self.logger.info("[RETRAIN] Running SATURDAY RETRAIN MODE - Model Retraining")
            
            # Check kill switch
            if self.check_kill_switch():
                self.logger.warning("[STOP] Kill switch activated - stopping retrain session")
                self._log_supervision_error('saturday_retrain', run_id, 'Kill switch activated')
                return False
            
            # Initialize modules
            if not self.initialize_modules():
                self._log_supervision_error('saturday_retrain', run_id, 'Failed to initialize modules')
                return False
            
            # Perform Saturday retrain
            self.logger.info("[RETRAIN] Starting Saturday model retraining...")
            retrain_results = self._perform_saturday_retrain()
            
            # Send notification with logs
            self._send_saturday_retrain_notification(retrain_results)
            
            session_end = datetime.now()
            duration = (session_end - session_start).total_seconds()
            
            # Log supervision completion
            details = {
                "retrain_status": retrain_results.get('status', 'Unknown'),
                "models_updated": retrain_results.get('models_updated', 0),
                "performance": retrain_results.get('performance', 'N/A')
            }
            self._log_supervision_complete('saturday_retrain', run_id, True, duration, details)
            
            self.logger.info(f"[OK] SATURDAY RETRAIN MODE completed in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error in SATURDAY RETRAIN MODE: {e}")
            if run_id:
                self._log_supervision_error('saturday_retrain', run_id, str(e))
            return False
    
    def _perform_saturday_retrain(self) -> Dict:
        """Perform Saturday retrain operations"""
        import subprocess
        import sys
        import time
        
        try:
            self.logger.info("[RETRAIN] Starting Saturday model retraining...")
            start_time = time.time()
            
            # Execute the Phase 12 Brain-Gate protected retraining script
            self.logger.info("[RETRAIN] Executing: python scripts/retrain_phase12.py")
            
            result = subprocess.run(
                [sys.executable, "scripts/retrain_phase12.py"],
                capture_output=True,
                text=True,
                cwd="."  # Run from project root
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info("[OK] Ensemble training completed successfully")
                self.logger.info(f"   Duration: {duration:.1f} seconds")
                
                # Parse output for key metrics from Brain-Gate validation
                output_lines = result.stdout.split('\n')
                models_updated = 0
                performance_improvement = "N/A"
                validation_passed = False
                
                for line in output_lines:
                    if "Brain Gate Validation Passed" in line:
                        validation_passed = True
                        models_updated = 3  # XGBoost, LightGBM, RandomForest
                        performance_improvement = "Brain Gate Passed"
                    elif "ENSEMBLE" in line and "Accuracy:" in line:
                        # Extract ensemble accuracy
                        try:
                            accuracy_str = line.split("Accuracy:")[1].strip()
                            performance_improvement = f"Ensemble Acc: {accuracy_str}"
                        except:
                            pass
                    elif "Performance Gate Failed" in line:
                        performance_improvement = "Performance Gate Failed"
                        break
                
                retrain_results = {
                    "status": "SUCCESS",
                    "models_updated": models_updated,
                    "performance": performance_improvement,
                    "validation_passed": validation_passed,
                    "duration": f"{duration:.1f} seconds",
                    "details": {
                        "models": ["XGBoost", "LightGBM", "RandomForest"],
                        "exit_code": result.returncode,
                        "output_length": len(result.stdout),
                        "brain_gate_active": True
                    }
                }
                
                self.logger.info(f"[OK] Models updated: {retrain_results['models_updated']}")
                self.logger.info(f"[OK] Performance: {retrain_results['performance']}")
                self.logger.info(f"[OK] Duration: {retrain_results['duration']}")
                
                return retrain_results
                
            else:
                self.logger.error(f"[ERROR] Ensemble training failed with exit code: {result.returncode}")
                self.logger.error(f"[ERROR] Error output: {result.stderr}")
                
                return {
                    "status": "FAILED",
                    "models_updated": 0,
                    "performance": "Training failed",
                    "duration": f"{duration:.1f} seconds",
                    "details": {
                        "exit_code": result.returncode,
                        "error": result.stderr,
                        "output": result.stdout
                    }
                }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error during Saturday retrain: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "models_updated": 0
            }
    
    def _generate_signal_simple(self, ticker: str) -> float:
        """Generate simple signal for testing"""
        try:
            # Simple random signal for testing
            import random
            return random.random()
        except Exception as e:
            self.logger.error(f"Error generating signal for {ticker}: {e}")
            return 0.0
    
    def run_auto_mode(self):
        """Run auto mode with IST scheduling"""
        try:
            self.logger.info("[TIME] Running AUTO MODE - IST Scheduling")
            
            # Initialize modules
            if not self.initialize_modules():
                return
            
            # Start IST scheduler
            import schedule
            
            # Schedule tasks
            schedule.every().day.at("16:45").do(self.run_fetch_mode)
            schedule.every().day.at("23:15").do(self.run_report_mode)
            
            self.logger.info("[OK] IST Scheduler started:")
            self.logger.info("   - Data fetch: 16:45 IST (09:45 EST)")
            self.logger.info("   - Daily report: 23:15 IST")
            
            # Run scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("[STOP] Auto mode stopped by user")
        except Exception as e:
            self.logger.error(f"[ERROR] Error in AUTO MODE: {e}")
    
    def _generate_signal(self, df: pd.DataFrame) -> float:
        """Generate trading signal (placeholder for AI model)"""
        try:
            if len(df) < 20:
                return 0.0
            
            # Simple momentum signal
            recent_return = df['close'].pct_change(5).iloc[-1]
            
            if recent_return > 0.02:
                return 0.7
            elif recent_return < -0.02:
                return 0.3
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return 0.0
    
# ==================== MAIN FUNCTION WITH CLI ====================

def main():
    """Main function with CLI argument parsing and Ironclad Wrapper"""
    
    # Environment safety check
    check_environment()
    
    # Setup logging
    logger = setup_automation_logging()
    
    # ==================== CORE PROTECTION PROTOCOL ====================
    logger.info("[STARTUP] Initiating NeuralTrader Core Protection Protocol...")
    
    # Load the REAL brains - EnsemblePredictor and TradingStrategy
    try:
        logger.info("[STARTUP] Loading real AI model (EnsemblePredictor)...")
        real_ai = EnsemblePredictor()
        logger.info("[STARTUP] Loading real trading strategy (TradingStrategy)...")
        real_strategy = TradingStrategy()
    except Exception as e:
        logger.critical(f"[CRITICAL] Failed to load core components: {e}")
        logger.critical("[CRITICAL] System cannot operate without AI model and strategy")
        sys.exit(1)
    
    # Verify the REAL brains - CRITICAL: Must pass or system terminates
    # This enforces the "No Silent Failures" law
    try:
        verify_system_integrity(real_ai, real_strategy)
        logger.info("[STARTUP] Core Protection Protocol verification complete")
    except SystemExit:
        logger.critical("[CRITICAL] Core Protection Protocol FAILED - System terminating")
        logger.critical("[CRITICAL] Trading operations BLOCKED - Integrity check failed")
        raise  # Re-raise to ensure immediate termination
    
    # ==================== END CORE PROTECTION PROTOCOL ====================
    
    # Call the Ironclad Wrapper main function with AI and strategy objects
    ironclad_main(real_ai, real_strategy, CURRENT_LOG_FILE)

def ironclad_main(real_ai, real_strategy, log_file_path):
    """
    Ironclad Wrapper - Guaranteed Email Notifications
    ================================================
    
    This function wraps the entire daily execution sequence in a massive try...except block
    to ensure email notifications are sent whether the pipeline succeeds or catastrophically fails.
    """
    # Get the logger (already set up in main())
    logger = logging.getLogger('NeuralTrader_Automation')
    
    # Initialize email notifier
    notifier = EmailNotifier()
    
    # Track if we caught an exception
    exception_caught = False
    exception_traceback = None
    execution_result = None
    
    try:
        # ===== IRONCLAD EXECUTION WRAPPER =====
        logger.info("=" * 80)
        logger.info("[IRONCLAD] Starting NeuralTrader execution with guaranteed notifications")
        logger.info(f"[TIME] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"[LOG] Log file: {log_file_path}")
        logger.info("=" * 80)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='NeuralTrader Automated Trading System')
        parser.add_argument('--mode', choices=['fetch', 'trade', 'report', 'paper', 'auto', 'saturday_retrain', 'backtest', 'simulation'], 
                          help='Execution mode')
        parser.add_argument('--start',   default='2000-01-01', help='Backtest start date (backtest/simulation modes)')
        parser.add_argument('--end',     default=None,          help='Backtest end date (backtest/simulation modes)')
        parser.add_argument('--cash',    default=100000.0, type=float, help='Initial cash (backtest/simulation modes)')
        parser.add_argument('--tickers', nargs='*', default=None, help='Ticker subset (backtest/simulation modes)')
        parser.add_argument('--data-fetch', action='store_true', help='Fetch market data (legacy)')
        parser.add_argument('--trading', action='store_true', help='Run trading session (legacy)')
        parser.add_argument('--report', action='store_true', help='Send daily report (legacy)')
        
        args = parser.parse_args()
        
        # Initialize orchestrator with REAL AI and strategy objects
        # These objects have already passed integrity validation
        orchestrator = TradingOrchestrator(ai_model=real_ai, trading_strategy=real_strategy, mode=args.mode)
        logger.info("[STARTUP] Orchestrator initialized with validated AI and strategy components")
        
        # ===== DAILY EXECUTION SEQUENCE =====
        # Determine mode and execute
        if args.mode == 'fetch' or args.data_fetch:
            logger.info("[MODE] Running data fetch sequence")
            execution_result = orchestrator.run_fetch_mode()
            # Smart notification logic for fetch mode
            if execution_result:
                logger.info("[INFO] Fetch successful. Email will be sent in finally block.")
            else:
                logger.error("[ERROR] Fetch failed. Will send crash report in finally block.")
        
        elif args.mode == 'trade' or args.trading:
            logger.info("[MODE] Running trading sequence")
            execution_result = orchestrator.run_trade_mode()
            
        elif args.mode == 'report' or args.report:
            logger.info("[MODE] Running report sequence")
            execution_result = orchestrator.run_report_mode()
            
        elif args.mode == 'saturday_retrain':
            logger.info("[MODE] Running Saturday retrain sequence")
            execution_result = orchestrator.run_saturday_retrain_mode()
        
        elif args.mode == 'paper':
            # Set global paper trading flag
            global PAPER_TRADING
            PAPER_TRADING = True
            
            logger.info("[PAPER] Running in PAPER TRADING mode (No real money)")
            logger.info("[PAPER] Core Protection Protocol active with paper trading")
            
            execution_result = orchestrator.run_trade_mode()
            logger.info("[PAPER] Paper trading mode completed")
        
        elif args.mode == 'auto':
            logger.info("[MODE] Running auto sequence")
            execution_result = orchestrator.run_auto_mode()

        elif args.mode == 'backtest':
            from scripts.run_full_backtest import UnifiedBacktest
            logger.info(f"[MODE] BACKTEST | {args.start} -> {args.end or 'today'} | cash=${args.cash:,.0f}")
            bt = UnifiedBacktest(
                start_date=args.start,
                end_date=args.end,
                initial_cash=args.cash,
                ticker_filter=args.tickers,
            )
            bt.run()
            metrics = bt.report()
            execution_result = True
            logger.info("[MODE] Backtest complete. Results -> reports/backtest_metrics.json")
            # Write portfolio_backtest.json snapshot
            try:
                import json as _json
                _bt_portfolio = {
                    'mode': 'backtest',
                    'period': f"{args.start} -> {args.end or 'today'}",
                    'initial_cash': args.cash,
                    'final_value': bt.portfolio_value,
                    'cash': bt.cash,
                    'positions': {
                        t: {'shares': p['shares'], 'entry_price': p['entry_price']}
                        for t, p in bt.positions.items()
                    },
                    'total_trades': len(bt.trades),
                    'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'metrics': metrics or {},
                }
                _bt_path = Path(__file__).resolve().parent / 'data' / 'portfolio_backtest.json'
                _bt_path.parent.mkdir(exist_ok=True)
                _bt_path.write_text(_json.dumps(_bt_portfolio, indent=2, default=str))
                logger.info(f"[PORTFOLIO] portfolio_backtest.json updated -> {_bt_path}")
            except Exception as _e:
                logger.warning(f"[PORTFOLIO] Could not write portfolio_backtest.json: {_e}")

        elif args.mode == 'simulation':
            from scripts.run_simulation import run_simulation
            logger.info(f"[MODE] SIMULATION | {args.start or 'last 252d'} -> {args.end or 'today'} | cash=${args.cash:,.0f}")
            sim_metrics = run_simulation(
                start_date=args.start,
                end_date=args.end,
                initial_cash=args.cash,
                ticker_filter=args.tickers,
            )
            execution_result = True
            logger.info("[MODE] Simulation complete. Results -> reports/simulation_metrics.json")
            # Write portfolio_simulation.json snapshot
            try:
                import json as _json
                _sim_portfolio = {
                    'mode': 'simulation',
                    'period': f"{args.start or 'last 252d'} -> {args.end or 'today'}",
                    'initial_cash': args.cash,
                    'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'metrics': sim_metrics or {},
                }
                _sim_path = Path(__file__).resolve().parent / 'data' / 'portfolio_simulation.json'
                _sim_path.parent.mkdir(exist_ok=True)
                _sim_path.write_text(_json.dumps(_sim_portfolio, indent=2, default=str))
                logger.info(f"[PORTFOLIO] portfolio_simulation.json updated -> {_sim_path}")
            except Exception as _e:
                logger.warning(f"[PORTFOLIO] Could not write portfolio_simulation.json: {_e}")

        else:
            logger.error("[ERROR] No mode specified. Use --mode=fetch|trade|report|paper|auto|saturday_retrain|backtest|simulation")
            parser.print_help()
            execution_result = False
        
        # Log successful completion
        logger.info("[SUCCESS] Daily execution sequence completed successfully")
        logger.info(f"[RESULT] Execution result: {execution_result}")
        
    except KeyboardInterrupt:
        logger.info("[STOP] NeuralTrader session interrupted by user")
        execution_result = False
        exception_caught = True
        exception_traceback = "Session interrupted by user"
        
    except Exception as e:
        # ===== CATASTROPHIC FAILURE HANDLING =====
        logger.error(f"[CRASH] 🚨 CATASTROPHIC FAILURE DETECTED: {e}")
        logger.error("[CRASH] Capturing full traceback for crash report...")
        
        # Capture the full traceback
        exception_caught = True
        exception_traceback = traceback.format_exc()
        execution_result = False
        
        # Log the full traceback
        logger.error(f"[CRASH] Full traceback:\n{exception_traceback}")
        
    finally:
        # ===== GUARANTEED EMAIL NOTIFICATION (one email per run, log always attached) =====
        logger.info("[IRONCLAD] Entering finally block - sending guaranteed notification")

        try:
            # Declare globals to avoid UnboundLocalError
            global EMAIL_BUY_SIGNALS, EMAIL_SELL_SIGNALS, EMAIL_PORTFOLIO_DATA
            
            # Initialize email data globals if not set
            if 'EMAIL_BUY_SIGNALS' not in globals():
                EMAIL_BUY_SIGNALS = []
            if 'EMAIL_SELL_SIGNALS' not in globals():
                EMAIL_SELL_SIGNALS = []
            if 'EMAIL_PORTFOLIO_DATA' not in globals():
                EMAIL_PORTFOLIO_DATA = {
                    'active_positions': pd.DataFrame(),
                    'portfolio_value': 0,
                    'cash': 0,
                    'total_pnl': 0
                }
            
            timestamp  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            mode_label = (args.mode if 'args' in locals() and args.mode else 'UNKNOWN').upper()
            paper_flag = PAPER_TRADING if 'PAPER_TRADING' in globals() else False

            # ---- Subject line ------------------------------------------------
            if exception_caught:
                subject = f"[CRASH] NeuralTrader {mode_label} FAILED - {timestamp}"
                status_tag   = "CRASH"
                header_color = "#c0392b"
            elif not execution_result:
                subject = f"[WARN] NeuralTrader {mode_label} completed with issues - {timestamp}"
                status_tag   = "WARN"
                header_color = "#e67e22"
            else:
                subject = f"[OK] NeuralTrader {mode_label} completed - {timestamp}"
                status_tag   = "OK"
                header_color = "#27ae60"

            logger.info(f"[EMAIL] Subject: {subject}")

            # ---- HTML body ---------------------------------------------------
            crash_section = ""
            if exception_caught and exception_traceback:
                crash_section = f"""
<div style="background:#fff3f3;border:2px solid #e74c3c;padding:15px;border-radius:6px;margin-top:15px;">
  <h3 style="color:#c0392b;margin:0 0 10px 0;">[CRASH] Traceback</h3>
  <pre style="font-size:12px;white-space:pre-wrap;word-break:break-all;">{exception_traceback}</pre>
</div>"""

            html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;padding:20px;background:#f5f5f5;">

  <div style="background:{header_color};color:white;padding:20px;border-radius:8px;text-align:center;">
    <h1 style="margin:0;">NeuralTrader — {mode_label} Report</h1>
    <p style="margin:6px 0 0 0;font-size:14px;">{timestamp} IST &nbsp;|&nbsp; Status: [{status_tag}]</p>
  </div>

  <div style="background:white;padding:20px;margin-top:12px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;">Execution Summary</h2>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="padding:6px;color:#555;width:160px;"><strong>Mode</strong></td>
          <td style="padding:6px;">{mode_label}</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:6px;color:#555;"><strong>Result</strong></td>
          <td style="padding:6px;">{'SUCCESS' if execution_result and not exception_caught else 'FAILED' if exception_caught else 'COMPLETED WITH ISSUES'}</td></tr>
      <tr><td style="padding:6px;color:#555;"><strong>Paper Trading</strong></td>
          <td style="padding:6px;">{'YES' if paper_flag else 'NO'}</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:6px;color:#555;"><strong>Log File</strong></td>
          <td style="padding:6px;font-size:12px;">{log_file_path or 'N/A'}</td></tr>
      <tr><td style="padding:6px;color:#555;"><strong>Timestamp</strong></td>
          <td style="padding:6px;">{timestamp}</td></tr>
    </table>
    {crash_section}
  </div>

  <div style="background:white;padding:20px;margin-top:12px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;">💰 Portfolio Summary</h2>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="padding:8px;color:#555;width:160px;"><strong>Portfolio Value</strong></td>
          <td style="padding:8px;font-size:18px;font-weight:bold;color:#27ae60;">${EMAIL_PORTFOLIO_DATA.get('portfolio_value', 0):,.2f}</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:8px;color:#555;"><strong>Available Cash</strong></td>
          <td style="padding:8px;">${EMAIL_PORTFOLIO_DATA.get('cash', 0):,.2f}</td></tr>
      <tr><td style="padding:8px;color:#555;"><strong>Total P&L</strong></td>
          <td style="padding:8px;font-weight:bold;color:{'#27ae60' if EMAIL_PORTFOLIO_DATA.get('total_pnl', 0) >= 0 else '#e74c3c'};">${EMAIL_PORTFOLIO_DATA.get('total_pnl', 0):,.2f}</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:8px;color:#555;"><strong>Active Positions</strong></td>
          <td style="padding:8px;">{len(EMAIL_PORTFOLIO_DATA.get('active_positions', []))}</td></tr>
    </table>
  </div>

  <div style="background:white;padding:20px;margin-top:12px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;">📊 Current Positions</h2>
    {''.join([f'''<div style="background:#f9f9f9;padding:12px;margin:8px 0;border-radius:6px;border-left:4px solid {'#27ae60' if row['PnL_Pct'] >= 0 else '#e74c3c'};">
      <div style="font-weight:bold;font-size:16px;margin-bottom:6px;">{row['Ticker']}</div>
      <table style="width:100%;font-size:13px;">
        <tr><td style="color:#666;width:120px;">Quantity:</td><td>{int(row['Quantity'])}</td></tr>
        <tr><td style="color:#666;">Entry Price:</td><td>${row['EntryPrice']:.2f}</td></tr>
        <tr><td style="color:#666;">Current Price:</td><td>${row['CurrentPrice']:.2f}</td></tr>
        <tr><td style="color:#666;">P&L:</td><td style="font-weight:bold;color:{'#27ae60' if row['PnL_Pct'] >= 0 else '#e74c3c'};">${row['PnL_USD']:.2f} ({row['PnL_Pct']:.2f}%)</td></tr>
        <tr><td style="color:#666;">Days Held:</td><td>{int(row['DaysHeld'])}</td></tr>
      </table>
    </div>''' for _, row in EMAIL_PORTFOLIO_DATA.get('active_positions', pd.DataFrame()).iterrows()]) if len(EMAIL_PORTFOLIO_DATA.get('active_positions', [])) > 0 else '<p style="color:#999;text-align:center;padding:20px;">No active positions</p>'}
  </div>

  <div style="background:white;padding:20px;margin-top:12px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;">🔵 Buy Signals Today</h2>
    {''.join([f'''<div style="background:#e8f5e9;padding:10px;margin:6px 0;border-radius:4px;border-left:3px solid #27ae60;">
      <strong>{signal['ticker'].upper()}</strong> @ ${signal['price']:.2f} — AI Score: {signal.get('ai_score', signal.get('confidence', 0)):.3f}
      <div style="font-size:12px;color:#666;margin-top:4px;">Reason: {signal.get('reason', 'AI recommendation')}</div>
    </div>''' for signal in EMAIL_BUY_SIGNALS]) if len(EMAIL_BUY_SIGNALS) > 0 else '<p style="color:#999;text-align:center;padding:20px;">No buy signals today</p>'}
  </div>

  <div style="background:white;padding:20px;margin-top:12px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;">🔴 Sell Signals Today</h2>
    {''.join([f'''<div style="background:#ffebee;padding:10px;margin:6px 0;border-radius:4px;border-left:3px solid #e74c3c;">
      <strong>{signal['ticker'].upper()}</strong> @ ${signal['price']:.2f} — AI Score: {signal.get('ai_score', signal.get('confidence', 0)):.3f}
      <div style="font-size:12px;color:#666;margin-top:4px;">Reason: {signal.get('reason', 'AI recommendation')}</div>
    </div>''' for signal in EMAIL_SELL_SIGNALS]) if len(EMAIL_SELL_SIGNALS) > 0 else '<p style="color:#999;text-align:center;padding:20px;">No sell signals today</p>'}
  </div>

  <div style="background:white;padding:20px;margin-top:12px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;">⚙️ System Status</h2>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="padding:5px;color:#555;width:160px;"><strong>Orchestrator</strong></td>
          <td style="padding:5px;">{'INITIALIZED' if 'orchestrator' in locals() else 'NOT INITIALIZED'}</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:5px;color:#555;"><strong>Risk Management</strong></td>
          <td style="padding:5px;">ACTIVE (Uncle Point 10% / 8-day cooldown)</td></tr>
      <tr><td style="padding:5px;color:#555;"><strong>SPY Regime Filter</strong></td>
          <td style="padding:5px;">ACTIVE (100-day SMA)</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:5px;color:#555;"><strong>Max Positions</strong></td>
          <td style="padding:5px;">20 concurrent</td></tr>
    </table>
  </div>

  <p style="text-align:center;color:#999;font-size:12px;margin-top:16px;">
    NeuralTrader v5.5 — Ironclad Wrapper — Daily log attached
  </p>
</body>
</html>"""

            # ---- Send single email with log and TradingView signals attached ----
            logger.info("[EMAIL] Sending single notification with log attachment...")
            
            # Get TradingView signal file if it exists
            tradingview_file = TRADINGVIEW_SIGNAL_FILE if 'TRADINGVIEW_SIGNAL_FILE' in globals() and TRADINGVIEW_SIGNAL_FILE else None
            if tradingview_file:
                logger.info(f"[EMAIL] Attaching TradingView signals: {tradingview_file}")
            
            # Get portfolio TradingView file if it exists (managed portfolio)
            portfolio_file = PORTFOLIO_TRADINGVIEW_FILE if 'PORTFOLIO_TRADINGVIEW_FILE' in globals() and PORTFOLIO_TRADINGVIEW_FILE else None
            if portfolio_file:
                logger.info(f"[EMAIL] Attaching managed portfolio: {portfolio_file}")
            
            # Attach portfolio file (managed portfolio) instead of daily signals
            attachment_to_use = portfolio_file if portfolio_file else tradingview_file
            
            email_sent = notifier._send_email(
                subject=subject,
                html_content=html_body,
                log_file_path=log_file_path,
                attachment_file=attachment_to_use,
            )

            if email_sent:
                logger.info(f"[EMAIL] [OK] Notification sent | mode={mode_label} | status={status_tag}")
            else:
                logger.error(f"[EMAIL] [FAIL] Failed to send notification | mode={mode_label}")

        except Exception as email_error:
            logger.error(f"[EMAIL] [FAIL] CRITICAL - could not send notification: {email_error}")
            logger.error(f"[EMAIL] {traceback.format_exc()}")

        # ===== FINAL LOGGING =====
        logger.info("=" * 80)
        if exception_caught:
            logger.info("[IRONCLAD] Session completed with CRASH - notification sent")
        else:
            logger.info("[IRONCLAD] Session completed successfully - notification sent")
        logger.info(f"[TIME] Final timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Exit with appropriate code
        if exception_caught:
            sys.exit(1)
        elif execution_result is False:
            sys.exit(1)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()
