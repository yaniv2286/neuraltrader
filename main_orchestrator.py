"""
NeuralTrader Main Orchestrator - The Pilot
==========================================

Master script that automates the complete trading workflow:
Update Data → Score → Risk Check → Execute

Features:
- Complete automated trading workflow
- Kill Switch (STOP.txt file)
- Comprehensive logging and error handling
- Windows Task Scheduler integration
- Production-grade safety checks

Usage:
    python main_orchestrator.py
    
    # For Windows Task Scheduler (9:45 AM EST daily):
    python main_orchestrator.py --auto
"""

import os
import sys
import logging
import argparse
from datetime import datetime, time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.data_manager import DataManager
from src.trading.risk_manager import RiskManager
from src.trading.virtual_engine import VirtualEngine
from src.utils.notifier import EmailNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """
    Main Trading Orchestrator - The Pilot (Shadow Trading Version)
    Coordinates all trading modules with safety and automation
    """
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.kill_switch_path = 'STOP.txt'
        self.log_dir = 'logs'
        self.reports_dir = 'reports'
        
        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize modules
        self.data_manager = None
        self.risk_manager = None
        self.virtual_engine = None
        self.email_notifier = None
        
        logger.info("Trading Orchestrator initialized (Shadow Trading)")
    
    def check_kill_switch(self) -> bool:
        """
        Check if kill switch is activated
        
        Returns:
            True if kill switch is active (should stop), False otherwise
        """
        if os.path.exists(self.kill_switch_path):
            logger.warning("🚨 KILL SWITCH ACTIVATED - STOP.txt file found")
            logger.warning("Trading halted immediately")
            return True
        
        return False
    
    def initialize_modules(self) -> bool:
        """
        Initialize all trading modules
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing trading modules...")
            
            # Initialize Data Manager
            self.data_manager = DataManager()
            logger.info("✅ Data Manager initialized")
            
            # Initialize Risk Manager
            self.risk_manager = RiskManager()
            logger.info("✅ Risk Manager initialized")
            
            # Initialize Virtual Engine
            self.virtual_engine = VirtualEngine()
            logger.info("✅ Virtual Engine initialized")
            
            # Initialize Email Notifier
            self.email_notifier = EmailNotifier()
            logger.info("✅ Email Notifier initialized")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize modules: {e}")
            return False
    
    def get_trading_signals(self, market_data: dict) -> list:
        """
        Generate trading signals (placeholder for ML model)
        
        Args:
            market_data: Dictionary of market data
            
        Returns:
            List of trading signals
        """
        # TODO: Integrate with actual ML model
        # For now, return empty list (no trades)
        
        logger.info("Generating trading signals...")
        
        # Placeholder: No signals for safety
        signals = []
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    def run_trading_session(self) -> dict:
        """
        Run complete trading session
        
        Returns:
            Dictionary with session results
        """
        session_start = datetime.now()
        logger.info("🚀 Starting trading session")
        
        session_results = {
            'start_time': session_start,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': [],
            'success': False
        }
        
        try:
            # Step 1: Check Kill Switch
            if self.check_kill_switch():
                logger.warning("Session aborted by kill switch")
                return session_results
            
            # Step 2: Initialize Modules
            if not self.initialize_modules():
                session_results['errors'].append("Failed to initialize modules")
                return session_results
            
            # Step 3: Update Data
            logger.info("📊 Step 1: Updating market data...")
            
            # Check trading window and daily execution limit
            market_status = self.data_manager.get_market_status()
            
            if not market_status['in_trading_window']:
                logger.info(f"Outside trading window: {market_status['ny_time']} EST / {market_status['israel_time']} IST")
                session_results['success'] = True
                return session_results
            
            if not self.data_manager.check_daily_execution_limit():
                logger.info("Bot already executed today")
                session_results['success'] = True
                return session_results
            
            # Log daily execution
            self.data_manager.log_daily_execution()
            
            # Fetch S&P 100 data with 53 indicators
            market_data = self.data_manager.fetch_daily_data()
            
            if not market_data:
                session_results['errors'].append("No market data available")
                return session_results
            
            # Calculate 53 indicators for all tickers
            logger.info("🧮 Calculating 53 technical indicators...")
            for ticker, df in market_data.items():
                market_data[ticker] = self.data_manager.calculate_53_indicators(df)
            
            logger.info(f"✅ Market data updated: {len(market_data)} tickers with 53 indicators")
            
            logger.info(f"Market status: {'OPEN' if market_status['is_open'] else 'CLOSED'}")
            logger.info(f"Trading window: {market_status['ny_time']} EST / {market_status['israel_time']} IST")
            
            if not market_status['is_open']:
                logger.info("Market is closed - no trading today")
                session_results['success'] = True
                return session_results
            
            # Step 5: Generate Signals
            logger.info("🧠 Step 2: Generating trading signals...")
            signals = self.get_trading_signals(market_data)
            session_results['signals_generated'] = len(signals)
            
            if not signals:
                logger.info("No trading signals generated - session complete")
                session_results['success'] = True
                return session_results
            
            # Step 6: Get Account and Position Info
            logger.info("💰 Step 3: Getting account information...")
            account_info = self.risk_manager.get_account_info()
            current_positions = self.risk_manager.get_current_positions()
            
            logger.info(f"Portfolio value: ${account_info.get('portfolio_value', 0):,.2f}")
            logger.info(f"Current positions: {len(current_positions)}")
            
            # Step 7: Risk Check and Execution
            logger.info("⚖️ Step 4: Risk evaluation and execution...")
            trades_executed = 0
            
            for signal in signals:
                ticker = signal['ticker']
                action = signal['action']  # 'buy' or 'sell'
                
                try:
                    # Get current price
                    current_price = signal.get('price')
                    if not current_price:
                        prices = self.data_manager.get_latest_prices([ticker])
                        current_price = prices.get(ticker)
                    
                    if not current_price:
                        logger.warning(f"No price available for {ticker}")
                        continue
                    
                    # Risk evaluation
                    if action == 'buy':
                        decision, details = self.risk_manager.evaluate_trade(
                            ticker, current_price, account_info, current_positions, market_data
                        )
                        
                        if decision == 'APPROVED':
                            # Execute trade
                            position_size = details['position_size']
                            result = self.execution_manager.execute_order(
                                ticker, 'buy', position_size, current_price, 
                                f"ML Signal: {signal.get('score', 'N/A')}"
                            )
                            
                            if result['status'] == 'SUCCESS':
                                trades_executed += 1
                                logger.info(f"✅ Trade executed: {ticker}")
                            else:
                                logger.error(f"❌ Trade failed: {ticker} - {result.get('error', 'Unknown')}")
                        else:
                            logger.info(f"🚫 Trade rejected: {ticker} - {details.get('reason', 'Unknown')}")
                    
                    elif action == 'sell':
                        # For sell signals, check if we own the position
                        owned_position = None
                        for pos in current_positions:
                            if pos['symbol'] == ticker and pos['side'] == 'long':
                                owned_position = pos
                                break
                        
                        if owned_position:
                            result = self.execution_manager.execute_order(
                                ticker, 'sell', int(owned_position['qty']), current_price,
                                f"ML Sell Signal: {signal.get('score', 'N/A')}"
                            )
                            
                            if result['status'] == 'SUCCESS':
                                trades_executed += 1
                                logger.info(f"✅ Sell executed: {ticker}")
                            else:
                                logger.error(f"❌ Sell failed: {ticker} - {result.get('error', 'Unknown')}")
                        else:
                            logger.info(f"🚫 No position to sell: {ticker}")
                
                except Exception as e:
                    logger.error(f"Error processing signal for {ticker}: {e}")
                    session_results['errors'].append(f"Signal processing error for {ticker}: {str(e)}")
            
            session_results['trades_executed'] = trades_executed
            
            # Step 8: Send Daily Executive Brief
            logger.info("📧 Step 5: Sending Daily Executive Brief...")
            self._send_daily_brief(account_info, current_positions, session_results)
            
            # Step 9: Final Summary
            logger.info("📊 Step 6: Generating session summary...")
            self._log_session_summary(session_start, session_results, account_info, current_positions)
            
            session_results['success'] = True
            logger.info("✅ Trading session completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Trading session failed: {e}")
            session_results['errors'].append(f"Session error: {str(e)}")
        
        finally:
            session_results['end_time'] = datetime.now()
            session_results['duration'] = session_results['end_time'] - session_results['start_time']
        
        return session_results
    
    def _send_daily_brief(self, account_info: Dict, current_positions: List[Dict], session_results: Dict):
        """Send Daily Executive Brief via email"""
        try:
            # Get today's trades from execution manager
            trades_today = []
            if self.execution_manager:
                summary = self.execution_manager.get_shadow_ledger_summary()
                if 'recent_trades' in summary:
                    # Filter trades from today
                    today = datetime.now().strftime('%Y-%m-%d')
                    trades_today = [
                        trade for trade in summary['recent_trades']
                        if trade['timestamp'].startswith(today)
                    ]
            
            # Prepare risk summary
            risk_summary = {
                'risk_per_trade': '0.9%',
                'max_sector_exposure': '30%',
                'black_swan_status': 'Active',
                'duplicate_protection': 'Active'
            }
            
            # Send email
            success = self.email_notifier.send_daily_brief(
                account_info, current_positions, trades_today, risk_summary
            )
            
            if success:
                logger.info("✅ Daily Executive Brief sent successfully")
            else:
                logger.error("❌ Failed to send Daily Executive Brief")
                
        except Exception as e:
            logger.error(f"Error sending daily brief: {e}")
    
    def _log_session_summary(self, start_time: datetime, results: dict, 
                           account_info: dict, current_positions: list):
        """Log comprehensive session summary"""
        logger.info("=" * 60)
        logger.info("📊 TRADING SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Session Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Session Duration: {results.get('duration', 'N/A')}")
        logger.info(f"Signals Generated: {results.get('signals_generated', 0)}")
        logger.info(f"Trades Executed: {results.get('trades_executed', 0)}")
        logger.info(f"Errors: {len(results.get('errors', []))}")
        
        if account_info:
            logger.info(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            logger.info(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        
        logger.info(f"Current Positions: {len(current_positions)}")
        
        # Log current positions
        if current_positions:
            logger.info("Current Holdings:")
            for pos in current_positions:
                logger.info(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos.get('cost_basis', 0):.2f}")
        
        # Log execution summary
        if self.execution_manager:
            self.execution_manager.log_execution_summary()
        
        # Log errors if any
        if results.get('errors'):
            logger.warning("Session Errors:")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        
        logger.info("=" * 60)
    
    def run_automated_session(self):
        """Run automated session (for Task Scheduler)"""
        try:
            # Check if it's the right time (9:45 AM EST ± 15 minutes)
            now = datetime.now()
            target_time = time(9, 45)  # 9:45 AM
            
            # Simple time check (can be enhanced for timezone handling)
            current_time = now.time()
            time_diff = abs((current_time.hour * 60 + current_time.minute) - (target_time.hour * 60 + target_time.minute))
            
            if time_diff > 15:  # More than 15 minutes away
                logger.info(f"Not within trading window (current: {current_time}, target: {target_time})")
                return
            
            logger.info("🤖 Running automated trading session")
            
            # Run the session
            results = self.run_trading_session()
            
            if results['success']:
                logger.info("🎉 Automated session completed successfully")
            else:
                logger.error("❌ Automated session failed")
        
        except Exception as e:
            logger.error(f"❌ Automated session error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NeuralTrader Trading Orchestrator')
    parser.add_argument('--auto', action='store_true', help='Run in automated mode (for Task Scheduler)')
    parser.add_argument('--force', action='store_true', help='Force run regardless of time')
    args = parser.parse_args()
    
    try:
        orchestrator = TradingOrchestrator()
        
        if args.auto:
            orchestrator.run_automated_session()
        else:
            results = orchestrator.run_trading_session()
            
            if results['success']:
                logger.info("🎉 Trading session completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Trading session failed")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Trading session interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
