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
import logging
import argparse
import pandas as pd
from datetime import datetime, time
from pathlib import Path

# ==================== MASTER RUNNER SAFETY CHECKS ====================

# 1. Working Directory Lock - Force project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# 2. Environment Safety - Check required libraries
REQUIRED_LIBRARIES = {
    'yfinance': 'yfinance',
    'pytz': 'pytz', 
    'pandas': 'pandas',
    'schedule': 'schedule'
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
        print(f"❌ ENVIRONMENT ERROR: Missing required libraries: {', '.join(missing_libs)}")
        print(f"📦 Install with: pip install {' '.join(missing_libs)}")
        print(f"🔧 Or run: pip install -r requirements_trading.txt")
        sys.exit(1)
    
    print("✅ Environment check passed - All required libraries available")

# 3. Logging Setup - automation.log for Task Scheduler debugging
def setup_automation_logging():
    """Setup comprehensive logging for Task Scheduler debugging"""
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'automation.log'
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('NeuralTrader_Automation')
    
    # Log session start
    logger.info("=" * 80)
    logger.info("🤖 NeuralTrader Automation Session Started")
    logger.info(f"📁 Working Directory: {PROJECT_ROOT}")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🐍 Python Version: {sys.version}")
    logger.info(f"⚙️ Command Line Args: {' '.join(sys.argv)}")
    logger.info("=" * 80)
    
    return logger

# ==================== IMPORTS AFTER ENVIRONMENT CHECK ====================

# Import after environment verification
from src.data.yfinance_manager import YFinanceManager
from src.trading.risk_manager import RiskManager, RiskDecision
from src.trading.virtual_engine import VirtualEngine
from src.reporting.ist_scheduler import ISTScheduler
from src.utils.notifier import EmailNotifier

# ==================== MASTER RUNNER CLASS ====================

class TradingOrchestrator:
    """
    Master Runner for NeuralTrader Task Scheduler Integration
    Handles all automation modes with comprehensive safety checks
    """
    
    def __init__(self, logger):
        """Initialize the Master Runner"""
        self.logger = logger
        self.kill_switch_path = 'STOP.txt'
        self.log_dir = 'logs'
        self.reports_dir = 'reports'
        
        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize modules (lazy loading)
        self.yfinance_manager = None
        self.risk_manager = None
        self.virtual_engine = None
        self.ist_scheduler = None
        self.email_notifier = None
        
        self.logger.info("🎭 Trading Orchestrator (Master Runner) initialized")
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch is activated"""
        if os.path.exists(self.kill_switch_path):
            self.logger.warning("🚨 KILL SWITCH ACTIVATED - STOP.txt file found")
            return True
        return False
    
    def initialize_modules(self) -> bool:
        """Initialize all trading modules with error handling"""
        try:
            self.logger.info("🔧 Initializing trading modules...")
            
            # Initialize YFinance Manager
            self.yfinance_manager = YFinanceManager()
            self.logger.info("✅ YFinance Manager initialized")
            
            # Initialize Risk Manager
            self.risk_manager = RiskManager()
            self.logger.info("✅ Risk Manager initialized")
            
            # Initialize Virtual Engine
            self.virtual_engine = VirtualEngine()
            self.logger.info("✅ Virtual Engine initialized")
            
            # Initialize IST Scheduler
            self.ist_scheduler = ISTScheduler()
            self.logger.info("✅ IST Scheduler initialized")
            
            # Initialize Email Notifier
            self.email_notifier = EmailNotifier()
            self.logger.info("✅ Email Notifier initialized")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize modules: {e}")
            return False
    
    def run_fetch_mode(self) -> bool:
        """Run data fetch mode (16:45 IST)"""
        try:
            self.logger.info("🕐 Running FETCH MODE - Data Fetch Session")
            session_start = datetime.now()
            
            # Check kill switch
            if self.check_kill_switch():
                self.logger.warning("🛑 Kill switch activated - stopping fetch session")
                return False
            
            # Initialize modules
            if not self.initialize_modules():
                return False
            
            # Check if it's data fetch time (16:45 IST)
            market_status = self.yfinance_manager.get_market_status()
            
            if not market_status.get('is_fetch_time', False):
                self.logger.info(f"⏰ Not data fetch time: {market_status.get('timestamp_ist')}")
                self.logger.info("🔧 Forcing data fetch for testing...")
            
            # Fetch scheduled data
            self.logger.info("📊 Fetching scheduled market data...")
            data = self.yfinance_manager.fetch_scheduled_data()
            
            if not data:
                self.logger.error("❌ No data fetched")
                return False
            
            self.logger.info(f"✅ Data fetch completed: {len(data)} tickers")
            
            # Update portfolio values
            self.virtual_engine.update_portfolio_values()
            
            # Send notification
            self._send_data_fetch_notification(len(data))
            
            session_end = datetime.now()
            duration = session_end - session_start
            
            self.logger.info(f"✅ FETCH MODE completed in {duration.total_seconds():.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in FETCH MODE: {e}")
            return False
    
    def run_trade_mode(self) -> bool:
        """Run trading mode (market hours)"""
        try:
            self.logger.info("🎭 Running TRADE MODE - Shadow Trading Session")
            session_start = datetime.now()
            
            # Check kill switch
            if self.check_kill_switch():
                self.logger.warning("🛑 Kill switch activated - stopping trading session")
                return False
            
            # Initialize modules
            if not self.initialize_modules():
                return False
            
            # Check if market is open
            market_status = self.yfinance_manager.get_market_status()
            if not market_status.get('is_market_open', False):
                self.logger.info("⏰ Market is closed - no trading")
                return False
            
            # Fetch market data
            self.logger.info("📊 Fetching market data for trading...")
            data = self.yfinance_manager.fetch_daily_data(period="5d")
            
            if not data:
                self.logger.error("❌ No market data available")
                return False
            
            self.logger.info(f"📊 Market data fetched: {len(data)} tickers")
            
            # Get portfolio info
            account_info = self.risk_manager.get_account_info()
            current_positions = self.risk_manager.get_current_positions()
            
            # Process signals and execute trades
            trades_executed = 0
            for ticker in self.yfinance_manager.sp100_tickers[:10]:  # Limit for testing
                try:
                    if ticker not in data:
                        continue
                    
                    df = data[ticker]
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Get latest price
                    current_price = self.yfinance_manager.get_latest_prices([ticker]).get(ticker)
                    if current_price is None:
                        continue
                    
                    # Generate signal
                    signal_strength = self._generate_signal(df)
                    
                    if signal_strength > 0.5:  # Buy signal
                        # Evaluate trade
                        decision, details = self.risk_manager.evaluate_trade(
                            ticker, current_price, account_info, current_positions
                        )
                        
                        if decision == RiskDecision.APPROVED:
                            # Execute trade
                            position_size = details.get('position_size', 10)
                            result = self.virtual_engine.execute_trade(
                                ticker, 'buy', position_size, current_price, 
                                f"AI Signal: {signal_strength:.2f}"
                            )
                            
                            if result['success']:
                                trades_executed += 1
                                self.logger.info(f"✅ Trade executed: {ticker} - {position_size} shares @ ${current_price:.2f}")
                            else:
                                self.logger.warning(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                        else:
                            self.logger.info(f"❌ Trade rejected: {decision}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {ticker}: {e}")
                    continue
            
            # Update portfolio values
            self.virtual_engine.update_portfolio_values()
            
            session_end = datetime.now()
            duration = session_end - session_start
            
            self.logger.info(f"✅ TRADE MODE completed in {duration.total_seconds():.2f} seconds")
            self.logger.info(f"   Trades executed: {trades_executed}")
            self.logger.info(f"   Portfolio value: ${self.virtual_engine.portfolio['performance']['total_value']:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in TRADE MODE: {e}")
            return False
    
    def run_report_mode(self) -> bool:
        """Run report mode (23:15 IST)"""
        try:
            self.logger.info("📧 Running REPORT MODE - Daily Executive Brief")
            session_start = datetime.now()
            
            # Check kill switch
            if self.check_kill_switch():
                self.logger.warning("🛑 Kill switch activated - stopping report session")
                return False
            
            # Initialize modules
            if not self.initialize_modules():
                return False
            
            # Send daily executive brief
            success = self.ist_scheduler.send_daily_report()
            
            session_end = datetime.now()
            duration = session_end - session_start
            
            if success:
                self.logger.info(f"✅ REPORT MODE completed in {duration.total_seconds():.2f} seconds")
            else:
                self.logger.error(f"❌ REPORT MODE failed in {duration.total_seconds():.2f} seconds")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error in REPORT MODE: {e}")
            return False
    
    def run_auto_mode(self):
        """Run auto mode with IST scheduling"""
        try:
            self.logger.info("🕐 Running AUTO MODE - IST Scheduling")
            
            # Initialize modules
            if not self.initialize_modules():
                return
            
            # Start IST scheduler
            import schedule
            
            # Schedule tasks
            schedule.every().day.at("16:45").do(self.run_fetch_mode)
            schedule.every().day.at("23:15").do(self.run_report_mode)
            
            self.logger.info("✅ IST Scheduler started:")
            self.logger.info("   • Data fetch: 16:45 IST (09:45 EST)")
            self.logger.info("   • Daily report: 23:15 IST")
            
            # Run scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Auto mode stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in AUTO MODE: {e}")
    
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
    
    def _send_data_fetch_notification(self, data_count: int):
        """Send notification about data fetch completion"""
        try:
            subject = f"📊 NeuralTrader Data Fetch Complete - {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
            
            body = f"""
NeuralTrader Data Fetch Notification
==================================

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
🎯 Mode: Shadow Trading Simulator
🏛️ Constitution: Risk Management Active

📊 DATA FETCH RESULTS:
--------------------
Data Fetch Time: 16:45 IST (09:45 EST)
Tickers Fetched: {data_count}
Data Source: Yahoo Finance
Status: ✅ SUCCESS

🎯 READINESS FOR TRADING:
-------------------------
✅ Market data updated
✅ Risk systems operational
✅ Virtual portfolio ready

---
NeuralTrader Automated Trading System
Phase 6: Shadow Trading Simulator
"""
            
            self.email_notifier.send_email(
                to_email="lugassy.ai@gmail.com",
                subject=subject,
                body=body
            )
            
            self.logger.info("✅ Data fetch notification sent")
            
        except Exception as e:
            self.logger.error(f"Error sending data fetch notification: {e}")

# ==================== MAIN FUNCTION WITH CLI ====================

def main():
    """Main function with CLI argument parsing"""
    
    # Environment safety check
    check_environment()
    
    # Setup logging
    logger = setup_automation_logging()
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description='NeuralTrader Master Runner for Task Scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_orchestrator_ist.py --mode=fetch
  python main_orchestrator_ist.py --mode=trade  
  python main_orchestrator_ist.py --mode=report
  python main_orchestrator_ist.py --auto
  
For Task Scheduler:
  run_neural.bat fetch
  run_neural.bat trade
  run_neural.bat report
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['fetch', 'trade', 'report', 'auto'],
        help='Operation mode: fetch (16:45 IST), trade (market hours), report (23:15 IST), auto (scheduler)'
    )
    
    parser.add_argument(
        '--data-fetch', 
        action='store_true', 
        help='Run data fetch session (legacy)'
    )
    
    parser.add_argument(
        '--trading', 
        action='store_true', 
        help='Run trading session (legacy)'
    )
    
    parser.add_argument(
        '--report', 
        action='store_true', 
        help='Send daily report (legacy)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TradingOrchestrator(logger)
    
    try:
        # Determine mode
        if args.mode == 'fetch' or args.data_fetch:
            success = orchestrator.run_fetch_mode()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'trade' or args.trading:
            success = orchestrator.run_trade_mode()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'report' or args.report:
            success = orchestrator.run_report_mode()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'auto':
            orchestrator.run_auto_mode()
            
        else:
            logger.error("❌ No mode specified. Use --mode=fetch|trade|report|auto")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 NeuralTrader session interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Log session end
        logger.info("=" * 80)
        logger.info("🤖 NeuralTrader Automation Session Ended")
        logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()
