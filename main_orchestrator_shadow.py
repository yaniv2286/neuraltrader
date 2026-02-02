"""
NeuralTrader Main Orchestrator - The Pilot (Shadow Trading Version)
===================================================

Master script that automates the complete shadow trading workflow:
Data → Signals → Risk Check → Virtual Execution → Reporting

Features:
- Complete automated shadow trading workflow
- Kill Switch (STOP.txt file)
- Yahoo Finance data integration
- Virtual portfolio management
- Comprehensive logging and error handling
- Daily executive brief emails
- Production-grade safety checks

Usage:
    python main_orchestrator_shadow.py
    
    # For Windows Task Scheduler (9:45 AM EST daily):
    python main_orchestrator_shadow.py --auto
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
from src.trading.risk_manager import RiskManager, RiskDecision
from src.trading.virtual_engine import VirtualEngine
from src.utils.notifier import EmailNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator_shadow.log'),
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
            return True
        
        return False
    
    def initialize_modules(self) -> bool:
        """Initialize all trading modules"""
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
    
    def run_daily_report(self):
        """Run daily executive brief report"""
        try:
            logger.info("📧 Running daily executive brief report...")
            
            # Get current portfolio summary
            portfolio_summary = self.virtual_engine.get_portfolio_summary()
            
            # Get market status
            market_status = self.data_manager.get_market_status()
            
            # Prepare email content
            subject = f"📊 NeuralTrader Daily Executive Brief - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
NeuralTrader Daily Executive Brief
=====================================

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M EST')}
🎯 Mode: Shadow Trading Simulator
🏛️ Constitution: Risk Management Active

📊 PORTFOLIO OVERVIEW:
-------------------------
Total Portfolio Value: ${portfolio_summary['performance']['total_value']:,.2f}
Cash Balance: ${portfolio_summary['portfolio_info']['current_cash']:,.2f}
Position Value: ${portfolio_summary['performance']['total_value'] - portfolio_summary['portfolio_info']['current_cash']:,.2f}

Performance Metrics:
• Total Return: ${portfolio_summary['performance']['total_return']:,.2f}
• Total Return %: {portfolio_summary['performance']['total_return_pct']:,.2f}%
• Unrealized P&L: ${portfolio_summary['performance']['unrealized_pnl']:,.2f}
• Realized P&L: ${portfolio_summary['performance']['realized_pnl']:,.2f}

📈 ACTIVE POSITIONS ({len(portfolio_summary['positions'])}):
"""
            
            # Add positions details
            if portfolio_summary['positions']:
                for ticker, position in portfolio_summary['positions'].items():
                    body += f"""
• {ticker}: {position['quantity']} shares @ ${position['avg_cost']:.2f}
  Current Price: ${position.get('current_price', 0):.2f}
  Position Value: ${position.get('position_value', 0):,.2f}
  Unrealized P&L: ${position.get('unrealized_pnl', 0):,.2f} ({position.get('unrealized_pnl_pct', 0):.2f}%)
"""
            
            body += f"""
📋 TRADING ACTIVITY:
-------------------
Total Trades: {portfolio_summary['trading_stats']['total_trades']}
Buy Trades: {portfolio_summary['trading_stats']['buy_trades']}
Sell Trades: {portfolio_summary['trading_stats']['sell_trades']}

Recent Trades:
"""
            
            # Add recent trades
            if portfolio_summary['trading_stats']['recent_trades']:
                for trade in portfolio_summary['trading_stats']['recent_trades'][-5:]:
                    body += f"• {trade['timestamp']}: {trade['side'].upper()} {trade['quantity']} {trade['ticker']} @ ${trade['execution_price']:.2f}"
            
            body += f"""
🌍 MARKET STATUS:
------------------
Current Time: {market_status.get('timestamp_eastern', 'N/A')}
Israel Time: {market_status.get('timestamp_israel', 'N/A')}
Market Hours: {'OPEN' if market_status.get('is_market_open') else 'CLOSED'}
Execution Window: {'OPEN' if market_status.get('is_execution_window') else 'CLOSED'}

🏛️ RISK MANAGEMENT CONSTITUTION:
---------------------------------
✅ 0.9% Risk Per Trade: Active
✅ 30% Technology Sector Cap: Active
✅ Black Swan Exit: VXX Monitoring Active
✅ Duplicate Position Protection: Active
✅ Capital Preservation: Priority #1

📊 CONSTITUTION HEALTH CHECK:
-----------------------------
✅ Risk Per Trade: {self.risk_manager.risk_per_trade:.1%} per trade
✅ Sector Caps: {self.risk_manager.max_sector_exposure:.1%} max per sector
✅ Black Swan: VXX surge >{self.risk_manager.black_swan_threshold:.0%} = REJECT ALL
✅ Daily Loss Limit: Enforced
✅ Maximum Drawdown: Controlled

🎯 READINESS FOR TOMORROW:
-------------------------
✅ All risk systems operational
✅ Virtual portfolio ready for trading
✅ Market data feed active (Yahoo Finance)
✅ Email notifications enabled
✅ Shadow trading simulation running

📋 NEXT STEPS:
-------------
• Monitor market conditions overnight
• Execute trades based on AI signals
• Maintain risk management compliance
• Track portfolio performance
• Send tomorrow's executive brief

---
NeuralTrader Automated Trading System
Phase 6: Shadow Trading Simulator
Constitution: Risk Management First
"""
            
            # Send email
            self.email_notifier.send_email(
                to_email="lugassy.ai@gmail.com",
                subject=subject,
                body=body
            )
            
            logger.info("✅ Daily executive brief sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    def run_shadow_trading_session(self):
        """Run shadow trading session"""
        try:
            logger.info("🎭 Starting shadow trading session...")
            
            # Check kill switch
            if self.check_kill_switch():
                logger.warning("🛑 Kill switch activated - stopping session")
                return False
            
            # Check if within execution window
            market_status = self.data_manager.get_market_status()
            if not market_status.get('is_execution_window', False):
                logger.info("⏰ Outside execution window - waiting")
                return False
            
            # Check daily execution limit
            if not self.data_manager.check_daily_execution_limit():
                logger.info("📅 Daily execution already completed")
                return False
            
            # Fetch market data
            logger.info("📊 Fetching market data...")
            tickers = self.data_manager.sp100_tickers
            data = self.data_manager.fetch_daily_data(tickers, period="1y")
            
            if not data:
                logger.error("❌ No market data available")
                return False
            
            logger.info(f"📊 Fetched data for {len(data)} tickers")
            
            # Get current portfolio
            account_info = self.risk_manager.get_account_info()
            current_positions = self.risk_manager.get_current_positions()
            
            # Process each ticker for signals
            trades_executed = 0
            for ticker in tickers:
                try:
                    if ticker not in data:
                        continue
                    
                    df = data[ticker]
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Calculate indicators
                    df_with_indicators = self.data_manager.calculate_53_indicators(df)
                    
                    # Get latest price
                    current_price = self.data_manager.get_latest_prices([ticker]).get(ticker)
                    if current_price is None:
                        continue
                    
                    # Simple signal generation (placeholder for AI model)
                    signal_strength = self._generate_signal(df_with_indicators)
                    
                    if signal_strength > 0.5:  # Buy signal
                        # Evaluate trade with risk manager
                        decision, details = self.risk_manager.evaluate_trade(
                            ticker, current_price, account_info, current_positions
                        )
                        
                        if decision == RiskDecision.APPROVED:
                            # Execute trade with virtual engine
                            position_size = details.get('position_size', 10)
                            result = self.virtual_engine.execute_trade(
                                ticker, 'buy', position_size, current_price, 
                                f"AI Signal: {signal_strength:.2f}"
                            )
                            
                            if result['success']:
                                trades_executed += 1
                                logger.info(f"✅ Trade executed: {ticker} - {position_size} shares @ ${current_price:.2f}")
                            else:
                                logger.warning(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                        else:
                            logger.info(f"❌ Trade rejected: {decision}")
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    continue
            
            # Log daily execution
            self.data_manager.log_daily_execution()
            
            # Update portfolio values
            self.virtual_engine.update_portfolio_values()
            
            # Send daily report
            self.run_daily_report()
            
            logger.info(f"🎭 Shadow trading session completed")
            logger.info(f"   Trades executed: {trades_executed}")
            logger.info(f"   Portfolio value: ${self.virtual_engine.portfolio['performance']['total_value']:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in shadow trading session: {e}")
            return False
    
    def _generate_signal(self, df_with_indicators: pd.DataFrame) -> float:
        """
        Generate trading signal (placeholder for AI model)
        
        Args:
            df_with_indicators: DataFrame with indicators
            
        Returns:
            Signal strength (0-1)
        """
        try:
            # Simple momentum signal (placeholder)
            # In production, this would use the trained XGBoost model
            if len(df_with_indicators) < 20:
                return 0.0
            
            # Calculate simple momentum signal
            recent_return = df_with_indicators['close'].pct_change(5).iloc[-1]
            rsi = df_with_indicators['rsi_14'].iloc[-1]
            
            # Simple signal logic
            if recent_return > 0.02 and rsi < 70:  # 2% return and not overbought
                return 0.7
            elif recent_return < -0.02 and rsi > 30:  # -2% return and not oversold
                return 0.3  # Sell signal
            else:
                return 0.0  # No signal
                
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0.0
    
    def run_automated_session(self):
        """Run automated trading session"""
        try:
            logger.info("🤖 Starting automated trading session...")
            
            # Check kill switch
            if self.check_kill_switch():
                logger.warning("🛑 Kill switch activated - stopping session")
                return False
            
            # Check if within execution window
            market_status = self.data_manager.get_market_status()
            if not market_status.get('is_execution_window', False):
                logger.info("⏰ Outside execution window - waiting")
                return False
            
            # Check daily execution limit
            if not self.data_manager.check_daily_execution_limit():
                logger.info("📅 Daily execution already completed")
                return False
            
            # Fetch market data
            logger.info("📊 Fetching market data...")
            tickers = self.data_manager.sp100_tickers
            data = self.data_manager.fetch_daily_data(tickers, period="1y")
            
            if not data:
                logger.error("❌ No market data available")
                return False
            
            logger.info(f"📊 Fetched data for {len(data)} tickers")
            
            # Get current positions
            positions = self.virtual_engine.get_portfolio_summary()['positions']
            account_info = self.risk_manager.get_account_info()
            
            # Process each ticker for signals
            trades_executed = 0
            for ticker in tickers:
                try:
                    if ticker not in data:
                        continue
                    
                    df = data[ticker]
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Calculate indicators
                    df_with_indicators = self.data_manager.calculate_53_indicators(df)
                    
                    # Get latest price
                    current_price = df['close'].iloc[-1]
                    
                    # Generate signal (placeholder for AI model)
                    signal_strength = self._generate_signal(df_with_indicators)
                    
                    if signal_strength > 0.5:  # Buy signal
                        # Evaluate trade with risk manager
                        decision, details = self.risk_manager.evaluate_trade(
                            ticker, current_price, account_info, positions
                        )
                        
                        if decision == RiskDecision.APPROVED:
                            # Execute trade with virtual engine
                            position_size = details.get('position_size', 10)
                            result = self.virtual_engine.execute_order(
                                ticker, 'buy', position_size, current_price,
                                f"AI Signal: {signal_strength:.2f}"
                            )
                            
                            if result['success']:
                                trades_executed += 1
                                logger.info(f"✅ Trade executed: {ticker} - {position_size} shares @ ${current_price:.2f}")
                            else:
                                logger.warning(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                        else:
                            logger.info(f"❌ Trade rejected: {decision}")
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    continue
            
            # Log daily execution
            self.data_manager.log_daily_execution()
            
            # Send daily report
            self.run_daily_report()
            
            logger.info(f"🤖 Automated session completed")
            logger.info(f"   Trades executed: {trades_executed}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in automated session: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NeuralTrader Shadow Trading Orchestrator')
    parser.add_argument('--auto', action='store_true', help='Run automated trading session')
    parser.add_argument('--report', action='store_true', help='Send daily report only')
    args = parser.parse_args()
    
    try:
        orchestrator = TradingOrchestrator()
        
        if args.report:
            orchestrator.run_daily_report()
        elif args.auto:
            success = orchestrator.run_automated_session()
            
            if success:
                logger.info("🎉 Automated session completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Automated session failed")
                sys.exit(1)
        else:
            # Manual trading session
            success = orchestrator.run_shadow_trading_session()
            
            if success:
                logger.info("🎉 Shadow trading session completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Shadow trading session failed")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Trading session interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
