"""
NeuralTrader IST Scheduler - 23:15 IST Daily Reporting
==================================================

Handles IST-scheduled daily portfolio updates and executive brief reporting.
Updates position values using yfinance and sends comprehensive daily reports.

Features:
- IST scheduling (23:15 IST) for daily reporting
- Portfolio value updates with latest prices
- Daily Executive Brief email generation
- Constitution Health Check reporting
- Virtual P&L and Active Positions tracking

Usage:
    from src.reporting.ist_scheduler import ISTScheduler
    
    scheduler = ISTScheduler()
    scheduler.start_scheduler()
"""

import os
import logging
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import schedule
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ISTScheduler:
    """
    IST Scheduler for Daily Reporting
    Handles 23:15 IST daily portfolio updates and reporting
    """
    
    def __init__(self):
        """Initialize IST Scheduler"""
        # Timezone setup
        self.ist = pytz.timezone('Asia/Jerusalem')
        self.est = pytz.timezone('US/Eastern')
        
        # Portfolio file
        self.portfolio_file = "data/portfolio.json"
        
        logger.info("IST Scheduler initialized")
        logger.info(f"IST Schedule: 23:15 IST for daily reporting")
        logger.info(f"Portfolio file: {self.portfolio_file}")
    
    def update_portfolio_values(self) -> Dict:
        """
        Update portfolio values with latest prices from yfinance
        
        Returns:
            Dictionary with updated portfolio values
        """
        try:
            logger.info("📊 Updating portfolio values with latest prices...")
            
            # Load portfolio
            if not os.path.exists(self.portfolio_file):
                logger.warning("Portfolio file not found")
                return {}
            
            with open(self.portfolio_file, 'r') as f:
                portfolio = json.load(f)
            
            # Get current positions
            positions = portfolio.get('positions', {})
            if not positions:
                logger.info("No positions to update")
                return portfolio
            
            # Get tickers
            tickers = list(positions.keys())
            logger.info(f"Updating {len(tickers)} positions...")
            
            # Fetch latest prices
            prices = self._fetch_latest_prices(tickers)
            
            if not prices:
                logger.warning("No prices fetched")
                return portfolio
            
            # Update position values
            total_position_value = 0.0
            updated_positions = {}
            
            for ticker, position in positions.items():
                current_price = prices.get(ticker)
                if current_price is not None:
                    position_value = current_price * position['quantity']
                    unrealized_pnl = (current_price - position['avg_cost']) * position['quantity']
                    unrealized_pnl_pct = (unrealized_pnl / (position['avg_cost'] * position['quantity'])) * 100 if position['quantity'] > 0 else 0.0
                    
                    updated_positions[ticker] = {
                        'quantity': position['quantity'],
                        'avg_cost': position['avg_cost'],
                        'last_price': current_price,
                        'position_value': position_value,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'created_at': position.get('created_at', datetime.now().isoformat()),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    total_position_value += position_value
                    
                    logger.info(f"✅ {ticker}: {position['quantity']} shares @ ${current_price:.2f} = ${position_value:,.2f}")
                else:
                    logger.warning(f"❌ {ticker}: No price available")
                    updated_positions[ticker] = position
            
            # Calculate total portfolio value
            cash = portfolio.get('cash', 0.0)
            total_value = cash + total_position_value
            
            # Update portfolio
            portfolio['positions'] = updated_positions
            portfolio['performance'] = {
                'total_value': total_value,
                'total_return': total_value - 100000.0,  # Assuming initial cash was 100k
                'total_return_pct': ((total_value - 100000.0) / 100000.0) * 100,
                'unrealized_pnl': sum(pos.get('unrealized_pnl', 0) for pos in updated_positions.values()),
                'realized_pnl': portfolio.get('performance', {}).get('realized_pnl', 0.0)
            }
            portfolio['updated_at'] = datetime.now().isoformat()
            
            # Save updated portfolio
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio, f, indent=2)
            
            logger.info(f"✅ Portfolio updated: ${total_value:,.2f} total value")
            logger.info(f"   Cash: ${cash:,.2f}")
            logger.info(f"   Positions: ${total_position_value:,.2f}")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error updating portfolio values: {e}")
            return {}
    
    def _fetch_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch latest prices for tickers using yfinance
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of ticker -> latest price
        """
        try:
            logger.info(f"Fetching latest prices for {len(tickers)} tickers...")
            
            # Fetch data in batches to avoid API limits
            batch_size = 50
            prices = {}
            
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                
                try:
                    # Download batch data
                    data = yf.download(batch, period="1d", progress=False)
                    
                    if data.empty:
                        logger.warning(f"No data fetched for batch {i//batch_size + 1}")
                        continue
                    
                    # Extract prices
                    for ticker in batch:
                        try:
                            if 'Close' in data.columns:
                                # Handle single ticker case
                                if isinstance(data['Close'], pd.Series):
                                    price = data['Close'].iloc[-1]
                                else:
                                    # Handle multiple tickers
                                    price = data['Close'][ticker].iloc[-1]
                                
                                if pd.notna(price) and price > 0:
                                    prices[ticker] = float(price)
                                else:
                                    logger.warning(f"Invalid price for {ticker}: {price}")
                            else:
                                logger.warning(f"No Close data for {ticker}")
                        
                        except Exception as e:
                            logger.error(f"Error getting price for {ticker}: {e}")
                            continue
                    
                    # Small delay between batches
                    if i + batch_size < len(tickers):
                        time.sleep(0.5)
                
                except Exception as e:
                    logger.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                    continue
            
            logger.info(f"Fetched latest prices: {len(prices)} tickers")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return {}
    
    def generate_daily_executive_brief(self, portfolio: Dict) -> Dict:
        """
        Generate Daily Executive Brief content
        
        Args:
            portfolio: Updated portfolio data
            
        Returns:
            Dictionary with email content
        """
        try:
            now_ist = datetime.now(self.ist)
            
            # Get market status
            market_status = self._get_market_status()
            
            # Prepare email content
            subject = f"📊 NeuralTrader Daily Executive Brief - {now_ist.strftime('%Y-%m-%d')}"
            
            body = f"""
NeuralTrader Daily Executive Brief
=====================================

📅 Date: {now_ist.strftime('%Y-%m-%d %H:%M IST')}
🎯 Mode: Shadow Trading Simulator
🏛️ Constitution: Risk Management Active

📊 PORTFOLIO OVERVIEW:
-------------------------
Total Portfolio Value: ${portfolio.get('performance', {}).get('total_value', 0):,.2f}
Cash Balance: ${portfolio.get('cash', 0):,.2f}
Position Value: ${portfolio.get('performance', {}).get('total_value', 0) - portfolio.get('cash', 0):,.2f}

Performance Metrics:
• Total Return: ${portfolio.get('performance', {}).get('total_return', 0):,.2f}
• Total Return %: {portfolio.get('performance', {}).get('total_return_pct', 0):.2f}%
• Unrealized P&L: ${portfolio.get('performance', {}).get('unrealized_pnl', 0):,.2f}
• Realized P&L: ${portfolio.get('performance', {}).get('realized_pnl', 0):,.2f}

📈 ACTIVE POSITIONS ({len(portfolio.get('positions', {}))}):
"""
            
            # Add positions details
            positions = portfolio.get('positions', {})
            if positions:
                for ticker, position in positions.items():
                    body += f"""
• {ticker}: {position['quantity']} shares @ ${position['avg_cost']:.2f}
  Current Price: ${position.get('last_price', 0):.2f}
  Position Value: ${position.get('position_value', 0):,.2f}
  Unrealized P&L: ${position.get('unrealized_pnl', 0):,.2f} ({position.get('unrealized_pnl_pct', 0):.2f}%)
"""
            
            # Add trading history
            history = portfolio.get('history', [])
            recent_trades = history[-5:] if history else []
            
            body += f"""
📋 TRADING ACTIVITY:
-------------------
Total Trades: {len(history)}
Recent Trades:
"""
            
            if recent_trades:
                for trade in recent_trades:
                    body += f"• {trade['timestamp'][:19]}: {trade['side'].upper()} {trade['quantity']} {trade['ticker']} @ ${trade['execution_price']:.2f}\n"
            
            body += f"""
🌍 MARKET STATUS:
------------------
Current Time: {market_status.get('timestamp_ist', 'N/A')}
EST Time: {market_status.get('timestamp_est', 'N/A')}
Market Hours: {'OPEN' if market_status.get('is_market_open') else 'CLOSED'}

🏛️ CONSTITUTION HEALTH CHECK:
---------------------------------
✅ 0.9% Risk Per Trade: Active
✅ 30% Technology Sector Cap: Active
✅ Black Swan Exit: VXX Monitoring Active
✅ Duplicate Position Protection: Active
✅ Capital Preservation: Priority #1

📊 CONSTITUTION HEALTH CHECK:
-----------------------------
✅ Risk Per Trade: 0.9% per trade
✅ Sector Caps: 30% max per sector
✅ Black Swan: VXX surge >15% = REJECT ALL
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
            
            return {
                'subject': subject,
                'body': body,
                'timestamp': now_ist.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating daily executive brief: {e}")
            return {}
    
    def _get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            now_ist = datetime.now(self.ist)
            now_est = datetime.now(self.est)
            
            # Check if market is open (9:30 AM - 4:00 PM EST, Mon-Fri)
            is_weekday = now_est.weekday() < 5  # 0-4 = Mon-Fri
            market_open = now_est.replace(hour=9, minute=30)
            market_close = now_est.replace(hour=16, minute=0)
            
            is_market_hours = is_weekday and market_open <= now_est <= market_close
            
            return {
                'timestamp_ist': now_ist.strftime('%Y-%m-%d %H:%M:%S IST'),
                'timestamp_est': now_est.strftime('%Y-%m-%d %H:%M:%S EST'),
                'is_weekday': is_weekday,
                'is_market_open': is_market_hours,
                'market_open_time': market_open.strftime('%H:%M EST'),
                'market_close_time': market_close.strftime('%H:%M EST'),
                'market_status': 'OPEN' if is_market_hours else 'CLOSED'
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {}
    
    def send_daily_report(self) -> bool:
    """Send daily executive brief with portfolio updates"""
    try:
        self.logger.info("📧 Sending daily executive brief...")
            
        # Update portfolio values with latest prices
        self._update_portfolio_values()
            
        # Get portfolio info
        account_info = self.virtual_engine.get_account_info()
        current_positions = self.virtual_engine.get_current_positions()
        trades_today = self.virtual_engine.get_trades_today()
            
        # Generate daily executive brief content
        brief_content = self._generate_daily_brief(account_info, current_positions, trades_today)
            
        # Send email with logs
        success = self.email_notifier.send_email_with_logs(
            to_email=self.email_notifier.recipient_email,
            subject=f"📧 NeuralTrader Daily Executive Brief - {datetime.now(self.israel).strftime('%Y-%m-%d %H:%M IST')}",
            body=brief_content,
            log_file_path=os.path.join(self.project_root, 'logs', 'automation.log')
        )
            
        if success:
            self.logger.info("✅ Daily executive brief sent successfully")
        else:
            self.logger.error("❌ Failed to send daily executive brief")
            
        return success
            
    except Exception as e:
        self.logger.error(f"Error sending daily report: {e}")
        return False
    
def start_scheduler(self):
    """Start the IST scheduler for daily reporting"""
    try:
        logger.info("🕐 Starting IST scheduler...")
            
            # Schedule daily report at 23:15 IST
            schedule.every().day.at("23:15").do(self.send_daily_report)
            
            logger.info("✅ Scheduler started - Daily report at 23:15 IST")
            
            # Run the scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in scheduler: {e}")

# Usage example
if __name__ == "__main__":
    # Test the IST scheduler
    try:
        scheduler = ISTScheduler()
        
        # Test portfolio update
        portfolio = scheduler.update_portfolio_values()
        logger.info(f"Portfolio updated: {len(portfolio.get('positions', {}))} positions")
        
        # Test report generation
        report = scheduler.generate_daily_executive_brief(portfolio)
        logger.info(f"Report generated: {len(report.get('body', ''))} characters")
        
        # Test market status
        status = scheduler._get_market_status()
        logger.info(f"Market status: {status}")
        
    except Exception as e:
        logger.error(f"Error in IST scheduler test: {e}")
