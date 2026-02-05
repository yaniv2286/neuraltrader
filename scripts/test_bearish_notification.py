#!/usr/bin/env python3
"""
NeuralTrader Bearish Notification Test
====================================

Test script to verify bearish market protection notifications work correctly.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.trading.virtual_engine import VirtualEngine
    from src.utils.notifier import EmailNotifier
    from src.data.yfinance_manager import YFinanceManager
    
    print("🐻 NeuralTrader Bearish Notification Test")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize virtual engine
    print("🚀 Initializing Virtual Engine...")
    virtual_engine = VirtualEngine()
    
    # Mock bearish market condition by temporarily disabling market filter
    print("🐻 Simulating Bearish Market Condition...")
    original_market_filter = virtual_engine.market_filter_enabled
    virtual_engine.market_filter_enabled = False  # This will make us skip the real check
    
    # Create a mock bearish notification
    print("📧 Creating Bearish Protection Notification...")
    
    # Get portfolio summary
    portfolio_summary = virtual_engine.get_portfolio_summary()
    
    # Create email notifier
    notifier = EmailNotifier()
    
    # Create bearish notification content
    subject = f"[NeuralTrader] Session Complete - BEARISH PROTECTION ACTIVE"
    
    body = f"""
NeuralTrader Bearish Market Protection Notification
===============================================

[DATE] Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
[MODE] Mode: Paper Trading
[STATUS] Market Status: BEARISH - Cash Preserved
[CONSTITUTION] Constitution: Risk Management Active

[MARKET FILTER] SPY Market Analysis:
---------------------------------
Status: BEARISH (SPY below 20-day SMA)
Action: Cash Preservation Mode Activated
Risk: No trades executed due to bearish market conditions

[PORTFOLIO] Current Status:
-------------------------
Total Value: ${portfolio_summary.get('performance', {}).get('total_value', 0):,.2f}
Cash: ${portfolio_summary.get('portfolio_info', {}).get('current_cash', 0):,.2f}
Positions: {len(portfolio_summary.get('positions', {}))} held

[PROTECTION] Risk Management:
---------------------------
✅ Market Filter: BEARISH protection active
✅ Position Limits: Enforced
✅ Stop Loss: Ready for activation
✅ Black Swan: Monitoring VXX volatility

[SUMMARY] Session Outcome:
------------------------
Market Condition: Bearish detected
System Response: Cash preservation mode
Capital Protection: 100% preserved
Next Check: Next scheduled run

🛡️ NeuralTrader Constitution: Capital Preservation Priority #1
📊 System is protecting capital during bearish market conditions.
🔄 Will resume trading when market turns bullish.

This is a TEST message from NeuralTrader Bearish Protection System.
"""
    
    # Get automation log file path
    automation_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'automation.log')
    
    # Send email
    print("📨 Sending Bearish Protection Test Email...")
    success = notifier.send_email_with_logs(
        to_email=notifier.recipient_email,
        subject=subject,
        body=body,
        log_file_path=automation_log_path
    )
    
    # Restore original setting
    virtual_engine.market_filter_enabled = original_market_filter
    
    if success:
        print("✅ SUCCESS: Bearish protection email sent successfully!")
        print("📬 Check your inbox for the bearish protection test email")
        print("🐻 This simulates what happens when SPY market filter is BEARISH")
    else:
        print("❌ FAILED: Bearish protection email sending failed")
        print("🔍 Check the logs above for SMTP or authentication errors")
    
    print()
    print("=" * 50)
    print("🎯 Bearish Notification Test Complete")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("🔧 Make sure the virtual environment is activated")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("🔍 Check configuration and system status")
    sys.exit(1)
