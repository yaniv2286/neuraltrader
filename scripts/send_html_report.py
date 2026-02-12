#!/usr/bin/env python3
"""
NeuralTrader HTML Report Sender
Generates HTML dashboard and sends it via email with log attachment
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '_LEGACY_VAULT' / '_archive_src'))

from scripts.report_generator import HTMLDashboardGenerator
from utils.notifier import EmailNotifier

def send_html_dashboard_report():
    """Generate HTML dashboard and send via email with log attachment"""
    
    print("🤖 NeuralTrader HTML Report Generation")
    print("=" * 50)
    
    # Generate HTML dashboard
    print("📊 Generating HTML dashboard...")
    generator = HTMLDashboardGenerator()
    
    # Sample market activity data (in real implementation, this would come from trading system)
    buy_signals = ["AAPL - Strong momentum detected", "MSFT - Breakout pattern"]
    sell_signals = ["TSLA - Overbought condition"]
    hold_signals = ["GOOGL - Consolidation phase", "AMZN - Neutral trend"]
    trades_executed = [
        {"ticker": "AAPL", "action": "buy", "quantity": 10, "price": 176.50},
        {"ticker": "MSFT", "action": "buy", "quantity": 5, "price": 425.30}
    ]
    
    dashboard_file = generator.generate_dashboard(
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        hold_signals=hold_signals,
        trades_executed=trades_executed
    )
    
    print(f"✅ Dashboard generated: {dashboard_file}")
    
    # Get today's log file
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = project_root / "logs" / f"NeuralTrader_{today}.log"
    
    # Initialize email notifier
    print("📧 Initializing email notifier...")
    try:
        notifier = EmailNotifier()
        print("✅ Email notifier initialized")
    except Exception as e:
        print(f"❌ Failed to initialize email notifier: {e}")
        return False
    
    # Send email with HTML dashboard and log attachment
    print("📤 Sending HTML dashboard report...")
    
    subject = f"[NEURAL] HTML Dashboard Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Read HTML content
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Create email body
    body = f"""
NeuralTrader HTML Dashboard Report
=====================================

Please find the HTML dashboard attached to this email.

Dashboard Features:
- Executive Summary with portfolio metrics
- Current holdings with gain/loss analysis
- Market activity signals and trades executed
- Professional styling with responsive design

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
NeuralTrader Automated Trading System v5.0 - Sector Authority Active

---
Note: This is an HTML dashboard report. For best viewing experience, 
save the attachment and open it in a web browser.
"""
    
    try:
        # Send HTML email with dashboard attachment
        success = notifier.send_html_email_with_attachment(
            subject=subject,
            html_content=html_content,
            attachment_path=dashboard_file
        )
        
        if success:
            print("✅ HTML dashboard report sent successfully")
            
            # Also send with log file if available
            if log_file.exists():
                print(f"📎 Sending with log attachment: {log_file}")
                success_with_log = notifier.send_email_with_logs(
                    to_email=notifier.recipient_email,
                    subject=f"[NEURAL] Dashboard + Logs - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    body=body,
                    log_file_path=str(log_file)
                )
                
                if success_with_log:
                    print("✅ Dashboard + logs sent successfully")
                else:
                    print("⚠️ Dashboard sent, but log attachment failed")
            
            return True
        else:
            print("❌ Failed to send HTML dashboard report")
            return False
            
    except Exception as e:
        print(f"❌ Error sending HTML dashboard report: {e}")
        return False

if __name__ == "__main__":
    success = send_html_dashboard_report()
    
    if success:
        print("\n🎉 HTML Dashboard Report Generation Complete!")
        print("📧 Email sent successfully with dashboard attachment")
    else:
        print("\n❌ HTML Dashboard Report Generation Failed!")
        sys.exit(1)
