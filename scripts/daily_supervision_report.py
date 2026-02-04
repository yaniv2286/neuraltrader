#!/usr/bin/env python3
"""
NeuralTrader Daily Supervision Report Script
===========================================

Generates and sends daily supervision reports for Task Scheduler monitoring.
Can be run manually or scheduled to send daily summaries.

Usage:
    python scripts/daily_supervision_report.py [--date YYYY-MM-DD] [--send-email]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to generate daily supervision report"""
    parser = argparse.ArgumentParser(description='Generate NeuralTrader Daily Supervision Report')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--send-email', action='store_true', help='Send report via email')
    parser.add_argument('--print', action='store_true', help='Print report to console')
    
    args = parser.parse_args()
    
    # Parse date
    if args.date:
        try:
            date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        date = datetime.now()
    
    try:
        # Import daily supervision logger
        from src.utils.daily_logger import DailySupervisionLogger
        
        # Initialize logger
        supervision_logger = DailySupervisionLogger()
        
        # Generate report
        print(f"📊 Generating daily supervision report for {date.strftime('%Y-%m-%d')}...")
        report = supervision_logger.create_daily_report(date)
        
        # Print to console if requested
        if args.print or not args.send_email:
            print(report)
        
        # Send email if requested
        if args.send_email:
            print("📧 Sending daily supervision report via email...")
            success = supervision_logger.send_daily_supervision_report(date)
            
            if success:
                print("✅ Daily supervision report sent successfully!")
            else:
                print("❌ Failed to send daily supervision report")
                sys.exit(1)
        
        print("✅ Daily supervision report completed")
        
    except Exception as e:
        print(f"❌ Error generating daily supervision report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
