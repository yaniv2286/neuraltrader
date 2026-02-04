#!/usr/bin/env python3
"""
NeuralTrader Supervision Dashboard
==================================

Interactive dashboard for monitoring Task Scheduler runs and system health.
Provides real-time status and historical analysis.

Usage:
    python scripts/supervision_dashboard.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def clear_screen():
    """Clear console screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print dashboard header"""
    print("=" * 80)
    print("🤖 NEURALTRADER SUPERVISION DASHBOARD")
    print("=" * 80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def load_daily_data(date: datetime) -> Dict:
    """Load daily supervision data"""
    try:
        from src.utils.daily_logger import DailySupervisionLogger
        logger = DailySupervisionLogger()
        return logger.get_daily_summary(date)
    except Exception as e:
        print(f"❌ Error loading daily data: {e}")
        return {"runs": [], "summary": {}, "mode_stats": {}}

def print_daily_summary(daily_data: Dict, date: datetime):
    """Print daily summary"""
    summary = daily_data["summary"]
    mode_stats = daily_data["mode_stats"]
    
    print(f"📊 DAILY SUMMARY - {date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    print(f"Total Runs: {summary.get('total_runs', 0)}")
    print(f"✅ Successful: {summary.get('successful_runs', 0)}")
    print(f"❌ Failed: {summary.get('failed_runs', 0)}")
    
    if summary.get('total_runs', 0) > 0:
        success_rate = (summary.get('successful_runs', 0) / summary.get('total_runs', 1)) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")
    
    print()
    
    if mode_stats:
        print("🎭 MODE BREAKDOWN:")
        for mode, stats in mode_stats.items():
            print(f"  {mode.upper()}: {stats['total']} runs, {stats['success_rate']:.1f}% success")
        print()

def print_recent_runs(daily_data: Dict, limit: int = 10):
    """Print recent runs"""
    runs = daily_data.get("runs", [])
    if not runs:
        print("📝 No runs found for today")
        return
    
    print(f"📝 RECENT RUNS (Last {limit}):")
    print("-" * 50)
    
    for run in runs[-limit:]:
        status_emoji = "✅" if run.get("status") == "SUCCESS" else "❌"
        timestamp = run.get("timestamp_ist", "Unknown time")
        mode = run.get("mode", "Unknown")
        duration = run.get("duration_formatted", "N/A")
        
        print(f"{status_emoji} {timestamp} - {mode} ({duration})")
        
        if run.get("error"):
            print(f"   ❌ Error: {run.get('error', 'Unknown error')}")
    print()

def print_week_summary():
    """Print week summary"""
    print("📅 WEEK SUMMARY (Last 7 Days):")
    print("-" * 50)
    
    try:
        from src.utils.daily_logger import DailySupervisionLogger
        logger = DailySupervisionLogger()
        
        total_runs = 0
        total_success = 0
        total_failed = 0
        
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            daily_data = logger.get_daily_summary(date)
            summary = daily_data["summary"]
            
            total_runs += summary.get('total_runs', 0)
            total_success += summary.get('successful_runs', 0)
            total_failed += summary.get('failed_runs', 0)
            
            date_str = date.strftime('%Y-%m-%d')
            runs = summary.get('total_runs', 0)
            success = summary.get('successful_runs', 0)
            
            print(f"  {date_str}: {runs} runs, {success} success")
        
        print()
        print(f"📊 WEEK TOTAL: {total_runs} runs, {total_success} success, {total_failed} failed")
        
        if total_runs > 0:
            week_success_rate = (total_success / total_runs) * 100
            print(f"📈 WEEK SUCCESS RATE: {week_success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error generating week summary: {e}")
    
    print()

def print_system_status():
    """Print system status"""
    print("🔧 SYSTEM STATUS:")
    print("-" * 50)
    
    # Check if automation.log exists and has recent entries
    automation_log = project_root / "logs" / "automation.log"
    if automation_log.exists():
        try:
            with open(automation_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print(f"📝 Last automation log: {last_line}")
                else:
                    print("📝 Automation log is empty")
        except Exception as e:
            print(f"❌ Error reading automation log: {e}")
    else:
        print("📝 Automation log not found")
    
    # Check portfolio status
    portfolio_file = project_root / "data" / "portfolio.json"
    if portfolio_file.exists():
        try:
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                portfolio = json.load(f)
                total_value = portfolio.get('performance', {}).get('total_value', 0)
                print(f"💰 Portfolio Value: ${total_value:,.2f}")
        except Exception as e:
            print(f"❌ Error reading portfolio: {e}")
    else:
        print("💰 Portfolio file not found")
    
    print()

def interactive_menu():
    """Interactive menu for dashboard"""
    while True:
        print("🎛️  INTERACTIVE OPTIONS:")
        print("1. Refresh Dashboard")
        print("2. View Specific Date")
        print("3. Send Daily Report Email")
        print("4. View Automation Log")
        print("5. Exit")
        print()
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            return  # Refresh dashboard
        elif choice == "2":
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                daily_data = load_daily_data(date)
                print_daily_summary(daily_data, date)
                print_recent_runs(daily_data)
                input("Press Enter to continue...")
            except ValueError:
                print("❌ Invalid date format")
                input("Press Enter to continue...")
        elif choice == "3":
            try:
                from src.utils.daily_logger import DailySupervisionLogger
                logger = DailySupervisionLogger()
                success = logger.send_daily_supervision_report()
                if success:
                    print("✅ Daily report sent successfully!")
                else:
                    print("❌ Failed to send daily report")
                input("Press Enter to continue...")
            except Exception as e:
                print(f"❌ Error sending report: {e}")
                input("Press Enter to continue...")
        elif choice == "4":
            try:
                automation_log = project_root / "logs" / "automation.log"
                if automation_log.exists():
                    with open(automation_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print("📝 RECENT AUTOMATION LOG (Last 20 lines):")
                        print("-" * 50)
                        for line in lines[-20:]:
                            print(line.rstrip())
                else:
                    print("❌ Automation log not found")
                input("Press Enter to continue...")
            except Exception as e:
                print(f"❌ Error reading log: {e}")
                input("Press Enter to continue...")
        elif choice == "5":
            print("👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice")
            input("Press Enter to continue...")

def main():
    """Main dashboard function"""
    while True:
        clear_screen()
        print_header()
        
        # Load today's data
        today = datetime.now()
        daily_data = load_daily_data(today)
        
        # Print dashboard sections
        print_daily_summary(daily_data, today)
        print_recent_runs(daily_data)
        print_week_summary()
        print_system_status()
        
        # Interactive menu
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Dashboard interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        sys.exit(1)
