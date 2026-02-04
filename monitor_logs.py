#!/usr/bin/env python3
"""
NeuralTrader Production Log Monitor
==================================

Watches the logs/ directory for the latest timestamped file and alerts
if it detects 'ERROR' or 'API_DISCONNECT' strings.

Usage:
    python monitor_logs.py
"""

import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

class LogMonitor:
    """Production log monitor for NeuralTrader"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.last_file: Optional[Path] = None
        self.last_position: int = 0
        self.alert_keywords: Set[str] = {"ERROR", "API_DISCONNECT"}
        self.status_file = "logs/monitor_status.txt"
        
        # Ensure logs directory exists
        self.logs_dir.mkdir(exist_ok=True)
        
        print(f"[MONITOR] Log monitor initialized")
        print(f"[MONITOR] Watching directory: {self.logs_dir.absolute()}")
        print(f"[MONITOR] Alert keywords: {', '.join(self.alert_keywords)}")
        print(f"[MONITOR] Status file: {self.status_file}")
        print("-" * 60)
    
    def get_latest_log_file(self) -> Optional[Path]:
        """Get the most recent timestamped log file"""
        try:
            # Look for automation_*.log files
            log_files = list(self.logs_dir.glob("automation_*.log"))
            
            if not log_files:
                return None
            
            # Sort by modification time (most recent first)
            latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
            return latest_file
            
        except Exception as e:
            print(f"[ERROR] Error finding latest log file: {e}")
            return None
    
    def write_status(self, status: str, message: str = ""):
        """Write status to status file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_content = f"[{timestamp}] {status}"
            if message:
                status_content += f": {message}"
            
            with open(self.status_file, 'w') as f:
                f.write(status_content + "\n")
                
        except Exception as e:
            print(f"[ERROR] Error writing status file: {e}")
    
    def check_for_alerts(self, file_path: Path, new_content: str) -> bool:
        """Check new content for alert keywords"""
        alerts_found = []
        
        for keyword in self.alert_keywords:
            if keyword in new_content:
                alerts_found.append(keyword)
        
        if alerts_found:
            alert_msg = f"[ALERT] Detected in {file_path.name}: {', '.join(alerts_found)}"
            print(alert_msg)
            self.write_status("ALERT", f"{file_path.name} - {', '.join(alerts_found)}")
            return True
        
        return False
    
    def monitor_once(self) -> bool:
        """Monitor once - returns True if new content was processed"""
        current_file = self.get_latest_log_file()
        
        if not current_file:
            print("[INFO] No log files found yet...")
            time.sleep(2)
            return False
        
        # Check if we have a new file
        if current_file != self.last_file:
            print(f"[INFO] New log file detected: {current_file.name}")
            self.last_file = current_file
            self.last_position = 0
            self.write_status("NEW_FILE", current_file.name)
        
        try:
            # Get current file size
            current_size = current_file.stat().st_size
            
            # Check if there's new content
            if current_size > self.last_position:
                # Read new content
                with open(current_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.last_position)
                    new_content = f.read()
                
                # Check for alerts
                self.check_for_alerts(current_file, new_content)
                
                # Update position
                self.last_position = current_size
                
                # Show last line for context
                lines = new_content.strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    print(f"[LATEST] {last_line}")
                
                return True
            else:
                # No new content
                return False
                
        except Exception as e:
            print(f"[ERROR] Error reading log file: {e}")
            return False
    
    def run(self, check_interval: int = 5):
        """Run the monitor continuously"""
        print(f"[MONITOR] Starting continuous monitoring (interval: {check_interval}s)")
        print(f"[MONITOR] Press Ctrl+C to stop")
        print("-" * 60)
        
        self.write_status("STARTED", f"Monitoring every {check_interval}s")
        
        try:
            while True:
                self.monitor_once()
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n[MONITOR] Monitoring stopped by user")
            self.write_status("STOPPED", "User interrupt")
        except Exception as e:
            print(f"[ERROR] Monitor crashed: {e}")
            self.write_status("CRASHED", str(e))
            sys.exit(1)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuralTrader Production Log Monitor")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5,
        help="Check interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Logs directory to watch (default: logs)"
    )
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = LogMonitor(logs_dir=args.logs_dir)
    monitor.run(check_interval=args.interval)

if __name__ == "__main__":
    main()
