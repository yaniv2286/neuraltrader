"""
NeuralTrader Daily Supervision Logger
====================================

Creates daily log summaries for Task Scheduler runs supervision.
Consolidates all automation logs into daily reports for easy monitoring.

Features:
- Daily run summaries
- Consolidated log analysis
- Performance metrics tracking
- Error monitoring and alerts
- Historical trend analysis
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pytz

class DailySupervisionLogger:
    """
    Daily supervision logger for Task Scheduler runs
    """
    
    def __init__(self, logs_dir: str = None):
        """Initialize daily supervision logger"""
        self.project_root = Path(__file__).parent.parent.parent
        self.logs_dir = Path(logs_dir) if logs_dir else self.project_root / "logs"
        self.supervision_dir = self.logs_dir / "supervision"
        self.supervision_dir.mkdir(exist_ok=True)
        
        # Timezone handling
        self.eastern = pytz.timezone('US/Eastern')
        self.israel = pytz.timezone('Asia/Jerusalem')
        
        # Configure logging
        self.logger = logging.getLogger('DailySupervision')
        
    def get_daily_log_file(self, date: datetime = None) -> Path:
        """Get daily supervision log file path"""
        if date is None:
            date = datetime.now(self.israel)
        
        date_str = date.strftime('%Y-%m-%d')
        return self.supervision_dir / f"daily_supervision_{date_str}.json"
    
    def log_run_start(self, mode: str, run_id: str = None) -> Dict:
        """Log the start of a Task Scheduler run"""
        timestamp = datetime.now(self.israel)
        run_id = run_id or f"{mode}_{timestamp.strftime('%H%M%S')}"
        
        log_entry = {
            "run_id": run_id,
            "mode": mode,
            "status": "STARTED",
            "timestamp": timestamp.isoformat(),
            "timestamp_ist": timestamp.strftime('%Y-%m-%d %H:%M:%S IST'),
            "timestamp_est": timestamp.astimezone(self.eastern).strftime('%Y-%m-%d %H:%M:%S EST'),
            "start_time": timestamp.isoformat()
        }
        
        self._write_daily_log(log_entry)
        self.logger.info(f"🚀 Run started: {mode} [{run_id}]")
        
        return log_entry
    
    def log_run_complete(self, mode: str, run_id: str, success: bool, 
                        duration: float, details: Dict = None) -> Dict:
        """Log the completion of a Task Scheduler run"""
        timestamp = datetime.now(self.israel)
        
        log_entry = {
            "run_id": run_id,
            "mode": mode,
            "status": "SUCCESS" if success else "FAILED",
            "timestamp": timestamp.isoformat(),
            "timestamp_ist": timestamp.strftime('%Y-%m-%d %H:%M:%S IST'),
            "timestamp_est": timestamp.astimezone(self.eastern).strftime('%Y-%m-%d %H:%M:%S EST'),
            "end_time": timestamp.isoformat(),
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration),
            "details": details or {}
        }
        
        self._write_daily_log(log_entry)
        
        status_emoji = "✅" if success else "❌"
        self.logger.info(f"{status_emoji} Run completed: {mode} [{run_id}] in {duration:.2f}s")
        
        return log_entry
    
    def log_error(self, mode: str, run_id: str, error: str, error_type: str = None) -> Dict:
        """Log an error during Task Scheduler run"""
        timestamp = datetime.now(self.israel)
        
        log_entry = {
            "run_id": run_id,
            "mode": mode,
            "status": "ERROR",
            "timestamp": timestamp.isoformat(),
            "timestamp_ist": timestamp.strftime('%Y-%m-%d %H:%M:%S IST'),
            "timestamp_est": timestamp.astimezone(self.eastern).strftime('%Y-%m-%d %H:%M:%S EST'),
            "error": error,
            "error_type": error_type or type(error).__name__
        }
        
        self._write_daily_log(log_entry)
        self.logger.error(f"❌ Error in {mode} [{run_id}]: {error}")
        
        return log_entry
    
    def _write_daily_log(self, log_entry: Dict):
        """Write log entry to daily supervision file"""
        daily_file = self.get_daily_log_file()
        
        # Load existing daily logs
        daily_logs = []
        if daily_file.exists():
            try:
                with open(daily_file, 'r', encoding='utf-8') as f:
                    daily_logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                daily_logs = []
        
        # Add new entry
        daily_logs.append(log_entry)
        
        # Save updated logs
        with open(daily_file, 'w', encoding='utf-8') as f:
            json.dump(daily_logs, f, indent=2, ensure_ascii=False)
    
    def get_daily_summary(self, date: datetime = None) -> Dict:
        """Get daily summary of all Task Scheduler runs"""
        daily_file = self.get_daily_log_file(date)
        
        if not daily_file.exists():
            return {"date": date.strftime('%Y-%m-%d'), "runs": [], "summary": {}}
        
        with open(daily_file, 'r', encoding='utf-8') as f:
            daily_logs = json.load(f)
        
        # Calculate summary
        summary = {
            "total_runs": len(daily_logs),
            "successful_runs": len([r for r in daily_logs if r.get("status") == "SUCCESS"]),
            "failed_runs": len([r for r in daily_logs if r.get("status") in ["FAILED", "ERROR"]]),
            "modes": list(set([r.get("mode") for r in daily_logs])),
            "first_run": min([r.get("timestamp") for r in daily_logs]) if daily_logs else None,
            "last_run": max([r.get("timestamp") for r in daily_logs]) if daily_logs else None
        }
        
        # Calculate success rate by mode
        mode_stats = {}
        for run in daily_logs:
            mode = run.get("mode", "unknown")
            if mode not in mode_stats:
                mode_stats[mode] = {"total": 0, "success": 0, "failed": 0}
            
            mode_stats[mode]["total"] += 1
            if run.get("status") == "SUCCESS":
                mode_stats[mode]["success"] += 1
            elif run.get("status") in ["FAILED", "ERROR"]:
                mode_stats[mode]["failed"] += 1
        
        # Calculate success rates
        for mode, stats in mode_stats.items():
            stats["success_rate"] = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            stats["avg_duration"] = self._calculate_avg_duration([r for r in daily_logs if r.get("mode") == mode])
        
        return {
            "date": date.strftime('%Y-%m-%d') if date else datetime.now(self.israel).strftime('%Y-%m-%d'),
            "runs": daily_logs,
            "summary": summary,
            "mode_stats": mode_stats
        }
    
    def create_daily_report(self, date: datetime = None) -> str:
        """Create a formatted daily supervision report"""
        daily_data = self.get_daily_summary(date)
        summary = daily_data["summary"]
        mode_stats = daily_data["mode_stats"]
        
        report = f"""
📊 NEURALTRADER DAILY SUPERVISION REPORT
=====================================
📅 Date: {daily_data['date']}
🕐 Generated: {datetime.now(self.israel).strftime('%Y-%m-%d %H:%M:%S IST')}

📈 EXECUTION SUMMARY:
---------------------
Total Runs: {summary['total_runs']}
✅ Successful: {summary['successful_runs']}
❌ Failed: {summary['failed_runs']}
📊 Success Rate: {(summary['successful_runs'] / summary['total_runs'] * 100) if summary['total_runs'] > 0 else 0:.1f}%

🎭 MODE BREAKDOWN:
-----------------"""
        
        for mode, stats in mode_stats.items():
            report += f"""
{mode.upper()}:
  Total: {stats['total']} runs
  Success: {stats['success']} ({stats['success_rate']:.1f}%)
  Failed: {stats['failed']}
  Avg Duration: {stats['avg_duration']}"""
        
        if summary['first_run'] and summary['last_run']:
            report += f"""
⏰ ACTIVITY PERIOD:
------------------
First Run: {datetime.fromisoformat(summary['first_run']).strftime('%H:%M:%S IST')}
Last Run: {datetime.fromisoformat(summary['last_run']).strftime('%H:%M:%S IST')}"""
        
        # Add recent errors if any
        failed_runs = [r for r in daily_data['runs'] if r.get('status') in ['FAILED', 'ERROR']]
        if failed_runs:
            report += f"""

❌ RECENT ERRORS:
-----------------"""
            for error_run in failed_runs[-3:]:  # Show last 3 errors
                report += f"""
{error_run.get('timestamp_ist', 'Unknown time')} - {error_run.get('mode', 'Unknown mode')}:
  Error: {error_run.get('error', 'Unknown error')}
  Type: {error_run.get('error_type', 'Unknown')}"""
        
        report += f"""

📧 EMAIL NOTIFICATIONS:
---------------------
This report is automatically generated and sent to lugassy.ai@gmail.com

🔍 DETAILED LOGS:
-----------------
Full logs available at: logs/supervision/daily_supervision_{daily_data['date']}.json
Automation logs: logs/automation.log

---
NeuralTrader Supervision System
Phase 6.2: Task Scheduler Integration
"""
        
        return report
    
    def _format_duration(self, duration: float) -> str:
        """Format duration in human readable format"""
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _calculate_avg_duration(self, runs: List[Dict]) -> str:
        """Calculate average duration for a set of runs"""
        durations = [r.get("duration_seconds", 0) for r in runs if r.get("duration_seconds")]
        if not durations:
            return "N/A"
        
        avg_duration = sum(durations) / len(durations)
        return self._format_duration(avg_duration)
    
    def send_daily_supervision_report(self, date: datetime = None) -> bool:
        """Send daily supervision report via email"""
        try:
            from .notifier import EmailNotifier
            
            # Create report
            report = self.create_daily_report(date)
            
            # Send email
            notifier = EmailNotifier()
            subject = f"📊 NeuralTrader Daily Supervision Report - {date.strftime('%Y-%m-%d') if date else datetime.now(self.israel).strftime('%Y-%m-%d')}"
            
            success = notifier.send_email(
                to_email=notifier.recipient_email,
                subject=subject,
                body=report
            )
            
            if success:
                self.logger.info("📧 Daily supervision report sent successfully")
            else:
                self.logger.error("❌ Failed to send daily supervision report")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error sending daily supervision report: {e}")
            return False
