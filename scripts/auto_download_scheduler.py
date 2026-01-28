#!/usr/bin/env python3
"""
Automatic Download Scheduler
Runs download script at the start of every hour (specifically 7:00 AM)
"""

import time
import schedule
import subprocess
import os
from datetime import datetime, timedelta

def run_download_script():
    """Run the download script and capture results"""
    print("="*70)
    print(f"🚀 AUTO SCHEDULER - Running Download Script")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(__file__))
    os.chdir(project_dir)
    
    # Run the download script
    try:
        result = subprocess.run(
            ['uv', 'run', 'python', 'scripts/download_all_tickers_fixed.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        print("📊 DOWNLOAD RESULTS:")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Return code: {result.returncode}")
        
        # Send notification (optional)
        if result.returncode == 0:
            print("✅ Download completed successfully!")
        else:
            print("❌ Download failed - check logs")
            
    except subprocess.TimeoutExpired:
        print("⏰ Download script timed out after 1 hour")
    except Exception as e:
        print(f"❌ Error running download script: {e}")

def check_api_status():
    """Check if API is ready by making a small test request"""
    try:
        import requests
        
        # Test API with a small request
        token = "72e14af10f4c32db4a7631275929617481aed281"
        test_url = f"https://api.tiingo.com/tiingo/daily/SPY/prices?token={token}&startDate=2026-01-26&endDate=2026-01-27"
        
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200 and len(response.text) > 100:
            print("✅ API is ready - good to download")
            return True
        else:
            print(f"⚠️ API not ready - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API check failed: {e}")
        return False

def scheduled_download():
    """Main scheduled download function"""
    print(f"\n🕐 Scheduled download triggered at {datetime.now().strftime('%H:%M:%S')}")
    
    # Check API status first
    if check_api_status():
        run_download_script()
    else:
        print("⏰ API not ready - will retry next hour")

def main():
    """Main scheduler function"""
    print("="*70)
    print("🤖 AUTOMATIC DOWNLOAD SCHEDULER")
    print("="*70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⏰ Schedule: Run at the start of every hour")
    print("🎯 Target: Complete 80 ticker download")
    print("🔄 Will continue until all tickers downloaded")
    print("="*70)
    
    # Schedule to run at the start of every hour
    schedule.every().hour.at(":00").do(scheduled_download)
    
    # Also schedule for 7:00 AM specifically (user's renewal time)
    schedule.every().day.at("07:00").do(scheduled_download)
    
    print("📅 Scheduled jobs:")
    print("  - Every hour at :00")
    print("  - Daily at 07:00 (user renewal)")
    print("\n🔄 Waiting for scheduled time...")
    
    # Run immediately if it's close to 7:00 AM
    now = datetime.now()
    if now.hour == 6 and now.minute >= 50:
        print("⏰ Close to 7:00 AM - running immediately!")
        scheduled_download()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
        
        # Check if download is complete
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
        csv_files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
        
        if len(csv_files) >= 116:
            print("🎉 All 116 tickers downloaded! Scheduler stopping.")
            break

if __name__ == "__main__":
    main()
