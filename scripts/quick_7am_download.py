#!/usr/bin/env python3
"""
Quick 7:00 AM Download Script
Simple script to run download at 7:00 AM when API renews
"""

import time
import subprocess
import os
from datetime import datetime

def wait_until_7am():
    """Wait until 7:00 AM"""
    now = datetime.now()
    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    
    # If it's already past 7:00 AM, schedule for tomorrow
    if now >= target:
        target = target + timedelta(days=1)
    
    wait_seconds = (target - now).total_seconds()
    
    print(f"⏰ Current time: {now.strftime('%H:%M:%S')}")
    print(f"🎯 Target time: {target.strftime('%H:%M %A')}")
    print(f"⏱️ Waiting {wait_seconds/60:.1f} minutes...")
    
    return wait_seconds

def run_download():
    """Run the download script"""
    print("\n" + "="*70)
    print("🚀 RUNNING 7:00 AM DOWNLOAD")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(__file__))
    os.chdir(project_dir)
    
    # Run the download script
    try:
        result = subprocess.run(
            ['uv', 'run', 'python', 'scripts/download_all_tickers_fixed.py'],
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("✅ Download completed successfully!")
        else:
            print("❌ Download failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("="*70)
    print("⏰ 7:00 AM AUTO DOWNLOAD")
    print("="*70)
    
    # Wait until 7:00 AM
    wait_seconds = wait_until_7am()
    
    # Wait with progress updates
    start_time = time.time()
    while time.time() - start_time < wait_seconds:
        elapsed = time.time() - start_time
        remaining = wait_seconds - elapsed
        
        if int(elapsed) % 300 == 0:  # Update every 5 minutes
            print(f"⏳ {remaining/60:.1f} minutes remaining...")
        
        time.sleep(60)  # Check every minute
    
    # Run the download
    run_download()
    
    print("\n🎉 7:00 AM download complete!")

if __name__ == "__main__":
    main()
