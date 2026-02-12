#!/usr/bin/env python3
"""
Data Freshness Sentinel - Pre-execution Safety Gate for Daily Orchestrator
Enforces 'No Silent Failures' protocol for data freshness validation
"""

import pandas as pd
from datetime import datetime
import sys
import os
import logging

# Setup Fail-Fast Logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DataSentinel")


def verify_data():
    """
    Verify data freshness before allowing execution
    
    Enforces 'No Silent Failures' protocol:
    - CRITICAL FAIL on missing file
    - CRITICAL FAIL on empty file  
    - CRITICAL FAIL on stale data
    - Exit code 1 for any failure
    - Exit code 0 only for fresh data
    """
    path = 'data/processed/modern_era_universe.parquet'
    
    logger.info("[CHECK] DataSentinel: Starting freshness validation...")
    
    # Check 1: File existence
    if not os.path.exists(path):
        logger.error(f"[FAIL] CRITICAL FAIL: Universe file missing at {path}")
        logger.error("[INFO] DataSync (P1) likely failed or hasn't finished yet.")
        sys.exit(1)

    logger.info(f"[PASS] File found: {path}")

    try:
        # Read only the index (dates) to save memory and time
        # Using index instead of 'date' column since parquet files typically use datetime index
        df = pd.read_parquet(path)
        
        if df.empty:
            logger.error("[FAIL] CRITICAL FAIL: Universe file is empty.")
            logger.error("[INFO] DataSync (P1) failed to populate data.")
            sys.exit(1)

        logger.info(f"[PASS] Data loaded: {len(df)} records")

        # Get the latest date in the data
        if 'Date' in df.columns:
            latest_date = pd.to_datetime(df['Date']).max().date()
        elif 'date' in df.columns:
            latest_date = pd.to_datetime(df['date']).max().date()
        elif hasattr(df.index, 'max') and hasattr(df.index.max(), 'date'):
            latest_date = df.index.max().date()
        elif 'timestamp' in df.columns:
            latest_date = pd.to_datetime(df['timestamp']).max().date()
        else:
            logger.error("[FAIL] CRITICAL FAIL: No date/timestamp column found in data.")
            logger.error(f"[INFO] Available columns: {df.columns.tolist()}")
            sys.exit(1)

        today = datetime.now().date()
        
        logger.info(f"[DATA] Latest data date: {latest_date}")
        logger.info(f"[DATA] Today's date: {today}")

        # Check 2: Data freshness (allow 1-day lag for market data)
        from datetime import timedelta
        one_day_ago = today - timedelta(days=1)
        
        if latest_date < one_day_ago:
            logger.error(f"[FAIL] CRITICAL FAIL: Data is TOO STALE. Latest date in data: {latest_date}, Expected: {one_day_ago} or newer")
            logger.error("[INFO] DataSync (P1) likely failed or hasn't finished yet.")
            logger.error("[STOP] EXECUTION ABORTED: Stale data detected for safety.")
            sys.exit(1)
        
        if latest_date == today:
            logger.info(f"[PASS] DATA FRESH: Latest record matches today ({latest_date}). Proceeding to Execution.")
        elif latest_date == one_day_ago:
            logger.info(f"[PASS] DATA FRESH: Latest record from previous trading day ({latest_date}). Proceeding to Execution.")
        else:
            logger.info(f"[PASS] DATA ACCEPTABLE: Latest record ({latest_date}) within acceptable range.")
        logger.info("[SUCCESS] DataSentinel: Validation PASSED - System GO for execution")
        sys.exit(0)

    except Exception as e:
        logger.error(f"[FAIL] CRITICAL FAIL: Sentinel crashed while reading data: {e}")
        logger.error("[STOP] EXECUTION ABORTED: Data validation failed.")
        sys.exit(1)


if __name__ == "__main__":
    verify_data()
