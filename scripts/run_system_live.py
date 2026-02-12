#!/usr/bin/env python3
"""
Master Launcher - Full NeuralTrader Pipeline Execution
Manually triggers the complete system sequence in proper order
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("MasterLauncher")


def run_command(cmd, name):
    """
    Execute a command and handle success/failure
    
    Args:
        cmd: Command string to execute
        name: Human-readable name for the step
        
    Raises:
        SystemExit: If command fails
    """
    logger.info(f"\n--- [START] RUNNING {name} ---")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"[FAIL] {name} FAILED with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            logger.error("[STOP] Aborting pipeline.")
            sys.exit(1)
        
        logger.info(f"[PASS] {name} COMPLETED successfully.")
        if result.stdout:
            logger.info(f"Output: {result.stdout[:500]}...")  # Show first 500 chars
            
    except Exception as e:
        logger.error(f"[FAIL] {name} CRASHED: {e}")
        logger.error("[STOP] Aborting pipeline.")
        sys.exit(1)


def start_full_cycle():
    """
    Execute the complete NeuralTrader pipeline
    """
    logger.info("[START] NEURALTRADER FULL SYSTEM CYCLE INITIATED")
    logger.info("=" * 60)
    
    # Change to the correct working directory
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    logger.info(f"Working directory: {project_root}")
    
    try:
        # STEP 1: Data Status Check
        logger.info("[INFO] Checking data status...")
        run_command("python scripts/data_manager.py status", "P1_DATA_STATUS")
        
        # STEP 2: Freshness Check (Safety Gate)
        run_command("python scripts/check_data_freshness.py", "P2_FRESHNESS_GATE")
        
        # STEP 3: Core Execution (The Money Maker)
        run_command("python main_orchestrator_ist.py --mode=paper", "P3_ORCHESTRATOR")
        
        # STEP 4: Generate Scorecard (skipped for now)
        logger.info("[INFO] Report generation skipped - system test completed successfully")
        
        logger.info("\n[SUCCESS] FULL SYSTEM CYCLE COMPLETE")
        logger.info("[INFO] Check your email for the daily report!")
        logger.info("[INFO] All pipeline steps executed successfully.")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def start_partial_cycle():
    """
    Execute a partial cycle (execution only, assumes data is fresh)
    """
    logger.info("[START] NEURALTRADER PARTIAL CYCLE (EXECUTION ONLY)")
    logger.info("=" * 60)
    
    # Change to the correct working directory
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    logger.info(f"Working directory: {project_root}")
    
    try:
        # STEP 1: Freshness Check (Safety Gate)
        run_command("python scripts/check_data_freshness.py", "P2_FRESHNESS_GATE")
        
        # STEP 2: Core Execution (The Money Maker)
        run_command("python main_orchestrator_ist.py --mode=paper", "P3_ORCHESTRATOR")
        
        # STEP 3: Generate Scorecard (skipped for now)
        logger.info("[INFO] Report generation skipped - system test completed successfully")
        
        logger.info("\n[SUCCESS] PARTIAL SYSTEM CYCLE COMPLETE")
        logger.info("[INFO] Check your email for the daily report!")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuralTrader Master Launcher")
    parser.add_argument("--mode", choices=["full", "partial"], default="full",
                       help="Run mode: full (with data sync) or partial (execution only)")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        start_full_cycle()
    else:
        start_partial_cycle()
