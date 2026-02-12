#!/usr/bin/env python3
"""
Send Report Script - Wrapper for NeuralTrader reporting functionality
Simple wrapper that calls the main orchestrator in report mode
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Main wrapper function to generate and send daily report"""
    try:
        print(f"[REPORT] Starting daily report generation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Import and call the orchestrator in report mode
        from main_orchestrator_ist import main as orchestrator_main
        
        # Set sys.argv to simulate command line arguments for report mode
        sys.argv = ['main_orchestrator_ist.py', '--mode=report']
        
        # Call the orchestrator main function
        result = orchestrator_main()
        
        if result == 0:
            print(f"[REPORT] Daily report completed successfully - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"[ERROR] Daily report failed with exit code {result}")
        
        return result
        
    except ImportError as e:
        print(f"[ERROR] Failed to import orchestrator: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error in report generation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
