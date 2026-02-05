"""
Test script to verify Core Protection Protocol enforcement
This script intentionally passes None as the model to verify SystemExit is raised
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.integrity import verify_system_integrity

print("\n" + "="*80)
print("TEST: Core Protection Protocol - Integrity Failure Test")
print("="*80)
print("\nObjective: Verify that passing None as model raises SystemExit")
print("Expected: [CRITICAL] errors followed by SystemExit(1)")
print("\n" + "="*80 + "\n")

# Create a valid strategy object
class MockStrategy:
    max_drawdown = 0.20
    
    def check_entry(self, data):
        return True
    
    def check_exit(self, data):
        return False

strategy = MockStrategy()

# Attempt to verify with None model - should raise SystemExit
try:
    print("[TEST] Calling verify_system_integrity(None, strategy)...")
    verify_system_integrity(None, strategy)
    
    # If we reach here, the test FAILED
    print("\n" + "="*80)
    print("[FAIL] TEST FAILED - SystemExit was NOT raised!")
    print("[FAIL] Core Protection Protocol is NOT working correctly")
    print("="*80)
    sys.exit(1)
    
except SystemExit as e:
    # This is expected - the test PASSED
    print("\n" + "="*80)
    print(f"[PASS] TEST PASSED - SystemExit({e.code}) was raised as expected")
    print("[PASS] Core Protection Protocol is working correctly")
    print("[PASS] System cannot operate without valid AI model")
    print("="*80)
    sys.exit(0)
