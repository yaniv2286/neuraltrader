"""
Test script for core integrity validation
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.integrity import verify_system_integrity

# Test Case 1: Valid model and strategy
print("\n" + "="*80)
print("TEST 1: Valid model and strategy (should PASS)")
print("="*80)

class MockModel:
    def predict(self, X):
        return [0.5]

class MockStrategy:
    max_drawdown = 0.20
    
    def check_entry(self, data):
        return True
    
    def check_exit(self, data):
        return False

try:
    model = MockModel()
    strategy = MockStrategy()
    result = verify_system_integrity(model, strategy)
    print(f"Result: {result}")
    print("TEST 1: PASSED\n")
except SystemExit as e:
    print(f"TEST 1: FAILED - Unexpected SystemExit: {e}")

# Test Case 2: Model is None (should FAIL)
print("\n" + "="*80)
print("TEST 2: Model is None (should FAIL with SystemExit)")
print("="*80)

try:
    result = verify_system_integrity(None, strategy)
    print("TEST 2: FAILED - Should have raised SystemExit")
except SystemExit as e:
    print(f"TEST 2: PASSED - Correctly raised SystemExit({e.code})\n")

# Test Case 3: Model missing predict method (should FAIL)
print("\n" + "="*80)
print("TEST 3: Model missing predict method (should FAIL with SystemExit)")
print("="*80)

class InvalidModel:
    pass

try:
    invalid_model = InvalidModel()
    result = verify_system_integrity(invalid_model, strategy)
    print("TEST 3: FAILED - Should have raised SystemExit")
except SystemExit as e:
    print(f"TEST 3: PASSED - Correctly raised SystemExit({e.code})\n")

# Test Case 4: Strategy missing check_entry (should FAIL)
print("\n" + "="*80)
print("TEST 4: Strategy missing check_entry (should FAIL with SystemExit)")
print("="*80)

class InvalidStrategy1:
    max_drawdown = 0.20
    
    def check_exit(self, data):
        return False

try:
    invalid_strategy = InvalidStrategy1()
    result = verify_system_integrity(model, invalid_strategy)
    print("TEST 4: FAILED - Should have raised SystemExit")
except SystemExit as e:
    print(f"TEST 4: PASSED - Correctly raised SystemExit({e.code})\n")

# Test Case 5: Strategy missing check_exit (should FAIL)
print("\n" + "="*80)
print("TEST 5: Strategy missing check_exit (should FAIL with SystemExit)")
print("="*80)

class InvalidStrategy2:
    max_drawdown = 0.20
    
    def check_entry(self, data):
        return True

try:
    invalid_strategy = InvalidStrategy2()
    result = verify_system_integrity(model, invalid_strategy)
    print("TEST 5: FAILED - Should have raised SystemExit")
except SystemExit as e:
    print(f"TEST 5: PASSED - Correctly raised SystemExit({e.code})\n")

# Test Case 6: Strategy missing max_drawdown (should FAIL)
print("\n" + "="*80)
print("TEST 6: Strategy missing max_drawdown (should FAIL with SystemExit)")
print("="*80)

class InvalidStrategy3:
    def check_entry(self, data):
        return True
    
    def check_exit(self, data):
        return False

try:
    invalid_strategy = InvalidStrategy3()
    result = verify_system_integrity(model, invalid_strategy)
    print("TEST 6: FAILED - Should have raised SystemExit")
except SystemExit as e:
    print(f"TEST 6: PASSED - Correctly raised SystemExit({e.code})\n")

# Test Case 7: Strategy max_drawdown is None (should FAIL)
print("\n" + "="*80)
print("TEST 7: Strategy max_drawdown is None (should FAIL with SystemExit)")
print("="*80)

class InvalidStrategy4:
    max_drawdown = None
    
    def check_entry(self, data):
        return True
    
    def check_exit(self, data):
        return False

try:
    invalid_strategy = InvalidStrategy4()
    result = verify_system_integrity(model, invalid_strategy)
    print("TEST 7: FAILED - Should have raised SystemExit")
except SystemExit as e:
    print(f"TEST 7: PASSED - Correctly raised SystemExit({e.code})\n")

print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)
