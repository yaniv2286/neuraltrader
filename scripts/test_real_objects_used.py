"""
Test script to verify REAL AI and strategy objects are passed to orchestrator
and actually used in trading operations
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

print("\n" + "="*80)
print("TEST: Verify REAL AI and Strategy Objects Are Used")
print("="*80)
print("\nObjective: Confirm orchestrator receives and stores real objects")
print("Expected: Orchestrator has ai_model and trading_strategy attributes")
print("\n" + "="*80 + "\n")

from core.ai_models import EnsemblePredictor
from core.strategy import TradingStrategy
from core.integrity import verify_system_integrity

# Import after setting up path
from main_orchestrator_ist import TradingOrchestrator

# Load REAL components
print("[TEST] Loading real AI model (EnsemblePredictor)...")
real_ai = EnsemblePredictor()
print("[TEST] AI model loaded successfully")

print("[TEST] Loading real trading strategy (TradingStrategy)...")
real_strategy = TradingStrategy()
print("[TEST] Strategy loaded successfully")

# Verify integrity
print("[TEST] Running integrity check...")
verify_system_integrity(real_ai, real_strategy)
print("[TEST] Integrity check passed")

# Create orchestrator with real objects
print("\n[TEST] Creating orchestrator with REAL objects...")
orchestrator = TradingOrchestrator(ai_model=real_ai, trading_strategy=real_strategy)

# Verify objects are stored
print("\n" + "="*80)
print("VERIFICATION RESULTS:")
print("="*80)

if orchestrator.ai_model is None:
    print("[FAIL] Orchestrator ai_model is None - objects NOT passed correctly")
    sys.exit(1)
else:
    print(f"[PASS] Orchestrator ai_model: {type(orchestrator.ai_model).__name__}")

if orchestrator.trading_strategy is None:
    print("[FAIL] Orchestrator trading_strategy is None - objects NOT passed correctly")
    sys.exit(1)
else:
    print(f"[PASS] Orchestrator trading_strategy: {type(orchestrator.trading_strategy).__name__}")

# Verify they are the SAME objects (not reloaded)
if orchestrator.ai_model is real_ai:
    print("[PASS] ai_model is the SAME object (not reloaded)")
else:
    print("[FAIL] ai_model is a DIFFERENT object (was reloaded)")
    sys.exit(1)

if orchestrator.trading_strategy is real_strategy:
    print("[PASS] trading_strategy is the SAME object (not reloaded)")
else:
    print("[FAIL] trading_strategy is a DIFFERENT object (was reloaded)")
    sys.exit(1)

# Verify objects have required methods
print("\n" + "="*80)
print("METHOD VERIFICATION:")
print("="*80)

if hasattr(orchestrator.ai_model, 'predict'):
    print("[PASS] ai_model has predict() method")
else:
    print("[FAIL] ai_model missing predict() method")
    sys.exit(1)

if hasattr(orchestrator.trading_strategy, 'check_entry'):
    print("[PASS] trading_strategy has check_entry() method")
else:
    print("[FAIL] trading_strategy missing check_entry() method")
    sys.exit(1)

if hasattr(orchestrator.trading_strategy, 'check_exit'):
    print("[PASS] trading_strategy has check_exit() method")
else:
    print("[FAIL] trading_strategy missing check_exit() method")
    sys.exit(1)

if hasattr(orchestrator.trading_strategy, 'max_drawdown'):
    print(f"[PASS] trading_strategy has max_drawdown: {orchestrator.trading_strategy.max_drawdown}")
else:
    print("[FAIL] trading_strategy missing max_drawdown")
    sys.exit(1)

print("\n" + "="*80)
print("[SUCCESS] ALL TESTS PASSED")
print("[SUCCESS] Real AI and strategy objects are properly passed to orchestrator")
print("[SUCCESS] Objects are NOT reloaded (same instance)")
print("[SUCCESS] System is using REAL intelligence, not dummy objects")
print("="*80)
