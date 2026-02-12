#!/usr/bin/env python3
"""
Test Traffic Light Logic
========================

Quick test to verify regime classifier integration
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.market_regime import RegimeClassifier

def test_regime_logic():
    """Test the regime classifier with different market states"""
    print("=" * 60)
    print("TESTING TRAFFIC LIGHT LOGIC")
    print("=" * 60)
    
    # Initialize regime classifier
    rc = RegimeClassifier()
    
    if not rc.load_model():
        print("ERROR: Could not load HMM model")
        return
    
    print("✓ HMM model loaded successfully")
    
    # Test different market scenarios
    test_scenarios = [
        # (return, vix, mmtw, expected_state)
        (0.01, 15.0, 0.7, "BULL"),      # Good returns, low vol, high breadth
        (0.00, 40.0, 0.4, "CHOP"),      # Flat returns, high vol, low breadth  
        (-0.05, 100.0, 0.3, "BEAR"),     # Bad returns, extreme vol, very low breadth
    ]
    
    print("\nTesting different market scenarios:")
    print("-" * 60)
    
    for i, (ret, vix, mmtw, expected) in enumerate(test_scenarios, 1):
        state = rc.predict_state((ret, vix, mmtw))
        state_name = rc.get_regime_name(state) if state is not None else "ERROR"
        
        # Determine traffic light action
        if state == 0:  # GREEN
            action = "NORMAL - Use standard threshold"
            threshold = "0.45"
        elif state == 1:  # YELLOW
            action = "SNIPER MODE - Use strict threshold"
            threshold = "0.60"
        elif state == 2:  # RED
            action = "FORCE CASH - Close all positions"
            threshold = "N/A"
        else:
            action = "ERROR - Default to normal"
            threshold = "0.45"
        
        print(f"\nScenario {i}:")
        print(f"  Input: Return={ret*100:.1f}%, VIX={vix:.1f}, MMTW={mmtw:.2f}")
        print(f"  Predicted: {state_name} (State {state})")
        print(f"  Expected: {expected}")
        print(f"  Action: {action}")
        print(f"  Threshold: {threshold}")
        
        # Verify prediction
        if state_name == expected:
            print(f"  ✓ CORRECT")
        else:
            print(f"  ✗ INCORRECT")
    
    print("\n" + "=" * 60)
    print("TRAFFIC LIGHT LOGIC TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_regime_logic()
