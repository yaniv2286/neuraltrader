#!/usr/bin/env python3
"""
NeuralTrader 2.0 - One-Click Daily Signals Runner
================================================

Quick execution script for daily trading signals.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from execution.live_ranker import LiveRanker

def main():
    """Run daily signals with one click."""
    print("🚀 NeuralTrader 2.0 - Daily Signals Generator")
    print("=" * 50)
    
    try:
        # Initialize and run live ranker
        ranker = LiveRanker()
        ranker.run()
        
        print("\n✅ Daily signals generated successfully!")
        print("📁 Check logs/daily_signals.txt for the report")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
