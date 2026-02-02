#!/usr/bin/env python3
"""
Continuous Alpha Factory Runner
================================

Run the Alpha Factory in continuous mode to process new data as it's downloaded.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml.features.alpha_factory import AlphaFactory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run continuous Alpha Factory processing."""
    
    print("🚀 Starting Continuous Alpha Factory...")
    print("📁 Monitoring: data/raw/ for new .parquet files")
    print("⚙️  Processing: RSI(14), ATR(14), SMA Distances, Volume Z-Scores")
    print("🎯 Output: data/processed/master_feature_matrix.parquet")
    print("⏱️  Check Interval: 30 seconds")
    print("🛑 Press Ctrl+C to stop")
    print("="*60)
    
    # Initialize factory
    factory = AlphaFactory(raw_path="data/raw", processed_path="data/processed")
    
    # Run continuous processing
    try:
        feature_matrix = factory.run_continuous(
            max_tickers=None,  # Process all tickers
            check_interval=30  # Check every 30 seconds
        )
        
        if not feature_matrix.empty:
            print("\n🎉 Continuous processing completed!")
            print(f"Final matrix: {feature_matrix.shape}")
        
    except KeyboardInterrupt:
        print("\n🛑 Continuous processing stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
