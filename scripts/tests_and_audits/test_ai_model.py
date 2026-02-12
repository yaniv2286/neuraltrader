#!/usr/bin/env python3
"""
Quick Test - Verify AI Model is Working
======================================
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_models import EnsemblePredictor
from core.feature_engineer import FeatureEngineer
from scripts.data_manager import DataManager

# Load sample data
dm = DataManager()
ticker = "AAPL"
pattern = Path(dm.cache_dir) / f"{ticker}_sp100_emergency_*.csv"
import glob
files = glob.glob(str(pattern))

if files:
    df = pd.read_csv(files[0])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.tail(1000)  # Use last 1000 days
    
    print(f"Loaded {len(df)} rows for {ticker}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Create features
    fe = FeatureEngineer()
    features, _ = fe.create_features(df)
    print(f"Created {len(features.columns)} features")
    
    # Test predictions
    ensemble = EnsemblePredictor()
    
    # Test on a few samples
    for i in range(5):
        sample_features = features.iloc[i:i+1]
        signal, confidence, details = ensemble.predict(sample_features)
        print(f"Sample {i}: Signal={signal}, Confidence={confidence:.3f}")
        print(f"  Details: {details}")
    
    # Check signal distribution
    signals = []
    confidences = []
    for i in range(min(100, len(features))):
        sample_features = features.iloc[i:i+1]
        signal, confidence, _ = ensemble.predict(sample_features)
        signals.append(signal)
        confidences.append(confidence)
    
    print(f"\nSignal Distribution (last 100 predictions):")
    print(f"  BUY: {signals.count('BUY')} ({signals.count('BUY')/len(signals)*100:.1f}%)")
    print(f"  SELL: {signals.count('SELL')} ({signals.count('SELL')/len(signals)*100:.1f}%)")
    print(f"  HOLD: {signals.count('HOLD')} ({signals.count('HOLD')/len(signals)*100:.1f}%)")
    print(f"  Avg Confidence: {np.mean(confidences):.3f}")
    print(f"  Min/Max Confidence: {np.min(confidences):.3f}/{np.max(confidences):.3f}")
    
else:
    print(f"No data file found for {ticker}")
