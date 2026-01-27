"""
Phase 3 Step 4: Verify Feature Optimization
Test that redundant features are removed correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data.enhanced_preprocess import build_enhanced_model_input

print("="*70)
print("PHASE 3 STEP 4: VERIFY FEATURE OPTIMIZATION")
print("="*70)

# Load data with features
print("\n📊 Loading AAPL data with optimized features...")
df = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2020-01-01',
    end='2023-12-31',
    validate_data=True,
    create_features=True
)

print(f"\n✅ Loaded {len(df)} rows with {len(df.columns)} features")

# Check which features were removed
from features.feature_selector import FeatureSelector
redundant_features = FeatureSelector.REDUNDANT_FEATURES

print("\n" + "="*70)
print("VERIFICATION: Redundant Features Removed")
print("="*70)

removed_count = 0
kept_count = 0

for feat in redundant_features:
    if feat in df.columns:
        print(f"❌ STILL PRESENT: {feat}")
        kept_count += 1
    else:
        print(f"✅ REMOVED: {feat}")
        removed_count += 1

print(f"\n📊 Summary:")
print(f"   ✅ Successfully removed: {removed_count}/{len(redundant_features)}")
print(f"   ❌ Still present: {kept_count}/{len(redundant_features)}")

# Check for highly correlated features
print("\n" + "="*70)
print("CORRELATION CHECK: Remaining Features")
print("="*70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr().abs()

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"\nRemaining high correlation pairs (>0.95): {len(high_corr_pairs)}")

if high_corr_pairs:
    print("\nTop 10 remaining high correlations:")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {feat1:20} ↔ {feat2:20} : {corr:.4f}")
else:
    print("✅ No highly correlated pairs remaining!")

# Feature count comparison
print("\n" + "="*70)
print("FEATURE COUNT COMPARISON")
print("="*70)
print(f"Before optimization: ~63 features")
print(f"After optimization:  {len(df.columns)} features")
print(f"Reduction: {63 - len(df.columns)} features removed")

# List all remaining features
print("\n" + "="*70)
print(f"ALL REMAINING FEATURES ({len(df.columns)} total)")
print("="*70)

for i, col in enumerate(sorted(df.columns), 1):
    print(f"{i:2}. {col}")

print("\n✅ Verification complete!")
