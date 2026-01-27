"""
Phase 3 Step 1: Detailed Feature Analysis
Identify exact duplicates and highly correlated features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data.enhanced_preprocess import build_enhanced_model_input

print("="*70)
print("PHASE 3 STEP 1: DETAILED FEATURE ANALYSIS")
print("="*70)

# Load data with features
print("\n📊 Loading AAPL data with all features...")
df = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2020-01-01',
    end='2023-12-31',
    validate_data=True,
    create_features=True
)

print(f"✅ Loaded {len(df)} rows with {len(df.columns)} features")

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n📊 Analyzing {len(numeric_cols)} numeric features...")

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr().abs()

# Find highly correlated pairs
print("\n" + "="*70)
print("FINDING REDUNDANT FEATURES (correlation > 0.95)")
print("="*70)

high_corr_pairs = []
features_to_remove = set()
feature_groups = {}  # Group identical features together

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        
        if corr_val > 0.95:
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            high_corr_pairs.append((feat1, feat2, corr_val))
            
            # Group features by correlation
            if corr_val > 0.999:  # Nearly identical
                # Find or create group
                group_found = False
                for key in feature_groups:
                    if feat1 in feature_groups[key] or feat2 in feature_groups[key]:
                        feature_groups[key].add(feat1)
                        feature_groups[key].add(feat2)
                        group_found = True
                        break
                
                if not group_found:
                    feature_groups[f"group_{len(feature_groups)}"] = {feat1, feat2}

# Print feature groups (identical features)
print("\n🔍 IDENTICAL FEATURE GROUPS (correlation > 0.999):")
print("-" * 70)

for group_name, features in feature_groups.items():
    if len(features) > 1:
        features_list = sorted(list(features))
        print(f"\n{group_name.upper()}:")
        
        # Show correlation values
        for feat in features_list:
            print(f"  • {feat}")
        
        # Recommend which to keep
        # Keep the simplest name (shortest, no prefixes)
        keep = min(features_list, key=lambda x: (len(x), '_' in x, x))
        remove = [f for f in features_list if f != keep]
        
        print(f"  ✅ KEEP: {keep}")
        print(f"  ❌ REMOVE: {', '.join(remove)}")
        
        features_to_remove.update(remove)

# Print all high correlation pairs
print("\n" + "="*70)
print(f"ALL HIGH CORRELATION PAIRS (>0.95): {len(high_corr_pairs)} total")
print("="*70)

# Sort by correlation
high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

for feat1, feat2, corr in high_corr_pairs[:30]:  # Show top 30
    print(f"{feat1:25} ↔ {feat2:25} : {corr:.4f}")

if len(high_corr_pairs) > 30:
    print(f"\n... and {len(high_corr_pairs) - 30} more pairs")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total features: {len(numeric_cols)}")
print(f"Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
print(f"Features recommended for removal: {len(features_to_remove)}")

print("\n📋 COMPLETE LIST OF FEATURES TO REMOVE:")
print("-" * 70)
for i, feat in enumerate(sorted(features_to_remove), 1):
    print(f"{i:2}. {feat}")

# Save to file for reference
with open('tests/phase3_features_to_remove.txt', 'w') as f:
    f.write("Features to Remove (Redundant)\n")
    f.write("="*70 + "\n\n")
    for feat in sorted(features_to_remove):
        f.write(f"{feat}\n")

print(f"\n✅ Analysis complete. Results saved to: tests/phase3_features_to_remove.txt")
