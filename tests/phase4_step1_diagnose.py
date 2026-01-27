"""
Phase 4 Step 1: Diagnose Current Model Performance
Identify exactly why models are failing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data.enhanced_preprocess import build_enhanced_model_input
from models import create_optimal_model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 4 STEP 1: DIAGNOSE MODEL PERFORMANCE ISSUES")
print("="*70)

# Load data with optimized features
print("\n📊 Loading AAPL data (2020-2023) with optimized features...")
df = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2020-01-01',
    end='2023-12-31',
    validate_data=True,
    create_features=True
)

print(f"✅ Loaded {len(df)} days with {len(df.columns)} features")

# Prepare data for modeling
print("\n" + "="*70)
print("CURRENT APPROACH: Predicting Raw Prices")
print("="*70)

# Remove non-numeric and target columns
feature_cols = [col for col in df.columns if col not in ['close', 'low', 'volume', 'market_regime']]
feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]

X = df[feature_cols].fillna(0)
y = df['close']  # Predicting raw close price

print(f"\n📊 Features: {len(feature_cols)}")
print(f"📊 Target: close price (raw)")
print(f"📊 Samples: {len(X)}")

# Split data: 60% train, 20% val, 20% test
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print(f"\n📊 Train: {len(X_train)} samples ({train_size/len(X)*100:.1f}%)")
print(f"📊 Val: {len(X_val)} samples ({val_size/len(X)*100:.1f}%)")
print(f"📊 Test: {len(X_test)} samples ({(len(X)-train_size-val_size)/len(X)*100:.1f}%)")

# Train XGBoost model
print("\n" + "="*70)
print("TRAINING XGBOOST MODEL (Current Approach)")
print("="*70)

from models.model_selector import ModelSelector
selector = ModelSelector()
recommendations = selector.get_recommended_models(task_type='stock_prediction')
model_class = recommendations['primary']['model']
model_params = recommendations['primary']['params']
model = model_class(**model_params)
print(f"\n🤖 Model: {model.model_name}")

print("\n⏳ Training...")
model.fit(X_train.values, y_train.values)

# Evaluate
print("\n📊 EVALUATION RESULTS:")
print("-" * 70)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Train performance
y_train_pred = model.predict(X_train.values)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Val performance
y_val_pred = model.predict(X_val.values)
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Test performance
y_test_pred = model.predict(X_test.values)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Directional accuracy
train_dir = np.mean((np.diff(y_train) > 0) == (np.diff(y_train_pred) > 0)) * 100
val_dir = np.mean((np.diff(y_val) > 0) == (np.diff(y_val_pred) > 0)) * 100
test_dir = np.mean((np.diff(y_test) > 0) == (np.diff(y_test_pred) > 0)) * 100

print(f"\nTRAIN Performance:")
print(f"  R²: {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  Directional Accuracy: {train_dir:.2f}%")

print(f"\nVAL Performance:")
print(f"  R²: {val_r2:.4f}")
print(f"  RMSE: ${val_rmse:.2f}")
print(f"  Directional Accuracy: {val_dir:.2f}%")

print(f"\nTEST Performance:")
print(f"  R²: {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  Directional Accuracy: {test_dir:.2f}%")

# Diagnose issues
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

issues = []

if train_r2 > 0.95:
    issues.append("🔴 SEVERE OVERFITTING: Train R² > 0.95 (memorizing training data)")

if test_r2 < 0:
    issues.append("🔴 CRITICAL: Test R² < 0 (worse than predicting mean)")

if train_r2 - test_r2 > 0.3:
    issues.append(f"🔴 GENERALIZATION GAP: {train_r2 - test_r2:.2f} difference between train and test")

if test_dir < 55:
    issues.append(f"🔴 POOR DIRECTION: {test_dir:.1f}% accuracy (barely better than random)")

if issues:
    print("\n❌ CRITICAL ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ No critical issues found")

# Root cause analysis
print("\n" + "="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

print("\n🔍 Why models fail:")
print("\n1. ❌ PREDICTING RAW PRICES (non-stationary)")
print("   - Prices trend upward over time")
print("   - Model memorizes the trend, not the pattern")
print("   - Fails when trend changes")

print("\n2. ❌ TOO MANY FEATURES (53 features)")
print("   - Many features are noise")
print("   - Model overfits to noise")
print("   - Need feature selection")

print("\n3. ❌ LIMITED TRAINING DATA (only 2020-2023)")
print("   - Only 4 years of data")
print("   - Missing 2008 crisis, other bear markets")
print("   - Can't learn bear market patterns")

print("\n4. ❌ NO REGULARIZATION")
print("   - Model too complex for data")
print("   - Needs stronger regularization")

# Solution preview
print("\n" + "="*70)
print("SOLUTION: What We'll Fix in Phase 4")
print("="*70)

print("\n✅ STEP 2: Switch to LOG RETURNS")
print("   - Predict returns instead of prices")
print("   - Makes data stationary")
print("   - Model learns patterns, not trends")

print("\n✅ STEP 3: FEATURE SELECTION")
print("   - Select top 20-30 most important features")
print("   - Remove noise")
print("   - Reduce overfitting")

print("\n✅ STEP 4: FULL 20-YEAR DATASET")
print("   - Train on 2004-2020 (includes 2008 crisis)")
print("   - Validate on 2020-2022 (includes COVID + bear)")
print("   - Test on 2023-2024")

print("\n✅ STEP 5: ADD REGULARIZATION")
print("   - Increase XGBoost regularization (alpha, lambda)")
print("   - Reduce max_depth")
print("   - Add early stopping")

print("\n✅ STEP 6: VALIDATE ON BULL/BEAR SEPARATELY")
print("   - Test on 2008-2009 (bear)")
print("   - Test on 2020 Q1 (bear)")
print("   - Test on 2022 (bear)")
print("   - Ensure profitability in BOTH conditions")

print("\n" + "="*70)
print("READY TO PROCEED TO STEP 2")
print("="*70)
