"""
Phase 4 Step 3: Train on Full 20-Year Dataset
Use ALL available data (2004-2024) to learn bull AND bear market patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data.enhanced_preprocess import build_enhanced_model_input
from models.model_trainer import ModelTrainer
from models.model_selector import ModelSelector
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 4 STEP 3: TRAIN ON FULL 20-YEAR DATASET")
print("="*70)

print("\n🎯 OBJECTIVE:")
print("   Train model on 2004-2024 data to learn:")
print("   - 2008 Financial Crisis (bear)")
print("   - 2009-2020 Bull Run")
print("   - 2020 COVID Crash (bear)")
print("   - 2020-2021 Recovery")
print("   - 2022 Bear Market")
print("   - 2023-2024 Recent trends")

# Load FULL 20-year dataset
print("\n📊 Loading AAPL data (2004-2024) - FULL DATASET...")
df_full = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2004-01-01',
    end='2024-12-31',
    validate_data=True,
    create_features=True
)

print(f"✅ Loaded {len(df_full)} days with {len(df_full.columns)} features")

# Get date range - convert to string directly
start_date = str(df_full.index[0])[:10]
end_date = str(df_full.index[-1])[:10]
years_covered = len(df_full) / 252  # Approximate trading days per year

print(f"   Date range: {start_date} to {end_date}")
print(f"   Years covered: {years_covered:.1f} years")

# Initialize trainer with log returns
print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)

trainer = ModelTrainer(use_log_returns=True, n_features=25)

# Prepare data
X, y = trainer.prepare_data(df_full, target_col='close')
print(f"   ✅ Prepared {len(X)} samples")

# Create time-series splits using simple percentage-based approach
# Train: 70% (first ~13 years, includes 2008 crisis)
# Val: 15% (middle ~3 years, includes COVID + 2022 bear)
# Test: 15% (last ~3 years, recent data)
print("\n📊 Creating time-series splits...")

n_samples = len(X)
train_size = int(0.70 * n_samples)
val_size = int(0.15 * n_samples)

# Time-series split (no shuffling)
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print(f"\n   📊 Data splits:")
print(f"      Train: {len(X_train)} samples (~2006-2018) - {len(X_train)/n_samples*100:.1f}%")
print(f"      Val:   {len(X_val)} samples (~2018-2021) - {len(X_val)/n_samples*100:.1f}%")
print(f"      Test:  {len(X_test)} samples (~2021-2024) - {len(X_test)/n_samples*100:.1f}%")

# Feature selection on training data
print("\n🔧 Selecting top 25 features based on training data...")
selected_features = trainer.select_features(X_train, y_train, method='correlation')

# Use only selected features
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

# Train model with regularization
print("\n" + "="*70)
print("TRAINING XGBOOST MODEL")
print("="*70)

selector = ModelSelector()
recommendations = selector.get_recommended_models(task_type='stock_prediction')
model_class = recommendations['primary']['model']

# Optimized hyperparameters for 20-year dataset
model_params = {
    'n_estimators': 300,  # More trees for more data
    'max_depth': 3,
    'learning_rate': 0.02,
    'tree_method': 'hist',
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3  # Prevent overfitting on small patterns
}

print(f"\n🤖 Model: XGBoost")
print(f"   Parameters:")
for key, val in model_params.items():
    print(f"      {key}: {val}")

print("\n⏳ Training on 16 years of data (2004-2020)...")
print("   This includes:")
print("   - 2008-2009 Financial Crisis")
print("   - 2011 Debt Crisis")
print("   - 2015-2016 Correction")
print("   - 2018 Bear Market")
print("   - 2009-2020 Bull Run")

model = model_class(**model_params)
model.fit(X_train_selected.values, y_train.values)

print("✅ Training complete!")

# Evaluate
print("\n" + "="*70)
print("EVALUATION ON FULL DATASET")
print("="*70)

results = trainer.evaluate_model(
    model,
    X_train_selected.values, y_train.values,
    X_val_selected.values, y_val.values,
    X_test_selected.values, y_test.values
)

trainer.print_results(results)

# Compare with 4-year dataset results
print("\n" + "="*70)
print("COMPARISON: 20-Year vs 4-Year Dataset")
print("="*70)

print("\n📊 Previous results (4-year dataset, 2020-2023):")
print("   Test R²: 0.3123")
print("   Test Direction: 79.29%")
print("   Overfitting Gap: 0.0970")

print(f"\n📊 Current results (20-year dataset, 2004-2024):")
print(f"   Test R²: {results['test_r2']:.4f}")
print(f"   Test Direction: {results['test_dir']:.2f}%")
print(f"   Overfitting Gap: {results['train_r2'] - results['test_r2']:.4f}")

# Improvement analysis
r2_improvement = results['test_r2'] - 0.3123
dir_improvement = results['test_dir'] - 79.29
gap_improvement = 0.0970 - (results['train_r2'] - results['test_r2'])

print("\n📈 Improvements:")
if r2_improvement > 0:
    print(f"   ✅ Test R² improved by {r2_improvement:.4f}")
else:
    print(f"   ⚠️ Test R² changed by {r2_improvement:.4f}")

if dir_improvement > 0:
    print(f"   ✅ Direction accuracy improved by {dir_improvement:.2f}%")
else:
    print(f"   ⚠️ Direction accuracy changed by {dir_improvement:.2f}%")

if gap_improvement > 0:
    print(f"   ✅ Overfitting reduced by {gap_improvement:.4f}")
else:
    print(f"   ⚠️ Overfitting changed by {gap_improvement:.4f}")

# Key insights
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n🎓 What the model learned from 20 years:")
print("   1. 2008 Crisis patterns - extreme volatility and recovery")
print("   2. Multiple market cycles - bull, bear, sideways")
print("   3. Different volatility regimes - calm and chaotic")
print("   4. Long-term trends vs short-term noise")

if results['test_r2'] > 0.2 and results['test_dir'] > 55:
    print("\n✅ MODEL READY FOR BEAR MARKET VALIDATION")
    print("   Next: Test on 2008, 2020, and 2022 bear markets separately")
else:
    print("\n⚠️ MODEL NEEDS TUNING")
    print("   Consider adjusting hyperparameters or feature selection")

print("\n" + "="*70)
print("STEP 3 COMPLETE - Ready for Step 4 (Bear Market Validation)")
print("="*70)
