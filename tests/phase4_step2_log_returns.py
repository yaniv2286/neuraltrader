"""
Phase 4 Step 2: Test Log Returns Approach
Compare raw prices vs log returns performance
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
print("PHASE 4 STEP 2: LOG RETURNS APPROACH")
print("="*70)

# Load data
print("\n📊 Loading AAPL data (2020-2023)...")
df = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2020-01-01',
    end='2023-12-31',
    validate_data=True,
    create_features=True
)

print(f"✅ Loaded {len(df)} days with {len(df.columns)} features")

# Initialize trainer with log returns
print("\n" + "="*70)
print("APPROACH 1: LOG RETURNS (Recommended)")
print("="*70)

trainer = ModelTrainer(use_log_returns=True, n_features=25)

# Prepare data
X, y = trainer.prepare_data(df, target_col='close')
print(f"   ✅ Prepared {len(X)} samples")

# Create splits
X_train, X_val, X_test, y_train, y_val, y_test = trainer.create_time_series_splits(X, y)

# Feature selection
selected_features = trainer.select_features(X_train, y_train, method='correlation')

# Use only selected features
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

# Train model
print("\n⏳ Training XGBoost with log returns...")
selector = ModelSelector()
recommendations = selector.get_recommended_models(task_type='stock_prediction')
model_class = recommendations['primary']['model']

# Add stronger regularization
model_params = {
    'n_estimators': 200,
    'max_depth': 3,  # Reduced from 4
    'learning_rate': 0.02,  # Reduced from 0.03
    'tree_method': 'hist',
    'reg_alpha': 0.5,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'subsample': 0.8,  # Use 80% of data per tree
    'colsample_bytree': 0.8  # Use 80% of features per tree
}

model = model_class(**model_params)
model.fit(X_train_selected.values, y_train.values)

# Evaluate
results = trainer.evaluate_model(
    model,
    X_train_selected.values, y_train.values,
    X_val_selected.values, y_val.values,
    X_test_selected.values, y_test.values
)

trainer.print_results(results)

# Compare with raw prices approach
print("\n" + "="*70)
print("COMPARISON: RAW PRICES (Old Approach)")
print("="*70)

trainer_raw = ModelTrainer(use_log_returns=False, n_features=25)
X_raw, y_raw = trainer_raw.prepare_data(df, target_col='close')
X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = trainer_raw.create_time_series_splits(X_raw, y_raw)

selected_features_raw = trainer_raw.select_features(X_train_raw, y_train_raw, method='correlation')
X_train_raw_sel = X_train_raw[selected_features_raw]
X_val_raw_sel = X_val_raw[selected_features_raw]
X_test_raw_sel = X_test_raw[selected_features_raw]

print("\n⏳ Training XGBoost with raw prices...")
model_raw = model_class(**model_params)
model_raw.fit(X_train_raw_sel.values, y_train_raw.values)

results_raw = trainer_raw.evaluate_model(
    model_raw,
    X_train_raw_sel.values, y_train_raw.values,
    X_val_raw_sel.values, y_val_raw.values,
    X_test_raw_sel.values, y_test_raw.values
)

trainer_raw.print_results(results_raw)

# Final comparison
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

print(f"\n{'Metric':<25} {'Log Returns':>15} {'Raw Prices':>15} {'Winner':>15}")
print("-" * 70)

metrics = [
    ('Test R²', results['test_r2'], results_raw['test_r2'], 'higher'),
    ('Test Direction Acc', results['test_dir'], results_raw['test_dir'], 'higher'),
    ('Overfitting Gap', results['train_r2'] - results['test_r2'], results_raw['train_r2'] - results_raw['test_r2'], 'lower')
]

for metric_name, log_val, raw_val, better in metrics:
    if better == 'higher':
        winner = 'Log Returns' if log_val > raw_val else 'Raw Prices'
        symbol = '✅' if log_val > raw_val else '❌'
    else:
        winner = 'Log Returns' if log_val < raw_val else 'Raw Prices'
        symbol = '✅' if log_val < raw_val else '❌'
    
    print(f"{metric_name:<25} {log_val:>15.4f} {raw_val:>15.4f} {symbol} {winner:>12}")

print("\n" + "="*70)
if results['test_r2'] > results_raw['test_r2'] and results['test_dir'] > results_raw['test_dir']:
    print("✅ LOG RETURNS APPROACH IS SUPERIOR")
    print("   - Better generalization")
    print("   - More stable predictions")
    print("   - Ready for production")
else:
    print("⚠️ RESULTS INCONCLUSIVE - May need more tuning")

print("="*70)
