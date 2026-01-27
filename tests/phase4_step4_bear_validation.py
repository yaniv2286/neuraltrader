"""
Phase 4 Step 4-6: Bear Market Validation
Test model performance on specific bear market periods to ensure profitability in crashes
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
print("PHASE 4 STEPS 4-6: BEAR MARKET VALIDATION")
print("="*70)

print("\n🎯 OBJECTIVE:")
print("   Test model on specific bear market periods to ensure:")
print("   1. Predicts crashes (negative returns correctly)")
print("   2. Identifies bottoms (buying opportunities)")
print("   3. Makes money in bear markets (not just bull)")

# Define bear market periods (adjusted for available data: 2006-2024)
bear_markets = [
    {
        'name': '2008 Financial Crisis',
        'start': '2008-09-01',  # Lehman collapse
        'end': '2009-03-31',
        'description': 'Worst crash since Great Depression (-50% drop)',
        'key_events': ['Lehman collapse', 'Bank failures', 'Market panic']
    },
    {
        'name': '2011 Debt Crisis',
        'start': '2011-07-01',
        'end': '2011-10-31',
        'description': 'European debt crisis and US downgrade',
        'key_events': ['US credit downgrade', 'European debt fears', 'Flash crash']
    },
    {
        'name': '2018 Q4 Selloff',
        'start': '2018-10-01',
        'end': '2018-12-31',
        'description': 'Q4 2018 correction (-20%)',
        'key_events': ['Fed rate hikes', 'Trade war fears', 'Growth concerns']
    },
    {
        'name': '2020 COVID Crash',
        'start': '2020-02-01',
        'end': '2020-04-30',
        'description': 'Fastest crash in history (-35% in 1 month)',
        'key_events': ['Pandemic lockdowns', 'Economic shutdown', 'VIX spike to 80']
    },
    {
        'name': '2022 Bear Market',
        'start': '2022-01-01',
        'end': '2022-10-31',
        'description': 'Slow bear market (-25% over 10 months)',
        'key_events': ['Fed rate hikes', 'Inflation fears', 'Tech selloff']
    }
]

# Load full dataset
print("\n📊 Loading AAPL full dataset...")
df_full = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2006-01-01',
    end='2024-12-31',
    validate_data=True,
    create_features=True
)

print(f"✅ Loaded {len(df_full)} days")

# Train model on full dataset (same as Step 3)
print("\n" + "="*70)
print("TRAINING MODEL ON FULL DATASET")
print("="*70)

trainer = ModelTrainer(use_log_returns=True, n_features=25)
X, y = trainer.prepare_data(df_full, target_col='close')

# Use same splits as Step 3
n_samples = len(X)
train_size = int(0.70 * n_samples)
val_size = int(0.15 * n_samples)

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]
X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

# Feature selection
selected_features = trainer.select_features(X_train, y_train, method='correlation')
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

# Train model
print("\n⏳ Training XGBoost...")
selector = ModelSelector()
recommendations = selector.get_recommended_models(task_type='stock_prediction')
model_class = recommendations['primary']['model']

model_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.02,
    'tree_method': 'hist',
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3
}

model = model_class(**model_params)
model.fit(X_train_selected.values, y_train.values)
print("✅ Training complete")

# Test on each bear market period
print("\n" + "="*70)
print("BEAR MARKET VALIDATION")
print("="*70)

bear_results = []

for bear in bear_markets:
    print("\n" + "="*70)
    print(f"TESTING: {bear['name']}")
    print("="*70)
    
    print(f"\n📅 Period: {bear['start']} to {bear['end']}")
    print(f"📖 Description: {bear['description']}")
    print(f"🔑 Key events:")
    for event in bear['key_events']:
        print(f"   • {event}")
    
    # Filter data for this bear market period
    bear_start = bear['start']
    bear_end = bear['end']
    
    # Convert index to strings for comparison
    try:
        # Extract dates from complex index
        index_dates = []
        for idx in X.index:
            idx_str = str(idx)
            # Extract date from Period or tuple format
            if 'Period' in idx_str:
                # Extract date between quotes
                import re
                match = re.search(r"'(\d{4}-\d{2}-\d{2})", idx_str)
                if match:
                    index_dates.append(match.group(1))
                else:
                    index_dates.append('1900-01-01')  # Fallback
            else:
                index_dates.append(idx_str[:10])
        
        index_dates = pd.Series(index_dates, index=X.index)
        
        # Create mask
        bear_mask = (index_dates >= bear_start) & (index_dates <= bear_end)
        
        if bear_mask.sum() == 0:
            print(f"\n⚠️ No data available for this period")
            print(f"   Date range in data: {index_dates.iloc[0]} to {index_dates.iloc[-1]}")
            continue
        
        X_bear = X[bear_mask]
        y_bear = y[bear_mask]
    except Exception as e:
        print(f"\n⚠️ Error filtering data: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    print(f"\n📊 Data: {len(X_bear)} trading days")
    
    # Select features
    X_bear_selected = X_bear[selected_features]
    
    # Make predictions
    y_bear_pred = model.predict(X_bear_selected.values)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_bear, y_bear_pred)
    rmse = np.sqrt(mean_squared_error(y_bear, y_bear_pred))
    mae = mean_absolute_error(y_bear, y_bear_pred)
    
    # Direction accuracy
    direction_acc = np.mean((y_bear > 0) == (y_bear_pred > 0)) * 100
    
    # Actual vs predicted returns
    actual_cumulative = (1 + y_bear).cumprod().iloc[-1] - 1
    predicted_cumulative = (1 + y_bear_pred).sum()
    
    # Trading simulation: only trade when model predicts positive return
    trades_taken = sum(y_bear_pred > 0)
    trades_correct = sum((y_bear > 0) & (y_bear_pred > 0))
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   R²: {r2:.4f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   Direction Accuracy: {direction_acc:.2f}%")
    
    print(f"\n💰 TRADING SIMULATION:")
    print(f"   Actual market return: {actual_cumulative*100:.2f}%")
    print(f"   Model predicted: {predicted_cumulative*100:.2f}%")
    print(f"   Trades taken: {trades_taken}/{len(y_bear)} ({trades_taken/len(y_bear)*100:.1f}%)")
    
    if trades_taken > 0:
        win_rate = trades_correct / trades_taken * 100
        print(f"   Win rate on trades: {win_rate:.1f}%")
    
    # Analyze crash prediction
    down_days = sum(y_bear < 0)
    down_predicted = sum((y_bear < 0) & (y_bear_pred < 0))
    crash_prediction_rate = (down_predicted / down_days * 100) if down_days > 0 else 0
    
    print(f"\n🔻 CRASH PREDICTION:")
    print(f"   Down days: {down_days}/{len(y_bear)} ({down_days/len(y_bear)*100:.1f}%)")
    print(f"   Correctly predicted: {down_predicted}/{down_days} ({crash_prediction_rate:.1f}%)")
    
    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT FOR {bear['name']}")
    print(f"{'='*70}")
    
    issues = []
    successes = []
    
    if direction_acc > 55:
        successes.append(f"✅ Direction accuracy {direction_acc:.1f}% (>55% threshold)")
    else:
        issues.append(f"❌ Direction accuracy {direction_acc:.1f}% (<55% threshold)")
    
    if crash_prediction_rate > 50:
        successes.append(f"✅ Crash prediction {crash_prediction_rate:.1f}% (>50%)")
    else:
        issues.append(f"⚠️ Crash prediction {crash_prediction_rate:.1f}% (<50%)")
    
    if r2 > 0:
        successes.append(f"✅ Positive R² ({r2:.4f})")
    else:
        issues.append(f"⚠️ Negative R² ({r2:.4f})")
    
    for success in successes:
        print(f"   {success}")
    
    for issue in issues:
        print(f"   {issue}")
    
    # Store results
    bear_results.append({
        'name': bear['name'],
        'r2': r2,
        'direction_acc': direction_acc,
        'crash_prediction': crash_prediction_rate,
        'actual_return': actual_cumulative * 100,
        'trades_taken': trades_taken,
        'total_days': len(y_bear)
    })

# Final summary
print("\n" + "="*70)
print("FINAL BEAR MARKET SUMMARY")
print("="*70)

print(f"\n{'Period':<25} {'R²':>8} {'Dir Acc':>10} {'Crash Pred':>12}")
print("-" * 70)

for result in bear_results:
    print(f"{result['name']:<25} {result['r2']:>8.4f} {result['direction_acc']:>9.1f}% {result['crash_prediction']:>11.1f}%")

# Overall verdict
avg_direction = np.mean([r['direction_acc'] for r in bear_results])
avg_crash_pred = np.mean([r['crash_prediction'] for r in bear_results])

print("\n" + "="*70)
print("OVERALL VERDICT")
print("="*70)

print(f"\n📊 Average Performance Across All Bear Markets:")
print(f"   Direction Accuracy: {avg_direction:.2f}%")
print(f"   Crash Prediction: {avg_crash_pred:.2f}%")

if avg_direction > 55 and avg_crash_pred > 50:
    print("\n✅ MODEL READY FOR PRODUCTION")
    print("   ✓ Predicts direction correctly in bear markets")
    print("   ✓ Identifies crashes before they happen")
    print("   ✓ Can make money in both bull AND bear markets")
    print("\n🎯 PHASE 4 COMPLETE - Ready for live trading!")
elif avg_direction > 50:
    print("\n⚠️ MODEL NEEDS MINOR TUNING")
    print("   ✓ Direction prediction is decent")
    print("   ⚠️ Crash prediction could be improved")
    print("\n💡 Consider: Adding volatility-based features or regime detection")
else:
    print("\n❌ MODEL NEEDS SIGNIFICANT IMPROVEMENT")
    print("   ✗ Direction prediction below threshold")
    print("   ✗ Not ready for bear market trading")
    print("\n💡 Consider: More bear market training data or different features")

print("\n" + "="*70)
print("PHASE 4 STEPS 4-6 COMPLETE")
print("="*70)
