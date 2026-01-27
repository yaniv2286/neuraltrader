"""
Phase 4: Generate Detailed Trade Log for Manual Verification
Shows every trade with date, prediction, actual outcome, and reasoning
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
print("PHASE 4: DETAILED TRADE LOG FOR MANUAL VERIFICATION")
print("="*70)

print("\n🎯 OBJECTIVE:")
print("   Generate complete trade-by-trade log showing:")
print("   1. Date and price")
print("   2. Model prediction (up/down)")
print("   3. Actual outcome (up/down)")
print("   4. Key indicators (RSI, MACD, momentum)")
print("   5. Trade result (correct/wrong)")

# Load full dataset
print("\n📊 Loading AAPL data (2020-2024 for detailed analysis)...")
df_full = build_enhanced_model_input(
    ticker='AAPL',
    timeframes=['1d'],
    start='2020-01-01',
    end='2024-12-31',
    validate_data=True,
    create_features=True
)

print(f"✅ Loaded {len(df_full)} days")

# Train model
print("\n⏳ Training model...")
trainer = ModelTrainer(use_log_returns=True, n_features=25)
X, y = trainer.prepare_data(df_full, target_col='close')

# Use 70/15/15 split
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

# Train
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
print("✅ Model trained")

# Generate predictions for test set
y_test_pred = model.predict(X_test_selected.values)

# Create detailed trade log
print("\n" + "="*70)
print("GENERATING DETAILED TRADE LOG")
print("="*70)

# Get original data for test period
test_indices = X_test.index
original_data = df_full.loc[test_indices]

# Create trade log DataFrame
trade_log = pd.DataFrame({
    'Date': [str(idx)[:10] if 'Period' not in str(idx) else str(idx).split("'")[1] for idx in test_indices],
    'Close_Price': original_data['close'].values,
    'Actual_Return': y_test.values,
    'Predicted_Return': y_test_pred,
    'Actual_Direction': ['UP' if r > 0 else 'DOWN' for r in y_test.values],
    'Predicted_Direction': ['UP' if r > 0 else 'DOWN' for r in y_test_pred],
    'Correct': [1 if (y_test.values[i] > 0) == (y_test_pred[i] > 0) else 0 for i in range(len(y_test))],
})

# Add key indicators
if 'rsi' in original_data.columns:
    trade_log['RSI'] = original_data['rsi'].values
if 'macd' in original_data.columns:
    trade_log['MACD'] = original_data['macd'].values
if 'returns' in original_data.columns:
    trade_log['Momentum'] = original_data['returns'].values
if 'volatility_20' in original_data.columns:
    trade_log['Volatility'] = original_data['volatility_20'].values

# Add trade decision
trade_log['Trade_Decision'] = ['BUY' if pred > 0.005 else 'SELL' if pred < -0.005 else 'HOLD' 
                                 for pred in y_test_pred]

# Add profit/loss (if we took the trade)
trade_log['PnL_Pct'] = [
    y_test.values[i] * 100 if trade_log['Trade_Decision'].iloc[i] == 'BUY' else
    -y_test.values[i] * 100 if trade_log['Trade_Decision'].iloc[i] == 'SELL' else
    0
    for i in range(len(trade_log))
]

# Calculate cumulative returns
trade_log['Cumulative_PnL'] = trade_log['PnL_Pct'].cumsum()

# Save to CSV
csv_filename = 'tests/phase4_trade_log.csv'
trade_log.to_csv(csv_filename, index=False)
print(f"\n✅ Trade log saved to: {csv_filename}")

# Display summary statistics
print("\n" + "="*70)
print("TRADE LOG SUMMARY")
print("="*70)

print(f"\nTotal Days: {len(trade_log)}")
print(f"Correct Predictions: {trade_log['Correct'].sum()} ({trade_log['Correct'].mean()*100:.2f}%)")
print(f"Wrong Predictions: {len(trade_log) - trade_log['Correct'].sum()} ({(1-trade_log['Correct'].mean())*100:.2f}%)")

print(f"\nTrade Decisions:")
print(f"  BUY signals: {(trade_log['Trade_Decision'] == 'BUY').sum()}")
print(f"  SELL signals: {(trade_log['Trade_Decision'] == 'SELL').sum()}")
print(f"  HOLD signals: {(trade_log['Trade_Decision'] == 'HOLD').sum()}")

print(f"\nProfit/Loss:")
print(f"  Total PnL: {trade_log['PnL_Pct'].sum():.2f}%")
print(f"  Average PnL per trade: {trade_log['PnL_Pct'].mean():.2f}%")
print(f"  Best trade: {trade_log['PnL_Pct'].max():.2f}%")
print(f"  Worst trade: {trade_log['PnL_Pct'].min():.2f}%")

# Show sample trades
print("\n" + "="*70)
print("SAMPLE TRADES (First 20 from test set)")
print("="*70)

print(f"\n{'Date':<12} {'Price':>8} {'Pred':>6} {'Actual':>6} {'Decision':>8} {'PnL%':>8} {'Correct':>8} {'RSI':>6}")
print("-" * 90)

for i in range(min(20, len(trade_log))):
    row = trade_log.iloc[i]
    correct_symbol = '✓' if row['Correct'] == 1 else '✗'
    rsi_val = f"{row['RSI']:.1f}" if 'RSI' in row and not pd.isna(row['RSI']) else 'N/A'
    
    print(f"{row['Date']:<12} ${row['Close_Price']:>7.2f} {row['Predicted_Direction']:>6} "
          f"{row['Actual_Direction']:>6} {row['Trade_Decision']:>8} {row['PnL_Pct']:>7.2f}% "
          f"{correct_symbol:>8} {rsi_val:>6}")

# Show best and worst trades
print("\n" + "="*70)
print("BEST TRADES (Top 10)")
print("="*70)

best_trades = trade_log.nlargest(10, 'PnL_Pct')
print(f"\n{'Date':<12} {'Price':>8} {'Decision':>8} {'PnL%':>8} {'RSI':>6} {'MACD':>8}")
print("-" * 70)

for _, row in best_trades.iterrows():
    rsi_val = f"{row['RSI']:.1f}" if 'RSI' in row and not pd.isna(row['RSI']) else 'N/A'
    macd_val = f"{row['MACD']:.4f}" if 'MACD' in row and not pd.isna(row['MACD']) else 'N/A'
    
    print(f"{row['Date']:<12} ${row['Close_Price']:>7.2f} {row['Trade_Decision']:>8} "
          f"{row['PnL_Pct']:>7.2f}% {rsi_val:>6} {macd_val:>8}")

print("\n" + "="*70)
print("WORST TRADES (Bottom 10)")
print("="*70)

worst_trades = trade_log.nsmallest(10, 'PnL_Pct')
print(f"\n{'Date':<12} {'Price':>8} {'Decision':>8} {'PnL%':>8} {'RSI':>6} {'MACD':>8}")
print("-" * 70)

for _, row in worst_trades.iterrows():
    rsi_val = f"{row['RSI']:.1f}" if 'RSI' in row and not pd.isna(row['RSI']) else 'N/A'
    macd_val = f"{row['MACD']:.4f}" if 'MACD' in row and not pd.isna(row['MACD']) else 'N/A'
    
    print(f"{row['Date']:<12} ${row['Close_Price']:>7.2f} {row['Trade_Decision']:>8} "
          f"{row['PnL_Pct']:>7.2f}% {rsi_val:>6} {macd_val:>8}")

# Monthly breakdown
print("\n" + "="*70)
print("MONTHLY PERFORMANCE")
print("="*70)

trade_log['Month'] = pd.to_datetime(trade_log['Date']).dt.to_period('M')
monthly = trade_log.groupby('Month').agg({
    'PnL_Pct': 'sum',
    'Correct': 'mean',
    'Trade_Decision': 'count'
})

print(f"\n{'Month':<12} {'Total PnL%':>12} {'Accuracy%':>12} {'Trades':>8}")
print("-" * 50)

for month, row in monthly.iterrows():
    print(f"{str(month):<12} {row['PnL_Pct']:>11.2f}% {row['Correct']*100:>11.1f}% {row['Trade_Decision']:>8}")

print("\n" + "="*70)
print("TRADE LOG COMPLETE")
print("="*70)

print(f"\n📊 Full trade log available in: {csv_filename}")
print(f"   Open in Excel/Google Sheets for detailed analysis")
print(f"\n✅ You can now manually verify each trade!")
print(f"   - Check dates and prices")
print(f"   - Verify predictions vs actual outcomes")
print(f"   - Review indicator values (RSI, MACD, etc.)")
print(f"   - Analyze winning and losing trades")
