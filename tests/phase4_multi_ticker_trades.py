"""
Phase 4: Multi-Ticker Trade Log
Shows trades across multiple stocks with ticker identification
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
print("MULTI-TICKER TRADE LOG")
print("="*70)

print("\n🎯 OBJECTIVE:")
print("   Generate trade log for MULTIPLE stocks showing:")
print("   1. Ticker symbol for each trade")
print("   2. Individual stock performance")
print("   3. Portfolio-level aggregation")

# Test on multiple tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

print(f"\n📊 Testing on {len(tickers)} stocks: {', '.join(tickers)}")

# Store all trades
all_trades = []

# Train model for each ticker
for ticker in tickers:
    print(f"\n{'='*70}")
    print(f"Processing: {ticker}")
    print(f"{'='*70}")
    
    try:
        # Load data
        df = build_enhanced_model_input(
            ticker=ticker,
            timeframes=['1d'],
            start='2023-01-01',
            end='2024-12-31',
            validate_data=True,
            create_features=True
        )
        
        print(f"✅ Loaded {len(df)} days for {ticker}")
        
        # Train model
        trainer = ModelTrainer(use_log_returns=True, n_features=25)
        X, y = trainer.prepare_data(df, target_col='close')
        
        # Split
        n_samples = len(X)
        train_size = int(0.70 * n_samples)
        val_size = int(0.15 * n_samples)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        # Feature selection
        selected_features = trainer.select_features(X_train, y_train, method='correlation')
        X_train_selected = X_train[selected_features]
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
        
        # Predict
        y_test_pred = model.predict(X_test_selected.values)
        
        # Get original data
        test_indices = X_test.index
        original_data = df.loc[test_indices]
        
        # Create trades for this ticker
        for i in range(len(y_test)):
            idx = test_indices[i]
            date_str = str(idx)[:10] if 'Period' not in str(idx) else str(idx).split("'")[1]
            
            actual_return = y_test.iloc[i]
            pred_return = y_test_pred[i]
            
            trade = {
                'Ticker': ticker,
                'Date': date_str,
                'Close_Price': original_data['close'].iloc[i],
                'Actual_Return': actual_return,
                'Predicted_Return': pred_return,
                'Actual_Direction': 'UP' if actual_return > 0 else 'DOWN',
                'Predicted_Direction': 'UP' if pred_return > 0 else 'DOWN',
                'Correct': 1 if (actual_return > 0) == (pred_return > 0) else 0,
                'Trade_Decision': 'BUY' if pred_return > 0.005 else 'SELL' if pred_return < -0.005 else 'HOLD',
            }
            
            # Calculate PnL
            if trade['Trade_Decision'] == 'BUY':
                trade['PnL_Pct'] = actual_return * 100
            elif trade['Trade_Decision'] == 'SELL':
                trade['PnL_Pct'] = -actual_return * 100
            else:
                trade['PnL_Pct'] = 0
            
            all_trades.append(trade)
        
        accuracy = sum(1 for t in all_trades if t['Ticker'] == ticker and t['Correct'] == 1) / len([t for t in all_trades if t['Ticker'] == ticker]) * 100
        print(f"✅ {ticker}: {accuracy:.1f}% accuracy, {len([t for t in all_trades if t['Ticker'] == ticker])} trades")
        
    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")
        continue

# Create DataFrame
trade_log = pd.DataFrame(all_trades)

# Save to CSV
csv_filename = 'tests/phase4_multi_ticker_trades.csv'
trade_log.to_csv(csv_filename, index=False)

print(f"\n{'='*70}")
print("MULTI-TICKER SUMMARY")
print(f"{'='*70}")

print(f"\n📊 Total Trades: {len(trade_log)}")
print(f"📊 Tickers: {len(trade_log['Ticker'].unique())}")

# Per-ticker summary
print(f"\n{'Ticker':<8} {'Trades':>8} {'Accuracy':>10} {'Total PnL':>12} {'Avg PnL':>10}")
print("-" * 60)

for ticker in trade_log['Ticker'].unique():
    ticker_trades = trade_log[trade_log['Ticker'] == ticker]
    accuracy = ticker_trades['Correct'].mean() * 100
    total_pnl = ticker_trades['PnL_Pct'].sum()
    avg_pnl = ticker_trades['PnL_Pct'].mean()
    
    print(f"{ticker:<8} {len(ticker_trades):>8} {accuracy:>9.1f}% {total_pnl:>11.2f}% {avg_pnl:>9.2f}%")

# Overall summary
print(f"\n{'='*70}")
print("PORTFOLIO SUMMARY")
print(f"{'='*70}")

print(f"\nOverall Accuracy: {trade_log['Correct'].mean()*100:.2f}%")
print(f"Total Portfolio PnL: {trade_log['PnL_Pct'].sum():.2f}%")
print(f"Average PnL per Trade: {trade_log['PnL_Pct'].mean():.2f}%")

# Show sample trades from different tickers
print(f"\n{'='*70}")
print("SAMPLE TRADES (First 5 from each ticker)")
print(f"{'='*70}")

print(f"\n{'Ticker':<8} {'Date':<12} {'Price':>10} {'Pred':>6} {'Actual':>6} {'Decision':>8} {'PnL%':>8} {'✓/✗':>4}")
print("-" * 80)

for ticker in trade_log['Ticker'].unique():
    ticker_trades = trade_log[trade_log['Ticker'] == ticker].head(5)
    for _, row in ticker_trades.iterrows():
        correct_symbol = '✓' if row['Correct'] == 1 else '✗'
        print(f"{row['Ticker']:<8} {row['Date']:<12} ${row['Close_Price']:>9.2f} "
              f"{row['Predicted_Direction']:>6} {row['Actual_Direction']:>6} "
              f"{row['Trade_Decision']:>8} {row['PnL_Pct']:>7.2f}% {correct_symbol:>4}")

print(f"\n{'='*70}")
print(f"✅ Multi-ticker trade log saved to: {csv_filename}")
print(f"{'='*70}")

print(f"\n📊 Now you can see:")
print(f"   - Which stock each trade is for (Ticker column)")
print(f"   - Performance across multiple stocks")
print(f"   - Portfolio-level aggregation")
print(f"   - Compare AAPL vs MSFT vs GOOGL, etc.")
