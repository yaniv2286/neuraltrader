"""
Simple ML Threshold Optimization Test
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def simple_ml_optimization():
    """Simple ML-based threshold optimization"""
    print("Simple ML Threshold Optimization")
    print("=" * 50)
    
    # Load 20 years of AAPL data
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'tiingo')
    aapl_file = os.path.join(cache_dir, 'AAPL_1d_20y.csv')
    
    if not os.path.exists(aapl_file):
        print(f"✗ Data file not found: {aapl_file}")
        return
    
    print(f"✓ Loading AAPL data...")
    df = pd.read_csv(aapl_file, index_col=0, parse_dates=True)
    print(f"  Data shape: {df.shape}")
    
    # Use last 10 years for faster training
    df_recent = df.iloc[-2520:]  # Last 10 years
    print(f"  Using last {len(df_recent)} days for training")
    
    # Create features
    print("Creating features...")
    features = pd.DataFrame(index=df_recent.index)
    
    # Basic features
    features['returns'] = df_recent['close'].pct_change()
    features['volatility_20d'] = features['returns'].rolling(20).std()
    features['sma_20'] = df_recent['close'].rolling(20).mean()
    features['sma_ratio'] = df_recent['close'] / features['sma_20']
    
    # RSI
    delta = df_recent['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    features['volume_ratio'] = df_recent['volume'] / df_recent['volume'].rolling(20).mean()
    
    # Lagged returns
    for lag in [1, 5, 10]:
        features[f'return_lag_{lag}d'] = features['returns'].shift(lag)
    
    # Target: future 5-day return
    features['target'] = features['returns'].shift(-5)
    
    # Remove NaN values
    clean_features = features.dropna()
    print(f"  Clean features shape: {clean_features.shape}")
    
    if len(clean_features) < 100:
        print("✗ Not enough clean data for training")
        return
    
    # Prepare training data
    feature_cols = [col for col in clean_features.columns if col != 'target']
    X = clean_features[feature_cols]
    y = clean_features['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(importance.head(10))
    
    # Optimize thresholds using ML predictions
    print("\nOptimizing thresholds...")
    
    # Get predictions for test set
    y_pred = rf.predict(X_test)
    
    # Calculate optimal thresholds based on predictions
    pred_returns = pd.Series(y_pred, index=X_test.index)
    
    # Grid search for optimal thresholds
    buy_thresholds = np.linspace(0.005, 0.03, 20)
    sell_thresholds = np.linspace(0.005, 0.03, 20)
    
    best_sharpe = -np.inf
    best_buy = 0.01
    best_sell = 0.01
    
    actual_returns = y_test
    
    for buy_thr in buy_thresholds:
        for sell_thr in sell_thresholds:
            # Generate signals
            signals = pd.Series('HOLD', index=X_test.index)
            signals.loc[pred_returns < -buy_thr] = 'BUY'
            signals.loc[pred_returns > sell_thr] = 'SELL'
            
            # Calculate strategy returns
            positions = signals.map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
            strategy_returns = positions.shift(1) * actual_returns
            
            # Calculate Sharpe ratio
            if strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_buy = buy_thr
                best_sell = sell_thr
    
    print(f"\n✓ Optimization Complete!")
    print(f"Best Buy Threshold: {best_buy:.4f}")
    print(f"Best Sell Threshold: {best_sell:.4f}")
    print(f"Best Sharpe Ratio: {best_sharpe:.3f}")
    
    # Calculate performance with optimized thresholds
    signals_opt = pd.Series('HOLD', index=X_test.index)
    signals_opt.loc[pred_returns < -best_buy] = 'BUY'
    signals_opt.loc[pred_returns > best_sell] = 'SELL'
    
    positions_opt = signals_opt.map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
    strategy_returns_opt = positions_opt.shift(1) * actual_returns
    
    total_return = (1 + strategy_returns_opt).prod() - 1
    final_sharpe = strategy_returns_opt.mean() / strategy_returns_opt.std() * np.sqrt(252) if strategy_returns_opt.std() > 0 else 0
    
    print(f"\nPerformance with Optimized Thresholds:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {final_sharpe:.3f}")
    print(f"  Total Trades: {len(positions_opt[positions_opt != 0])}")
    
    # Save model
    import joblib
    joblib.dump(rf, 'simple_ml_model.pkl')
    print(f"\n✓ Model saved to simple_ml_model.pkl")
    
    return {
        'best_buy_threshold': best_buy,
        'best_sell_threshold': best_sell,
        'best_sharpe': best_sharpe,
        'total_return': total_return,
        'final_sharpe': final_sharpe,
        'model_score': test_score,
        'feature_importance': importance
    }

if __name__ == "__main__":
    simple_ml_optimization()
