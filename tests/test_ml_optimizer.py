"""
Test ML Threshold Optimizer with 20-year data
"""

import os
import pandas as pd
import numpy as np
from ml.integrated_optimizer import IntegratedMLThresholdOptimizer, ThresholdConfig

def main():
    """Test ML optimizer with 20-year AAPL data"""
    print("Testing ML Threshold Optimizer")
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
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Use last 10 years for faster training
    df_recent = df.iloc[-2520:]  # Last 10 years
    print(f"  Using last {len(df_recent)} days for training")
    
    # Create simple features
    print("Creating features...")
    features = pd.DataFrame(index=df_recent.index)
    
    # Basic price features
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
    
    # Future returns (targets)
    features['future_return_5d'] = features['returns'].shift(-5)
    features['future_return_10d'] = features['returns'].shift(-10)
    
    # Remove NaN values
    clean_features = features.dropna()
    print(f"  Clean features shape: {clean_features.shape}")
    
    if len(clean_features) < 100:
        print("✗ Not enough clean data for training")
        return
    
    # Initialize optimizer
    print("\nInitializing ML optimizer...")
    optimizer = IntegratedMLThresholdOptimizer()
    
    # Train models
    print("Training models...")
    try:
        optimizer.ml_optimizer.train_models(clean_features, 'future_return_5d')
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Test optimization
    print("\nTesting optimization...")
    base_config = ThresholdConfig(0.01, 0.01, 0.95, 1.05, 1.0, 0.5, 0.0, 0.02)
    
    # Create mock signals and positions for testing
    signals = pd.Series('HOLD', index=clean_features.index)
    positions = pd.Series(0, index=clean_features.index)
    
    # Simple strategy: buy when return < -2%, sell when return > 2%
    signals.loc[clean_features['returns'] < -0.02] = 'BUY'
    signals.loc[clean_features['returns'] > 0.02] = 'SELL'
    positions.loc[signals == 'BUY'] = 1
    positions.loc[signals == 'SELL'] = -1
    
    # Test optimization
    try:
        result = optimizer.optimize_thresholds_ensemble(clean_features, base_config)
        
        print(f"\n✓ Optimization Successful!")
        print(f"Best Method: {result.optimization_method}")
        print(f"Buy Threshold: {result.best_config.buy_threshold:.4f}")
        print(f"Sell Threshold: {result.best_config.sell_threshold:.4f}")
        print(f"Position Size: {result.best_config.position_size:.2f}")
        print(f"Expected Return: {result.best_config.expected_return:.4f}")
        print(f"Risk Score: {result.best_config.risk_score:.4f}")
        print(f"Confidence: {result.confidence_score:.3f}")
        
        print(f"\nPerformance:")
        perf = result.ensemble_performance
        print(f"  Total Return: {perf.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
        
        # Save models
        optimizer.save_models()
        
        # Show threshold evolution
        print(f"\nThreshold Evolution:")
        print("Window | Buy Thr | Sell Thr | Method")
        print("-" * 40)
        print(f"Current | {result.best_config.buy_threshold:.4f} | {result.best_config.sell_threshold:.4f} | {result.optimization_method[:3].upper()}")
        
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
