"""
NeuralTrader Demo System
Demonstrates the AI trading system with synthetic data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_synthetic_stock_data(ticker: str, days: int = 252):
    """Create synthetic stock data for testing"""
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)
    
    # Starting price
    initial_price = 100.0
    
    # Generate returns with some trend and volatility
    trend = np.random.normal(0.0001, 0.001, days)  # Small upward trend
    volatility = np.random.normal(0, 0.02, days)      # Daily volatility
    
    # Calculate prices
    returns = trend + volatility
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate intraday variation
        intraday_vol = close * 0.02 * np.random.randn()
        
        open_price = close * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def demo_neuraltrader():
    """Demonstrate NeuralTrader with synthetic data"""
    
    print("🧠 NeuralTrader Demo System")
    print("=" * 50)
    print("📊 Using synthetic data for demonstration")
    print("=" * 50)
    
    # Test tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        # Import NeuralTrader components
        from models import create_optimal_model, print_model_status
        from data.enhanced_preprocess import apply_basic_indicators, create_sequences
        
        print("✅ NeuralTrader components imported successfully")
        
        # Show model status
        print_model_status()
        
        # Process each ticker
        results = {}
        
        for ticker in tickers:
            print(f"\n📈 Processing {ticker}...")
            
            # Create synthetic data
            df = create_synthetic_stock_data(ticker, days=252)
            print(f"   ✅ Generated synthetic data: {df.shape}")
            
            # Add basic features
            df_features = apply_basic_indicators(df)
            print(f"   ✅ Added features: {len(df_features.columns)} columns")
            
            # Create sequences
            target_col = 'close'
            X, y = create_sequences(df_features, target_col=target_col, sequence_length=30)
            
            if len(X) > 0:
                print(f"   ✅ Created sequences: X={X.shape}, y={y.shape}")
                
                # Flatten sequences for traditional ML models
                X_flat = X.reshape(X.shape[0], -1)  # (samples, timesteps * features)
                print(f"   ✅ Flattened for ML: X={X_flat.shape}")
                
                # Create and train model
                model = create_optimal_model('stock_prediction', 'primary')
                
                # Train on subset
                train_size = min(100, len(X_flat))
                model.fit(X_flat[:train_size], y[:train_size])
                
                # Test predictions
                test_size = min(20, len(X_flat) - train_size)
                if test_size > 0:
                    predictions = model.predict(X_flat[train_size:train_size+test_size])
                    
                    # Evaluate
                    metrics = model.evaluate(X_flat[train_size:train_size+test_size], y[train_size:train_size+test_size])
                    
                    results[ticker] = {
                        'model_type': type(model).__name__,
                        'train_samples': train_size,
                        'test_samples': test_size,
                        'r2_score': metrics['r2'],
                        'rmse': metrics['rmse'],
                        'directional_accuracy': metrics['directional_accuracy']
                    }
                    
                    print(f"   ✅ Model: {results[ticker]['model_type']}")
                    print(f"   ✅ R² Score: {results[ticker]['r2_score']:.4f}")
                    print(f"   ✅ RMSE: {results[ticker]['rmse']:.4f}")
                    print(f"   ✅ Directional Accuracy: {results[ticker]['directional_accuracy']:.2%}")
            else:
                print(f"   ❌ Not enough data for sequences")
        
        # Summary
        print(f"\n📊 DEMO RESULTS SUMMARY")
        print("=" * 50)
        
        if results:
            avg_r2 = np.mean([r['r2_score'] for r in results.values()])
            avg_rmse = np.mean([r['rmse'] for r in results.values()])
            avg_dir_acc = np.mean([r['directional_accuracy'] for r in results.values()])
            
            print(f"📈 Processed Tickers: {len(results)}")
            print(f"📊 Average R² Score: {avg_r2:.4f}")
            print(f"📊 Average RMSE: {avg_rmse:.4f}")
            print(f"📊 Average Directional Accuracy: {avg_dir_acc:.2%}")
            
            print(f"\n🎉 NeuralTrader Demo Completed Successfully!")
            print(f"🚀 AI models working correctly with synthetic data")
            
        else:
            print(f"❌ No results generated")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print(f"🚀 Starting NeuralTrader Demo...")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    results = demo_neuraltrader()
    
    print(f"\n⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}")
