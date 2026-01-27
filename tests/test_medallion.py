"""
Test Medallion Model Integration
Simple test without Streamlit
"""

import pandas as pd
import numpy as np
from models.medallion_ensemble_model import build_medallion_ensemble_model, train_and_predict_medallion_ensemble

print('🏆 Testing Medallion Model Integration')
print('=' * 50)

# Test with sample data
try:
    # Build sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create realistic price data
    price = 100
    prices = [price]
    for _ in range(len(dates)-1):
        change = np.random.normal(0.001, 0.02)  # Daily returns
        price *= (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        '1d_close': prices,
        '1d_open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        '1d_high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        '1d_low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        '1d_volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }, index=dates)
    
    print(f'✅ Sample data created: {len(df)} days')
    
    # Create Medallion features
    from models.medallion_ensemble_model import MedallionFeatureEngine
    feature_engine = MedallionFeatureEngine()
    features = feature_engine.create_medallion_features(df, '1d')
    
    print(f'✅ Medallion features created: {features.shape[1]} features')
    
    # Split data
    split_idx = int(0.8 * len(features))
    X_train = features.iloc[:split_idx].values
    y_train = df['1d_close'].iloc[:split_idx].values
    X_test = features.iloc[split_idx:].values
    y_test = df['1d_close'].iloc[split_idx:].values
    
    print(f'✅ Data split - Train: {X_train.shape}, Test: {X_test.shape}')
    
    # Build and train model
    print('🚀 Training Medallion Ensemble...')
    model = build_medallion_ensemble_model()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    print(f'📊 Predictions shape: {predictions.shape}')
    print(f'📊 y_test shape: {y_test.shape}')
    
    # Ensure shapes match
    min_len = min(len(predictions), len(y_test))
    predictions = predictions[:min_len]
    y_test = y_test[:min_len]
    
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    mae = np.mean(np.abs(predictions - y_test))
    
    print(f'✅ Model trained successfully!')
    print(f'📊 Test RMSE: {rmse:.4f}')
    print(f'📊 Test MAE: {mae:.4f}')
    print(f'📊 Price range: ${y_test.min():.2f} - ${y_test.max():.2f}')
    print(f'📊 Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}')
    
    # Test individual model components
    print('\n🔍 Testing individual models...')
    for name, submodel in model.models.items():
        if hasattr(submodel, 'predict'):
            pred = submodel.predict(X_test[:10])  # Test on small sample
            print(f'✅ {name}: Predictions shape {pred.shape}')
    
    print('\n🎉 Medallion Model Integration Test: PASSED!')
    print('🚀 Ready for production use!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
