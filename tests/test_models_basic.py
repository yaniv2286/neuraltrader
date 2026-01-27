"""
Basic Models Test
Test CPU models without requiring data cache
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_models_basic():
    """Test basic model functionality without data cache"""
    print("🧪 BASIC MODELS TEST")
    print("=" * 40)
    
    results = {
        'imports': False,
        'random_forest': False,
        'xgboost': False,
        'model_selection': False,
        'training': False,
        'prediction': False
    }
    
    try:
        # Test 1: Imports
        print(f"\n📦 Test 1: Imports")
        print("-" * 20)
        
        from models.cpu_models import RandomForestModel, XGBoostModel
        from models import ModelSelector, create_optimal_model
        
        print("   ✅ All imports successful")
        results['imports'] = True
        
        # Test 2: Model Creation
        print(f"\n🏗️ Test 2: Model Creation")
        print("-" * 20)
        
        rf_model = RandomForestModel()
        xgb_model = XGBoostModel()
        
        print(f"   ✅ Random Forest created: {type(rf_model).__name__}")
        print(f"   ✅ XGBoost created: {type(xgb_model).__name__}")
        
        results['random_forest'] = True
        results['xgboost'] = True
        
        # Test 3: Model Selection
        print(f"\n🤖 Test 3: Model Selection")
        print("-" * 20)
        
        selector = ModelSelector()
        hardware_info = selector.get_hardware_info()
        
        print(f"   ✅ Hardware: {hardware_info['recommended_approach']}")
        print(f"   ✅ Performance tier: {hardware_info['performance_tier']}")
        
        optimal_model = create_optimal_model('stock_prediction', 'primary')
        print(f"   ✅ Optimal model: {type(optimal_model).__name__}")
        
        results['model_selection'] = True
        
        # Test 4: Training
        print(f"\n🏋️ Test 4: Training")
        print("-" * 20)
        
        # Generate sample data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        
        print(f"   ✅ Random Forest trained")
        print(f"   ✅ XGBoost trained")
        
        results['training'] = True
        
        # Test 5: Prediction
        print(f"\n🔮 Test 5: Prediction")
        print("-" * 20)
        
        X_test = np.random.randn(20, 10)
        
        rf_pred = rf_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)
        
        print(f"   ✅ RF predictions: {rf_pred.shape}")
        print(f"   ✅ XGBoost predictions: {xgb_pred.shape}")
        
        # Test evaluation
        rf_metrics = rf_model.evaluate(X_test[:10], y_train[:10])
        xgb_metrics = xgb_model.evaluate(X_test[:10], y_train[:10])
        
        print(f"   ✅ RF R²: {rf_metrics['r2']:.4f}")
        print(f"   ✅ XGBoost R²: {xgb_metrics['r2']:.4f}")
        
        results['prediction'] = True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n📊 BASIC MODELS TEST RESULTS")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.title():<15}: {status}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 ALL BASIC MODELS TESTS PASSED!")
        print(f"🚀 CPU models working correctly!")
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
    
    return results

if __name__ == "__main__":
    print(f"🚀 Starting Basic Models Test...")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    results = test_models_basic()
    
    print(f"\n⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}")
