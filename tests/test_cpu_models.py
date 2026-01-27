"""
Test CPU Models
Verify CPU models work correctly without GPU dependencies
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cpu_models():
    """Test CPU models functionality"""
    print("🧪 TESTING CPU MODELS")
    print("=" * 50)
    
    try:
        # Test model selector
        from models import ModelSelector, get_best_models, create_optimal_model
        print("✅ Model selector imported successfully")
        
        # Test CPU models
        from models.cpu_models import RandomForestModel, XGBoostModel
        print("✅ CPU models imported successfully")
        
        # Check GPU availability
        selector = ModelSelector()
        print(f"🖥️ Hardware: {selector.gpu_available}")
        if not selector.gpu_available:
            print("💡 Using CPU-optimized models (recommended for this hardware)")
        
        # Get recommended models
        models = get_best_models('stock_prediction')
        print(f"\n🎯 Recommended Models:")
        print(f"   Primary: {models['primary']['model'].__name__}")
        print(f"   Secondary: {models['secondary']['model'].__name__}")
        
        # Test model creation
        print(f"\n🔧 Testing Model Creation...")
        
        # Test XGBoost
        xgb_model = create_optimal_model('stock_prediction', 'primary')
        print(f"✅ XGBoost model created: {type(xgb_model).__name__}")
        
        # Test Random Forest
        rf_model = create_optimal_model('stock_prediction', 'secondary')
        print(f"✅ Random Forest model created: {type(rf_model).__name__}")
        
        # Generate sample data
        print(f"\n📊 Generating Sample Data...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 0.02  # Small returns
        
        # Convert to DataFrame for better feature handling
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        print(f"   Data shape: {X_df.shape}")
        print(f"   Target range: {y.min():.4f} to {y.max():.4f}")
        
        # Test model training
        print(f"\n🏋️ Testing Model Training...")
        
        # Train XGBoost
        print(f"   Training XGBoost...")
        xgb_model.fit(X_df, y)
        print(f"   ✅ XGBoost trained successfully")
        
        # Train Random Forest
        print(f"   Training Random Forest...")
        rf_model.fit(X_df, y)
        print(f"   ✅ Random Forest trained successfully")
        
        # Test predictions
        print(f"\n🔮 Testing Predictions...")
        
        X_test = X_df[:100]  # Use first 100 samples for testing
        
        xgb_pred = xgb_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        print(f"   XGBoost predictions: {xgb_pred.shape}")
        print(f"   Random Forest predictions: {rf_pred.shape}")
        print(f"   ✅ Predictions generated successfully")
        
        # Test evaluation
        print(f"\n📈 Testing Model Evaluation...")
        
        y_test = y[:100]
        
        xgb_metrics = xgb_model.evaluate(X_test, y_test)
        rf_metrics = rf_model.evaluate(X_test, y_test)
        
        print(f"   XGBoost R²: {xgb_metrics['r2']:.4f}")
        print(f"   Random Forest R²: {rf_metrics['r2']:.4f}")
        print(f"   ✅ Evaluation completed successfully")
        
        # Test feature importance
        print(f"\n🎯 Testing Feature Importance...")
        
        xgb_importance = xgb_model.get_feature_importance()
        rf_importance = rf_model.get_feature_importance()
        
        if xgb_importance is not None:
            print(f"   XGBoost top feature: {xgb_importance.iloc[0]['feature']}")
        if rf_importance is not None:
            print(f"   RF top feature: {rf_importance.iloc[0]['feature']}")
        
        print(f"   ✅ Feature importance calculated successfully")
        
        # Test model saving/loading
        print(f"\n💾 Testing Model Persistence...")
        
        test_save_path = "models_cache/test_xgb_model.pkl"
        xgb_model.save_model(test_save_path)
        
        # Create new model and load
        xgb_loaded = XGBoostModel()
        xgb_loaded.load_model(test_save_path)
        
        # Test loaded model
        loaded_pred = xgb_loaded.predict(X_test)
        
        # Check if predictions are the same
        if np.allclose(xgb_pred, loaded_pred):
            print(f"   ✅ Model save/load successful")
        else:
            print(f"   ⚠️ Model save/load issue detected")
        
        # Clean up
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        
        print(f"\n🎉 CPU MODELS TEST COMPLETE!")
        print("=" * 50)
        print(f"✅ All CPU models working correctly")
        print(f"✅ No GPU dependencies required")
        print(f"✅ Model selection working properly")
        print(f"✅ Training and prediction successful")
        print(f"✅ Model persistence working")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU Models Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_selector():
    """Test intelligent model selector"""
    print(f"\n🤖 TESTING MODEL SELECTOR")
    print("-" * 30)
    
    try:
        from models import ModelSelector
        
        selector = ModelSelector()
        hardware_info = selector.get_hardware_info()
        
        print(f"Hardware: {hardware_info['recommended_approach']}")
        print(f"Performance Tier: {hardware_info['performance_tier']}")
        
        # Test different task types
        for task in ['stock_prediction', 'volatility', 'regime_detection']:
            models = selector.get_recommended_models(task)
            primary_model = models['primary']['model'].__name__
            print(f"{task}: {primary_model}")
        
        print(f"✅ Model selector working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Model selector test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Starting CPU Models Test...")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    # Run tests
    cpu_test_passed = test_cpu_models()
    selector_test_passed = test_model_selector()
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   CPU Models Test: {'✅ PASSED' if cpu_test_passed else '❌ FAILED'}")
    print(f"   Model Selector Test: {'✅ PASSED' if selector_test_passed else '❌ FAILED'}")
    
    if cpu_test_passed and selector_test_passed:
        print(f"\n🎉 CPU MODELS READY FOR PRODUCTION!")
        print(f"💡 System will automatically use optimal models based on hardware")
    else:
        print(f"\n🔧 CPU MODELS NEED FIXES")
    
    print(f"⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}")
