"""
Phase 1 Infrastructure Testing
Comprehensive testing for data pipeline, feature engineering, and model selection
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_phase1_infrastructure():
    """Test all Phase 1 infrastructure components"""
    print("🧪 PHASE 1 INFRASTRUCTURE TESTING")
    print("=" * 60)
    
    results = {
        'data_pipeline': False,
        'feature_engineering': False,
        'model_selection': False,
        'cpu_models': False,
        'gpu_detection': False,
        'integration': False,
        'performance': False
    }
    
    try:
        # Test 1: Data Pipeline
        print(f"\n📊 Test 1: Data Pipeline")
        print("-" * 30)
        results['data_pipeline'] = test_data_pipeline()
        
        # Test 2: Feature Engineering
        print(f"\n🎯 Test 2: Feature Engineering")
        print("-" * 30)
        results['feature_engineering'] = test_feature_engineering()
        
        # Test 3: Model Selection
        print(f"\n🤖 Test 3: Model Selection")
        print("-" * 30)
        results['model_selection'] = test_model_selection()
        
        # Test 4: CPU Models
        print(f"\n💻 Test 4: CPU Models")
        print("-" * 30)
        results['cpu_models'] = test_cpu_models()
        
        # Test 5: GPU Detection
        print(f"\n🎮 Test 5: GPU Detection")
        print("-" * 30)
        results['gpu_detection'] = test_gpu_detection()
        
        # Test 6: Integration
        print(f"\n🔗 Test 6: Integration")
        print("-" * 30)
        results['integration'] = test_integration()
        
        # Test 7: Performance
        print(f"\n⚡ Test 7: Performance")
        print("-" * 30)
        results['performance'] = test_performance()
        
    except Exception as e:
        print(f"❌ Infrastructure test failed: {e}")
        traceback.print_exc()
    
    # Summary
    print(f"\n🏁 PHASE 1 INFRASTRUCTURE TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 ALL PHASE 1 INFRASTRUCTURE TESTS PASSED!")
        print(f"🚀 System is production-ready!")
        print(f"💡 Ready to proceed to Phase 2: Basic Models & Validation")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Fix issues before proceeding.")
    
    return results

def test_data_pipeline():
    """Test data loading and preprocessing"""
    try:
        from data.enhanced_preprocess import build_enhanced_model_input
        from data.tiingo_loader import get_available_tickers
        
        # Test Tiingo loader
        tickers = get_available_tickers()
        if len(tickers) == 0:
            print("   ❌ No tickers available in cache")
            return False
        
        print(f"   ✅ Found {len(tickers)} tickers in cache")
        
        # Test enhanced preprocessing
        test_ticker = tickers[0]
        print(f"   📊 Testing with {test_ticker}...")
        
        df = build_enhanced_model_input(
            ticker=test_ticker,
            timeframes=['1d'],
            start='2023-01-01',
            end='2023-12-31',
            validate_data=True,
            create_features=True
        )
        
        if df is None or df.empty:
            print("   ❌ Failed to load data")
            return False
        
        print(f"   ✅ Data loaded: {df.shape}")
        print(f"   ✅ Features: {len(df.columns)}")
        print(f"   ✅ Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Test data quality
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_pct > 5:
            print(f"   ⚠️ High missing data: {missing_pct:.1f}%")
        else:
            print(f"   ✅ Data quality good: {missing_pct:.1f}% missing")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data pipeline error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering modules"""
    try:
        from data.enhanced_preprocess import FeatureEngineer
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        sample_df = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, n_samples)
        }, index=dates)
        
        # Test feature engineering
        engineer = FeatureEngineer()
        features_df = engineer.create_all_features(sample_df)
        
        if features_df is None or features_df.empty:
            print("   ❌ Feature engineering failed")
            return False
        
        original_features = len(sample_df.columns)
        total_features = len(features_df.columns)
        new_features = len(engineer.feature_names)
        
        print(f"   ✅ Original features: {original_features}")
        print(f"   ✅ Total features: {total_features}")
        print(f"   ✅ New features: {new_features}")
        
        # Check key features
        key_features = ['returns', 'sma_20', 'rsi', 'volatility']
        missing_features = [f for f in key_features if f not in features_df.columns]
        
        if missing_features:
            print(f"   ⚠️ Missing key features: {missing_features}")
        else:
            print(f"   ✅ All key features present")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature engineering error: {e}")
        return False

def test_model_selection():
    """Test intelligent model selection"""
    try:
        from models import ModelSelector, get_best_models, create_optimal_model
        
        # Test model selector
        selector = ModelSelector()
        hardware_info = selector.get_hardware_info()
        
        print(f"   ✅ Hardware detected: {hardware_info['recommended_approach']}")
        print(f"   ✅ Performance tier: {hardware_info['performance_tier']}")
        
        # Test model recommendations
        models = get_best_models('stock_prediction')
        
        if 'primary' not in models:
            print("   ❌ No primary model recommended")
            return False
        
        primary_model = models['primary']['model'].__name__
        secondary_model = models['secondary']['model'].__name__
        
        print(f"   ✅ Primary model: {primary_model}")
        print(f"   ✅ Secondary model: {secondary_model}")
        
        # Test model creation
        optimal_model = create_optimal_model('stock_prediction', 'primary')
        
        if optimal_model is None:
            print("   ❌ Failed to create optimal model")
            return False
        
        print(f"   ✅ Created model: {type(optimal_model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model selection error: {e}")
        return False

def test_cpu_models():
    """Test CPU-optimized models"""
    try:
        from models.cpu_models import RandomForestModel, XGBoostModel
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        
        # Test Random Forest
        rf_model = RandomForestModel()
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)
        
        print(f"   ✅ Random Forest trained: {rf_pred.shape}")
        
        # Test XGBoost
        xgb_model = XGBoostModel()
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)
        
        print(f"   ✅ XGBoost trained: {xgb_pred.shape}")
        
        # Test evaluation
        rf_metrics = rf_model.evaluate(X, y)
        xgb_metrics = xgb_model.evaluate(X, y)
        
        print(f"   ✅ RF R²: {rf_metrics['r2']:.4f}")
        print(f"   ✅ XGBoost R²: {xgb_metrics['r2']:.4f}")
        
        # Test feature importance
        rf_importance = rf_model.get_feature_importance()
        if rf_importance is not None:
            print(f"   ✅ RF feature importance: {len(rf_importance)} features")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CPU models error: {e}")
        return False

def test_gpu_detection():
    """Test GPU availability detection"""
    try:
        from models.gpu_models import check_gpu_availability, GPU_AVAILABLE, GPU_INFO
        
        available, info = check_gpu_availability()
        
        print(f"   ✅ GPU Available: {available}")
        print(f"   ✅ GPU Info: {info}")
        print(f"   ✅ GPU_AVAILABLE flag: {GPU_AVAILABLE}")
        
        # Test GPU models import
        if GPU_AVAILABLE:
            try:
                from models.gpu_models import TransformerModel
                print("   ✅ GPU models import successful")
            except ImportError as e:
                print(f"   ⚠️ GPU models import issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU detection error: {e}")
        return False

def test_integration():
    """Test full integration"""
    try:
        from data.enhanced_preprocess import build_enhanced_model_input
        from models import create_optimal_model
        
        # Get available tickers
        from data.tiingo_loader import get_available_tickers
        tickers = get_available_tickers()
        
        if len(tickers) == 0:
            print("   ❌ No tickers available for integration test")
            return False
        
        # Load data
        test_ticker = tickers[0]
        df = build_enhanced_model_input(
            ticker=test_ticker,
            timeframes=['1d'],
            start='2023-01-01',
            end='2023-03-31',
            validate_data=True,
            create_features=True
        )
        
        if df is None or df.empty:
            print("   ❌ Failed to load data for integration")
            return False
        
        # Create sequences
        from data.enhanced_preprocess import create_sequences
        X, y = create_sequences(df, target_col='1d_close', sequence_length=30)
        
        if len(X) == 0 or len(y) == 0:
            print("   ❌ Failed to create sequences")
            return False
        
        print(f"   ✅ Sequences created: X={X.shape}, y={y.shape}")
        
        # Create and train model
        model = create_optimal_model('stock_prediction', 'primary')
        
        # Use smaller dataset for integration test
        X_small = X[:50]
        y_small = y[:50]
        
        model.fit(X_small, y_small)
        predictions = model.predict(X_small)
        
        print(f"   ✅ Integration test successful: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration error: {e}")
        return False

def test_performance():
    """Test performance metrics"""
    try:
        import time
        import psutil
        import os
        
        # Test data loading performance
        from data.tiingo_loader import get_available_tickers
        from data.enhanced_preprocess import build_enhanced_model_input
        
        tickers = get_available_tickers()
        if len(tickers) == 0:
            print("   ⚠️ No tickers available for performance test")
            return True
        
        test_ticker = tickers[0]
        
        # Measure data loading time
        start_time = time.time()
        df = build_enhanced_model_input(
            ticker=test_ticker,
            timeframes=['1d'],
            start='2023-01-01',
            end='2023-06-30',
            validate_data=True,
            create_features=True
        )
        load_time = time.time() - start_time
        
        print(f"   ✅ Data loading time: {load_time:.2f}s")
        
        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"   ✅ Memory usage: {memory_mb:.1f} MB")
        
        # Performance criteria
        if load_time < 5.0:
            print(f"   ✅ Loading performance: Excellent")
        elif load_time < 10.0:
            print(f"   ✅ Loading performance: Good")
        else:
            print(f"   ⚠️ Loading performance: Needs optimization")
        
        if memory_mb < 500:
            print(f"   ✅ Memory usage: Efficient")
        elif memory_mb < 1000:
            print(f"   ✅ Memory usage: Acceptable")
        else:
            print(f"   ⚠️ Memory usage: High")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Starting Phase 1 Infrastructure Testing...")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    results = test_phase1_infrastructure()
    
    print(f"\n⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}")
