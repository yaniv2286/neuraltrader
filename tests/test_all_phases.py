"""
NeuralTrader - Comprehensive Phase Testing
Tests all completed phases systematically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

def test_phase_1_project_structure():
    """Phase 1: Verify project structure and model interfaces"""
    print("\n" + "="*60)
    print("📋 PHASE 1: Project Structure & Infrastructure")
    print("="*60)
    
    try:
        # Test imports
        from models import create_optimal_model, get_best_models, print_model_status
        from data.enhanced_preprocess import build_enhanced_model_input
        print("✅ Core imports successful")
        
        # Test model status
        print("\n🤖 Model Status:")
        print_model_status()
        
        # Test model creation
        model = create_optimal_model('stock_prediction', 'primary')
        print(f"\n✅ Created optimal model: {type(model).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_2_data_collection():
    """Phase 2: Test data collection and preprocessing"""
    print("\n" + "="*60)
    print("📊 PHASE 2: Data Collection & Preprocessing")
    print("="*60)
    
    try:
        from data.tiingo_loader import TiingoDataLoader
        from data.enhanced_preprocess import build_enhanced_model_input
        
        # Test data loader
        loader = TiingoDataLoader()
        print(f"✅ TiingoDataLoader initialized")
        print(f"   Cache directory: {loader.cache_dir}")
        
        # Test loading data for a ticker
        ticker = 'AAPL'
        print(f"\n📈 Testing data load for {ticker}...")
        
        df = build_enhanced_model_input(
            ticker=ticker,
            timeframes=['1d'],
            start='2023-01-01',
            end='2023-12-31',
            validate_data=True,
            create_features=False  # Test basic load first
        )
        
        if df is not None and not df.empty:
            print(f"✅ Loaded {len(df)} days of data")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            return True
        else:
            print(f"⚠️ No data loaded for {ticker}")
            return False
            
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_3_feature_engineering():
    """Phase 3: Test feature engineering components"""
    print("\n" + "="*60)
    print("🔧 PHASE 3: Feature Engineering")
    print("="*60)
    
    try:
        from data.enhanced_preprocess import build_enhanced_model_input, apply_basic_indicators
        from features.regime import detect_regime
        from features.momentum import calculate_rsi, calculate_macd
        
        print("✅ Feature modules imported")
        
        # Load data with features - use more data for testing
        ticker = 'AAPL'
        print(f"\n📈 Testing feature engineering for {ticker}...")
        
        df = build_enhanced_model_input(
            ticker=ticker,
            timeframes=['1d'],
            start='2020-01-01',  # Use more data
            end='2023-12-31',
            validate_data=True,
            create_features=True  # Enable feature engineering
        )
        
        if df is not None and not df.empty:
            print(f"✅ Loaded data with {len(df.columns)} features")
            
            # Check for key features
            feature_checks = {
                'RSI': any('rsi' in col.lower() for col in df.columns),
                'MACD': any('macd' in col.lower() for col in df.columns),
                'Moving Averages': any('sma' in col.lower() or 'ema' in col.lower() for col in df.columns),
                'Volume': any('volume' in col.lower() for col in df.columns),
                'Regime': any('regime' in col.lower() for col in df.columns)
            }
            
            print("\n📊 Feature Categories:")
            for feature, exists in feature_checks.items():
                status = "✅" if exists else "❌"
                print(f"   {status} {feature}")
            
            return all(feature_checks.values())
        else:
            print(f"❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_4_core_models():
    """Phase 4: Test core ML models"""
    print("\n" + "="*60)
    print("🤖 PHASE 4: Core ML Models (LSTM, RF, XGBoost)")
    print("="*60)
    
    try:
        from models import create_optimal_model
        from data.enhanced_preprocess import build_enhanced_model_input, create_sequences
        
        # Load data - use more data for model training
        ticker = 'AAPL'
        print(f"\n📈 Testing models on {ticker}...")
        
        df = build_enhanced_model_input(
            ticker=ticker,
            timeframes=['1d'],
            start='2020-01-01',  # Use 4 years of data
            end='2023-12-31',
            validate_data=True,
            create_features=True
        )
        
        if df is None or df.empty:
            print(f"❌ No data available")
            return False
        
        # Create sequences with shorter length for testing
        target_col = [col for col in df.columns if 'close' in col.lower()][0]
        X, y = create_sequences(df, target_col=target_col, sequence_length=10)
        
        if len(X) == 0:
            print(f"❌ Not enough data for sequences")
            return False
        
        print(f"✅ Created sequences: X={X.shape}, y={y.shape}")
        
        # Flatten for traditional ML
        X_flat = X.reshape(X.shape[0], -1)
        
        # Test models
        model_types = ['primary', 'secondary']
        results = {}
        
        for model_type in model_types:
            print(f"\n🧪 Testing {model_type} model...")
            
            model = create_optimal_model('stock_prediction', model_type)
            print(f"   Model: {type(model).__name__}")
            
            # Train/test split
            train_size = int(0.8 * len(X_flat))
            X_train, X_test = X_flat[:train_size], X_flat[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train
            model.fit(X_train, y_train)
            print(f"   ✅ Training complete")
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            results[model_type] = metrics
            
            print(f"   📊 R² Score: {metrics['r2']:.4f}")
            print(f"   📊 RMSE: {metrics['rmse']:.4f}")
            print(f"   📊 Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        
        print(f"\n✅ All models tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_5_ensemble():
    """Phase 5: Test ensemble modeling"""
    print("\n" + "="*60)
    print("🎯 PHASE 5: Ensemble Modeling")
    print("="*60)
    
    try:
        # Check if ensemble model exists
        from models.cpu_models import MedallionEnsembleModel
        print("✅ Ensemble model imported")
        
        # Test ensemble creation
        ensemble = MedallionEnsembleModel()
        print(f"✅ Created ensemble model: {type(ensemble).__name__}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Phase 5 not fully implemented: {e}")
        return False

def test_phase_7_streamlit():
    """Phase 7: Check Streamlit UI"""
    print("\n" + "="*60)
    print("🖥️ PHASE 7: Streamlit UI")
    print("="*60)
    
    try:
        import os
        streamlit_files = []
        
        # Search for streamlit files
        for root, dirs, files in os.walk('src'):
            for file in files:
                if 'streamlit' in file.lower() or 'app' in file.lower():
                    streamlit_files.append(os.path.join(root, file))
        
        if streamlit_files:
            print(f"✅ Found Streamlit files:")
            for f in streamlit_files:
                print(f"   • {f}")
            return True
        else:
            print(f"⚠️ No Streamlit UI files found")
            return False
            
    except Exception as e:
        print(f"❌ Phase 7 check failed: {e}")
        return False

def main():
    """Run all phase tests"""
    print("🧪 NEURALTRADER - COMPREHENSIVE PHASE TESTING")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {
        'Phase 1: Project Structure': test_phase_1_project_structure(),
        'Phase 2: Data Collection': test_phase_2_data_collection(),
        'Phase 3: Feature Engineering': test_phase_3_feature_engineering(),
        'Phase 4: Core Models': test_phase_4_core_models(),
        'Phase 5: Ensemble': test_phase_5_ensemble(),
        'Phase 7: Streamlit UI': test_phase_7_streamlit(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for phase, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {phase}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n📈 Overall: {passed}/{total} phases passed ({passed/total*100:.1f}%)")
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    main()
