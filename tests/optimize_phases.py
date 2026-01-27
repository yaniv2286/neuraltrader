"""
NeuralTrader - Comprehensive Phase Optimization
Systematically review and optimize each phase before GPU deployment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔍 NEURALTRADER - COMPREHENSIVE PHASE OPTIMIZATION")
print("=" * 70)
print("Goal: Optimize CPU-only performance before GPU models")
print("=" * 70)

# ============================================================================
# PHASE 1: PROJECT STRUCTURE & MODEL INTERFACES
# ============================================================================
def optimize_phase_1():
    """Review and optimize project structure and model interfaces"""
    print("\n" + "="*70)
    print("📋 PHASE 1: PROJECT STRUCTURE & MODEL INTERFACES")
    print("="*70)
    
    issues = []
    optimizations = []
    
    try:
        from models import create_optimal_model, get_best_models
        from models.cpu_models import XGBoostModel, RandomForestModel
        
        # Test 1: Verify model interfaces
        print("\n🔍 Testing model interfaces...")
        for model_type in ['primary', 'secondary']:
            model = create_optimal_model('stock_prediction', model_type)
            
            # Check required methods
            required_methods = ['fit', 'predict', 'evaluate']
            for method in required_methods:
                if not hasattr(model, method):
                    issues.append(f"{type(model).__name__} missing {method} method")
                    
        if not issues:
            print("   ✅ All models have required interfaces")
        
        # Test 2: Check model parameters
        print("\n🔍 Checking model parameters...")
        models_config = get_best_models('stock_prediction')
        
        for model_name, config in models_config.items():
            print(f"\n   📊 {model_name}:")
            print(f"      Model: {config['model'].__name__}")
            print(f"      Params: {config['params']}")
            print(f"      Reason: {config['reason']}")
        
        optimizations.append("Model interfaces verified and documented")
        
    except Exception as e:
        issues.append(f"Phase 1 error: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'phase': 'Phase 1: Project Structure',
        'status': 'PASS' if not issues else 'FAIL',
        'issues': issues,
        'optimizations': optimizations
    }

# ============================================================================
# PHASE 2: DATA COLLECTION & PREPROCESSING
# ============================================================================
def optimize_phase_2():
    """Review and optimize data loading and preprocessing"""
    print("\n" + "="*70)
    print("📊 PHASE 2: DATA COLLECTION & PREPROCESSING")
    print("="*70)
    
    issues = []
    optimizations = []
    
    try:
        from data.tiingo_loader import TiingoDataLoader
        from data.enhanced_preprocess import build_enhanced_model_input
        
        loader = TiingoDataLoader()
        
        # Test 1: Check available data
        print("\n🔍 Checking available data...")
        available_tickers = loader.get_available_tickers()
        print(f"   ✅ Available tickers: {len(available_tickers)}")
        print(f"   📊 Tickers: {', '.join(available_tickers[:10])}")
        
        if len(available_tickers) < 3:
            issues.append("Limited ticker data available")
        
        # Test 2: Data quality check for each ticker
        print("\n🔍 Testing data quality...")
        data_quality = {}
        
        for ticker in available_tickers[:3]:  # Test first 3
            df = loader.load_ticker_data(ticker, start_date='2020-01-01')
            if df is not None:
                # Check data completeness
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                
                data_quality[ticker] = {
                    'rows': len(df),
                    'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
                    'missing_pct': missing_pct,
                    'columns': list(df.columns)
                }
                
                print(f"\n   📈 {ticker}:")
                print(f"      Rows: {data_quality[ticker]['rows']}")
                print(f"      Range: {data_quality[ticker]['date_range']}")
                print(f"      Missing: {missing_pct:.2f}%")
                
                if missing_pct > 1:
                    issues.append(f"{ticker} has {missing_pct:.2f}% missing data")
        
        # Test 3: Preprocessing speed
        print("\n🔍 Testing preprocessing speed...")
        import time
        
        start = time.time()
        df = build_enhanced_model_input(
            ticker='AAPL',
            timeframes=['1d'],
            start='2020-01-01',
            end='2023-12-31',
            validate_data=True,
            create_features=False  # Basic only
        )
        basic_time = time.time() - start
        
        print(f"   ⏱️ Basic preprocessing: {basic_time:.2f}s")
        
        if basic_time > 5:
            optimizations.append("Consider caching preprocessed data")
        
        optimizations.append(f"Data loading optimized: {basic_time:.2f}s per ticker")
        
    except Exception as e:
        issues.append(f"Phase 2 error: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'phase': 'Phase 2: Data Collection',
        'status': 'PASS' if not issues else 'FAIL',
        'issues': issues,
        'optimizations': optimizations,
        'data_quality': data_quality if 'data_quality' in locals() else {}
    }

# ============================================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================================
def optimize_phase_3():
    """Review and optimize feature engineering"""
    print("\n" + "="*70)
    print("🔧 PHASE 3: FEATURE ENGINEERING")
    print("="*70)
    
    issues = []
    optimizations = []
    
    try:
        from data.enhanced_preprocess import build_enhanced_model_input
        import time
        
        # Test 1: Feature creation speed
        print("\n🔍 Testing feature creation speed...")
        start = time.time()
        df = build_enhanced_model_input(
            ticker='AAPL',
            timeframes=['1d'],
            start='2020-01-01',
            end='2023-12-31',
            validate_data=True,
            create_features=True
        )
        feature_time = time.time() - start
        
        print(f"   ⏱️ Feature engineering: {feature_time:.2f}s")
        print(f"   📊 Total features: {len(df.columns)}")
        
        # Test 2: Feature correlation analysis
        print("\n🔍 Analyzing feature correlations...")
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated features (> 0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"   ⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
            for feat1, feat2, corr in high_corr_pairs[:5]:
                print(f"      • {feat1} ↔ {feat2}: {corr:.3f}")
            optimizations.append(f"Remove {len(high_corr_pairs)} redundant features")
        else:
            print("   ✅ No highly correlated features found")
        
        # Test 3: Missing values after feature creation
        print("\n🔍 Checking missing values...")
        missing_by_feature = df.isnull().sum()
        features_with_missing = missing_by_feature[missing_by_feature > 0]
        
        if len(features_with_missing) > 0:
            print(f"   ⚠️ {len(features_with_missing)} features have missing values:")
            for feat, count in features_with_missing.head(5).items():
                pct = count / len(df) * 100
                print(f"      • {feat}: {count} ({pct:.1f}%)")
            
            if features_with_missing.max() / len(df) > 0.2:
                issues.append("Some features have >20% missing values")
        else:
            print("   ✅ No missing values in features")
        
        # Test 4: Feature importance preview (using correlation with target)
        print("\n🔍 Analyzing feature importance (correlation with close price)...")
        
        if 'close' in df.columns:
            correlations = df[numeric_cols].corrwith(df['close']).abs().sort_values(ascending=False)
            
            print("   📊 Top 10 features by correlation with price:")
            for feat, corr in correlations.head(10).items():
                if feat != 'close':
                    print(f"      • {feat}: {corr:.3f}")
            
            # Check if we have weak features
            weak_features = correlations[correlations < 0.01]
            if len(weak_features) > 10:
                optimizations.append(f"Consider removing {len(weak_features)} very weak features (<0.01 correlation)")
        
        optimizations.append(f"Feature engineering optimized: {len(df.columns)} features in {feature_time:.2f}s")
        
    except Exception as e:
        issues.append(f"Phase 3 error: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'phase': 'Phase 3: Feature Engineering',
        'status': 'PASS' if not issues else 'FAIL',
        'issues': issues,
        'optimizations': optimizations
    }

# ============================================================================
# PHASE 4: MODEL PERFORMANCE
# ============================================================================
def optimize_phase_4():
    """Review and optimize model performance - THIS IS CRITICAL"""
    print("\n" + "="*70)
    print("🤖 PHASE 4: MODEL PERFORMANCE (CRITICAL)")
    print("="*70)
    
    issues = []
    optimizations = []
    
    try:
        from models import create_optimal_model
        from data.enhanced_preprocess import build_enhanced_model_input, create_sequences
        from sklearn.model_selection import train_test_split
        
        # Load data
        print("\n🔍 Loading data for model testing...")
        df = build_enhanced_model_input(
            ticker='AAPL',
            timeframes=['1d'],
            start='2020-01-01',
            end='2023-12-31',
            validate_data=True,
            create_features=True
        )
        
        print(f"   ✅ Loaded {len(df)} days with {len(df.columns)} features")
        
        # Create sequences
        target_col = 'close'
        X, y = create_sequences(df, target_col=target_col, sequence_length=10)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Split data properly
        train_size = int(0.7 * len(X_flat))
        val_size = int(0.15 * len(X_flat))
        
        X_train = X_flat[:train_size]
        y_train = y[:train_size]
        X_val = X_flat[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X_flat[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"\n📊 Data split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        # Test both models
        models_to_test = {
            'XGBoost': create_optimal_model('stock_prediction', 'primary'),
            'RandomForest': create_optimal_model('stock_prediction', 'secondary')
        }
        
        results = {}
        
        for model_name, model in models_to_test.items():
            print(f"\n🧪 Testing {model_name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate on all sets
            train_metrics = model.evaluate(X_train, y_train)
            val_metrics = model.evaluate(X_val, y_val)
            test_metrics = model.evaluate(X_test, y_test)
            
            results[model_name] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
            
            print(f"   📊 {model_name} Results:")
            print(f"      Train R²: {train_metrics['r2']:.4f} | RMSE: {train_metrics['rmse']:.4f}")
            print(f"      Val   R²: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f}")
            print(f"      Test  R²: {test_metrics['r2']:.4f} | RMSE: {test_metrics['rmse']:.4f}")
            print(f"      Test Dir Acc: {test_metrics['directional_accuracy']:.2%}")
            
            # Check for issues
            if train_metrics['r2'] > 0.5 and test_metrics['r2'] < 0:
                issues.append(f"{model_name}: Severe overfitting (train R²={train_metrics['r2']:.2f}, test R²={test_metrics['r2']:.2f})")
            elif test_metrics['r2'] < 0:
                issues.append(f"{model_name}: Negative test R² - model worse than baseline")
            elif test_metrics['directional_accuracy'] < 0.52:
                issues.append(f"{model_name}: Poor directional accuracy ({test_metrics['directional_accuracy']:.2%})")
        
        # Suggest optimizations
        if issues:
            optimizations.append("CRITICAL: Implement log returns instead of raw prices")
            optimizations.append("Add feature selection to reduce overfitting")
            optimizations.append("Implement hyperparameter tuning")
            optimizations.append("Try different sequence lengths (5, 20, 30 days)")
        
    except Exception as e:
        issues.append(f"Phase 4 error: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'phase': 'Phase 4: Model Performance',
        'status': 'PASS' if not issues else 'NEEDS OPTIMIZATION',
        'issues': issues,
        'optimizations': optimizations,
        'results': results if 'results' in locals() else {}
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run comprehensive phase optimization"""
    
    results = []
    
    # Run each phase optimization
    print("\n🚀 Starting comprehensive optimization...")
    
    results.append(optimize_phase_1())
    results.append(optimize_phase_2())
    results.append(optimize_phase_3())
    results.append(optimize_phase_4())
    
    # Summary
    print("\n" + "="*70)
    print("📊 OPTIMIZATION SUMMARY")
    print("="*70)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'NEEDS OPTIMIZATION' else "❌"
        print(f"\n{status_icon} {result['phase']}: {result['status']}")
        
        if result['issues']:
            print("   🔴 Issues:")
            for issue in result['issues']:
                print(f"      • {issue}")
        
        if result['optimizations']:
            print("   💡 Optimizations:")
            for opt in result['optimizations']:
                print(f"      • {opt}")
    
    # Overall assessment
    total_issues = sum(len(r['issues']) for r in results)
    total_optimizations = sum(len(r['optimizations']) for r in results)
    
    print("\n" + "="*70)
    print(f"🎯 OVERALL ASSESSMENT")
    print("="*70)
    print(f"Total Issues Found: {total_issues}")
    print(f"Total Optimizations Suggested: {total_optimizations}")
    
    if total_issues == 0:
        print("\n✅ System is optimized and ready for GPU models!")
    else:
        print(f"\n⚠️ {total_issues} issues need attention before GPU deployment")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
