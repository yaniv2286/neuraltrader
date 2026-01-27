"""
Phase 4 Comprehensive Test: Test ALL Available Tickers
Run Phase 4 tests on all 79 tickers in the Tiingo cache to get complete performance metrics
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.enhanced_preprocess import build_enhanced_model_input
from models.model_trainer import ModelTrainer
from models.model_selector import ModelSelector

print("="*80)
print("PHASE 4 COMPREHENSIVE TEST: ALL 79 TICKERS")
print("="*80)

# Get all available tickers from cache
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))

# Extract tickers from filenames
ALL_TICKERS = sorted(list(set([os.path.basename(f).split('_')[0].upper() for f in cache_files])))
print(f"\n📊 Found {len(ALL_TICKERS)} tickers in cache")
print(f"📋 Tickers: {', '.join(ALL_TICKERS[:10])}{'...' if len(ALL_TICKERS) > 10 else ''}")

# Test configuration
TEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'use_log_returns': True,
    'n_features': 25,
    'model_params': {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.02,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3
    }
}

def test_single_ticker(ticker):
    """Test a single ticker and return results"""
    try:
        print(f"\n🔍 Testing {ticker}...")
        
        # Load data
        df = build_enhanced_model_input(
            ticker=ticker,
            timeframes=['1d'],
            start=TEST_CONFIG['start_date'],
            end=TEST_CONFIG['end_date'],
            validate_data=True,
            create_features=True
        )
        
        if df is None or len(df) == 0:
            print(f"   ❌ No data available for {ticker}")
            return None
        
        print(f"   ✅ Loaded {len(df)} days")
        
        # Initialize trainer
        trainer = ModelTrainer(
            use_log_returns=TEST_CONFIG['use_log_returns'],
            n_features=TEST_CONFIG['n_features']
        )
        
        # Prepare data
        X, y = trainer.prepare_data(df, target_col='close')
        if X is None or len(X) == 0:
            print(f"   ❌ Failed to prepare data for {ticker}")
            return None
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.create_time_series_splits(X, y)
        
        # Feature selection
        selected_features = trainer.select_features(X_train, y_train, method='correlation')
        
        # Train model
        model = ModelSelector().get_recommended_models('stock_prediction')['primary']['model'](**TEST_CONFIG['model_params'])
        model.fit(X_train.values, y_train.values)
        
        # Evaluate
        y_train_pred = model.predict(X_train.values)
        y_val_pred = model.predict(X_val.values)
        y_test_pred = model.predict(X_test.values)
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Directional accuracy
        train_dir = np.mean((np.diff(y_train) > 0) == (np.diff(y_train_pred) > 0)) * 100
        val_dir = np.mean((np.diff(y_val) > 0) == (np.diff(y_val_pred) > 0)) * 100
        test_dir = np.mean((np.diff(y_test) > 0) == (np.diff(y_test_pred) > 0)) * 100
        
        # Generalization gap
        gen_gap = train_r2 - test_r2
        
        # Determine if model is good
        is_good = (
            test_r2 > 0.3 and
            test_dir > 60 and
            gen_gap < 0.2
        )
        
        return {
            'ticker': ticker,
            'samples': len(X),
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_dir': train_dir,
            'val_dir': val_dir,
            'test_dir': test_dir,
            'gen_gap': gen_gap,
            'is_good': is_good,
            'data_quality': 'good' if len(df) > 500 else 'limited'
        }
        
    except Exception as e:
        print(f"   ❌ Error testing {ticker}: {str(e)}")
        return None

def run_comprehensive_test():
    """Run tests on all available tickers"""
    print(f"\n🚀 Starting comprehensive test on {len(ALL_TICKERS)} tickers...")
    print(f"📅 Date range: {TEST_CONFIG['start_date']} to {TEST_CONFIG['end_date']}")
    print(f"🎯 Target: Log returns with {TEST_CONFIG['n_features']} features")
    
    results = []
    good_models = []
    failed_tickers = []
    
    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"Progress: {i}/{len(ALL_TICKERS)} ({i/len(ALL_TICKERS)*100:.1f}%)")
        
        result = test_single_ticker(ticker)
        if result:
            results.append(result)
            if result['is_good']:
                good_models.append(ticker)
        else:
            failed_tickers.append(ticker)
    
    # Create summary DataFrame
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total tickers tested: {len(results)}")
        print(f"   Good models: {len(good_models)} ({len(good_models)/len(results)*100:.1f}%)")
        print(f"   Failed tickers: {len(failed_tickers)} ({len(failed_tickers)/len(results)*100:.1f}%)")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Average Test R²: {df_results['test_r2'].mean():.4f}")
        print(f"   Average Test Direction: {df_results['test_dir'].mean():.2f}%")
        print(f"   Average Generalization Gap: {df_results['gen_gap'].mean():.4f}")
        
        print(f"\n🏆 BEST PERFORMING TICKERS:")
        best_performers = df_results.nlargest(5, 'test_r2')
        for _, row in best_performers.iterrows():
            print(f"   {row['ticker']:8s}: R²={row['test_r2']:.4f}, Dir={row['test_dir']:.1f}%, Gap={row['gen_gap']:.3f}")
        
        print(f"\n⚠️ WORST PERFORMING TICKERS:")
        worst_performers = df_results.nsmallest(5, 'test_r2')
        for _, row in worst_performers.iterrows():
            print(f"   {row['ticker']:8s}: R²={row['test_r2']:.4f}, Dir={row['test_dir']:.1f}%, Gap={row['gen_gap']:.3f}")
        
        # Save detailed results
        output_file = f"tests/phase4_comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Create summary report
        with open(f"tests/phase4_comprehensive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write("PHASE 4 COMPREHENSIVE TEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tickers Tested: {len(results)}\n")
            f.write(f"Date Range: {TEST_CONFIG['start_date']} to {TEST_CONFIG['end_date']}\n")
            f.write(f"Configuration: Log Returns, {TEST_CONFIG['n_features']} features\n\n")
            f.write("RESULTS:\n")
            f.write(f"Good Models: {len(good_models) ({len(good_models)/len(results)*100:.1f}%)\n")
            f.write(f"Failed Tickers: {len(failed_tickers)} ({len(failed_tickers)/len(results)*100:.1f}%)\n")
            f.write(f"Average Test R²: {df_results['test_r2'].mean():.4f}\n")
            f.write(f"Average Test Direction: {df_results['test_dir'].mean():.2f}%\n")
            f.write(f"Average Generalization Gap: {df_results['gen_gap'].mean():.4f}\n\n")
            f.write("BEST PERFORMERS:\n")
            for _, row in best_performers.iterrows():
                f.write(f"  {row['ticker']}: R²={row['test_r2']:.4f}, Dir={row['test_dir']:.1f}%\n")
            f.write("\nFAILED TICKERS:\n")
            for ticker in failed_tickers:
                f.write(f"  {ticker}\n")
        
        print(f"📄 Summary report saved to: tests/phase4_comprehensive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        return df_results
    else:
        print("\n❌ No successful tests completed!")
        return None

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    if results is not None:
        print("\n" + "="*80)
        print("TEST COMPLETE!")
        print("="*80)
        print(f"✅ Successfully tested {len(results)} tickers")
        print(f"📊 Results saved to CSV and summary files")
        print(f"🎯 Ready for analysis and comparison with previous runs")
