"""
Phase 4 Comprehensive Test: Test ALL Available Tickers (Simple Version)
Run Phase 4 tests on all 79 tickers in the Tiingo cache without complex dependencies
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

print("="*80)
print("PHASE 4 COMPREHENSIVE TEST: ALL 79 TICKERS (Simple Version)")
print("="*80)

# Get all available tickers from cache
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))

# Extract tickers from filenames
ALL_TICKERS = sorted(list(set([os.path.basename(f).split('_')[0].upper() for f in cache_files])))
print(f"\n📊 Found {len(ALL_TICKERS)} tickers in cache")
print(f"📋 First 10: {', '.join(ALL_TICKERS[:10])}")
print(f"📋 Last 10: {', '.join(ALL_TICKERS[-10:])}")

# Test configuration
TEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'min_samples': 500  # Minimum samples required
}

def test_single_ticker_simple(ticker):
    """Test a single ticker with simple approach"""
    try:
        print(f"\n🔍 Testing {ticker}...")
        
        # Find the data file
        ticker_files = [f for f in cache_files if ticker in os.path.basename(f)]
        if not ticker_files:
            print(f"   ❌ No data file found for {ticker}")
            return None
        
        data_file = ticker_files[0]
        
        # Load data
        df = pd.read_csv(data_file)
        
        if len(df) == 0:
            print(f"   ❌ Empty data file for {ticker}")
            return None
        
        # Check if we have required columns
        required_cols = ['date', 'close', 'open', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ❌ Missing columns for {ticker}: {missing_cols}")
            return None
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Filter by date range
        start_date = pd.to_datetime(TEST_CONFIG['start_date'])
        end_date = pd.to_datetime(TEST_CONFIG['end_date'])
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(df) < TEST_CONFIG['min_samples']:
            print(f"   ❌ Insufficient data for {ticker}: {len(df)} days (need {TEST_CONFIG['min_samples']})")
            return None
        
        print(f"   ✅ Loaded {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Calculate basic metrics
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < TEST_CONFIG['min_samples']:
            print(f"   ❌ Insufficient data after cleaning for {ticker}: {len(df)} days")
            return None
        
        # Simple correlation test (predict next day's return from today's features)
        # Use simple features: price changes, volume, moving averages
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['price_to_ma5'] = df['close'] / df['ma_5']
        df['price_to_ma20'] = df['close'] / df['ma_20']
        
        # Drop NaN values from moving averages
        df = df.dropna()
        
        if len(df) < TEST_CONFIG['min_samples']:
            print(f"   ❌ Insufficient data after feature creation for {ticker}: {len(df)} days")
            return None
        
        # Prepare features and target
        feature_cols = ['price_change', 'volume_change', 'price_to_ma5', 'price_to_ma20']
        X = df[feature_cols]
        y = df['log_returns'].shift(-1)  # Predict next day's log return
        
        # Remove last row (no target)
        X = X[:-1]
        y = y[:-1]
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Directional accuracy
        train_dir = np.mean((y_train > 0) == (y_train_pred > 0)) * 100
        val_dir = np.mean((y_val > 0) == (y_val_pred > 0)) * 100
        test_dir = np.mean((y_test > 0) == (y_test_pred > 0)) * 100
        
        # Generalization gap
        gen_gap = train_r2 - test_r2
        
        # Determine if model is good
        is_good = (
            test_r2 > 0.1 and  # Lower threshold for simple model
            test_dir > 55 and
            gen_gap < 0.3
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
    print(f"🎯 Target: Simple linear regression with basic features")
    
    results = []
    good_models = []
    failed_tickers = []
    
    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"Progress: {i}/{len(ALL_TICKERS)} ({i/len(ALL_TICKERS)*100:.1f}%)")
        
        result = test_single_ticker_simple(ticker)
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
        print("COMPREHENSIVE TEST RESULTS (Simple Model)")
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"tests/phase4_comprehensive_simple_results_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Create summary report
        summary_file = f"tests/phase4_comprehensive_simple_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PHASE 4 COMPREHENSIVE TEST SUMMARY (Simple Model)\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tickers Tested: {len(results)}\n")
            f.write(f"Date Range: {TEST_CONFIG['start_date']} to {TEST_CONFIG['end_date']}\n")
            f.write(f"Model: Simple Linear Regression\n")
            f.write(f"Features: Basic price/volume/MA indicators\n\n")
            f.write("RESULTS:\n")
            f.write(f"Good Models: {len(good_models)} ({len(good_models)/len(results)*100:.1f}%)\n")
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
        
        print(f"📄 Summary report saved to: {summary_file}")
        
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
        print(f"✅ Successfully tested {len(results)} tickers with simple model")
        print(f"📊 Results saved to CSV and summary files")
        print(f"🎯 This gives us a baseline for comparison with complex models")
        print(f"📈 Ready for analysis and comparison with previous runs")
