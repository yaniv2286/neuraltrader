"""
Phase 2: Complete Testing & Finalization
Data Collection & Preprocessing

Tests:
1. Data availability and completeness
2. Data quality (missing values, outliers, gaps)
3. Data loading speed and caching
4. Date range accuracy
5. Column standardization
6. Preprocessing consistency
"""

import sys
import os
# Add parent directory's src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 2: COMPLETE TESTING & FINALIZATION")
print("Data Collection & Preprocessing")
print("="*70)

# ============================================================================
# TEST 1: Data Availability
# ============================================================================
def test_data_availability():
    """Test that data is available and accessible"""
    print("\n" + "="*70)
    print("TEST 1: Data Availability")
    print("="*70)
    
    import glob
    import pandas as pd
    
    test_results = []
    
    # Check cache directory exists
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    print(f"\n🔍 Cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        print(f"   ❌ FAILED - Cache directory does not exist")
        return [{'test': 'cache_dir', 'status': 'FAIL', 'issues': ['Cache directory missing']}]
    else:
        print(f"   ✅ Cache directory exists")
    
    # Get available tickers from cache files
    cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    available_tickers = []
    
    for file in cache_files:
        # Extract ticker from filename like "AAL_1d_full_20260128.csv"
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            # Handle crypto tickers
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            available_tickers.append(ticker)
    
    available_tickers = sorted(list(set(available_tickers)))
    print(f"\n🔍 Available tickers: {len(available_tickers)}")
    
    if len(available_tickers) == 0:
        print(f"   ❌ FAILED - No tickers available")
        test_results.append({'test': 'ticker_count', 'status': 'FAIL', 'issues': ['No tickers available']})
    elif len(available_tickers) < 3:
        print(f"   ⚠️ WARNING - Only {len(available_tickers)} tickers available")
        print(f"   Tickers: {', '.join(available_tickers)}")
        test_results.append({'test': 'ticker_count', 'status': 'WARNING', 'issues': [f'Only {len(available_tickers)} tickers']})
    else:
        print(f"   ✅ {len(available_tickers)} tickers available")
        print(f"   Tickers: {', '.join(available_tickers[:10])}")
        if len(available_tickers) > 10:
            print(f"   ... and {len(available_tickers) - 10} more")
        test_results.append({'test': 'ticker_count', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 2: Data Quality
# ============================================================================
def test_data_quality():
    """Test data quality for each ticker"""
    print("\n" + "="*70)
    print("TEST 2: Data Quality")
    print("="*70)
    
    import glob
    import pandas as pd
    
    # Get available tickers from cache files
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    available_tickers = []
    
    for file in cache_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            available_tickers.append(ticker)
    
    available_tickers = sorted(list(set(available_tickers)))
    test_results = []
    
    for ticker in available_tickers[:5]:  # Test first 5 tickers
        print(f"\n🔍 Testing {ticker}...")
        
        # Find the data file for this ticker
        ticker_files = [f for f in cache_files if ticker in os.path.basename(f)]
        if not ticker_files:
            print(f"   ❌ FAILED - No data file found for {ticker}")
            test_results.append({'ticker': ticker, 'status': 'FAIL', 'issues': ['No data file found']})
            continue
        
        data_file = ticker_files[0]
        
        try:
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            if df is None or df.empty:
                print(f"   ❌ FAILED - No data loaded")
                test_results.append({'ticker': ticker, 'status': 'FAIL', 'issues': ['No data loaded']})
                continue
                
        except Exception as e:
            print(f"   ❌ FAILED - Error loading data: {e}")
            test_results.append({'ticker': ticker, 'status': 'FAIL', 'issues': [f'Error loading data: {e}']})
            continue
        
        issues = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.2f}%")
        elif missing_pct > 0:
            print(f"   ⚠️ Missing data: {missing_pct:.2f}%")
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(f"Duplicate dates: {dup_count}")
        
        # Check for non-positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    neg_count = (df[col] <= 0).sum()
                    issues.append(f"Non-positive {col}: {neg_count} rows")
        
        # Check OHLC logic (High >= Low, Close between High and Low)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                issues.append(f"Invalid High/Low: {invalid_hl} rows")
            
            # Close should generally be between high and low (allow small tolerance for rounding)
            invalid_close = ((df['close'] > df['high'] * 1.001) | (df['close'] < df['low'] * 0.999)).sum()
            if invalid_close > len(df) * 0.01:  # More than 1% invalid
                issues.append(f"Close outside High/Low: {invalid_close} rows")
        
        # Check date continuity (should be trading days)
        date_gaps = df.index.to_series().diff()
        large_gaps = (date_gaps > timedelta(days=7)).sum()  # Gaps > 1 week
        if large_gaps > 10:
            issues.append(f"Large date gaps: {large_gaps}")
        
        # Summary
        print(f"   Rows: {len(df)}")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"   Missing: {missing_pct:.2f}%")
        
        if issues:
            print(f"   ❌ FAILED:")
            for issue in issues:
                print(f"      • {issue}")
            test_results.append({'ticker': ticker, 'status': 'FAIL', 'issues': issues})
        else:
            print(f"   ✅ PASSED - Data quality is good")
            test_results.append({'ticker': ticker, 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 3: Data Loading Speed
# ============================================================================
def test_loading_speed():
    """Test data loading performance"""
    print("\n" + "="*70)
    print("TEST 3: Data Loading Speed")
    print("="*70)
    
    import glob
    import pandas as pd
    
    # Get available tickers from cache files
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    available_tickers = []
    
    for file in cache_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            available_tickers.append(ticker)
    
    available_tickers = sorted(list(set(available_tickers)))
    
    if len(available_tickers) == 0:
        print("\n❌ No tickers available for speed test")
        return [{'test': 'loading_speed', 'status': 'FAIL', 'issues': ['No tickers available']}]
    
    # Test with first available ticker
    test_ticker = available_tickers[0]
    print(f"\n🔍 Testing loading speed for {test_ticker}...")
    
    # Find the data file for this ticker
    ticker_files = [f for f in cache_files if test_ticker in os.path.basename(f)]
    if not ticker_files:
        print(f"\n❌ No data file found for {test_ticker}")
        return [{'test': 'loading_speed', 'status': 'FAIL', 'issues': ['No data file found']}]
    
    data_file = ticker_files[0]
    
    # Cold load
    start = time.time()
    df_cold = pd.read_csv(data_file)
    df_cold['date'] = pd.to_datetime(df_cold['date'])
    df_cold = df_cold.set_index('date')
    cold_time = time.time() - start
    
    # Warm load
    start = time.time()
    df_warm = pd.read_csv(data_file)
    df_warm['date'] = pd.to_datetime(df_warm['date'])
    df_warm = df_warm.set_index('date')
    warm_time = time.time() - start
    
    print(f"   Cold load: {cold_time:.4f}s")
    print(f"   Warm load: {warm_time:.4f}s")
    
    issues = []
    
    if cold_time > 1.0:
        issues.append(f"Slow cold load: {cold_time:.2f}s")
    
    if warm_time > 0.5:
        issues.append(f"Slow warm load: {warm_time:.2f}s")
    
    # Check data consistency
    if df_cold is not None and df_warm is not None:
        if not df_cold.equals(df_warm):
            issues.append("Inconsistent data between loads")
    
    if issues:
        print(f"   ⚠️ WARNINGS:")
        for issue in issues:
            print(f"      • {issue}")
        test_results.append({'test': 'loading_speed', 'status': 'WARNING', 'issues': issues})
    else:
        print(f"   ✅ PASSED - Loading speed is good")
        test_results.append({'test': 'loading_speed', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 4: Date Range Filtering
# ============================================================================
def test_date_filtering():
    """Test date range filtering accuracy"""
    print("\n" + "="*70)
    print("TEST 4: Date Range Filtering")
    print("="*70)
    
    import glob
    import pandas as pd
    
    # Get available tickers from cache files
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'cache', 'tiingo')
    cache_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    available_tickers = []
    
    for file in cache_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 2:
            ticker = parts[0]
            if ticker.endswith('USD'):
                ticker = ticker.replace('_USD', '-USD')
            available_tickers.append(ticker)
    
    available_tickers = sorted(list(set(available_tickers)))
    
    if len(available_tickers) == 0:
        print("\n❌ No tickers available for date filtering test")
        return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': ['No tickers available']}]
    
    # Test with first available ticker
    test_ticker = available_tickers[0]
    print(f"\n🔍 Testing date filtering with {test_ticker}...")
    
    # Find the data file for this ticker
    ticker_files = [f for f in cache_files if test_ticker in os.path.basename(f)]
    if not ticker_files:
        print(f"\n❌ No data file found for {test_ticker}")
        return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': ['No data file found']}]
    
    data_file = ticker_files[0]
    
    try:
        # Load full data
        df_full = pd.read_csv(data_file)
        df_full['date'] = pd.to_datetime(df_full['date'])
        df_full = df_full.set_index('date')
        
        # Test 2020 only
        df_2020 = df_full.loc['2020-01-01':'2020-12-31']
        if df_2020.empty:
            print(f"   ❌ FAILED - No data loaded for 2020")
            return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': ['No data loaded for 2020']}]
        
        range_2020 = f"{df_2020.index.min().date()} to {df_2020.index.max().date()}"
        print(f"   Range: {range_2020}")
        print(f"   Rows: {len(df_2020)}")
        
        # Test from 2022
        df_from_2022 = df_full.loc['2022-01-01':]
        if df_from_2022.empty:
            print(f"   ❌ FAILED - No data loaded from 2022")
            return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': ['No data loaded from 2022']}]
        
        range_from_2022 = f"{df_from_2022.index.min().date()} to {df_from_2022.index.max().date()}"
        print(f"   Range: {range_from_2022}")
        print(f"   Rows: {len(df_from_2022)}")
        
        # Test until 2021
        df_until_2021 = df_full.loc[:'2021-12-31']
        if df_until_2021.empty:
            print(f"   ❌ FAILED - No data loaded until 2021")
            return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': ['No data loaded until 2021']}]
        
        range_until_2021 = f"{df_until_2021.index.min().date()} to {df_until_2021.index.max().date()}"
        print(f"   Range: {range_until_2021}")
        print(f"   Rows: {len(df_until_2021)}")
        
        # Validate ranges
        issues = []
        
        # Check 2020 data
        if not (df_2020.index.min().year == 2020 and df_2020.index.max().year == 2020):
            issues.append("2020 data range incorrect")
        
        # Check from 2022 data
        if df_from_2022.index.min().year < 2022:
            issues.append("From 2022 data range incorrect")
        
        # Check until 2021 data
        if df_until_2021.index.max().year > 2021:
            issues.append("Until 2021 data range incorrect")
        
        if issues:
            print(f"   ❌ FAILED:")
            for issue in issues:
                print(f"      • {issue}")
            return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': issues}]
        else:
            print(f"   ✅ PASSED")
            return [{'test': 'date_filtering', 'status': 'PASS', 'issues': []}]
            
    except Exception as e:
        print(f"   ❌ FAILED - Error: {e}")
        return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': [f'Error: {e}']}]

# ============================================================================
# TEST 5: Enhanced Preprocessing
# ============================================================================
def test_enhanced_preprocessing():
    """Test enhanced preprocessing pipeline"""
    print("\n" + "="*70)
    print("TEST 5: Enhanced Preprocessing")
    print("="*70)
    
    from data.enhanced_preprocess import build_enhanced_model_input
    
    test_results = []
    
    # Test basic preprocessing (no features)
    print("\n🔍 Testing basic preprocessing...")
    
    start = time.time()
    df_basic = build_enhanced_model_input(
        ticker='MSFT',
        timeframes=['1d'],
        start='2023-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=False
    )
    basic_time = time.time() - start
    
    if df_basic is None or df_basic.empty:
        print(f"   ❌ FAILED - No data returned")
        test_results.append({'test': 'basic_preprocessing', 'status': 'FAIL', 'issues': ['No data returned']})
    else:
        print(f"   ✅ Loaded {len(df_basic)} rows in {basic_time:.2f}s")
        print(f"   Columns: {list(df_basic.columns)}")
        test_results.append({'test': 'basic_preprocessing', 'status': 'PASS', 'issues': []})
    
    # Test consistency (load same data twice)
    print("\n🔍 Testing preprocessing consistency...")
    
    df_test1 = build_enhanced_model_input(
        ticker='MSFT',
        timeframes=['1d'],
        start='2023-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=False
    )
    
    df_test2 = build_enhanced_model_input(
        ticker='MSFT',
        timeframes=['1d'],
        start='2023-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=False
    )
    
    if df_test1 is not None and df_test2 is not None:
        if df_test1.equals(df_test2):
            print(f"   ✅ PASSED - Consistent results")
            test_results.append({'test': 'consistency', 'status': 'PASS', 'issues': []})
        else:
            print(f"   ❌ FAILED - Inconsistent results")
            test_results.append({'test': 'consistency', 'status': 'FAIL', 'issues': ['Inconsistent results']})
    
    return test_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all Phase 2 tests"""
    
    all_results = {
        'test_1_availability': test_data_availability(),
        'test_2_quality': test_data_quality(),
        'test_3_speed': test_loading_speed(),
        'test_4_filtering': test_date_filtering(),
        'test_5_preprocessing': test_enhanced_preprocessing()
    }
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 TEST SUMMARY")
    print("="*70)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warnings = 0
    all_issues = []
    
    for test_name, results in all_results.items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        for result in results:
            total_tests += 1
            status = result['status']
            test_id = result.get('ticker', result.get('test', 'unknown'))
            
            if status == 'PASS':
                passed_tests += 1
                print(f"   ✅ {test_id}")
            elif status == 'WARNING':
                warnings += 1
                print(f"   ⚠️ {test_id}")
                for issue in result['issues']:
                    print(f"      • {issue}")
                    all_issues.append(issue)
            else:
                failed_tests += 1
                print(f"   ❌ {test_id}")
                for issue in result['issues']:
                    print(f"      • {issue}")
                    all_issues.append(issue)
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"❌ Failed: {failed_tests}")
    
    if failed_tests == 0 and warnings == 0:
        print("\n🎉 PHASE 2: COMPLETE - ALL TESTS PASSED")
        print("✅ Ready to proceed to Phase 3")
        return True
    elif failed_tests == 0:
        print(f"\n⚠️ PHASE 2: COMPLETE WITH WARNINGS")
        print(f"   {warnings} warnings found but no critical failures")
        print("✅ Can proceed to Phase 3 (address warnings later)")
        return True
    else:
        print(f"\n❌ PHASE 2: INCOMPLETE")
        print(f"   {failed_tests} critical failures must be fixed")
        print("\nIssues to fix:")
        for issue in set(all_issues):
            print(f"   • {issue}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
