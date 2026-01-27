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
    
    from data.tiingo_loader import TiingoDataLoader
    
    test_results = []
    
    loader = TiingoDataLoader()
    
    # Check cache directory exists
    print(f"\n🔍 Cache directory: {loader.cache_dir}")
    if not os.path.exists(loader.cache_dir):
        print(f"   ❌ FAILED - Cache directory does not exist")
        return [{'test': 'cache_dir', 'status': 'FAIL', 'issues': ['Cache directory missing']}]
    else:
        print(f"   ✅ Cache directory exists")
    
    # Check available tickers
    available = loader.get_available_tickers()
    print(f"\n🔍 Available tickers: {len(available)}")
    
    if len(available) == 0:
        print(f"   ❌ FAILED - No tickers available")
        test_results.append({'test': 'ticker_count', 'status': 'FAIL', 'issues': ['No tickers available']})
    elif len(available) < 3:
        print(f"   ⚠️ WARNING - Only {len(available)} tickers available")
        print(f"   Tickers: {', '.join(available)}")
        test_results.append({'test': 'ticker_count', 'status': 'WARNING', 'issues': [f'Only {len(available)} tickers']})
    else:
        print(f"   ✅ {len(available)} tickers available")
        print(f"   Tickers: {', '.join(available)}")
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
    
    from data.tiingo_loader import TiingoDataLoader
    
    loader = TiingoDataLoader()
    available = loader.get_available_tickers()
    
    test_results = []
    
    for ticker in available[:5]:  # Test first 5 tickers
        print(f"\n🔍 Testing {ticker}...")
        
        df = loader.load_ticker_data(ticker, start_date='2020-01-01')
        
        if df is None or df.empty:
            print(f"   ❌ FAILED - No data loaded")
            test_results.append({'ticker': ticker, 'status': 'FAIL', 'issues': ['No data loaded']})
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
    
    from data.tiingo_loader import TiingoDataLoader
    
    loader = TiingoDataLoader()
    available = loader.get_available_tickers()
    
    if not available:
        return [{'test': 'loading_speed', 'status': 'FAIL', 'issues': ['No tickers to test']}]
    
    test_results = []
    
    # Test loading speed
    ticker = available[0]
    print(f"\n🔍 Testing loading speed for {ticker}...")
    
    # Cold load (first time)
    start = time.time()
    df1 = loader.load_ticker_data(ticker)
    cold_time = time.time() - start
    
    # Warm load (second time - should be faster if cached)
    start = time.time()
    df2 = loader.load_ticker_data(ticker)
    warm_time = time.time() - start
    
    print(f"   Cold load: {cold_time:.4f}s")
    print(f"   Warm load: {warm_time:.4f}s")
    
    issues = []
    
    if cold_time > 1.0:
        issues.append(f"Slow cold load: {cold_time:.2f}s")
    
    if warm_time > 0.5:
        issues.append(f"Slow warm load: {warm_time:.2f}s")
    
    # Check data consistency
    if df1 is not None and df2 is not None:
        if not df1.equals(df2):
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
    
    from data.tiingo_loader import TiingoDataLoader
    
    loader = TiingoDataLoader()
    available = loader.get_available_tickers()
    
    if not available:
        return [{'test': 'date_filtering', 'status': 'FAIL', 'issues': ['No tickers to test']}]
    
    test_results = []
    ticker = available[0]
    
    # Test different date ranges
    test_cases = [
        {'start': '2020-01-01', 'end': '2020-12-31', 'name': '2020 only'},
        {'start': '2022-01-01', 'end': None, 'name': 'From 2022'},
        {'start': None, 'end': '2021-12-31', 'name': 'Until 2021'},
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        df = loader.load_ticker_data(
            ticker,
            start_date=test_case['start'],
            end_date=test_case['end']
        )
        
        if df is None or df.empty:
            print(f"   ⚠️ No data for this range")
            continue
        
        issues = []
        
        # Check start date
        if test_case['start']:
            expected_start = pd.to_datetime(test_case['start'])
            actual_start = df.index.min()
            
            # Allow some tolerance for weekends/holidays
            if actual_start < expected_start - timedelta(days=7):
                issues.append(f"Start date too early: {actual_start.date()} < {expected_start.date()}")
        
        # Check end date
        if test_case['end']:
            expected_end = pd.to_datetime(test_case['end'])
            actual_end = df.index.max()
            
            if actual_end > expected_end + timedelta(days=7):
                issues.append(f"End date too late: {actual_end.date()} > {expected_end.date()}")
        
        print(f"   Range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"   Rows: {len(df)}")
        
        if issues:
            print(f"   ❌ FAILED:")
            for issue in issues:
                print(f"      • {issue}")
            test_results.append({'test': test_case['name'], 'status': 'FAIL', 'issues': issues})
        else:
            print(f"   ✅ PASSED")
            test_results.append({'test': test_case['name'], 'status': 'PASS', 'issues': []})
    
    return test_results

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
        ticker='AAPL',
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
        ticker='AAPL',
        timeframes=['1d'],
        start='2023-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=False
    )
    
    df_test2 = build_enhanced_model_input(
        ticker='AAPL',
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
