"""
Phase 3: Complete Testing & Optimization
Feature Engineering

Tests:
1. Feature creation completeness
2. Feature correlation analysis (identify redundancies)
3. Missing value handling
4. Feature importance analysis
5. Feature engineering speed
6. Feature consistency

Goal: Identify and remove 119 redundant features found in initial testing
"""

import sys
import os
# Add parent directory's src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 3: COMPLETE TESTING & OPTIMIZATION")
print("Feature Engineering")
print("="*70)

# ============================================================================
# TEST 1: Feature Creation Completeness
# ============================================================================
def test_feature_completeness():
    """Test that all expected features are created"""
    print("\n" + "="*70)
    print("TEST 1: Feature Creation Completeness")
    print("="*70)
    
    from data.enhanced_preprocess import build_enhanced_model_input
    
    test_results = []
    
    print("\n🔍 Creating features for AAPL...")
    
    df = build_enhanced_model_input(
        ticker='AAPL',
        timeframes=['1d'],
        start='2020-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=True
    )
    
    if df is None or df.empty:
        print("   ❌ FAILED - No data returned")
        return [{'test': 'feature_creation', 'status': 'FAIL', 'issues': ['No data returned']}]
    
    print(f"   ✅ Created {len(df.columns)} features")
    print(f"   📊 Data shape: {df.shape}")
    
    # Expected feature categories
    expected_categories = {
        'momentum': ['rsi', 'macd', 'sma', 'ema'],
        'volatility': ['atr', 'bb_', 'volatility'],
        'volume': ['volume'],
        'regime': ['regime'],
        'returns': ['return'],
        'price': ['close', 'open', 'high', 'low']
    }
    
    found_categories = {}
    
    for category, keywords in expected_categories.items():
        found = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                found.append(col)
        found_categories[category] = found
    
    print("\n📊 Feature Categories:")
    issues = []
    
    for category, features in found_categories.items():
        if features:
            print(f"   ✅ {category.title()}: {len(features)} features")
        else:
            print(f"   ⚠️ {category.title()}: No features found")
            issues.append(f"Missing {category} features")
    
    if issues:
        test_results.append({'test': 'feature_completeness', 'status': 'WARNING', 'issues': issues})
    else:
        test_results.append({'test': 'feature_completeness', 'status': 'PASS', 'issues': []})
    
    return test_results, df

# ============================================================================
# TEST 2: Feature Correlation Analysis (CRITICAL)
# ============================================================================
def test_feature_correlation(df):
    """Identify highly correlated features that should be removed"""
    print("\n" + "="*70)
    print("TEST 2: Feature Correlation Analysis (CRITICAL)")
    print("="*70)
    
    test_results = []
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n🔍 Analyzing {len(numeric_cols)} numeric features...")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Find highly correlated pairs (>0.95)
    high_corr_pairs = []
    features_to_remove = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                high_corr_pairs.append((feat1, feat2, corr_val))
                
                # Keep the simpler feature name (usually the base feature)
                if len(feat2) > len(feat1):
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)
    
    print(f"\n📊 Correlation Analysis Results:")
    print(f"   Total feature pairs analyzed: {len(numeric_cols) * (len(numeric_cols) - 1) // 2}")
    print(f"   Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
    print(f"   Features recommended for removal: {len(features_to_remove)}")
    
    if high_corr_pairs:
        print(f"\n   Top 10 highly correlated pairs:")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
            print(f"      • {feat1} ↔ {feat2}: {corr:.3f}")
    
    # Categorize features to remove
    if features_to_remove:
        print(f"\n   🔧 Recommended features to remove ({len(features_to_remove)}):")
        for feat in sorted(list(features_to_remove))[:20]:
            print(f"      • {feat}")
        
        if len(features_to_remove) > 20:
            print(f"      ... and {len(features_to_remove) - 20} more")
        
        test_results.append({
            'test': 'correlation_analysis',
            'status': 'ACTION_REQUIRED',
            'issues': [f'{len(features_to_remove)} redundant features found'],
            'features_to_remove': list(features_to_remove)
        })
    else:
        print(f"   ✅ No highly correlated features found")
        test_results.append({
            'test': 'correlation_analysis',
            'status': 'PASS',
            'issues': [],
            'features_to_remove': []
        })
    
    return test_results

# ============================================================================
# TEST 3: Missing Value Analysis
# ============================================================================
def test_missing_values(df):
    """Analyze missing values in features"""
    print("\n" + "="*70)
    print("TEST 3: Missing Value Analysis")
    print("="*70)
    
    test_results = []
    
    missing_counts = df.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    print(f"\n🔍 Missing Value Analysis:")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Features with missing values: {len(features_with_missing)}")
    
    issues = []
    
    if len(features_with_missing) > 0:
        print(f"\n   Features with missing values:")
        
        for feat, count in features_with_missing.head(10).items():
            pct = count / len(df) * 100
            print(f"      • {feat}: {count} ({pct:.1f}%)")
            
            if pct > 20:
                issues.append(f"{feat} has {pct:.1f}% missing values")
        
        if len(features_with_missing) > 10:
            print(f"      ... and {len(features_with_missing) - 10} more")
        
        # Check if missing values are at the start (due to rolling windows)
        first_valid_idx = df.first_valid_index()
        if first_valid_idx is not None:
            rows_until_valid = df.index.get_loc(first_valid_idx)
            print(f"\n   ℹ️ First {rows_until_valid} rows have missing values (expected for rolling windows)")
    else:
        print(f"   ✅ No missing values found")
    
    if issues:
        test_results.append({'test': 'missing_values', 'status': 'WARNING', 'issues': issues})
    else:
        test_results.append({'test': 'missing_values', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 4: Feature Importance (Correlation with Target)
# ============================================================================
def test_feature_importance(df):
    """Analyze feature importance using correlation with target"""
    print("\n" + "="*70)
    print("TEST 4: Feature Importance Analysis")
    print("="*70)
    
    test_results = []
    
    if 'close' not in df.columns:
        print("   ⚠️ No 'close' column found - skipping importance analysis")
        return [{'test': 'feature_importance', 'status': 'SKIP', 'issues': ['No target column']}]
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from features
    feature_cols = [col for col in numeric_cols if col != 'close']
    
    print(f"\n🔍 Analyzing importance of {len(feature_cols)} features...")
    
    # Calculate correlation with target
    correlations = df[feature_cols].corrwith(df['close']).abs().sort_values(ascending=False)
    
    # Top features
    print(f"\n   📊 Top 15 most important features (by correlation with price):")
    for feat, corr in correlations.head(15).items():
        print(f"      • {feat}: {corr:.3f}")
    
    # Weak features
    weak_features = correlations[correlations < 0.01]
    
    if len(weak_features) > 0:
        print(f"\n   ⚠️ {len(weak_features)} very weak features (<0.01 correlation):")
        for feat, corr in weak_features.head(10).items():
            print(f"      • {feat}: {corr:.4f}")
        
        if len(weak_features) > 10:
            print(f"      ... and {len(weak_features) - 10} more")
        
        test_results.append({
            'test': 'feature_importance',
            'status': 'WARNING',
            'issues': [f'{len(weak_features)} very weak features'],
            'weak_features': list(weak_features.index)
        })
    else:
        print(f"   ✅ All features have reasonable correlation with target")
        test_results.append({'test': 'feature_importance', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 5: Feature Engineering Speed
# ============================================================================
def test_feature_speed():
    """Test feature engineering performance"""
    print("\n" + "="*70)
    print("TEST 5: Feature Engineering Speed")
    print("="*70)
    
    from data.enhanced_preprocess import build_enhanced_model_input
    
    test_results = []
    
    print("\n🔍 Testing feature engineering speed...")
    
    # Test with features
    start = time.time()
    df_with_features = build_enhanced_model_input(
        ticker='AAPL',
        timeframes=['1d'],
        start='2020-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=True
    )
    time_with_features = time.time() - start
    
    # Test without features
    start = time.time()
    df_without_features = build_enhanced_model_input(
        ticker='AAPL',
        timeframes=['1d'],
        start='2020-01-01',
        end='2023-12-31',
        validate_data=True,
        create_features=False
    )
    time_without_features = time.time() - start
    
    feature_overhead = time_with_features - time_without_features
    
    print(f"   Without features: {time_without_features:.3f}s")
    print(f"   With features: {time_with_features:.3f}s")
    print(f"   Feature overhead: {feature_overhead:.3f}s")
    
    issues = []
    
    if time_with_features > 5.0:
        issues.append(f"Slow feature engineering: {time_with_features:.2f}s")
    
    if feature_overhead > 3.0:
        issues.append(f"High feature overhead: {feature_overhead:.2f}s")
    
    if issues:
        test_results.append({'test': 'feature_speed', 'status': 'WARNING', 'issues': issues})
    else:
        print(f"   ✅ Feature engineering speed is acceptable")
        test_results.append({'test': 'feature_speed', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 6: Feature Consistency
# ============================================================================
def test_feature_consistency():
    """Test that feature engineering produces consistent results"""
    print("\n" + "="*70)
    print("TEST 6: Feature Consistency")
    print("="*70)
    
    from data.enhanced_preprocess import build_enhanced_model_input
    
    test_results = []
    
    print("\n🔍 Testing feature consistency (3 runs)...")
    
    dfs = []
    for i in range(3):
        df = build_enhanced_model_input(
            ticker='AAPL',
            timeframes=['1d'],
            start='2023-01-01',
            end='2023-12-31',
            validate_data=True,
            create_features=True
        )
        dfs.append(df)
    
    # Check if all runs produced same results
    issues = []
    
    # Check column consistency
    cols_0 = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 1):
        cols_i = set(df.columns)
        if cols_0 != cols_i:
            issues.append(f"Run {i+1} has different columns than run 1")
    
    # Check value consistency (allowing for small floating point differences)
    if not issues:
        for col in dfs[0].columns:
            if dfs[0][col].dtype in [np.float64, np.float32]:
                for i in range(1, len(dfs)):
                    if not np.allclose(dfs[0][col].dropna(), dfs[i][col].dropna(), rtol=1e-10, equal_nan=True):
                        issues.append(f"Column '{col}' has inconsistent values")
                        break
    
    if issues:
        print(f"   ❌ FAILED:")
        for issue in issues:
            print(f"      • {issue}")
        test_results.append({'test': 'feature_consistency', 'status': 'FAIL', 'issues': issues})
    else:
        print(f"   ✅ PASSED - Features are consistent across runs")
        test_results.append({'test': 'feature_consistency', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all Phase 3 tests"""
    
    # Test 1: Feature completeness
    completeness_results, df = test_feature_completeness()
    
    # Test 2: Correlation analysis (CRITICAL)
    correlation_results = test_feature_correlation(df)
    
    # Test 3: Missing values
    missing_results = test_missing_values(df)
    
    # Test 4: Feature importance
    importance_results = test_feature_importance(df)
    
    # Test 5: Speed
    speed_results = test_feature_speed()
    
    # Test 6: Consistency
    consistency_results = test_feature_consistency()
    
    all_results = {
        'test_1_completeness': completeness_results,
        'test_2_correlation': correlation_results,
        'test_3_missing': missing_results,
        'test_4_importance': importance_results,
        'test_5_speed': speed_results,
        'test_6_consistency': consistency_results
    }
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 3 TEST SUMMARY")
    print("="*70)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warnings = 0
    action_required = 0
    all_issues = []
    features_to_remove = []
    weak_features = []
    
    for test_name, results in all_results.items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        for result in results:
            total_tests += 1
            status = result['status']
            test_id = result.get('test', 'unknown')
            
            if status == 'PASS':
                passed_tests += 1
                print(f"   ✅ {test_id}")
            elif status == 'WARNING':
                warnings += 1
                print(f"   ⚠️ {test_id}")
                for issue in result['issues']:
                    print(f"      • {issue}")
                    all_issues.append(issue)
                
                # Collect weak features
                if 'weak_features' in result:
                    weak_features.extend(result['weak_features'])
                    
            elif status == 'ACTION_REQUIRED':
                action_required += 1
                print(f"   🔧 {test_id} - ACTION REQUIRED")
                for issue in result['issues']:
                    print(f"      • {issue}")
                    all_issues.append(issue)
                
                # Collect features to remove
                if 'features_to_remove' in result:
                    features_to_remove.extend(result['features_to_remove'])
                    
            elif status == 'SKIP':
                print(f"   ⏭️ {test_id} - SKIPPED")
            else:
                failed_tests += 1
                print(f"   ❌ {test_id}")
                for issue in result['issues']:
                    print(f"      • {issue}")
                    all_issues.append(issue)
    
    # Optimization recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    if features_to_remove:
        print(f"\n🔧 CRITICAL: Remove {len(features_to_remove)} redundant features")
        print(f"   These features have >0.95 correlation with other features")
        print(f"   Removing them will:")
        print(f"      • Reduce overfitting")
        print(f"      • Speed up training")
        print(f"      • Improve model generalization")
    
    if weak_features:
        print(f"\n💡 OPTIONAL: Consider removing {len(weak_features)} weak features")
        print(f"   These features have <0.01 correlation with target")
        print(f"   They add noise without predictive value")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"🔧 Action Required: {action_required}")
    print(f"❌ Failed: {failed_tests}")
    
    if failed_tests == 0 and action_required == 0 and warnings == 0:
        print("\n🎉 PHASE 3: COMPLETE - ALL TESTS PASSED")
        print("✅ Ready to proceed to Phase 4")
        return True
    elif failed_tests == 0 and action_required > 0:
        print(f"\n🔧 PHASE 3: OPTIMIZATION REQUIRED")
        print(f"   {action_required} critical optimizations needed")
        print(f"   Must remove redundant features before Phase 4")
        return False
    elif failed_tests == 0:
        print(f"\n⚠️ PHASE 3: COMPLETE WITH WARNINGS")
        print(f"   {warnings} warnings found but no critical failures")
        print("✅ Can proceed to Phase 4 (address warnings later)")
        return True
    else:
        print(f"\n❌ PHASE 3: INCOMPLETE")
        print(f"   {failed_tests} critical failures must be fixed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
