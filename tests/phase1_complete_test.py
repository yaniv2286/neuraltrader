"""
Phase 1: Complete Testing & Finalization
Project Structure & Model Interfaces

Tests:
1. Model interface compliance
2. Model parameter validation
3. Error handling
4. Model creation consistency
5. Hardware detection accuracy
"""

import sys
import os
# Add parent directory's src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 1: COMPLETE TESTING & FINALIZATION")
print("Project Structure & Model Interfaces")
print("="*70)

# ============================================================================
# TEST 1: Model Interface Compliance
# ============================================================================
def test_model_interfaces():
    """Test that all models implement required interface"""
    print("\n" + "="*70)
    print("TEST 1: Model Interface Compliance")
    print("="*70)
    
    from models import create_optimal_model
    from models.cpu_models import XGBoostModel, RandomForestModel
    
    required_methods = ['fit', 'predict', 'evaluate', 'prepare_data']
    required_attributes = []
    
    test_results = []
    
    # Test CPU models
    cpu_models = {
        'XGBoost': XGBoostModel(),
        'RandomForest': RandomForestModel()
    }
    
    for model_name, model in cpu_models.items():
        print(f"\n🔍 Testing {model_name}...")
        issues = []
        
        # Check methods
        for method in required_methods:
            if not hasattr(model, method):
                issues.append(f"Missing method: {method}")
            elif not callable(getattr(model, method)):
                issues.append(f"{method} is not callable")
        
        # Check method signatures
        try:
            import inspect
            
            # Check fit signature
            fit_sig = inspect.signature(model.fit)
            if 'X' not in fit_sig.parameters or 'y' not in fit_sig.parameters:
                issues.append("fit() missing X or y parameters")
            
            # Check predict signature
            pred_sig = inspect.signature(model.predict)
            if 'X' not in pred_sig.parameters:
                issues.append("predict() missing X parameter")
            
            # Check evaluate signature
            eval_sig = inspect.signature(model.evaluate)
            if 'X' not in eval_sig.parameters or 'y' not in eval_sig.parameters:
                issues.append("evaluate() missing X or y parameters")
                
        except Exception as e:
            issues.append(f"Signature inspection failed: {e}")
        
        if issues:
            print(f"   ❌ FAILED:")
            for issue in issues:
                print(f"      • {issue}")
            test_results.append({'model': model_name, 'status': 'FAIL', 'issues': issues})
        else:
            print(f"   ✅ PASSED - All required methods present")
            test_results.append({'model': model_name, 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 2: Model Parameter Validation
# ============================================================================
def test_model_parameters():
    """Test that model parameters are valid and reasonable"""
    print("\n" + "="*70)
    print("TEST 2: Model Parameter Validation")
    print("="*70)
    
    from models import get_best_models
    
    test_results = []
    
    # Get recommended models
    models_config = get_best_models('stock_prediction')
    
    for model_type, config in models_config.items():
        print(f"\n🔍 Testing {model_type} configuration...")
        issues = []
        
        # Check config structure
        if 'model' not in config:
            issues.append("Missing 'model' key in config")
        if 'params' not in config:
            issues.append("Missing 'params' key in config")
        if 'reason' not in config:
            issues.append("Missing 'reason' key in config")
        
        if issues:
            print(f"   ❌ FAILED:")
            for issue in issues:
                print(f"      • {issue}")
            test_results.append({'config': model_type, 'status': 'FAIL', 'issues': issues})
            continue
        
        # Validate parameters
        params = config['params']
        model_class = config['model']
        
        print(f"   Model: {model_class.__name__}")
        print(f"   Params: {params}")
        
        # XGBoost specific validation
        if 'XGBoost' in model_class.__name__:
            if 'n_estimators' in params and params['n_estimators'] < 50:
                issues.append("n_estimators too low (< 50)")
            if 'learning_rate' in params and params['learning_rate'] > 0.3:
                issues.append("learning_rate too high (> 0.3)")
            if 'max_depth' in params and params['max_depth'] > 15:
                issues.append("max_depth too high (> 15) - may overfit")
        
        # RandomForest specific validation
        if 'RandomForest' in model_class.__name__:
            if 'n_estimators' in params and params['n_estimators'] < 50:
                issues.append("n_estimators too low (< 50)")
            if 'max_depth' in params and params['max_depth'] > 20:
                issues.append("max_depth too high (> 20) - may overfit")
        
        if issues:
            print(f"   ⚠️ WARNINGS:")
            for issue in issues:
                print(f"      • {issue}")
            test_results.append({'config': model_type, 'status': 'WARNING', 'issues': issues})
        else:
            print(f"   ✅ PASSED - Parameters are reasonable")
            test_results.append({'config': model_type, 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 3: Error Handling
# ============================================================================
def test_error_handling():
    """Test that models handle errors gracefully"""
    print("\n" + "="*70)
    print("TEST 3: Error Handling")
    print("="*70)
    
    from models import create_optimal_model
    
    test_results = []
    
    print("\n🔍 Testing invalid inputs...")
    
    model = create_optimal_model('stock_prediction', 'primary')
    
    # Test 1: Empty data
    print("\n   Test 3.1: Empty arrays")
    try:
        X_empty = np.array([])
        y_empty = np.array([])
        model.fit(X_empty, y_empty)
        print("   ❌ FAILED - Should raise error on empty data")
        test_results.append({'test': 'empty_data', 'status': 'FAIL', 'issues': ['No error on empty data']})
    except Exception as e:
        print(f"   ✅ PASSED - Correctly raised error: {type(e).__name__}")
        test_results.append({'test': 'empty_data', 'status': 'PASS', 'issues': []})
    
    # Test 2: Mismatched dimensions
    print("\n   Test 3.2: Mismatched X and y dimensions")
    try:
        X_mismatch = np.random.rand(10, 5)
        y_mismatch = np.random.rand(15)
        model.fit(X_mismatch, y_mismatch)
        print("   ❌ FAILED - Should raise error on dimension mismatch")
        test_results.append({'test': 'dimension_mismatch', 'status': 'FAIL', 'issues': ['No error on dimension mismatch']})
    except Exception as e:
        print(f"   ✅ PASSED - Correctly raised error: {type(e).__name__}")
        test_results.append({'test': 'dimension_mismatch', 'status': 'PASS', 'issues': []})
    
    # Test 3: NaN values
    print("\n   Test 3.3: NaN values in data")
    try:
        X_nan = np.random.rand(100, 5)
        X_nan[0, 0] = np.nan
        y_nan = np.random.rand(100)
        model.fit(X_nan, y_nan)
        print("   ⚠️ WARNING - Model accepted NaN values (may handle internally)")
        test_results.append({'test': 'nan_values', 'status': 'WARNING', 'issues': ['NaN values accepted']})
    except Exception as e:
        print(f"   ✅ PASSED - Correctly raised error: {type(e).__name__}")
        test_results.append({'test': 'nan_values', 'status': 'PASS', 'issues': []})
    
    # Test 4: Predict before fit
    print("\n   Test 3.4: Predict before fit")
    try:
        fresh_model = create_optimal_model('stock_prediction', 'primary')
        X_test = np.random.rand(10, 5)
        fresh_model.predict(X_test)
        print("   ⚠️ WARNING - Model allowed prediction before training")
        test_results.append({'test': 'predict_before_fit', 'status': 'WARNING', 'issues': ['Prediction before fit allowed']})
    except Exception as e:
        print(f"   ✅ PASSED - Correctly raised error: {type(e).__name__}")
        test_results.append({'test': 'predict_before_fit', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# TEST 4: Model Creation Consistency
# ============================================================================
def test_model_creation():
    """Test that model creation is consistent"""
    print("\n" + "="*70)
    print("TEST 4: Model Creation Consistency")
    print("="*70)
    
    from models import create_optimal_model
    
    test_results = []
    
    print("\n🔍 Testing model creation consistency...")
    
    # Create same model multiple times
    models = []
    for i in range(3):
        model = create_optimal_model('stock_prediction', 'primary')
        models.append(model)
    
    # Check they're the same type
    model_types = [type(m).__name__ for m in models]
    if len(set(model_types)) == 1:
        print(f"   ✅ PASSED - Consistent model type: {model_types[0]}")
        test_results.append({'test': 'consistent_type', 'status': 'PASS', 'issues': []})
    else:
        print(f"   ❌ FAILED - Inconsistent model types: {model_types}")
        test_results.append({'test': 'consistent_type', 'status': 'FAIL', 'issues': ['Inconsistent model types']})
    
    # Test different model types
    print("\n🔍 Testing different model types...")
    model_types_to_test = ['primary', 'secondary', 'ensemble']
    
    for model_type in model_types_to_test:
        try:
            model = create_optimal_model('stock_prediction', model_type)
            print(f"   ✅ {model_type}: {type(model).__name__}")
            test_results.append({'test': f'create_{model_type}', 'status': 'PASS', 'issues': []})
        except Exception as e:
            print(f"   ❌ {model_type}: Failed - {e}")
            test_results.append({'test': f'create_{model_type}', 'status': 'FAIL', 'issues': [str(e)]})
    
    return test_results

# ============================================================================
# TEST 5: Hardware Detection
# ============================================================================
def test_hardware_detection():
    """Test hardware detection accuracy"""
    print("\n" + "="*70)
    print("TEST 5: Hardware Detection")
    print("="*70)
    
    from models.model_selector import ModelSelector
    
    test_results = []
    
    selector = ModelSelector()
    hw_info = selector.get_hardware_info()
    
    print(f"\n🔍 Detected Hardware:")
    print(f"   GPU Available: {hw_info['gpu_available']}")
    print(f"   Recommended: {hw_info['recommended_approach']}")
    print(f"   Performance Tier: {hw_info['performance_tier']}")
    
    if hw_info['gpu_available']:
        print(f"   GPU Info: {hw_info['gpu_info']}")
        print("\n   💡 GPU Recommendations:")
        for rec in hw_info['gpu_recommendations']:
            print(f"      • {rec}")
    else:
        print("\n   💡 CPU Recommendations:")
        for rec in hw_info['cpu_recommendations']:
            print(f"      • {rec}")
    
    # Validate detection logic
    issues = []
    
    if hw_info['gpu_available'] and hw_info['recommended_approach'] != 'GPU':
        issues.append("GPU detected but not recommended")
    
    if not hw_info['gpu_available'] and hw_info['recommended_approach'] != 'CPU':
        issues.append("No GPU but GPU approach recommended")
    
    if issues:
        print(f"\n   ❌ FAILED:")
        for issue in issues:
            print(f"      • {issue}")
        test_results.append({'test': 'hardware_detection', 'status': 'FAIL', 'issues': issues})
    else:
        print(f"\n   ✅ PASSED - Hardware detection is correct")
        test_results.append({'test': 'hardware_detection', 'status': 'PASS', 'issues': []})
    
    return test_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all Phase 1 tests"""
    
    all_results = {
        'test_1_interfaces': test_model_interfaces(),
        'test_2_parameters': test_model_parameters(),
        'test_3_error_handling': test_error_handling(),
        'test_4_creation': test_model_creation(),
        'test_5_hardware': test_hardware_detection()
    }
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1 TEST SUMMARY")
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
            
            if status == 'PASS':
                passed_tests += 1
                print(f"   ✅ {result.get('model', result.get('config', result.get('test', 'unknown')))}")
            elif status == 'WARNING':
                warnings += 1
                print(f"   ⚠️ {result.get('model', result.get('config', result.get('test', 'unknown')))}")
                for issue in result['issues']:
                    print(f"      • {issue}")
                    all_issues.append(issue)
            else:
                failed_tests += 1
                print(f"   ❌ {result.get('model', result.get('config', result.get('test', 'unknown')))}")
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
        print("\n🎉 PHASE 1: COMPLETE - ALL TESTS PASSED")
        print("✅ Ready to proceed to Phase 2")
        return True
    elif failed_tests == 0:
        print(f"\n⚠️ PHASE 1: COMPLETE WITH WARNINGS")
        print(f"   {warnings} warnings found but no critical failures")
        print("✅ Can proceed to Phase 2 (address warnings later)")
        return True
    else:
        print(f"\n❌ PHASE 1: INCOMPLETE")
        print(f"   {failed_tests} critical failures must be fixed")
        print("\nIssues to fix:")
        for issue in set(all_issues):
            print(f"   • {issue}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
