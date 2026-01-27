"""
Test Runner for NeuralTrader
Orchestrates all testing phases
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all available tests"""
    print("🧪 NEURALTRADER TEST RUNNER")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    test_results = {}
    
    # Phase 1 Infrastructure Tests
    print(f"\n🏗️ Phase 1: Infrastructure Tests")
    print("-" * 40)
    
    try:
        from test_phase1_infrastructure import test_phase1_infrastructure
        test_results['phase1_infrastructure'] = test_phase1_infrastructure()
    except Exception as e:
        print(f"❌ Phase 1 tests failed: {e}")
        test_results['phase1_infrastructure'] = False
    
    # Phase 2 CPU Models Tests
    print(f"\n💻 Phase 2: CPU Models Tests")
    print("-" * 40)
    
    try:
        from test_cpu_models import test_cpu_models
        test_results['cpu_models'] = test_cpu_models()
    except Exception as e:
        print(f"❌ CPU models tests failed: {e}")
        test_results['cpu_models'] = False
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25}: {status}")
    
    print(f"\n📈 Overall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"🚀 NeuralTrader is production-ready!")
        
        if test_results.get('phase1_infrastructure', False):
            print(f"⚠️ Phase 1 has issues - fix before proceeding to Phase 2")
        elif test_results.get('cpu_models', False):
            print(f"⚠️ CPU models have issues - fix before proceeding")
        else:
            print(f"✅ Ready for Phase 2: Basic Models & Validation!")
    else:
        print(f"\n❌ {total - passed} test suites failed")
        print(f"🔧 Fix issues before proceeding")
    
    return test_results

def run_phase1_only():
    """Run only Phase 1 tests"""
    print("🧪 PHASE 1 ONLY TEST RUNNER")
    print("=" * 40)
    
    try:
        from test_phase1_infrastructure import test_phase1_infrastructure
        results = test_phase1_infrastructure()
        
        if results:
            print(f"\n✅ Phase 1 Infrastructure: PASSED")
            print(f"🚀 Ready to proceed to Phase 2!")
        else:
            print(f"\n❌ Phase 1 Infrastructure: FAILED")
            print(f"🔧 Fix issues before proceeding")
            
        return results
        
    except Exception as e:
        print(f"❌ Phase 1 test error: {e}")
        return False

def run_cpu_models_only():
    """Run only CPU models tests"""
    print("💻 CPU MODELS ONLY TEST RUNNER")
    print("=" * 40)
    
    try:
        from test_cpu_models import test_cpu_models
        results = test_cpu_models()
        
        if results:
            print(f"\n✅ CPU Models: PASSED")
            print(f"🤖 Models working correctly without GPU")
        else:
            print(f"\n❌ CPU Models: FAILED")
            print(f"🔧 Fix model issues")
            
        return results
        
    except Exception as e:
        print(f"❌ CPU models test error: {e}")
        return False

def quick_smoke_test():
    """Quick smoke test to verify basic functionality"""
    print("💨 QUICK SMOKE TEST")
    print("=" * 30)
    
    try:
        # Test imports
        from data.tiingo_loader import get_available_tickers
        from models import print_model_status
        
        # Test data availability
        tickers = get_available_tickers()
        print(f"✅ Data cache: {len(tickers)} tickers available")
        
        # Test model status
        print_model_status()
        
        print(f"\n✅ Smoke test PASSED!")
        print(f"🚀 NeuralTrader basic functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Smoke test FAILED: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralTrader Test Runner')
    parser.add_argument('--phase1', action='store_true', help='Run Phase 1 tests only')
    parser.add_argument('--cpu', action='store_true', help='Run CPU models tests only')
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke test')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    
    args = parser.parse_args()
    
    if args.smoke:
        quick_smoke_test()
    elif args.phase1:
        run_phase1_only()
    elif args.cpu:
        run_cpu_models_only()
    else:
        run_all_tests()
