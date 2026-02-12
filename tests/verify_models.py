#!/usr/bin/env python3

from pathlib import Path

def verify_production_models():
    print("✅ VERIFYING PRODUCTION MODELS DIRECTORY...")
    
    production_dir = Path("models/production")
    
    # Expected files
    expected_files = {
        'xgboost_model.pkl',
        'lightgbm_model.pkl', 
        'rf_model.pkl',
        'ensemble_metadata.pkl',
        'feature_names.pkl',
        'feature_scaler.pkl'
    }
    
    # Get actual files
    actual_files = set()
    if production_dir.exists():
        actual_files = {f.name for f in production_dir.glob('*.pkl')} | {f.name for f in production_dir.glob('*.json')}
    
    print(f"Expected files: {len(expected_files)}")
    print(f"Actual files: {len(actual_files)}")
    print("")
    
    # Print file details
    print("📄 PRODUCTION MODELS:")
    for file_name in sorted(expected_files):
        file_path = production_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            status = "✅"
        else:
            size_mb = 0
            status = "❌"
        print(f"   {status} {file_name} ({size_mb:.1f} MB)")
    
    # Check for extra files
    extra_files = actual_files - expected_files
    if extra_files:
        print(f"\n⚠️  EXTRA FILES: {extra_files}")
    
    # Final verification
    missing_files = expected_files - actual_files
    if missing_files:
        print(f"\n❌ MISSING FILES: {missing_files}")
        print("❌ VERIFICATION FAILED")
        return False
    else:
        print(f"\n✅ ALL EXPECTED FILES PRESENT!")
        print("✅ VERIFICATION SUCCESSFUL!")
        return True

if __name__ == "__main__":
    verify_production_models()
