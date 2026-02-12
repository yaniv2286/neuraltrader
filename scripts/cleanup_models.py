#!/usr/bin/env python3
"""
Model Directory Cleanup & Standardization Script
==============================================

Cleans up the models/production/ directory and standardizes naming conventions
for the Modern Era brain.

Usage:
    python scripts/cleanup_models.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelCleanup:
    """Cleans up and standardizes the production models directory"""
    
    def __init__(self):
        """Initialize cleanup operation"""
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'models'
        self.production_dir = self.models_dir / 'production'
        
        # Create archive directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_Cleanup')
        self.archive_dir = self.models_dir / 'archive' / timestamp
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧹 Model Cleanup Initialized")
        logger.info(f"   Production Dir: {self.production_dir}")
        logger.info(f"   Archive Dir: {self.archive_dir}")
    
    def identify_new_brain_files(self) -> dict:
        """Identify the new Modern Era brain files"""
        new_brain_files = {
            'xgboost': 'xgboostmodel_model.pkl',
            'lightgbm': 'lightgbmmodel_model.pkl', 
            'rf': 'randomforestmodel_model.pkl',
            'metadata': 'ensemble_metadata.pkl',
            'feature_names': 'feature_names.pkl',
            'scaler': 'feature_scaler.pkl'
        }
        
        logger.info("🧠 Identifying New Modern Era Brain Files...")
        for key, filename in new_brain_files.items():
            filepath = self.production_dir / filename
            if filepath.exists():
                logger.info(f"   ✅ Found: {filename}")
            else:
                logger.warning(f"   ❌ Missing: {filename}")
        
        return new_brain_files
    
    def standardize_new_models(self, new_brain_files: dict) -> dict:
        """Rename new models to standard naming convention"""
        standard_names = {
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'rf': 'rf_model.pkl',
            'metadata': 'ensemble_metadata.pkl',  # Keep as is
            'feature_names': 'feature_names.pkl',  # Keep as is
            'scaler': 'feature_scaler.pkl'  # Keep as is
        }
        
        logger.info("📝 Standardizing New Model Names...")
        renamed_files = {}
        
        for key, old_name in new_brain_files.items():
            old_path = self.production_dir / old_name
            new_name = standard_names[key]
            new_path = self.production_dir / new_name
            
            if old_path.exists():
                if old_name != new_name:
                    # Rename the file
                    old_path.rename(new_path)
                    logger.info(f"   🔄 Renamed: {old_name} -> {new_name}")
                else:
                    logger.info(f"   ✅ Already standard: {new_name}")
                
                renamed_files[key] = new_name
            else:
                logger.warning(f"   ❌ File not found: {old_name}")
        
        return renamed_files
    
    def archive_old_files(self, keep_files: list) -> int:
        """Archive all files except the ones we want to keep"""
        logger.info("📦 Archiving Old Model Files...")
        
        archived_count = 0
        keep_set = set(keep_files)
        
        # Get all files in production directory
        all_files = list(self.production_dir.glob('*.pkl')) + list(self.production_dir.glob('*.json'))
        
        for file_path in all_files:
            if file_path.name not in keep_set:
                # Move to archive
                archive_path = self.archive_dir / file_path.name
                shutil.move(str(file_path), str(archive_path))
                logger.info(f"   📦 Archived: {file_path.name}")
                archived_count += 1
        
        return archived_count
    
    def update_loading_logic(self):
        """Update model loading logic to use standard names"""
        logger.info("🔧 Updating Model Loading Logic...")
        
        # Files to update
        files_to_check = [
            'core/brain_gate.py',
            'core/model_factory.py',
            'core/ensemble.py',
            'scripts/retrain_ensemble.py'
        ]
        
        updated_files = []
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    # Read the file
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if it needs updating (looks for old model names)
                    old_patterns = [
                        'xgboostmodel_model.pkl',
                        'lightgbmmodel_model.pkl',
                        'randomforestmodel_model.pkl'
                    ]
                    
                    needs_update = any(pattern in content for pattern in old_patterns)
                    
                    if needs_update:
                        # Replace old names with standard names
                        content = content.replace('xgboostmodel_model.pkl', 'xgboost_model.pkl')
                        content = content.replace('lightgbmmodel_model.pkl', 'lightgbm_model.pkl')
                        content = content.replace('randomforestmodel_model.pkl', 'rf_model.pkl')
                        
                        # Write back
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        logger.info(f"   ✅ Updated: {file_path}")
                        updated_files.append(file_path)
                    else:
                        logger.info(f"   ✅ No update needed: {file_path}")
                
                except Exception as e:
                    logger.error(f"   ❌ Failed to update {file_path}: {e}")
            else:
                logger.info(f"   ⚠️  File not found: {file_path}")
        
        return updated_files
    
    def verify_final_state(self) -> bool:
        """Verify the final state of the production directory"""
        logger.info("✅ Verifying Final Production Directory State...")
        
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
        if self.production_dir.exists():
            actual_files = {f.name for f in self.production_dir.glob('*.pkl')} | {f.name for f in self.production_dir.glob('*.json')}
        
        # Check if we have exactly the expected files
        logger.info(f"   Expected files: {len(expected_files)}")
        logger.info(f"   Actual files: {len(actual_files)}")
        
        # Print actual files
        for file_name in sorted(actual_files):
            file_path = self.production_dir / file_name
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"   📄 {file_name} ({size_mb:.1f} MB)")
        
        # Check for missing files
        missing_files = expected_files - actual_files
        if missing_files:
            logger.error(f"   ❌ Missing files: {missing_files}")
            return False
        
        # Check for extra files
        extra_files = actual_files - expected_files
        if extra_files:
            logger.warning(f"   ⚠️  Extra files: {extra_files}")
        
        # Check if all expected files exist and are the right size
        all_good = True
        for expected_file in expected_files:
            file_path = self.production_dir / expected_file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb < 0.1:  # Less than 100KB seems suspicious
                    logger.warning(f"   ⚠️  {expected_file} seems small: {size_mb:.1f} MB")
                    all_good = False
                else:
                    logger.info(f"   ✅ {expected_file} ({size_mb:.1f} MB)")
            else:
                logger.error(f"   ❌ {expected_file} not found!")
                all_good = False
        
        return all_good
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        logger.info("🚀 Starting Model Directory Cleanup...")
        
        try:
            # Step 1: Identify new brain files
            new_brain_files = self.identify_new_brain_files()
            
            # Step 2: Standardize new model names
            standard_files = self.standardize_new_models(new_brain_files)
            keep_files = list(standard_files.values())
            
            # Step 3: Archive old files
            archived_count = self.archive_old_files(keep_files)
            
            # Step 4: Update loading logic
            updated_files = self.update_loading_logic()
            
            # Step 5: Verify final state
            success = self.verify_final_state()
            
            # Summary
            logger.info("=" * 60)
            logger.info("🧹 MODEL CLEANUP SUMMARY")
            logger.info("=" * 60)
            logger.info(f"   Archive Directory: {self.archive_dir}")
            logger.info(f"   Files Archived: {archived_count}")
            logger.info(f"   Files Updated: {len(updated_files)}")
            logger.info(f"   Final State: {'✅ SUCCESS' if success else '❌ FAILED'}")
            logger.info("=" * 60)
            
            if success:
                logger.info("🎉 Model cleanup completed successfully!")
                logger.info("🧠 Modern Era brain is now standardized and ready!")
            else:
                logger.error("❌ Model cleanup failed - check logs above")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed with error: {e}")
            return False


def main():
    """Main execution function"""
    cleanup = ModelCleanup()
    success = cleanup.run_cleanup()
    
    if success:
        logger.info("🚀 Model cleanup completed successfully!")
    else:
        logger.error("❌ Model cleanup failed!")
        exit(1)


if __name__ == "__main__":
    main()
