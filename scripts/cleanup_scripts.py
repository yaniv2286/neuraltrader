#!/usr/bin/env python3
"""
Scripts Directory Cleanup - Archive Obsolete Files
==============================================

Archives redundant and obsolete Python scripts to prevent confusion
and maintain a clean, organized scripts directory.

Usage:
    python scripts/cleanup_scripts.py
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ScriptsCleanup')

class ScriptsCleanup:
    """Cleans up the scripts directory by archiving obsolete files"""
    
    def __init__(self):
        """Initialize scripts cleanup"""
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / 'scripts'
        self.archive_dir = self.scripts_dir / 'archive' / '20260210_Cleanup'
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧹 Scripts Cleanup Initialized")
        logger.info(f"   Scripts Directory: {self.scripts_dir}")
        logger.info(f"   Archive Directory: {self.archive_dir}")
    
    def identify_junk_files(self) -> list:
        """Identify obsolete files to archive"""
        logger.info("🔍 Identifying obsolete files for archiving...")
        
        # Kill List - Files to archive
        junk_files = [
            'visualize_shield.py',           # OBSOLETE - Replaced by visualize_shield_modern.py
            'generate_brain_dump.py',         # OBSOLETE - Old export tool
            'executive_briefing.py'           # OBSOLETE - Old reporting logic
        ]
        
        # Verify files exist before adding to list
        existing_junk = []
        for file_name in junk_files:
            file_path = self.scripts_dir / file_name
            if file_path.exists():
                existing_junk.append(file_name)
                logger.info(f"   🗑️ Found junk file: {file_name}")
            else:
                logger.info(f"   ⚠️  Junk file not found: {file_name}")
        
        return existing_junk
    
    def identify_essential_files(self) -> list:
        """Identify essential files to keep"""
        logger.info("📋 Identifying essential files to keep...")
        
        # Essential files - DO NOT TOUCH
        essential_files = [
            'retrain_ensemble.py',         # CRITICAL - Brain retraining
            'optimize_strategy.py',          # CRITICAL - Strategy optimization
            'visualize_shield_modern.py',      # CRITICAL - Modern Shield audit
            'cleanup_scripts.py'              # CURRENT - This script
        ]
        
        # Verify files exist before adding to list
        existing_essential = []
        for file_name in essential_files:
            file_path = self.scripts_dir / file_name
            if file_path.exists():
                existing_essential.append(file_name)
                logger.info(f"   ✅ Found essential file: {file_name}")
            else:
                logger.warning(f"   ❌ Essential file not found: {file_name}")
        
        return existing_essential
    
    def archive_junk_files(self, junk_files: list) -> int:
        """Move junk files to archive directory"""
        logger.info("📦 Archiving junk files...")
        
        archived_count = 0
        
        for file_name in junk_files:
            source_path = self.scripts_dir / file_name
            target_path = self.archive_dir / file_name
            
            try:
                shutil.move(str(source_path), str(target_path))
                logger.info(f"   📦 Archived: {file_name}")
                archived_count += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to archive {file_name}: {e}")
        
        return archived_count
    
    def verify_cleanup(self) -> bool:
        """Verify the cleanup was successful"""
        logger.info("✅ Verifying cleanup results...")
        
        # Get all files in scripts directory
        all_files = []
        if self.scripts_dir.exists():
            all_files = [f.name for f in self.scripts_dir.iterdir() if f.is_file()]
        
        # Check that junk files are gone
        remaining_junk = [f for f in all_files if f in ['visualize_shield.py', 'generate_brain_dump.py', 'executive_briefing.py']]
        if remaining_junk:
            logger.error(f"❌ Junk files still present: {remaining_junk}")
            return False
        
        # Check that essential files remain
        essential_files = ['retrain_ensemble.py', 'optimize_strategy.py', 'visualize_shield_modern.py', 'cleanup_scripts.py']
        remaining_essential = [f for f in all_files if f in essential_files]
        if len(remaining_essential) != len(essential_files):
            missing_files = set(essential_files) - set(remaining_essential)
            logger.error(f"❌ Missing essential files: {missing_files}")
            return False
        
        # Print final directory contents
        logger.info("📁 Final scripts directory contents:")
        for file_name in sorted(all_files):
            file_path = self.scripts_dir / file_name
            size_kb = file_path.stat().st_size / 1024
            status = "✅" if file_name in essential_files else "📄"
            logger.info(f"   {status} {file_name} ({size_kb:.1f} KB)")
        
        return True
    
    def run_cleanup(self):
        """Run the complete scripts cleanup process"""
        try:
            logger.info("🧹 Starting Scripts Directory Cleanup...")
            
            # Step 1: Identify junk files
            junk_files = self.identify_junk_files()
            
            if not junk_files:
                logger.info("✅ No junk files found - scripts directory is clean!")
                return True
            
            # Step 2: Archive junk files
            archived_count = self.archive_junk_files(junk_files)
            
            # Step 3: Verify cleanup
            success = self.verify_cleanup()
            
            # Summary
            logger.info("=" * 60)
            logger.info("🧹 SCRIPTS CLEANUP SUMMARY")
            logger.info("=" * 60)
            logger.info(f"   Archive Directory: {self.archive_dir}")
            logger.info(f"   Files Archived: {archived_count}")
            logger.info(f"   Final Scripts: {len([f for f in self.scripts_dir.iterdir() if f.is_file()])}")
            logger.info(f"   Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            logger.info("=" * 60)
            
            if success:
                logger.info("🚀 Scripts cleanup completed successfully!")
                logger.info("📁 Scripts directory is now clean and organized!")
            else:
                logger.error("❌ Scripts cleanup failed!")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed with error: {e}")
            return False


def main():
    """Main execution function"""
    cleanup = ScriptsCleanup()
    success = cleanup.run_cleanup()
    
    if success:
        logger.info("🚀 Scripts cleanup completed successfully!")
    else:
        logger.error("❌ Scripts cleanup failed!")
        exit(1)


if __name__ == "__main__":
    main()
