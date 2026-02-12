#!/usr/bin/env python3
"""
NeuralTrader Unified Integrity Test
================================

Critical path validation for the complete system.
This script replaces all individual test scripts and provides comprehensive validation.

Usage:
    python integrity_test.py
"""

import os
import sys
import logging
import requests
import yaml
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class UnifiedIntegrityTest:
    """Comprehensive integrity validation for NeuralTrader"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.checks_passed = []
        self.checks_failed = []
        self.critical_failures = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('IntegrityTest')
        
    def run_all_checks(self) -> bool:
        """Execute all integrity checks"""
        self.logger.info("=" * 80)
        self.logger.info("NeuralTrader Unified Integrity Test")
        self.logger.info("=" * 80)
        
        checks = [
            self.check_project_structure,
            self.check_configuration,
            self.check_environment,
            self.check_internet_connection,
            self.check_tiingo_api,
            self.check_core_modules,
            self.check_data_directories,
            self.check_model_files,
            self.check_ai_models,
            self.check_trading_components,
            self.check_notification_system
        ]
        
        for check in checks:
            try:
                if check():
                    self.checks_passed.append(check.__name__)
                else:
                    self.checks_failed.append(check.__name__)
                    if "critical" in check.__name__ or check.__name__ in [
                        'check_project_structure', 'check_configuration', 'check_core_modules'
                    ]:
                        self.critical_failures.append(check.__name__)
            except Exception as e:
                self.checks_failed.append(check.__name__)
                self.logger.error(f"Check {check.__name__} failed: {e}")
                if "critical" in check.__name__ or check.__name__ in [
                    'check_project_structure', 'check_configuration', 'check_core_modules'
                ]:
                    self.critical_failures.append(check.__name__)
        
        # Generate report
        self._generate_report()
        
        # Fail fast if critical failures
        if self.critical_failures:
            self.logger.critical(f"[CRITICAL] Critical failures: {self.critical_failures}")
            return False
        
        return len(self.checks_failed) == 0
    
    def check_project_structure(self) -> bool:
        """Check project directory structure"""
        self.logger.info("[STRUCTURE] Checking project structure...")
        
        required_dirs = [
            'core', 'src', 'data', 'models', 'docs', 'scripts'
        ]
        
        required_files = [
            'main_orchestrator_ist.py',
            '.gitignore',
            'requirements.txt',
            '.env.example'
        ]
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.logger.warning(f"[STRUCTURE] Missing directory: {dir_name} (will be created)")
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"[STRUCTURE] Created directory: {dir_name}")
                except:
                    self.logger.error(f"[STRUCTURE] Cannot create directory: {dir_name}")
                    return False
        
        # Check files
        for file_name in required_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                self.logger.error(f"[STRUCTURE] Missing file: {file_name}")
                return False
        
        self.logger.info("[STRUCTURE] Project structure valid")
        return True
    
    def check_configuration(self) -> bool:
        """Check configuration files"""
        self.logger.info("[CONFIG] Checking configuration...")
        
        # Check .gitignore
        gitignore_path = self.project_root / '.gitignore'
        if not gitignore_path.exists():
            self.logger.error("[CONFIG] Missing .gitignore file")
            return False
        
        # Check .env.example
        env_example_path = self.project_root / '.env.example'
        if not env_example_path.exists():
            self.logger.error("[CONFIG] Missing .env.example file")
            return False
        
        # Check if .env exists (should exist locally but be gitignored)
        env_path = self.project_root / '.env'
        if not env_path.exists():
            self.logger.warning("[CONFIG] .env file not found (create from .env.example)")
        
        self.logger.info("[CONFIG] Configuration files valid")
        return True
    
    def check_environment(self) -> bool:
        """Check Python environment and required packages"""
        self.logger.info("[ENV] Checking environment...")
        
        # Core packages that must exist
        critical_packages = ['pandas', 'numpy', 'yfinance', 'requests']
        
        # Optional packages (nice to have but not critical)
        optional_packages = ['pyyaml', 'xgboost', 'lightgbm', 'scikit-learn', 'pytz']
        
        missing_critical = []
        missing_optional = []
        
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing_critical.append(package)
        
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)
        
        if missing_critical:
            self.logger.error(f"[ENV] Missing critical packages: {', '.join(missing_critical)}")
            return False
        
        if missing_optional:
            self.logger.warning(f"[ENV] Missing optional packages: {', '.join(missing_optional)}")
        
        self.logger.info("[ENV] Environment check passed")
        return True
    
    def check_internet_connection(self) -> bool:
        """Check internet connectivity"""
        self.logger.info("[NET] Checking internet connection...")
        
        try:
            response = requests.get('https://google.com', timeout=10)
            if response.status_code == 200:
                self.logger.info("[NET] Internet connection available")
                return True
        except:
            pass
        
        self.logger.error("[NET] No internet connection")
        return False
    
    def check_tiingo_api(self) -> bool:
        """Check Tiingo API accessibility"""
        self.logger.info("[API] Checking Tiingo API...")
        
        # Load environment variables
        env_path = self.project_root / '.env'
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
        
        api_key = os.getenv('TIINGO_API_KEY')
        if not api_key:
            self.logger.warning("[API] Tiingo API key not found in .env")
            return False  # Not critical, but warning
        
        try:
            headers = {'Authorization': f'Bearer {api_key}'}
            response = requests.get('https://api.tiingo.com/api/test', 
                                  headers=headers, timeout=10)
            if response.status_code == 200:
                self.logger.info("[API] Tiingo API accessible")
                return True
        except:
            pass
        
        self.logger.warning("[API] Tiingo API not accessible")
        return False
    
    def check_core_modules(self) -> bool:
        """Check core modules can be imported"""
        self.logger.info("[CORE] Checking core modules...")
        
        core_modules = [
            'core.ai_models',
            'core.integrity',
            'src.data.tiingo_manager',
            'src.data.yfinance_manager',
            'src.trading.risk_manager',
            'src.trading.virtual_engine',
            'src.utils.notifier',
            'src.features.feature_engineer'
        ]
        
        failed_imports = []
        for module in core_modules:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
        
        if failed_imports:
            self.logger.error(f"[CORE] Failed imports: {failed_imports}")
            return False
        
        self.logger.info("[CORE] All core modules importable")
        return True
    
    def check_data_directories(self) -> bool:
        """Check data directories exist and are accessible"""
        self.logger.info("[DATA] Checking data directories...")
        
        data_dirs = [
            'data/cache',
            'data/cache/tiingo',
            'data/cache/yfinance'
        ]
        
        for dir_name in data_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"[DATA] Created directory: {dir_name}")
                except:
                    self.logger.error(f"[DATA] Cannot create directory: {dir_name}")
                    return False
        
        self.logger.info("[DATA] Data directories accessible")
        return True
    
    def check_model_files(self) -> bool:
        """Check model files exist and are loadable"""
        self.logger.info("[MODELS] Checking model files...")
        
        models_dir = self.project_root / 'models'
        production_dir = models_dir / 'production'
        
        # Check if production models exist
        if production_dir.exists():
            model_files = list(production_dir.glob("*.pkl"))
            if not model_files:
                self.logger.warning("[MODELS] No production models found")
                return True  # Not critical for first run
        else:
            production_dir.mkdir(parents=True, exist_ok=True)
        
        # Check main models directory
        main_models = ['xgboost_model.pkl', 'lightgbm_model.pkl', 'rf_model.pkl']
        for model_file in main_models:
            model_path = models_dir / model_file
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        pickle.load(f)
                    self.logger.info(f"[MODELS] Model loadable: {model_file}")
                except Exception as e:
                    self.logger.error(f"[MODELS] Cannot load {model_file}: {e}")
                    return False
        
        self.logger.info("[MODELS] Model files check completed")
        return True
    
    def check_ai_models(self) -> bool:
        """Check AI models can be instantiated"""
        self.logger.info("[AI] Checking AI models...")
        
        try:
            from core.ai_models import EnsemblePredictor
            
            # Try to create ensemble predictor
            ensemble = EnsemblePredictor()
            info = ensemble.get_ensemble_info()
            
            self.logger.info(f"[AI] Ensemble loaded: {info['n_models']} models")
            self.logger.info(f"[AI] Features: {info['n_features']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[AI] Failed to load AI models: {e}")
            return False
    
    def check_trading_components(self) -> bool:
        """Check trading components"""
        self.logger.info("[TRADING] Checking trading components...")
        
        try:
            from src.trading.risk_manager import RiskManager
            from src.trading.virtual_engine import VirtualEngine
            
            # Test Risk Manager
            risk_manager = RiskManager()
            self.logger.info("[TRADING] Risk Manager initialized")
            
            # Test Virtual Engine
            virtual_engine = VirtualEngine()
            self.logger.info("[TRADING] Virtual Engine initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[TRADING] Trading components failed: {e}")
            return False
    
    def check_notification_system(self) -> bool:
        """Check notification system"""
        self.logger.info("[NOTIFY] Checking notification system...")
        
        try:
            from src.utils.notifier import EmailNotifier
            
            # Try to create notifier (will fail if credentials not set)
            notifier = EmailNotifier()
            self.logger.info("[NOTIFY] Email notifier initialized")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"[NOTIFY] Notification system issue: {e}")
            return False  # Not critical
    
    def _generate_report(self):
        """Generate integrity test report"""
        total_checks = len(self.checks_passed) + len(self.checks_failed)
        passed_count = len(self.checks_passed)
        
        self.logger.info("=" * 80)
        self.logger.info("INTEGRITY TEST REPORT")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Checks: {total_checks}")
        self.logger.info(f"Passed: {passed_count}")
        self.logger.info(f"Failed: {len(self.checks_failed)}")
        
        if self.checks_passed:
            self.logger.info(f"Passed: {', '.join(self.checks_passed)}")
        
        if self.checks_failed:
            self.logger.info(f"Failed: {', '.join(self.checks_failed)}")
        
        if self.critical_failures:
            self.logger.error(f"Critical Failures: {', '.join(self.critical_failures)}")
        
        # Overall status
        if len(self.checks_failed) == 0:
            self.logger.info("[SUCCESS] All integrity checks passed")
        elif self.critical_failures:
            self.logger.error("[CRITICAL] Critical failures detected")
        else:
            self.logger.warning("[WARNING] Some checks failed")
        
        self.logger.info("=" * 80)

if __name__ == "__main__":
    test = UnifiedIntegrityTest()
    success = test.run_all_checks()
    sys.exit(0 if success else 1)
