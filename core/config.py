"""
Core Configuration Module - NeuralTrader Institutional Schedule
Contains all system configuration parameters and operational settings
"""

import os
import yaml
from datetime import datetime, time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class NeuralTraderConfig:
    """
    NeuralTrader Configuration with Institutional Schedule
    """
    
    def __init__(self):
        """Initialize configuration with institutional schedule parameters"""
        
        # [ARCH] INSTITUTIONAL SCHEDULE CONFIGURATION
        self.OPERATIONAL_WINDOW_START = time(17, 15)  # 5:15 PM institutional start
        self.OPERATIONAL_WINDOW_END = time(21, 0)    # 9:00 PM institutional end
        self.DATA_FRESHNESS_THRESHOLD_HOURS = 2      # Maximum age for data freshness
        
        # [DATA] DATA VALIDATION CONFIGURATION
        self.PROCESSED_DATA_PATH = "data/processed/modern_era_universe.parquet"
        self.REQUIRE_CURRENT_DATA = True              # Fail Fast if data is stale
        self.DATA_STALENESS_CRASH = True             # CRASH on stale data detection
        
        # [EXEC] EXECUTION SAFEGUARDS
        self.PRE_EXECUTION_DATA_CHECK = True         # Verify data freshness before trades
        self.FAIL_FAST_ON_STALE_DATA = True          # Immediate crash on data issues
        
        # 📋 CORE SYSTEM CONFIGURATION
        self.load_base_config()
        
        # Log institutional schedule
        logger.info("[ARCH] INSTITUTIONAL SCHEDULE CONFIGURED")
        logger.info(f"   Operational Window: {self.OPERATIONAL_WINDOW_START.strftime('%H:%M')} - {self.OPERATIONAL_WINDOW_END.strftime('%H:%M')}")
        logger.info(f"   Data Freshness Threshold: {self.DATA_FRESHNESS_THRESHOLD_HOURS} hours")
        logger.info(f"   Pre-execution Data Check: {self.PRE_EXECUTION_DATA_CHECK}")
        logger.info(f"   Fail Fast on Stale Data: {self.FAIL_FAST_ON_STALE_DATA}")
    
    def load_base_config(self):
        """Load base configuration from YAML file if available"""
        try:
            config_path = "_LEGACY_VAULT/config/config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.base_config = yaml.safe_load(f)
                logger.info("[OK] Base configuration loaded from YAML")
            else:
                self.base_config = {}
                logger.warning("[WARN] Base config file not found, using defaults")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load base config: {e}")
            self.base_config = {}
    
    def is_operational_window(self) -> bool:
        """
        Check if current time is within institutional operational window
        
        Returns:
            True if within operational window, False otherwise
        """
        current_time = datetime.now().time()
        
        # Check if current time is within operational window
        if self.OPERATIONAL_WINDOW_START <= current_time <= self.OPERATIONAL_WINDOW_END:
            return True
        
        logger.warning(f"[TIME] Current time {current_time.strftime('%H:%M')} outside operational window "
                       f"({self.OPERATIONAL_WINDOW_START.strftime('%H:%M')} - {self.OPERATIONAL_WINDOW_END.strftime('%H:%M')})")
        return False
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value from nested dictionary using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'risk.max_positions')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.base_config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"❌ Failed to get config value for {key_path}: {e}")
            return default
    
    def validate_data_freshness(self) -> bool:
        """
        Validate that processed data is current (today's date)
        
        Returns:
            True if data is fresh, False otherwise
            
        Raises:
            SystemExit: If data is stale and FAIL_FAST_ON_STALE_DATA is True
        """
        try:
            import pandas as pd
            
            if not os.path.exists(self.PROCESSED_DATA_PATH):
                error_msg = f"CRITICAL: Processed data file not found: {self.PROCESSED_DATA_PATH}"
                logger.error(error_msg)
                if self.FAIL_FAST_ON_STALE_DATA:
                    raise SystemExit(error_msg)
                return False
            
            # Load the data and check latest timestamp
            df = pd.read_parquet(self.PROCESSED_DATA_PATH)
            
            if df.empty:
                error_msg = "CRITICAL: Processed data file is empty"
                logger.error(error_msg)
                if self.FAIL_FAST_ON_STALE_DATA:
                    raise SystemExit(error_msg)
                return False
            
            # Get the latest date in the dataset
            latest_date = df.index.max()
            today = datetime.now().date()
            
            # Check if data is current (today's date)
            if latest_date.date() != today:
                error_msg = f"CRITICAL: Stale data detected. Latest data: {latest_date.date()}, Today: {today}. Execution aborted for safety."
                logger.error(error_msg)
                if self.FAIL_FAST_ON_STALE_DATA:
                    raise SystemExit(error_msg)
                return False
            
            logger.info(f"[PASS] Data freshness validated: Latest data {latest_date.date()} matches today {today}")
            return True
            
        except Exception as e:
            error_msg = f"CRITICAL: Data freshness validation failed: {e}"
            logger.error(error_msg)
            if self.FAIL_FAST_ON_STALE_DATA:
                raise SystemExit(error_msg)
            return False
    
    def pre_execution_check(self) -> bool:
        """
        Perform pre-execution checks including data freshness
        
        Returns:
            True if all checks pass, False otherwise
            
        Raises:
            SystemExit: If critical checks fail and FAIL_FAST_ON_STALE_DATA is True
        """
        logger.info("[CHECK] PRE-EXECUTION CHECKS STARTED")
        
        # Check operational window
        if not self.is_operational_window():
            logger.warning("[WARN] Outside operational window")
            return False
        
        # Check data freshness if required
        if self.PRE_EXECUTION_DATA_CHECK:
            if not self.validate_data_freshness():
                logger.error("[FAIL] Data freshness check failed")
                return False
        
        logger.info("[PASS] PRE-EXECUTION CHECKS PASSED")
        return True


# Global configuration instance
config = NeuralTraderConfig()
