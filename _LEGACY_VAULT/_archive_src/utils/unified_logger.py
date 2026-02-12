#!/usr/bin/env python3
"""
NeuralTrader Unified Logger
===========================

Centralized logging system for all 4 pillars.
Accumulative logging to single daily log file.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

class UnifiedLogger:
    """Centralized logging for all pillars"""
    
    def __init__(self, pillar_name: str, log_dir: str = "logs"):
        self.pillar_name = pillar_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate daily log file name
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"NT_{today}.log"
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with daily accumulative logging"""
        logger_name = f"NeuralTrader.{self.pillar_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler (append to daily log)
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler with pillar-specific format
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            f'[{self.pillar_name}] %(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_log_file_path(self) -> str:
        """Get current daily log file path"""
        return str(self.log_file)
    
    def log_pillar_start(self, operation: str):
        """Log pillar operation start"""
        self.logger.info(f"[{self.pillar_name}] Starting {operation}")
        self.logger.info("=" * 80)
    
    def log_pillar_complete(self, operation: str, duration: float, success: bool):
        """Log pillar operation completion"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"[{self.pillar_name}] {operation} {status} in {duration:.2f}s")
        self.logger.info("=" * 80)
    
    def log_error(self, operation: str, error: Exception):
        """Log error with context"""
        self.logger.error(f"[{self.pillar_name}] Error in {operation}: {error}")
        self.logger.error(f"[{self.pillar_name}] Error type: {type(error).__name__}")
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.pillar_name}] {message}")
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[{self.pillar_name}] {message}")
    
    def log_critical(self, message: str):
        """Log critical message"""
        self.logger.critical(f"[{self.pillar_name}] {message}")

# Usage example:
# logger = UnifiedLogger("DataMastery")
# logger.log_pillar_start("Tiingo data fetch")
# ... do work ...
# logger.log_pillar_complete("Tiingo data fetch", duration, success)
