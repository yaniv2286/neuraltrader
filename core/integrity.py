"""
Core System Integrity Validation
Strict validation module to verify critical system components before trading
"""

import logging
import sys
from typing import Any, Optional


def verify_system_integrity(model_object: Any, strategy_object: Any) -> bool:
    """
    Verify system integrity before allowing trading operations.
    
    Performs strict validation of:
    1. Model object has predict method
    2. Strategy object has check_entry and check_exit methods
    3. Strategy object has max_drawdown limit defined
    
    Args:
        model_object: ML model object to validate
        strategy_object: Trading strategy object to validate
        
    Returns:
        True if all checks pass
        
    Raises:
        SystemExit: If any validation check fails (critical error)
    """
    logger = logging.getLogger('CoreIntegrity')
    
    # Check 1: Model validation
    if model_object is None:
        logger.critical("[CRITICAL] Model object is None. System cannot operate without AI model.")
        logger.critical("[CRITICAL] ABORTING: Missing AI model component.")
        sys.exit(1)
    
    if not hasattr(model_object, 'predict'):
        logger.critical("[CRITICAL] Model object missing 'predict' method.")
        logger.critical("[CRITICAL] ABORTING: Invalid AI model interface.")
        sys.exit(1)
    
    # Check 2: Strategy validation - check_entry method
    if not hasattr(strategy_object, 'check_entry'):
        logger.critical("[CRITICAL] Strategy object missing 'check_entry' method.")
        logger.critical("[CRITICAL] ABORTING: Invalid strategy interface.")
        sys.exit(1)
    
    # Check 2: Strategy validation - check_exit method
    if not hasattr(strategy_object, 'check_exit'):
        logger.critical("[CRITICAL] Strategy object missing 'check_exit' method.")
        logger.critical("[CRITICAL] ABORTING: Invalid strategy interface.")
        sys.exit(1)
    
    # Check 3: Safety validation - max_drawdown limit
    if not hasattr(strategy_object, 'max_drawdown'):
        logger.critical("[CRITICAL] Strategy object missing 'max_drawdown' limit.")
        logger.critical("[CRITICAL] ABORTING: Missing critical safety constraint.")
        sys.exit(1)
    
    # Verify max_drawdown is defined and not None
    max_drawdown = getattr(strategy_object, 'max_drawdown', None)
    if max_drawdown is None:
        logger.critical("[CRITICAL] Strategy 'max_drawdown' is None.")
        logger.critical("[CRITICAL] ABORTING: Safety limit not configured.")
        sys.exit(1)
    
    # All checks passed
    logger.info("[PASS] Core Integrity Verified. AI Online.")
    return True


def verify_model_loaded(model_object: Any, model_name: str = "AI Model") -> bool:
    """
    Verify a model is properly loaded and functional.
    
    Args:
        model_object: Model object to validate
        model_name: Name of the model for logging
        
    Returns:
        True if model is valid
        
    Raises:
        SystemExit: If model validation fails
    """
    logger = logging.getLogger('CoreIntegrity')
    
    if model_object is None:
        logger.critical(f"[CRITICAL] {model_name} is None.")
        logger.critical("[CRITICAL] ABORTING: Model not loaded.")
        sys.exit(1)
    
    if not hasattr(model_object, 'predict'):
        logger.critical(f"[CRITICAL] {model_name} missing 'predict' method.")
        logger.critical("[CRITICAL] ABORTING: Invalid model interface.")
        sys.exit(1)
    
    logger.info(f"[PASS] {model_name} validated successfully.")
    return True


def verify_strategy_safety(strategy_object: Any) -> bool:
    """
    Verify strategy has all required safety constraints.
    
    Args:
        strategy_object: Strategy object to validate
        
    Returns:
        True if all safety constraints are present
        
    Raises:
        SystemExit: If safety validation fails
    """
    logger = logging.getLogger('CoreIntegrity')
    
    required_attributes = ['max_drawdown', 'max_position_size', 'stop_loss']
    missing_attributes = []
    
    for attr in required_attributes:
        if not hasattr(strategy_object, attr):
            missing_attributes.append(attr)
    
    if missing_attributes:
        logger.critical(f"[CRITICAL] Strategy missing safety attributes: {', '.join(missing_attributes)}")
        logger.critical("[CRITICAL] ABORTING: Incomplete safety configuration.")
        sys.exit(1)
    
    # Verify values are not None
    for attr in required_attributes:
        value = getattr(strategy_object, attr, None)
        if value is None:
            logger.critical(f"[CRITICAL] Strategy '{attr}' is None.")
            logger.critical("[CRITICAL] ABORTING: Safety limit not configured.")
            sys.exit(1)
    
    logger.info("[PASS] Strategy safety constraints validated.")
    return True


def verify_risk_limits(risk_manager: Any) -> bool:
    """
    Verify risk manager has required limits configured.
    
    Args:
        risk_manager: Risk manager object to validate
        
    Returns:
        True if risk limits are properly configured
        
    Raises:
        SystemExit: If risk validation fails
    """
    logger = logging.getLogger('CoreIntegrity')
    
    if risk_manager is None:
        logger.critical("[CRITICAL] Risk manager is None.")
        logger.critical("[CRITICAL] ABORTING: No risk management system.")
        sys.exit(1)
    
    required_methods = ['check_risk_limits', 'calculate_position_size']
    missing_methods = []
    
    for method in required_methods:
        if not hasattr(risk_manager, method):
            missing_methods.append(method)
    
    if missing_methods:
        logger.critical(f"[CRITICAL] Risk manager missing methods: {', '.join(missing_methods)}")
        logger.critical("[CRITICAL] ABORTING: Invalid risk manager interface.")
        sys.exit(1)
    
    logger.info("[PASS] Risk manager validated successfully.")
    return True


def verify_complete_system(model_object: Any, 
                          strategy_object: Any, 
                          risk_manager: Optional[Any] = None) -> bool:
    """
    Comprehensive system integrity verification.
    
    Validates all critical components before allowing trading operations.
    
    Args:
        model_object: ML model object
        strategy_object: Trading strategy object
        risk_manager: Optional risk manager object
        
    Returns:
        True if all validations pass
        
    Raises:
        SystemExit: If any validation fails
    """
    logger = logging.getLogger('CoreIntegrity')
    logger.info("[CHECK] Starting comprehensive system integrity verification...")
    
    # Core validation
    verify_system_integrity(model_object, strategy_object)
    
    # Additional safety validation
    verify_strategy_safety(strategy_object)
    
    # Risk manager validation (if provided)
    if risk_manager is not None:
        verify_risk_limits(risk_manager)
    
    logger.info("[PASS] Complete system integrity verified. All systems operational.")
    return True
