"""
Core Protection Protocol - Critical Trading Logic
==================================================

This package contains the protected core trading logic:
- AI Models: Machine learning inference engines
- Indicators: Technical indicators (RSI, ATR, volatility, momentum)
- Strategy: Entry/exit decision rules and risk management

All critical trading logic is isolated here for protection and maintainability.
"""

from .ai_models import (
    XGBoostInference,
    EnsemblePredictor,
    get_trading_signal,
    get_ensemble_signal
)

from .indicators import (
    TechnicalIndicators,
    calculate_volatility,
    calculate_historical_volatility,
    calculate_rsi,
    calculate_macd,
    calculate_momentum_score
)

from .strategy import (
    TradingStrategy,
    RiskManager
)

from .integrity import (
    verify_system_integrity,
    verify_model_loaded,
    verify_strategy_safety,
    verify_risk_limits,
    verify_complete_system
)

__all__ = [
    # AI Models
    'XGBoostInference',
    'EnsemblePredictor',
    'get_trading_signal',
    'get_ensemble_signal',
    
    # Indicators
    'TechnicalIndicators',
    'calculate_volatility',
    'calculate_historical_volatility',
    'calculate_rsi',
    'calculate_macd',
    'calculate_momentum_score',
    
    # Strategy
    'TradingStrategy',
    'RiskManager',
    
    # Integrity
    'verify_system_integrity',
    'verify_model_loaded',
    'verify_strategy_safety',
    'verify_risk_limits',
    'verify_complete_system',
]

__version__ = '1.0.0'
__author__ = 'NeuralTrader Team'
