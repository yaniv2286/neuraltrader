"""
NeuralTrader Trading Module (IBKR Version)
======================================

This module contains trading-related functionality for NeuralTrader Phase 6 International Deployment.

Modules:
- data_manager: Market data fetching and validation (The Sentry) - IBKR integration
- risk_manager: Risk management and position sizing (The Constitution) - Maintained logic
- execution_manager: Order execution and logging (The Bridge) - IBKR integration
- alpaca_paper_trading: Legacy Alpaca system (deprecated)

Usage:
    from src.trading import DataManager, RiskManager, ExecutionManager
    
    # Initialize modules (IBKR)
    dm = DataManager()
    rm = RiskManager()
    em = ExecutionManager()
    
    # Run trading workflow
    data = dm.fetch_daily_data()
    decision, details = rm.evaluate_trade('AAPL', 150.0, account, positions)
    result = em.execute_order('AAPL', 'buy', 100, 150.0)
"""

from .data_manager import DataManager
from .risk_manager import RiskManager, RiskDecision
from .execution_manager import ExecutionManager, ExecutionResult
from .alpaca_paper_trading import NeuralTraderPaperTrading

__all__ = [
    'DataManager',
    'RiskManager', 
    'RiskDecision',
    'ExecutionManager',
    'ExecutionResult',
    'NeuralTraderPaperTrading'
]
