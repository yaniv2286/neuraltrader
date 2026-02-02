"""
NeuralTrader Trading Module
==========================

This module contains trading-related functionality for NeuralTrader Phase 6 deployment.

Modules:
- data_manager: Market data fetching and validation (The Sentry)
- risk_manager: Risk management and position sizing (The Constitution)
- execution_manager: Order execution and logging (The Bridge)
- alpaca_paper_trading: Complete paper trading system

Usage:
    from src.trading import DataManager, RiskManager, ExecutionManager
    
    # Initialize modules
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
