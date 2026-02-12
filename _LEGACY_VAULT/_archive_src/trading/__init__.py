"""
NeuralTrader Trading Module (Shadow Trading Version)
=========================================

This module contains trading-related functionality for NeuralTrader Phase 6 Shadow Trading.

Modules:
- data_manager: Market data fetching and validation (The Sentry) - Yahoo Finance integration
- risk_manager: Risk management and position sizing (The Constitution) - Maintained logic
- virtual_engine: Virtual portfolio management (The Bridge) - Shadow trading
- alpaca_paper_trading: Legacy Alpaca system (deprecated)

Usage:
    from src.trading import DataManager, RiskManager, VirtualEngine
    
    # Initialize modules (Shadow Trading)
    dm = DataManager()
    rm = RiskManager()
    ve = VirtualEngine()
    
    # Run shadow trading workflow
    data = dm.fetch_daily_data(['AAPL', 'MSFT', 'NVDA'])
    decision, details = rm.evaluate_trade('AAPL', 150.0, account, positions)
    result = ve.execute_order('AAPL', 'buy', 100, 150.0)
"""

from .data_manager import DataManager
from .risk_manager import RiskManager, RiskDecision
from .virtual_engine import VirtualEngine

__all__ = [
    'DataManager',
    'RiskManager', 
    'RiskDecision',
    'VirtualEngine',
    'NeuralTraderPaperTrading'
]
