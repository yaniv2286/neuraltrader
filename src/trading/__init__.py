"""
NeuralTrader Trading Module
==========================

This module contains trading-related functionality for NeuralTrader.

Modules:
- alpaca_paper_trading: Paper trading implementation with Alpaca API
- (Future modules for live trading, order management, etc.)

Usage:
    from src.trading import NeuralTraderPaperTrading
    
    trader = NeuralTraderPaperTrading()
    trader.run_trading_session()
"""

from .alpaca_paper_trading import NeuralTraderPaperTrading

__all__ = ['NeuralTraderPaperTrading']
