"""
NeuralTrader Utilities
======================

Utility modules for NeuralTrader Phase 6.1 production environment.

Modules:
- notifier: Email notification system for daily briefs and alerts

Usage:
    from src.utils import EmailNotifier
    
    notifier = EmailNotifier()
    notifier.send_daily_brief(account_info, positions, trades)
"""

from .notifier import EmailNotifier

__all__ = ['EmailNotifier']
