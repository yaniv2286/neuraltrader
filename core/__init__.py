"""
NeuralTrader Core Engine
========================
Single namespace for all engine components.

Subpackages:
  core.sentiment/   - Sentiment intelligence (economic, news, social)
  core.execution/   - Risk manager (quant position sizing)
  core.utils/       - Email notifier, logging utilities

Modules:
  core.ai_models                 - EnsemblePredictor (XGBoost + LightGBM + RF)
  core.feature_engineer          - 64-feature vector generation
  core.sentiment_feature_engineer- 76-feature vector (sentiment mode)
  core.indicators                - Technical indicators (RSI, ATR, MACD ...)
  core.strategy                  - TradingStrategy (signal logic)
  core.ibkr_engine               - IBKR TWS integration
  core.integrity                 - verify_system_integrity()
"""

__all__ = [
    'EnsemblePredictor',
    'FeatureEngineer',
    'TradingStrategy',
]
