#!/usr/bin/env python3
"""
NeuralTrader Sentiment Analysis Module
Organized sentiment analysis for economic data, news, and social media
"""

from .economic.fred_integration import FREDIntegration
from .economic.economic_analyzer import EconomicAnalyzer
from .base_sentiment import BaseSentimentAnalyzer

__all__ = [
    'FREDIntegration',
    'EconomicAnalyzer', 
    'BaseSentimentAnalyzer'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "NeuralTrader Team"
__description__ = "Sentiment analysis module for economic data, news, and social media"

# Configure logging for sentiment module
import logging
sentiment_logger = logging.getLogger(__name__)
if not sentiment_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    sentiment_logger.addHandler(handler)
    sentiment_logger.setLevel(logging.INFO)
