#!/usr/bin/env python3
"""
News Sentiment Analysis Module
Real-time news processing and sentiment analysis
"""

from .news_sentiment_analyzer import NewsSentimentAnalyzer
from .news_processor import NewsProcessor
from .news_sources import NewsSources

__all__ = [
    'NewsSentimentAnalyzer',
    'NewsProcessor',
    'NewsSources'
]
