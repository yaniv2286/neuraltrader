#!/usr/bin/env python3
"""
Social Media Sentiment Analysis Module
Twitter, Reddit, and other social media sentiment analysis
"""

from .social_sentiment_analyzer import SocialSentimentAnalyzer
from .social_processor import SocialProcessor
from .social_sources import SocialSources

__all__ = [
    'SocialSentimentAnalyzer',
    'SocialProcessor',
    'SocialSources'
]
