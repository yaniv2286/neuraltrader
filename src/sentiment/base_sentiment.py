#!/usr/bin/env python3
"""
Base Sentiment Analyzer - Abstract base class for all sentiment analysis
Provides common interface and utilities for sentiment data processing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analysis components
    
    All sentiment analyzers must inherit from this class and implement
    the required methods for consistent interface and behavior.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize base sentiment analyzer
        
        Args:
            name: Name of the sentiment analyzer
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.last_update = None
        self.cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        
    @abstractmethod
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch sentiment data for the specified date range
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            DataFrame with sentiment data
        """
        pass
    
    @abstractmethod
    def analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze sentiment from the fetched data
        
        Args:
            data: DataFrame with raw sentiment data
            
        Returns:
            Dictionary with sentiment scores and metrics
        """
        pass
    
    @abstractmethod
    def get_market_signals(self, sentiment_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert sentiment analysis to market signals
        
        Args:
            sentiment_data: Sentiment analysis results
            
        Returns:
            Dictionary with market signals and recommendations
        """
        pass
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp')
        if not cache_time:
            return False
        
        age = (datetime.now() - cache_time).total_seconds()
        return age < self.cache_ttl
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if valid"""
        if self.is_cache_valid(cache_key):
            self.logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key].get('data')
        return None
    
    def cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        self.logger.debug(f"Cached data for {cache_key}")
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data for quality and completeness
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            self.logger.warning("Empty data received")
            return False
        
        # Check for required columns (to be implemented by subclasses)
        # Basic validation here
        if data.isnull().all().all():
            self.logger.warning("All data is null")
            return False
        
        return True
    
    def normalize_sentiment_score(self, score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """
        Normalize sentiment score to standard range
        
        Args:
            score: Raw sentiment score
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized score between -1 and 1
        """
        if score < min_val:
            return -1.0
        elif score > max_val:
            return 1.0
        
        # Linear normalization
        normalized = (score - min_val) / (max_val - min_val) * 2 - 1
        return max(-1.0, min(1.0, normalized))
    
    def calculate_sentiment_momentum(self, scores: List[float], window: int = 5) -> float:
        """
        Calculate sentiment momentum (rate of change)
        
        Args:
            scores: List of sentiment scores
            window: Window size for momentum calculation
            
        Returns:
            Momentum score
        """
        if len(scores) < window:
            return 0.0
        
        recent_scores = scores[-window:]
        if len(recent_scores) < 2:
            return 0.0
        
        # Simple momentum: recent average - older average
        recent_avg = np.mean(recent_scores[-2:])
        older_avg = np.mean(recent_scores[:-2]) if len(recent_scores) > 2 else recent_scores[0]
        
        momentum = recent_avg - older_avg
        return momentum
    
    def get_sentiment_regime(self, score: float, momentum: float) -> str:
        """
        Determine sentiment regime based on score and momentum
        
        Args:
            score: Current sentiment score
            momentum: Sentiment momentum
            
        Returns:
            Regime classification
        """
        if score > 0.3 and momentum > 0.1:
            return "BULLISH_STRONG"
        elif score > 0.1:
            return "BULLISH_MODERATE"
        elif score < -0.3 and momentum < -0.1:
            return "BEARISH_STRONG"
        elif score < -0.1:
            return "BEARISH_MODERATE"
        else:
            return "NEUTRAL"
    
    def generate_sentiment_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive sentiment report
        
        Args:
            analysis: Sentiment analysis results
            
        Returns:
            Formatted report dictionary
        """
        report = {
            'analyzer': self.name,
            'timestamp': datetime.now().isoformat(),
            'sentiment_score': analysis.get('score', 0.0),
            'regime': analysis.get('regime', 'NEUTRAL'),
            'momentum': analysis.get('momentum', 0.0),
            'confidence': analysis.get('confidence', 0.0),
            'signals': analysis.get('signals', {}),
            'metadata': {
                'data_points': analysis.get('data_points', 0),
                'update_frequency': self.config.get('update_frequency', 'unknown'),
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
        }
        
        return report
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"
