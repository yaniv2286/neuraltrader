#!/usr/bin/env python3
"""
Sentiment Integration Module
Main integration point for all sentiment analysis components
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from .economic.economic_analyzer import EconomicAnalyzer
from .news.news_sentiment_analyzer import NewsSentimentAnalyzer
from .social.social_sentiment_analyzer import SocialSentimentAnalyzer

logger = logging.getLogger(__name__)

class SentimentIntegration:
    """
    Main sentiment integration class
    
    Coordinates all sentiment analysis components and provides
    unified sentiment signals for trading decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sentiment integration
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzers
        self.economic_analyzer = None
        self.news_analyzer = None
        self.social_analyzer = None
        
        # Component status
        self.component_status = {
            'economic': False,
            'news': False,
            'social': False
        }
        
        # Initialize components
        self._initialize_components()
        
        # Cache for sentiment results
        self.sentiment_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        
        self.logger.info("Sentiment Integration initialized")
    
    def _initialize_components(self):
        """Initialize sentiment analysis components"""
        try:
            # Initialize economic analyzer
            if self.config.get('enable_economic', True):
                self.economic_analyzer = EconomicAnalyzer(self.config)
                self.component_status['economic'] = True
                self.logger.info("Economic analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize economic analyzer: {e}")
        
        # Initialize news analyzer (placeholder)
        if self.config.get('enable_news', False):
            try:
                news_config = {
                    'news_api_key': self.config.get('news_api_key'),
                    'news_sources': self.config.get('news_sources', ['reuters', 'bloomberg', 'cnbc']),
                    'sentiment_threshold': self.config.get('sentiment_threshold', 0.5),
                    'max_articles': self.config.get('max_articles', 100)
                }
                self.news_analyzer = NewsSentimentAnalyzer(news_config)
                self.component_status['news'] = True
                self.logger.info("News analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize news analyzer: {e}")
        
        # Initialize social analyzer (placeholder)
        if self.config.get('enable_social', False):
            try:
                social_config = {
                    'twitter_api_key': self.config.get('social_api_keys', {}).get('twitter_api_key'),
                    'reddit_api_key': self.config.get('social_api_keys', {}).get('reddit_api_key'),
                    'stocktwits_api_key': self.config.get('social_api_keys', {}).get('stocktwits_api_key'),
                    'max_posts': self.config.get('max_posts', 1000),
                    'confidence_threshold': self.config.get('confidence_threshold', 0.5)
                }
                self.social_analyzer = SocialSentimentAnalyzer(social_config)
                self.component_status['social'] = True
                self.logger.info("Social media analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize social analyzer: {e}")
    
    def get_comprehensive_sentiment(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis from all available sources
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with comprehensive sentiment analysis
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        cache_key = f"sentiment_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cache_time = self.sentiment_cache[cache_key].get('timestamp')
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                self.logger.debug("Using cached sentiment data")
                return self.sentiment_cache[cache_key]['data']
        
        # Collect sentiment from all components
        sentiment_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'components': {},
            'overall_sentiment': {},
            'market_signals': {},
            'component_status': self.component_status.copy()
        }
        
        # Economic sentiment
        if self.component_status['economic'] and self.economic_analyzer:
            try:
                economic_data = self.economic_analyzer.fetch_data(start_date, end_date)
                economic_sentiment = self.economic_analyzer.analyze_sentiment(economic_data)
                economic_signals = self.economic_analyzer.get_market_signals(economic_sentiment)
                
                sentiment_results['components']['economic'] = {
                    'sentiment': economic_sentiment,
                    'signals': economic_signals,
                    'status': 'active'
                }
                
                self.logger.info(f"Economic sentiment: {economic_sentiment.get('regime', 'UNKNOWN')}")
                
            except Exception as e:
                self.logger.error(f"Failed to get economic sentiment: {e}")
                sentiment_results['components']['economic'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # News sentiment (placeholder)
        if self.component_status['news'] and self.news_analyzer:
            try:
                news_data = self.news_analyzer.fetch_data(start_date, end_date)
                news_sentiment = self.news_analyzer.analyze_sentiment(news_data)
                news_signals = self.news_analyzer.get_market_signals(news_sentiment)
                
                sentiment_results['components']['news'] = {
                    'sentiment': news_sentiment,
                    'signals': news_signals,
                    'status': 'placeholder'
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get news sentiment: {e}")
                sentiment_results['components']['news'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Social sentiment (placeholder)
        if self.component_status['social'] and self.social_analyzer:
            try:
                social_data = self.social_analyzer.fetch_data(start_date, end_date)
                social_sentiment = self.social_analyzer.analyze_sentiment(social_data)
                social_signals = self.social_analyzer.get_market_signals(social_sentiment)
                
                sentiment_results['components']['social'] = {
                    'sentiment': social_sentiment,
                    'signals': social_signals,
                    'status': 'placeholder'
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get social sentiment: {e}")
                sentiment_results['components']['social'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate overall sentiment
        sentiment_results['overall_sentiment'] = self._calculate_overall_sentiment(
            sentiment_results['components']
        )
        
        # Generate unified market signals
        sentiment_results['market_signals'] = self._generate_unified_signals(
            sentiment_results['components']
        )
        
        # Cache the result
        self.sentiment_cache[cache_key] = {
            'data': sentiment_results,
            'timestamp': datetime.now()
        }
        
        return sentiment_results
    
    def _calculate_overall_sentiment(self, components: Dict) -> Dict[str, Any]:
        """
        Calculate overall sentiment from component results
        
        Args:
            components: Component sentiment results
            
        Returns:
            Overall sentiment analysis
        """
        scores = []
        confidences = []
        regimes = []
        
        for component_name, component_data in components.items():
            if component_data.get('status') in ['active', 'placeholder']:
                sentiment = component_data.get('sentiment', {})
                if sentiment:
                    scores.append(sentiment.get('score', 0.0))
                    confidences.append(sentiment.get('confidence', 0.0))
                    regimes.append(sentiment.get('regime', 'NEUTRAL'))
        
        if not scores:
            return {
                'score': 0.0,
                'regime': 'NEUTRAL',
                'confidence': 0.0,
                'components_used': 0
            }
        
        # Weighted average (economic gets higher weight)
        weights = {'economic': 0.7, 'news': 0.2, 'social': 0.1}
        weighted_scores = []
        
        for i, (component_name, component_data) in enumerate(components.items()):
            if component_data.get('status') in ['active', 'placeholder']:
                sentiment = component_data.get('sentiment', {})
                weight = weights.get(component_name, 0.1)
                weighted_scores.append(sentiment.get('score', 0.0) * weight)
        
        overall_score = sum(weighted_scores) if weighted_scores else 0.0
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine overall regime
        if overall_score > 0.3:
            overall_regime = 'BULLISH_MODERATE'
        elif overall_score < -0.3:
            overall_regime = 'BEARISH_MODERATE'
        else:
            overall_regime = 'NEUTRAL'
        
        return {
            'score': overall_score,
            'regime': overall_regime,
            'confidence': overall_confidence,
            'components_used': len(scores),
            'component_scores': dict(zip([c for c in components.keys() if components[c].get('status') in ['active', 'placeholder']], scores))
        }
    
    def _generate_unified_signals(self, components: Dict) -> Dict[str, Any]:
        """
        Generate unified market signals from all components
        
        Args:
            components: Component results
            
        Returns:
            Unified market signals
        """
        unified_signals = {
            'equity_bias': 'NEUTRAL',
            'risk_adjustment': 1.0,
            'sector_recommendations': {},
            'asset_allocation': {},
            'market_outlook': 'NEUTRAL',
            'confidence': 0.0
        }
        
        active_components = [c for c in components.values() if c.get('status') in ['active', 'placeholder']]
        
        if not active_components:
            return unified_signals
        
        # Combine signals from all components
        equity_biases = []
        risk_adjustments = []
        sector_recs = {}
        confidences = []
        
        for component_data in active_components:
            signals = component_data.get('signals', {})
            if signals:
                # Equity bias
                bias = signals.get('signals', {}).get('equity_bias', 'NEUTRAL')
                equity_biases.append(bias)
                
                # Risk adjustment
                risk_adj = signals.get('signals', {}).get('risk_adjustment', 1.0)
                risk_adjustments.append(risk_adj)
                
                # Sector recommendations
                sector_rec = signals.get('signals', {}).get('sector_recommendations', {})
                for sector, rec in sector_rec.items():
                    if sector not in sector_recs:
                        sector_recs[sector] = []
                    sector_recs[sector].append(rec)
                
                # Confidence
                conf = signals.get('confidence', 0.0)
                confidences.append(conf)
        
        # Determine unified equity bias
        if equity_biases:
            long_count = equity_biases.count('LONG')
            short_count = equity_biases.count('SHORT')
            
            if long_count > short_count:
                unified_signals['equity_bias'] = 'LONG'
            elif short_count > long_count:
                unified_signals['equity_bias'] = 'SHORT'
            else:
                unified_signals['equity_bias'] = 'NEUTRAL'
        
        # Average risk adjustment
        if risk_adjustments:
            unified_signals['risk_adjustment'] = sum(risk_adjustments) / len(risk_adjustments)
        
        # Consolidate sector recommendations
        for sector, recommendations in sector_recs.items():
            if recommendations:
                # Use majority vote
                overweight = recommendations.count('OVERWEIGHT')
                underweight = recommendations.count('UNDERWEIGHT')
                neutral = recommendations.count('NEUTRAL')
                
                if overweight > underweight and overweight > neutral:
                    unified_signals['sector_recommendations'][sector] = 'OVERWEIGHT'
                elif underweight > overweight and underweight > neutral:
                    unified_signals['sector_recommendations'][sector] = 'UNDERWEIGHT'
                else:
                    unified_signals['sector_recommendations'][sector] = 'NEUTRAL'
        
        # Overall confidence
        if confidences:
            unified_signals['confidence'] = sum(confidences) / len(confidences)
        
        # Market outlook based on equity bias and confidence
        if unified_signals['equity_bias'] == 'LONG' and unified_signals['confidence'] > 0.6:
            unified_signals['market_outlook'] = 'BULLISH'
        elif unified_signals['equity_bias'] == 'SHORT' and unified_signals['confidence'] > 0.6:
            unified_signals['market_outlook'] = 'BEARISH'
        else:
            unified_signals['market_outlook'] = 'NEUTRAL'
        
        return unified_signals
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """Get economic summary from economic analyzer"""
        if self.component_status['economic'] and self.economic_analyzer:
            return self.economic_analyzer.get_economic_summary()
        return {"status": "Economic analyzer not available"}
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all sentiment components"""
        return self.component_status.copy()
    
    def clear_cache(self):
        """Clear sentiment cache"""
        self.sentiment_cache.clear()
        if self.economic_analyzer:
            self.economic_analyzer.clear_cache()
        self.logger.info("Sentiment cache cleared")
    
    def validate_configuration(self) -> bool:
        """Validate sentiment configuration"""
        issues = []
        
        # Check economic analyzer
        if self.component_status['economic']:
            if not self.config.get('fred_api_key'):
                issues.append("FRED API key not configured")
        
        # Check news analyzer
        if self.component_status['news']:
            if not self.config.get('news_api_key'):
                issues.append("News API key not configured")
        
        # Check social analyzer
        if self.component_status['social']:
            if not self.config.get('twitter_api_key') and not self.config.get('reddit_api_key'):
                issues.append("Social media API keys not configured")
        
        if issues:
            self.logger.warning(f"Configuration issues: {issues}")
            return False
        
        return True
