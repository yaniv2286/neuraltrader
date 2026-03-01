#!/usr/bin/env python3
"""
News Sentiment Analyzer - Real-time news processing and sentiment analysis
Analyzes news articles and headlines for market sentiment signals
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import re
import os
from ..base_sentiment import BaseSentimentAnalyzer

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    News sentiment analyzer for market sentiment detection
    
    Analyzes news articles, headlines, and financial news to determine
    market sentiment and generate trading signals based on news flow.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize news sentiment analyzer
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__("NewsSentimentAnalyzer", config)
        
        # News API configuration
        self.api_key = config.get('news_api_key') or os.getenv('NEWS_API_KEY') if config else os.getenv('NEWS_API_KEY')
        self.news_sources = config.get('news_sources', ['reuters', 'bloomberg', 'cnbc']) if config else ['reuters', 'bloomberg', 'cnbc']
        
        # Sentiment analysis configuration
        self.sentiment_threshold = config.get('sentiment_threshold', 0.5) if config else 0.5
        self.max_articles = config.get('max_articles', 100) if config else 100
        
        # Financial keywords for market relevance
        self.market_keywords = {
            'bullish': [
                'rally', 'surge', 'jump', 'gain', 'rise', 'climb', 'soar', 'boom',
                'bull market', 'upward', 'positive', 'optimistic', 'growth', 'expansion',
                'record high', 'breakthrough', 'milestone', 'achievement', 'success'
            ],
            'bearish': [
                'plunge', 'slump', 'drop', 'fall', 'decline', 'crash', 'tumble',
                'bear market', 'downward', 'negative', 'pessimistic', 'recession',
                'record low', 'concern', 'worry', 'fear', 'panic', 'crisis'
            ],
            'neutral': [
                'stable', 'steady', 'flat', 'unchanged', 'mixed', 'volatile',
                'uncertain', 'cautious', 'wait-and-see', 'holding pattern'
            ]
        }
        
        # Sector keywords
        self.sector_keywords = {
            'Technology': ['tech', 'software', 'AI', 'cloud', 'semiconductor', 'apple', 'microsoft', 'google'],
            'Financial': ['bank', 'finance', 'investment', 'wall street', 'federal reserve', 'interest rates'],
            'Healthcare': ['health', 'pharmaceutical', 'medical', 'biotech', 'FDA', 'hospital'],
            'Energy': ['oil', 'gas', 'energy', 'petroleum', 'OPEC', 'renewable'],
            'Consumer': ['retail', 'consumer', 'shopping', 'Amazon', 'Walmart', 'Tesla'],
            'Industrial': ['manufacturing', 'industrial', 'production', 'factory', 'supply chain']
        }
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuralTrader-News/1.0'
        })
        
        logger.info("News Sentiment Analyzer initialized")
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch news data for sentiment analysis
        
        Args:
            start_date: Start date for news fetch
            end_date: End date for news fetch
            
        Returns:
            DataFrame with news articles and sentiment data
        """
        cache_key = f"news_data_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Use real News API
            articles = self._fetch_real_news_data(start_date, end_date)
            
            if not articles:
                logger.warning("No news articles fetched")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            
            # Add timestamp column
            df['timestamp'] = pd.to_datetime(df['published_at'])
            
            # Sort by timestamp
            df.sort_values('timestamp', inplace=True)
            
            # Validate data
            if not self.validate_data(df):
                raise Exception("Invalid news data")
            
            # Cache the result
            self.cache_data(cache_key, df)
            
            logger.info(f"Fetched {len(df)} news articles")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch news data: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze sentiment from news data
        
        Args:
            data: DataFrame with news articles
            
        Returns:
            Dictionary with sentiment scores and metrics
        """
        if data.empty:
            return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
        
        try:
            # Analyze sentiment for each article
            sentiments = []
            for _, article in data.iterrows():
                sentiment = self._analyze_article_sentiment(article)
                sentiments.append(sentiment)
            
            if not sentiments:
                return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
            
            # Calculate overall sentiment
            sentiment_scores = [s['score'] for s in sentiments]
            overall_score = np.mean(sentiment_scores)
            
            # Calculate momentum (change over recent articles)
            if len(sentiment_scores) >= 10:
                recent_avg = np.mean(sentiment_scores[-5:])
                older_avg = np.mean(sentiment_scores[-10:-5])
                momentum = recent_avg - older_avg
            else:
                momentum = 0.0
            
            # Determine regime
            regime = self.get_sentiment_regime(overall_score, momentum)
            
            # Calculate confidence based on article count and sentiment consistency
            article_count = len(sentiments)
            sentiment_std = np.std(sentiment_scores)
            confidence = min(article_count / 50.0, 1.0) * (1.0 - min(sentiment_std, 1.0))
            
            result = {
                'score': overall_score,
                'regime': regime,
                'momentum': momentum,
                'confidence': confidence,
                'articles_analyzed': article_count,
                'sentiment_distribution': self._calculate_sentiment_distribution(sentiments),
                'latest_date': data['timestamp'].max().isoformat() if not data.empty else None
            }
            
            logger.info(f"News sentiment analysis: {regime} (score: {overall_score:.3f}, articles: {article_count})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze news sentiment: {e}")
            return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
    
    def get_market_signals(self, sentiment_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert news sentiment to market signals
        
        Args:
            sentiment_data: News sentiment analysis results
            
        Returns:
            Dictionary with market signals and recommendations
        """
        score = sentiment_data.get('score', 0.0)
        regime = sentiment_data.get('regime', 'NEUTRAL')
        momentum = sentiment_data.get('momentum', 0.0)
        confidence = sentiment_data.get('confidence', 0.0)
        
        signals = {
            'equity_bias': self._get_equity_bias_from_news(score, regime),
            'volatility_expectation': self._get_volatility_from_news(sentiment_data),
            'sector_impact': self._get_sector_impact_from_news(sentiment_data),
            'time_horizon': self._get_time_horizon_from_news(momentum),
            'risk_level': self._get_risk_level_from_news(regime, confidence)
        }
        
        # Generate overall recommendation
        recommendation = self._generate_news_recommendation(signals, confidence)
        
        return {
            'signals': signals,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'regime': regime,
            'data_source': 'news_sentiment'
        }
    
    def _fetch_real_news_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch real news data from News API"""
        try:
            import requests
            
            api_key = self.config.get('news_api_key')
            if not api_key:
                logger.warning("No News API key configured, falling back to mock data")
                return self._fetch_mock_news_data(start_date, end_date)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'stock market OR finance OR economy OR trading',
                'domains': 'reuters.com,bloomberg.com,cnbc.com,wsj.com',
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'pageSize': 100,
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'content': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', '')
                })
            
            logger.info(f"Fetched {len(articles)} real news articles from News API")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch real news data: {e}")
            return self._fetch_mock_news_data(start_date, end_date)
    
    def _fetch_mock_news_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch mock news data for demonstration"""
        # In production, this would integrate with real news APIs
        mock_articles = [
            {
                'title': 'Tech Stocks Rally on AI Optimism',
                'content': 'Technology companies saw significant gains as investors expressed optimism about artificial intelligence developments.',
                'source': 'reuters',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'url': 'https://example.com/tech-rally'
            },
            {
                'title': 'Federal Reserve Holds Interest Rates Steady',
                'content': 'The Federal Reserve decided to maintain current interest rates amid economic uncertainty.',
                'source': 'bloomberg',
                'published_at': (datetime.now() - timedelta(hours=4)).isoformat(),
                'url': 'https://example.com/fed-rates'
            },
            {
                'title': 'Energy Sector Faces Headwinds',
                'content': 'Oil prices declined as concerns about global demand growth intensified.',
                'source': 'cnbc',
                'published_at': (datetime.now() - timedelta(hours=6)).isoformat(),
                'url': 'https://example.com/energy-headwinds'
            },
            {
                'title': 'Healthcare Stocks Show Mixed Performance',
                'content': 'Pharmaceutical companies reported mixed results, with some exceeding expectations while others missed.',
                'source': 'reuters',
                'published_at': (datetime.now() - timedelta(hours=8)).isoformat(),
                'url': 'https://example.com/healthcare-mixed'
            },
            {
                'title': 'Consumer Spending Remains Resilient',
                'content': 'Retail sales data showed consumer spending remained strong despite economic concerns.',
                'source': 'bloomberg',
                'published_at': (datetime.now() - timedelta(hours=10)).isoformat(),
                'url': 'https://example.com/consumer-spending'
            }
        ]
        
        return mock_articles
    
    def _analyze_article_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment of a single article"""
        title = article.get('title', '')
        content = article.get('content', '')
        full_text = f"{title} {content}".lower()
        
        # Count sentiment keywords
        bullish_count = sum(1 for keyword in self.market_keywords['bullish'] if keyword in full_text)
        bearish_count = sum(1 for keyword in self.market_keywords['bearish'] if keyword in full_text)
        neutral_count = sum(1 for keyword in self.market_keywords['neutral'] if keyword in full_text)
        
        # Calculate sentiment score
        total_sentiment_words = bullish_count + bearish_count + neutral_count
        
        if total_sentiment_words == 0:
            score = 0.0
        else:
            score = (bullish_count - bearish_count) / total_sentiment_words
        
        # Identify sectors mentioned
        sectors_mentioned = []
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                sectors_mentioned.append(sector)
        
        # Calculate relevance score based on financial keywords
        relevance = min(total_sentiment_words / 10.0, 1.0)
        
        return {
            'score': score,
            'bullish_words': bullish_count,
            'bearish_words': bearish_count,
            'neutral_words': neutral_count,
            'sectors': sectors_mentioned,
            'relevance': relevance
        }
    
    def _calculate_sentiment_distribution(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of sentiment types"""
        if not sentiments:
            return {'bullish': 0.0, 'bearish': 0.0, 'neutral': 1.0}
        
        total = len(sentiments)
        bullish = sum(1 for s in sentiments if s['score'] > 0.1)
        bearish = sum(1 for s in sentiments if s['score'] < -0.1)
        neutral = total - bullish - bearish
        
        return {
            'bullish': bullish / total,
            'bearish': bearish / total,
            'neutral': neutral / total
        }
    
    def _get_equity_bias_from_news(self, score: float, regime: str) -> str:
        """Get equity market bias from news sentiment"""
        if regime in ['BULLISH_STRONG', 'BULLISH_MODERATE']:
            return "LONG"
        elif regime in ['BEARISH_STRONG', 'BEARISH_MODERATE']:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def _get_volatility_from_news(self, sentiment_data: Dict) -> str:
        """Get volatility expectation from news sentiment"""
        regime = sentiment_data.get('regime', 'NEUTRAL')
        distribution = sentiment_data.get('sentiment_distribution', {})
        
        # High volatility if mixed sentiment or bearish regime
        if regime in ['BEARISH_STRONG'] or distribution.get('neutral', 0) > 0.6:
            return "HIGH_VOLATILITY"
        elif regime in ['BULLISH_STRONG']:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL_VOLATILITY"
    
    def _get_sector_impact_from_news(self, sentiment_data: Dict) -> Dict[str, str]:
        """Get sector impact from news sentiment"""
        # This would analyze sector-specific sentiment from articles
        # For now, return neutral recommendations
        return {
            'Technology': 'NEUTRAL',
            'Financial': 'NEUTRAL',
            'Healthcare': 'NEUTRAL',
            'Energy': 'NEUTRAL',
            'Consumer': 'NEUTRAL'
        }
    
    def _get_time_horizon_from_news(self, momentum: float) -> str:
        """Get recommended time horizon from news momentum"""
        if momentum > 0.1:
            return "SHORT_TERM"  # Positive momentum - shorter horizon
        elif momentum < -0.1:
            return "LONG_TERM"   # Negative momentum - longer horizon
        else:
            return "MEDIUM_TERM"
    
    def _get_risk_level_from_news(self, regime: str, confidence: float) -> str:
        """Get risk level from news sentiment"""
        if regime in ['BEARISH_STRONG'] and confidence > 0.7:
            return "HIGH_RISK"
        elif regime in ['BULLISH_STRONG'] and confidence > 0.7:
            return "LOW_RISK"
        else:
            return "MODERATE_RISK"
    
    def _generate_news_recommendation(self, signals: Dict, confidence: float) -> str:
        """Generate overall recommendation from news signals"""
        equity_bias = signals.get('equity_bias', 'NEUTRAL')
        risk_level = signals.get('risk_level', 'MODERATE_RISK')
        volatility = signals.get('volatility_expectation', 'NORMAL_VOLATILITY')
        
        if confidence < 0.3:
            return "LOW_CONFIDENCE - INSUFFICIENT NEWS DATA"
        
        if equity_bias == "LONG" and risk_level == "LOW_RISK":
            return "BULLISH - NEWS FLOW POSITIVE"
        elif equity_bias == "SHORT" and risk_level == "HIGH_RISK":
            return "BEARISH - NEWS FLOW NEGATIVE"
        elif volatility == "HIGH_VOLATILITY":
            return "CAUTION - HIGH NEWS VOLATILITY"
        else:
            return "NEUTRAL - MIXED NEWS SIGNALS"
    
    def get_news_summary(self) -> Dict[str, Any]:
        """Get current news summary"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        try:
            data = self.fetch_data(start_date, end_date)
            if data.empty:
                return {"status": "No news data available"}
            
            sentiment = self.analyze_sentiment(data)
            
            summary = {
                'status': 'active',
                'articles_analyzed': len(data),
                'time_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'overall_sentiment': sentiment.get('regime', 'NEUTRAL'),
                'sentiment_score': sentiment.get('score', 0.0),
                'confidence': sentiment.get('confidence', 0.0),
                'top_sources': data['source'].value_counts().head(3).to_dict() if not data.empty else {},
                'latest_article': data.iloc[-1]['title'] if not data.empty else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate news summary: {e}")
            return {"status": "Error", "error": str(e)}
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'session'):
            self.session.close()
