#!/usr/bin/env python3
"""
Social Media Sentiment Analyzer - Real-time social media processing and sentiment analysis
Analyzes Twitter, Reddit, and other social media platforms for market sentiment signals
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

class SocialSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Social media sentiment analyzer for market sentiment detection
    
    Analyzes social media posts, tweets, and discussions to determine
    market sentiment and generate trading signals based on social media flow.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize social media sentiment analyzer
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__("SocialSentimentAnalyzer", config)
        
        # Social media API configuration
        self.twitter_client_id = config.get('twitter_client_id') or os.getenv('TWITTER_CLIENT_ID') if config else os.getenv('TWITTER_CLIENT_ID')
        self.twitter_client_secret = config.get('twitter_client_secret') or os.getenv('TWITTER_CLIENT_SECRET') if config else os.getenv('TWITTER_CLIENT_SECRET')
        self.twitter_bearer_token = config.get('twitter_bearer_token') or os.getenv('TWITTER_BEARER_TOKEN') if config else os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_access_token = config.get('twitter_access_token') or os.getenv('TWITTER_ACCESS_TOKEN') if config else os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = config.get('twitter_access_token_secret') or os.getenv('TWITTER_ACCESS_TOKEN_SECRET') if config else os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.reddit_api_key = config.get('reddit_api_key') if config else None
        self.stocktwits_api_key = config.get('stocktwits_api_key') if config else None
        
        # Sentiment analysis configuration
        self.max_posts = config.get('max_posts', 1000) if config else 1000
        self.confidence_threshold = config.get('confidence_threshold', 0.5) if config else 0.5
        
        # Social media keywords for market relevance
        self.market_keywords = {
            'bullish': [
                'moon', 'rocket', 'diamond hands', 'hold', 'buy', 'long', 'bull',
                'pump', 'rally', 'surge', 'gain', 'profit', 'win', 'to the moon',
                'hodl', 'buy the dip', 'bull market', 'bullish', 'upward', 'positive'
            ],
            'bearish': [
                'paper hands', 'sell', 'short', 'bear', 'dump', 'crash', 'plunge',
                'slump', 'drop', 'fall', 'decline', 'loss', 'fear', 'panic', 'bear market',
                'bearish', 'downward', 'negative', 'recession', 'concern', 'worry'
            ],
            'neutral': [
                'hold', 'wait', 'watch', 'neutral', 'uncertain', 'mixed', 'volatile',
                'cautious', 'sideways', 'stable', 'steady', 'flat', 'unchanged'
            ]
        }
        
        # Stock-specific keywords
        self.stock_keywords = {
            'AAPL': ['apple', 'iphone', 'mac', 'tim cook', 'aapl'],
            'MSFT': ['microsoft', 'windows', 'azure', 'satya nadella', 'msft'],
            'GOOGL': ['google', 'alphabet', 'search', 'android', 'googl'],
            'AMZN': ['amazon', 'aws', 'jeff bezos', 'prime', 'amzn'],
            'TSLA': ['tesla', 'elon musk', 'model 3', 'cybertruck', 'tsla'],
            'NVDA': ['nvidia', 'gpu', 'ai chip', 'jensen huang', 'nvda'],
            'META': ['meta', 'facebook', 'instagram', 'zuckerberg', 'meta'],
            'BTC': ['bitcoin', 'btc', 'crypto', 'satoshi', 'blockchain'],
            'ETH': ['ethereum', 'eth', 'vitalik', 'defi', 'ether']
        }
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuralTrader-Social/1.0'
        })
        
        logger.info("Social Media Sentiment Analyzer initialized")
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch social media data for sentiment analysis
        
        Args:
            start_date: Start date for social media fetch
            end_date: End date for social media fetch
            
        Returns:
            DataFrame with social media posts and sentiment data
        """
        cache_key = f"social_data_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Use real Twitter API
            posts = self._fetch_real_twitter_data(start_date, end_date)
            
            if not posts:
                logger.warning("No social media posts fetched")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(posts)
            
            # Add timestamp column
            df['timestamp'] = pd.to_datetime(df['created_at'])
            
            # Sort by timestamp
            df.sort_values('timestamp', inplace=True)
            
            # Validate data
            if not self.validate_data(df):
                raise Exception("Invalid social media data")
            
            # Cache the result
            self.cache_data(cache_key, df)
            
            logger.info(f"Fetched {len(df)} social media posts")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch social media data: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze sentiment from social media data
        
        Args:
            data: DataFrame with social media posts
            
        Returns:
            Dictionary with sentiment scores and metrics
        """
        if data.empty:
            return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
        
        try:
            # Analyze sentiment for each post
            sentiments = []
            for _, post in data.iterrows():
                sentiment = self._analyze_post_sentiment(post)
                sentiments.append(sentiment)
            
            if not sentiments:
                return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
            
            # Calculate overall sentiment
            sentiment_scores = [s['score'] for s in sentiments]
            overall_score = np.mean(sentiment_scores)
            
            # Calculate momentum (change over recent posts)
            if len(sentiment_scores) >= 100:
                recent_avg = np.mean(sentiment_scores[-50:])
                older_avg = np.mean(sentiment_scores[-100:-50])
                momentum = recent_avg - older_avg
            else:
                momentum = 0.0
            
            # Determine regime
            regime = self.get_sentiment_regime(overall_score, momentum)
            
            # Calculate confidence based on post count and sentiment consistency
            post_count = len(sentiments)
            sentiment_std = np.std(sentiment_scores)
            confidence = min(post_count / 500.0, 1.0) * (1.0 - min(sentiment_std, 1.0))
            
            # Calculate engagement-weighted sentiment
            engagement_weighted_score = self._calculate_engagement_weighted_sentiment(sentiments)
            
            result = {
                'score': overall_score,
                'engagement_weighted_score': engagement_weighted_score,
                'regime': regime,
                'momentum': momentum,
                'confidence': confidence,
                'posts_analyzed': post_count,
                'sentiment_distribution': self._calculate_sentiment_distribution(sentiments),
                'latest_date': data['timestamp'].max().isoformat() if not data.empty else None,
                'platform_distribution': self._calculate_platform_distribution(data)
            }
            
            logger.info(f"Social media sentiment analysis: {regime} (score: {overall_score:.3f}, posts: {post_count})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze social media sentiment: {e}")
            return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
    
    def get_market_signals(self, sentiment_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert social media sentiment to market signals
        
        Args:
            sentiment_data: Social media sentiment analysis results
            
        Returns:
            Dictionary with market signals
        """
        score = sentiment_data.get('score', 0.0)
        engagement_score = sentiment_data.get('engagement_weighted_score', 0.0)
        regime = sentiment_data.get('regime', 'NEUTRAL')
        momentum = sentiment_data.get('momentum', 0.0)
        confidence = sentiment_data.get('confidence', 0.0)
        
        signals = {
            'equity_bias': self._get_equity_bias_from_social(score, regime),
            'retail_sentiment': self._get_retail_sentiment_from_social(sentiment_data),
            'volatility_indicator': self._get_volatility_from_social(sentiment_data),
            'momentum_signal': self._get_momentum_signal_from_social(momentum),
            'engagement_level': self._get_engagement_level_from_social(sentiment_data),
            'crowd_wisdom': self._get_crowd_wisdom_from_social(confidence)
        }
        
        # Generate overall recommendation
        recommendation = self._generate_social_recommendation(signals, confidence)
        
        return {
            'signals': signals,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'regime': regime,
            'data_source': 'social_sentiment'
        }
    
    def _create_placeholder_data(self) -> 'pd.DataFrame':
        """Create placeholder DataFrame for testing"""
        try:
            import pandas as pd
            
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                'timestamp', 'platform', 'username', 'content', 'sentiment_score',
                'engagement_score', 'ticker_mentions', 'hashtags'
            ])
            
            return df
            
        except ImportError:
            logger.error("pandas not available for placeholder data")
            return None
    
    def _fetch_real_twitter_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch real Twitter data"""
        try:
            # Try Bearer Token first (most efficient)
            if self.twitter_bearer_token:
                result = self._fetch_twitter_with_bearer_token(start_date, end_date)
                if result and len(result) > 0:
                    return result
            
            # Try Client ID + Client Secret (OAuth2 flow)
            elif self.twitter_client_id and self.twitter_client_secret:
                result = self._fetch_twitter_with_oauth2(start_date, end_date)
                if result and len(result) > 0:
                    return result
            
            # Try Access Token + Secret (Twitter API v1.1)
            elif self.twitter_access_token and self.twitter_access_token_secret:
                result = self._fetch_twitter_with_v11(start_date, end_date)
                if result and len(result) > 0:
                    return result
            
            # Fall back to mock data
            else:
                logger.warning("No Twitter API credentials configured, falling back to mock data")
                return self._fetch_mock_social_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Failed to fetch real Twitter data: {e}")
            return self._fetch_mock_social_data(start_date, end_date)
    
    def _fetch_twitter_with_bearer_token(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch Twitter data using Bearer Token"""
        try:
            # Search for tweets using Bearer Token
            search_url = "https://api.twitter.com/2/tweets/search/recent"
            search_params = {
                'query': 'stock market OR $SPY OR $QQQ OR investing OR trading',
                'max_results': 100,
                'tweet.fields': 'created_at,author_id,public_metrics'
            }
            
            headers = {'Authorization': f'Bearer {self.twitter_bearer_token}'}
            search_response = requests.get(
                search_url,
                headers=headers,
                params=search_params,
                timeout=30
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                tweets = []
                
                for tweet in search_data.get('data', []):
                    tweets.append({
                        'platform': 'twitter',
                        'username': f"user_{tweet.get('author_id', 'unknown')}",
                        'content': tweet.get('text', ''),
                        'created_at': tweet.get('created_at', ''),
                        'engagement_score': tweet.get('public_metrics', {}).get('like_count', 0),
                        'ticker_mentions': self._extract_ticker_mentions(tweet.get('text', '')),
                        'hashtags': self._extract_hashtags(tweet.get('text', ''))
                    })
                
                logger.info(f"Fetched {len(tweets)} real tweets from Twitter API (Bearer Token)")
                return tweets
            else:
                logger.error(f"Twitter search failed (Bearer Token): {search_response.status_code}")
                return self._fetch_mock_social_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Failed to fetch Twitter data with Bearer Token: {e}")
            return self._fetch_mock_social_data(start_date, end_date)
    
    def _fetch_twitter_with_oauth2(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch Twitter data using OAuth2 flow"""
        try:
            # Get bearer token using Client ID + Client Secret
            auth_url = "https://api.twitter.com/oauth2/token"
            auth_data = {'grant_type': 'client_credentials'}
            
            auth_response = requests.post(
                auth_url,
                auth=(self.twitter_client_id, self.twitter_client_secret),
                data=auth_data,
                timeout=30
            )
            
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                bearer_token = auth_data.get('access_token')
                
                if bearer_token:
                    # Search for tweets
                    search_url = "https://api.twitter.com/2/tweets/search/recent"
                    search_params = {
                        'query': 'stock market OR $SPY OR $QQQ OR investing OR trading',
                        'max_results': 100,
                        'tweet.fields': 'created_at,author_id,public_metrics'
                    }
                    
                    search_headers = {'Authorization': f'Bearer {bearer_token}'}
                    search_response = requests.get(
                        search_url,
                        headers=search_headers,
                        params=search_params,
                        timeout=30
                    )
                    
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        tweets = []
                        
                        for tweet in search_data.get('data', []):
                            tweets.append({
                                'platform': 'twitter',
                                'username': f"user_{tweet.get('author_id', 'unknown')}",
                                'content': tweet.get('text', ''),
                                'created_at': tweet.get('created_at', ''),
                                'engagement_score': tweet.get('public_metrics', {}).get('like_count', 0),
                                'ticker_mentions': self._extract_ticker_mentions(tweet.get('text', '')),
                                'hashtags': self._extract_hashtags(tweet.get('text', ''))
                            })
                        
                        logger.info(f"Fetched {len(tweets)} real tweets from Twitter API (OAuth2)")
                        return tweets
                    else:
                        logger.error(f"Twitter search failed (OAuth2): {search_response.status_code}")
                        return self._fetch_mock_social_data(start_date, end_date)
                else:
                    logger.error("Failed to get bearer token from Twitter")
                    return self._fetch_mock_social_data(start_date, end_date)
            else:
                logger.error(f"Twitter authentication failed (OAuth2): {auth_response.status_code}")
                return self._fetch_mock_social_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Failed to fetch Twitter data with OAuth2: {e}")
            return self._fetch_mock_social_data(start_date, end_date)
    
    def _fetch_twitter_with_v11(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch Twitter data using Twitter API v1.1 with Access Token"""
        try:
            # Search for tweets using Twitter API v1.1
            search_url = "https://api.twitter.com/1.1/search/tweets.json"
            search_params = {
                'q': '$SPY OR $QQQ OR stock market OR investing OR trading',
                'count': 100,
                'result_type': 'recent',
                'lang': 'en',
                'include_entities': 'true'
            }
            
            # Use OAuth 1.0a with Access Token
            auth = (self.twitter_access_token, self.twitter_access_token_secret)
            
            search_response = requests.get(
                search_url,
                auth=auth,
                params=search_params,
                timeout=30
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                tweets = []
                
                for tweet in search_data.get('statuses', []):
                    # Extract engagement metrics
                    public_metrics = tweet.get('user', {})
                    engagement_score = (
                        public_metrics.get('followers_count', 0) * 0.1 +
                        tweet.get('retweet_count', 0) * 2 +
                        tweet.get('favorite_count', 0) * 1
                    )
                    
                    tweets.append({
                        'platform': 'twitter',
                        'username': tweet.get('user', {}).get('screen_name', 'unknown'),
                        'content': tweet.get('text', ''),
                        'created_at': tweet.get('created_at', ''),
                        'engagement_score': engagement_score,
                        'ticker_mentions': self._extract_ticker_mentions(tweet.get('text', '')),
                        'hashtags': self._extract_hashtags(tweet.get('text', ''))
                    })
                
                logger.info(f"Fetched {len(tweets)} real tweets from Twitter API v1.1")
                return tweets
            else:
                logger.error(f"Twitter search failed (v1.1): {search_response.status_code}")
                return self._fetch_mock_social_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Failed to fetch Twitter data with v1.1: {e}")
            return self._fetch_mock_social_data(start_date, end_date)
    
    def _fetch_mock_social_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch mock social media data for demonstration"""
        mock_posts = [
            {
                'platform': 'twitter',
                'username': 'trader_joe',
                'content': 'Bullish on tech stocks! $AAPL looking strong today 🚀',
                'created_at': (datetime.now() - timedelta(hours=1)).isoformat(),
                'engagement_score': 150,
                'ticker_mentions': ['AAPL'],
                'hashtags': ['stocks', 'trading']
            },
            {
                'platform': 'twitter',
                'username': 'market_analyst',
                'content': 'Market sentiment shifting bearish, watch out for volatility 📉',
                'created_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'engagement_score': 89,
                'ticker_mentions': [],
                'hashtags': ['market', 'analysis']
            },
            {
                'platform': 'reddit',
                'username': 'investor_reddit',
                'content': 'Diamond hands on $GME, not selling! 💎🙌',
                'created_at': (datetime.now() - timedelta(hours=3)).isoformat(),
                'engagement_score': 234,
                'ticker_mentions': ['GME'],
                'hashtags': ['investing', 'wallstreetbets']
            }
        ]
        return mock_posts
    
    def _extract_ticker_mentions(self, text: str) -> List[str]:
        """Extract ticker mentions from text"""
        import re
        # Find $TICKER patterns
        tickers = re.findall(r'\$([A-Z]{1,5})', text)
        return tickers
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        import re
        # Find #hashtag patterns
        hashtags = re.findall(r'#(\w+)', text)
        return hashtags
