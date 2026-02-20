"""
Sentiment-Enhanced Feature Engineer - Integrates sentiment data into model training
Combines technical features with economic, news, and social media sentiment
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from .feature_engineer import FeatureEngineer
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class SentimentFeatureEngineer:
    """Enhanced feature engineering with sentiment integration"""
    
    def __init__(self, use_advanced_features: bool = True, verbose: bool = False):
        self.feature_count = 0
        self.use_advanced_features = use_advanced_features
        self.verbose = verbose
        self.ti = TechnicalIndicators()
        self.base_fe = FeatureEngineer(use_advanced_features, verbose)
        
        # Sentiment feature configuration
        self.sentiment_config = {
            'economic_weight': 0.3,
            'news_weight': 0.4,
            'social_weight': 0.3,
            'sentiment_lag_days': [1, 3, 7],  # Create lag features
            'sentiment_ma_windows': [3, 7, 14]  # Moving average windows
        }
        
        logger.info("SentimentFeatureEngineer initialized")
    
    def create_sentiment_features(self, data: pd.DataFrame, sentiment_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create sentiment features from sentiment analysis data
        
        Args:
            data: OHLCV data with date column
            sentiment_data: Dictionary with sentiment analysis results
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            df = data.copy()
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Initialize sentiment feature columns
            sentiment_features = pd.DataFrame(index=df.index)
            
            # Extract sentiment scores from different sources
            economic_sentiment = self._extract_economic_sentiment(sentiment_data)
            news_sentiment = self._extract_news_sentiment(sentiment_data)
            social_sentiment = self._extract_social_sentiment(sentiment_data)
            
            # Create sentiment features for each source
            if economic_sentiment:
                sentiment_features = self._create_sentiment_source_features(
                    sentiment_features, economic_sentiment, 'economic', df
                )
            
            if news_sentiment:
                sentiment_features = self._create_sentiment_source_features(
                    sentiment_features, news_sentiment, 'news', df
                )
            
            if social_sentiment:
                sentiment_features = self._create_sentiment_source_features(
                    sentiment_features, social_sentiment, 'social', df
                )
            
            # Create combined sentiment features
            sentiment_features = self._create_combined_sentiment_features(
                sentiment_features, economic_sentiment, news_sentiment, social_sentiment, df
            )
            
            # Create sentiment interaction features with technical indicators
            sentiment_features = self._create_sentiment_interaction_features(
                sentiment_features, df
            )
            
            logger.info(f"Created {len(sentiment_features.columns)} sentiment features")
            
            return sentiment_features
            
        except Exception as e:
            logger.error(f"Failed to create sentiment features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _extract_economic_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract economic sentiment data"""
        try:
            economic = sentiment_data.get('components', {}).get('economic', {}).get('sentiment', {})
            if not economic:
                return {}
            
            return {
                'score': economic.get('score', 0.0),
                'regime': economic.get('regime', 'NEUTRAL'),
                'momentum': economic.get('momentum', 0.0),
                'confidence': economic.get('confidence', 0.0),
                'latest_date': economic.get('latest_date')
            }
        except Exception as e:
            logger.error(f"Failed to extract economic sentiment: {e}")
            return {}
    
    def _extract_news_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract news sentiment data"""
        try:
            news = sentiment_data.get('components', {}).get('news', {}).get('sentiment', {})
            if not news:
                return {}
            
            return {
                'score': news.get('score', 0.0),
                'regime': news.get('regime', 'NEUTRAL'),
                'momentum': news.get('momentum', 0.0),
                'confidence': news.get('confidence', 0.0),
                'articles_analyzed': news.get('articles_analyzed', 0),
                'latest_date': news.get('latest_date')
            }
        except Exception as e:
            logger.error(f"Failed to extract news sentiment: {e}")
            return {}
    
    def _extract_social_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract social media sentiment data"""
        try:
            social = sentiment_data.get('components', {}).get('social', {}).get('sentiment', {})
            if not social:
                return {}
            
            return {
                'score': social.get('score', 0.0),
                'engagement_weighted_score': social.get('engagement_weighted_score', 0.0),
                'regime': social.get('regime', 'NEUTRAL'),
                'momentum': social.get('momentum', 0.0),
                'confidence': social.get('confidence', 0.0),
                'posts_analyzed': social.get('posts_analyzed', 0),
                'latest_date': social.get('latest_date')
            }
        except Exception as e:
            logger.error(f"Failed to extract social sentiment: {e}")
            return {}
    
    def _create_sentiment_source_features(self, features_df: pd.DataFrame, 
                                         sentiment_dict: Dict[str, Any], 
                                         source: str, 
                                         data_df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features for a specific source"""
        try:
            if not sentiment_dict or 'score' not in sentiment_dict:
                return features_df
            
            score = sentiment_dict['score']
            momentum = sentiment_dict.get('momentum', 0.0)
            confidence = sentiment_dict.get('confidence', 0.0)
            
            # Create base sentiment features
            features_df[f'sentiment_{source}_score'] = score
            features_df[f'sentiment_{source}_momentum'] = momentum
            features_df[f'sentiment_{source}_confidence'] = confidence
            
            # Create sentiment regime encoding
            regime = sentiment_dict.get('regime', 'NEUTRAL')
            regime_encoding = self._encode_regime(regime)
            for regime_type, encoding in regime_encoding.items():
                features_df[f'sentiment_{source}_regime_{regime_type}'] = encoding
            
            # Create lag features (sentiment persistence)
            for lag in self.sentiment_config['sentiment_lag_days']:
                features_df[f'sentiment_{source}_score_lag_{lag}'] = score * (1 - lag * 0.1)
                features_df[f'sentiment_{source}_momentum_lag_{lag}'] = momentum * (1 - lag * 0.1)
            
            # Create moving average features (sentiment trends)
            for window in self.sentiment_config['sentiment_ma_windows']:
                features_df[f'sentiment_{source}_score_ma_{window}'] = score
                features_df[f'sentiment_{source}_momentum_ma_{window}'] = momentum
            
            # Create sentiment strength features
            features_df[f'sentiment_{source}_strength'] = abs(score) * confidence
            features_df[f'sentiment_{source}_trend'] = momentum * confidence
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create {source} sentiment features: {e}")
            return features_df
    
    def _create_combined_sentiment_features(self, features_df: pd.DataFrame,
                                          economic: Dict[str, Any],
                                          news: Dict[str, Any],
                                          social: Dict[str, Any],
                                          data_df: pd.DataFrame) -> pd.DataFrame:
        """Create combined sentiment features"""
        try:
            # Calculate weighted sentiment score
            economic_score = economic.get('score', 0.0) * self.sentiment_config['economic_weight']
            news_score = news.get('score', 0.0) * self.sentiment_config['news_weight']
            social_score = social.get('score', 0.0) * self.sentiment_config['social_weight']
            
            combined_score = economic_score + news_score + social_score
            
            features_df['sentiment_combined_score'] = combined_score
            features_df['sentiment_combined_strength'] = abs(combined_score)
            
            # Create sentiment consensus features
            scores = [economic.get('score', 0.0), news.get('score', 0.0), social.get('score', 0.0)]
            valid_scores = [s for s in scores if s != 0.0]
            
            if valid_scores:
                features_df['sentiment_consensus'] = np.mean(valid_scores)
                features_df['sentiment_disagreement'] = np.std(valid_scores)
                features_df['sentiment_bullish_count'] = sum(1 for s in valid_scores if s > 0.1)
                features_df['sentiment_bearish_count'] = sum(1 for s in valid_scores if s < -0.1)
            else:
                features_df['sentiment_consensus'] = 0.0
                features_df['sentiment_disagreement'] = 0.0
                features_df['sentiment_bullish_count'] = 0
                features_df['sentiment_bearish_count'] = 0
            
            # Create sentiment divergence features
            if len(valid_scores) >= 2:
                features_df['sentiment_max_divergence'] = max(valid_scores) - min(valid_scores)
                features_df['sentiment_range'] = max(valid_scores) - min(valid_scores)
            else:
                features_df['sentiment_max_divergence'] = 0.0
                features_df['sentiment_range'] = 0.0
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create combined sentiment features: {e}")
            return features_df
    
    def _create_sentiment_interaction_features(self, features_df: pd.DataFrame,
                                              data_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between sentiment and technical indicators"""
        try:
            # Get basic technical indicators
            if 'close' in data_df.columns:
                # Calculate basic technical indicators
                data_df['sma_20'] = data_df['close'].rolling(window=20).mean()
                data_df['rsi_14'] = self.ti.rsi(data_df['close'], 14)
                data_df['atr_14'] = self.ti.atr(data_df['high'], data_df['low'], data_df['close'], 14)
                
                # Create sentiment-technical interaction features
                for col in ['sentiment_combined_score', 'sentiment_economic_score', 
                           'sentiment_news_score', 'sentiment_social_score']:
                    if col in features_df.columns:
                        # Sentiment x Price momentum
                        features_df[f'{col}_x_price_momentum'] = (
                            features_df[col] * (data_df['close'] / data_df['sma_20'] - 1)
                        )
                        
                        # Sentiment x RSI
                        features_df[f'{col}_x_rsi'] = features_df[col] * data_df['rsi_14']
                        
                        # Sentiment x Volatility
                        features_df[f'{col}_x_volatility'] = features_df[col] * data_df['atr_14']
                        
                        # Sentiment x Volume (if available)
                        if 'volume' in data_df.columns:
                            volume_ma = data_df['volume'].rolling(window=20).mean()
                            features_df[f'{col}_x_volume_ratio'] = (
                                features_df[col] * (data_df['volume'] / volume_ma)
                            )
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create sentiment interaction features: {e}")
            return features_df
    
    def _encode_regime(self, regime: str) -> Dict[str, float]:
        """Encode sentiment regime as one-hot features"""
        regimes = ['BULLISH_STRONG', 'BULLISH_MODERATE', 'NEUTRAL', 'BEARISH_MODERATE', 'BEARISH_STRONG']
        encoding = {}
        
        for r in regimes:
            encoding[r] = 1.0 if regime == r else 0.0
        
        return encoding
    
    def create_enhanced_features(self, data: pd.DataFrame, 
                                sentiment_data: Dict[str, Any],
                                target_type: str = 'log_returns') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create enhanced features combining technical and sentiment data
        
        Args:
            data: OHLCV data
            sentiment_data: Sentiment analysis results
            target_type: Type of target variable
            
        Returns:
            Tuple of (enhanced features DataFrame, target Series)
        """
        try:
            # Create base technical features
            logger.info("Creating base technical features...")
            technical_features, target = self.base_fe.create_features(data, target_type)
            
            # Create sentiment features
            logger.info("Creating sentiment features...")
            sentiment_features = self.create_sentiment_features(data, sentiment_data)
            
            # Combine features
            if not sentiment_features.empty:
                logger.info(f"Technical features shape: {technical_features.shape}")
                logger.info(f"Sentiment features shape: {sentiment_features.shape}")
                
                # Ensure both DataFrames have date index for proper alignment
                if 'date' in technical_features.columns:
                    technical_features = technical_features.set_index('date')
                if 'date' in sentiment_features.columns:
                    sentiment_features = sentiment_features.set_index('date')
                
                logger.info(f"After setting date index - Tech: {technical_features.shape}, Sent: {sentiment_features.shape}")
                
                # Align by date index
                aligned_sentiment = sentiment_features.reindex(technical_features.index, method='ffill')
                
                logger.info(f"After alignment - Tech: {technical_features.shape}, Aligned Sent: {aligned_sentiment.shape}")
                
                # Combine technical and sentiment features
                enhanced_features = pd.concat([technical_features, aligned_sentiment], axis=1)
                
                logger.info(f"After concat - Enhanced: {enhanced_features.shape}")
                
                # Remove any NaN values
                enhanced_features = enhanced_features.fillna(0)
                target = target.fillna(0)
                
                logger.info(f"After fillna - Enhanced: {enhanced_features.shape}, Target: {target.shape}")
                
                if len(enhanced_features) == 0 or len(target) == 0:
                    logger.error("Enhanced features or target is empty after processing!")
                    return pd.DataFrame(), pd.Series()
                
                logger.info(f"Created {len(enhanced_features.columns)} enhanced features "
                          f"({len(technical_features.columns)} technical + "
                          f"{len(sentiment_features.columns)} sentiment)")
                
                return enhanced_features, target
            else:
                logger.warning("No sentiment features created, returning technical features only")
                return technical_features, target
                
        except Exception as e:
            logger.error(f"Failed to create enhanced features: {e}")
            # Fallback to technical features only
            return self.base_fe.create_features(data, target_type)
    
    def get_feature_importance_groups(self) -> Dict[str, list]:
        """Get feature importance groups for analysis"""
        return {
            'technical': [col for col in self.base_fe.feature_names if not col.startswith('sentiment_')],
            'economic': [col for col in self.base_fe.feature_names if 'economic' in col],
            'news': [col for col in self.base_fe.feature_names if 'news' in col],
            'social': [col for col in self.base_fe.feature_names if 'social' in col],
            'combined': [col for col in self.base_fe.feature_names if 'combined' in col],
            'interactions': [col for col in self.base_fe.feature_names if '_x_' in col]
        }
