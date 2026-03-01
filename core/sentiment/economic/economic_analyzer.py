#!/usr/bin/env python3
"""
Economic Analyzer - Processes FRED data for sentiment analysis
Analyzes economic indicators and generates market signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from ..base_sentiment import BaseSentimentAnalyzer
from .fred_integration import FREDIntegration

logger = logging.getLogger(__name__)

class EconomicAnalyzer(BaseSentimentAnalyzer):
    """
    Economic sentiment analyzer using FRED data
    
    Analyzes key economic indicators to determine market sentiment
    and generate trading signals based on economic conditions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize economic analyzer
        
        Args:
            config: Configuration dictionary with FRED API settings
        """
        super().__init__("EconomicAnalyzer", config)
        
        # Initialize FRED integration
        self.fred = FREDIntegration(
            api_key=config.get('fred_api_key') if config else None,
            cache_dir=config.get('cache_dir', 'data/fred_cache') if config else 'data/fred_cache'
        )
        
        # Economic indicator weights for sentiment calculation
        self.indicator_weights = {
            'GDP': 0.15,                    # Economic growth
            'UNRATE': 0.10,                 # Employment
            'CPIAUCSL': 0.10,               # Inflation
            'FEDFUNDS': 0.08,               # Monetary policy
            'DGS10': 0.07,                  # Long-term rates
            'PAYEMS': 0.08,                 # Job market
            'INDPRO': 0.07,                 # Industrial production
            'UMCSENT': 0.06,                # Consumer confidence
            'HOUST': 0.05,                  # Housing market
            'RSXFS': 0.06,                  # Retail sales
            'VIXCLS': 0.08,                 # Market volatility
            'T10Y2Y': 0.10                  # Yield curve
        }
        
        # Sentiment thresholds
        self.thresholds = {
            'strong_bullish': 0.6,
            'bullish': 0.3,
            'neutral_high': 0.1,
            'neutral_low': -0.1,
            'bearish': -0.3,
            'strong_bearish': -0.6
        }
        
        # Data cache
        self.economic_data = {}
        self.last_data_fetch = None
        
        logger.info("Economic Analyzer initialized")
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch economic data from FRED
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            DataFrame with economic indicators
        """
        cache_key = f"economic_data_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Fetch key indicators
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching economic data from {start_str} to {end_str}")
            
            # Get data for all key indicators
            indicator_data = self.fred.get_key_indicators_data(start_str, end_str)
            
            if not indicator_data:
                raise Exception("No economic data fetched")
            
            # Combine all indicators into a single DataFrame
            combined_data = pd.DataFrame()
            
            for indicator, df in indicator_data.items():
                if df.empty:
                    continue
                
                # Rename value column to indicator name
                df_renamed = df.rename(columns={'value': indicator})
                
                # Merge with combined data
                if combined_data.empty:
                    combined_data = df_renamed
                else:
                    combined_data = pd.merge(combined_data, df_renamed, 
                                          left_index=True, right_index=True, 
                                          how='outer',
                                          suffixes=('', '_right'))
            
            # Sort by date
            combined_data.sort_index(inplace=True)
            
            # Forward fill missing values (economic data is often monthly/quarterly)
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            
            # Validate data
            if not self.validate_data(combined_data):
                raise Exception("Invalid economic data")
            
            # Cache the result
            self.cache_data(cache_key, combined_data)
            self.economic_data = combined_data
            self.last_data_fetch = datetime.now()
            
            logger.info(f"Fetched economic data with {len(combined_data)} observations and {len(combined_data.columns)} indicators")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Failed to fetch economic data: {e}")
            raise
    
    def analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze economic sentiment from indicator data
        
        Args:
            data: DataFrame with economic indicators
            
        Returns:
            Dictionary with sentiment scores and metrics
        """
        if data.empty:
            return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
        
        try:
            # Get most recent data
            latest_data = data.iloc[-1]
            
            # Calculate individual indicator sentiments
            indicator_scores = {}
            
            for indicator, weight in self.indicator_weights.items():
                if indicator not in latest_data:
                    continue
                
                score = self._calculate_indicator_sentiment(indicator, latest_data[indicator], data[indicator])
                indicator_scores[indicator] = score * weight
            
            # Calculate weighted sentiment score
            total_weight = sum(self.indicator_weights.get(ind, 0) for ind in indicator_scores.keys())
            if total_weight > 0:
                sentiment_score = sum(indicator_scores.values()) / total_weight
            else:
                sentiment_score = 0.0
            
            # Calculate momentum (change over last 3 periods)
            momentum = self._calculate_economic_momentum(data)
            
            # Determine regime
            regime = self.get_sentiment_regime(sentiment_score, momentum)
            
            # Calculate confidence based on data completeness
            available_indicators = len([ind for ind in self.indicator_weights.keys() if ind in latest_data])
            total_indicators = len(self.indicator_weights)
            confidence = available_indicators / total_indicators
            
            result = {
                'score': sentiment_score,
                'regime': regime,
                'momentum': momentum,
                'confidence': confidence,
                'indicator_scores': indicator_scores,
                'data_points': len(data),
                'latest_date': data.index[-1].isoformat()
            }
            
            logger.info(f"Economic sentiment analysis: {regime} (score: {sentiment_score:.3f}, momentum: {momentum:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze economic sentiment: {e}")
            return {'score': 0.0, 'regime': 'NEUTRAL', 'momentum': 0.0, 'confidence': 0.0}
    
    def get_market_signals(self, sentiment_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert economic sentiment to market signals
        
        Args:
            sentiment_data: Economic sentiment analysis results
            
        Returns:
            Dictionary with market signals and recommendations
        """
        score = sentiment_data.get('score', 0.0)
        regime = sentiment_data.get('regime', 'NEUTRAL')
        momentum = sentiment_data.get('momentum', 0.0)
        confidence = sentiment_data.get('confidence', 0.0)
        
        signals = {
            'equity_bias': self._get_equity_bias(score, regime),
            'sector_recommendations': self._get_sector_recommendations(sentiment_data),
            'risk_adjustment': self._get_risk_adjustment(score, momentum),
            'duration_bias': self._get_duration_bias(score),
            'volatility_expectation': self._get_volatility_expectation(sentiment_data),
            'currency_bias': self._get_currency_bias(sentiment_data),
            'commodity_bias': self._get_commodity_bias(sentiment_data)
        }
        
        # Generate overall market recommendation
        recommendation = self._generate_market_recommendation(signals, confidence)
        
        return {
            'signals': signals,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'regime': regime
        }
    
    def _calculate_indicator_sentiment(self, indicator: str, current_value: float, historical_data: pd.Series) -> float:
        """
        Calculate sentiment score for a single indicator
        
        Args:
            indicator: Indicator name
            current_value: Current value
            historical_data: Historical values
            
        Returns:
            Sentiment score between -1 and 1
        """
        if historical_data.empty:
            return 0.0
        
        try:
            # Different logic for different types of indicators
            if indicator == 'UNRATE':
                # Unemployment: lower is better
                percentile = (historical_data <= current_value).mean()
                return self.normalize_sentiment_score(1 - percentile)  # Invert
            
            elif indicator in ['GDP', 'INDPRO', 'PAYEMS', 'HOUST', 'RSXFS', 'UMCSENT', 'DSPIC96']:
                # Growth indicators: higher is better
                percentile = (historical_data <= current_value).mean()
                return self.normalize_sentiment_score(percentile)
            
            elif indicator == 'CPIAUCSL':
                # Inflation: moderate is best, very high or very low is bad
                # Calculate inflation rate (year-over-year change)
                if len(historical_data) >= 12:
                    inflation_rate = (current_value / historical_data.iloc[-12] - 1) * 100
                    # Optimal inflation around 2%
                    if 1.5 <= inflation_rate <= 2.5:
                        return 0.5
                    elif inflation_rate > 5:
                        return -0.8
                    elif inflation_rate < 0:
                        return -0.6
                    else:
                        return 0.2
                return 0.0
            
            elif indicator in ['FEDFUNDS', 'DGS10', 'DGS2']:
                # Interest rates: moderate is best
                percentile = (historical_data <= current_value).mean()
                # Normalize around 50th percentile (moderate rates)
                return self.normalize_sentiment_score(0.5 - abs(percentile - 0.5))
            
            elif indicator == 'VIXCLS':
                # VIX: lower is better (less volatility)
                percentile = (historical_data <= current_value).mean()
                return self.normalize_sentiment_score(1 - percentile)  # Invert
            
            elif indicator == 'T10Y2Y':
                # Yield curve: steeper is better (positive spread)
                if current_value > 0:
                    return 0.5
                elif current_value < -0.5:
                    return -0.8
                else:
                    return 0.0
            
            else:
                # Default: use percentile
                percentile = (historical_data <= current_value).mean()
                return self.normalize_sentiment_score(percentile)
                
        except Exception as e:
            logger.warning(f"Failed to calculate sentiment for {indicator}: {e}")
            return 0.0
    
    def _calculate_economic_momentum(self, data: pd.DataFrame) -> float:
        """
        Calculate economic momentum based on recent changes
        
        Args:
            data: Economic indicators DataFrame
            
        Returns:
            Momentum score
        """
        try:
            if len(data) < 3:
                return 0.0
            
            # Calculate momentum for key indicators
            momentum_scores = []
            
            for indicator in ['GDP', 'PAYEMS', 'INDPRO', 'UMCSENT']:
                if indicator not in data.columns:
                    continue
                
                series = data[indicator].dropna()
                if len(series) < 3:
                    continue
                
                # Calculate rate of change
                recent_change = (series.iloc[-1] / series.iloc[-3] - 1) if series.iloc[-3] != 0 else 0
                momentum_scores.append(recent_change)
            
            if momentum_scores:
                avg_momentum = np.mean(momentum_scores)
                # Normalize to -1 to 1 range
                return np.clip(avg_momentum * 10, -1, 1)  # Scale and clip
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate economic momentum: {e}")
            return 0.0
    
    def _get_equity_bias(self, score: float, regime: str) -> str:
        """Get equity market bias based on economic sentiment"""
        if regime in ['BULLISH_STRONG', 'BULLISH_MODERATE']:
            return "LONG"
        elif regime in ['BEARISH_STRONG', 'BEARISH_MODERATE']:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def _get_sector_recommendations(self, sentiment_data: Dict) -> Dict[str, str]:
        """Get sector recommendations based on economic conditions"""
        score = sentiment_data.get('score', 0.0)
        momentum = sentiment_data.get('momentum', 0.0)
        
        recommendations = {}
        
        # Technology: Strong in growth environments
        recommendations['Technology'] = "OVERWEIGHT" if score > 0.2 else "UNDERWEIGHT"
        
        # Financials: Benefit from higher rates
        recommendations['Financials'] = "OVERWEIGHT" if momentum > 0.1 else "NEUTRAL"
        
        # Consumer Discretionary: Strong in good economic times
        recommendations['Consumer'] = "OVERWEIGHT" if score > 0.3 else "UNDERWEIGHT"
        
        # Utilities: Defensive in poor conditions
        recommendations['Utilities'] = "OVERWEIGHT" if score < -0.2 else "NEUTRAL"
        
        # Energy: Inflation sensitive
        recommendations['Energy'] = "OVERWEIGHT" if score > 0.1 else "NEUTRAL"
        
        return recommendations
    
    def _get_risk_adjustment(self, score: float, momentum: float) -> float:
        """Get risk adjustment factor based on economic conditions"""
        # Reduce risk in poor economic conditions
        if score < -0.3:
            return 0.7  # Reduce position sizes by 30%
        elif score < -0.1:
            return 0.85  # Reduce by 15%
        elif score > 0.3:
            return 1.15  # Increase by 15%
        else:
            return 1.0  # No adjustment
    
    def _get_duration_bias(self, score: float) -> str:
        """Get duration bias for fixed income"""
        if score > 0.2:  # Good economy, rates may rise
            return "SHORT_DURATION"
        elif score < -0.2:  # Poor economy, rates may fall
            return "LONG_DURATION"
        else:
            return "NEUTRAL"
    
    def _get_volatility_expectation(self, sentiment_data: Dict) -> str:
        """Get volatility expectation based on economic conditions"""
        regime = sentiment_data.get('regime', 'NEUTRAL')
        momentum = sentiment_data.get('momentum', 0.0)
        
        if regime in ['BEARISH_STRONG'] or momentum < -0.2:
            return "HIGH_VOLATILITY"
        elif regime in ['BULLISH_STRONG']:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL_VOLATILITY"
    
    def _get_currency_bias(self, sentiment_data: Dict) -> Dict[str, str]:
        """Get currency bias based on economic conditions"""
        score = sentiment_data.get('score', 0.0)
        
        # USD strength based on US economic performance
        usd_bias = "STRONG" if score > 0.2 else "WEAK" if score < -0.2 else "NEUTRAL"
        
        return {
            'USD': usd_bias,
            'EUR': "WEAK" if usd_bias == "STRONG" else "STRONG" if usd_bias == "WEAK" else "NEUTRAL",
            'JPY': "WEAK" if usd_bias == "STRONG" else "STRONG" if usd_bias == "WEAK" else "NEUTRAL"
        }
    
    def _get_commodity_bias(self, sentiment_data: Dict) -> Dict[str, str]:
        """Get commodity bias based on economic conditions"""
        score = sentiment_data.get('score', 0.0)
        regime = sentiment_data.get('regime', 'NEUTRAL')
        
        # Gold: Safe haven in poor conditions
        gold_bias = "OVERWEIGHT" if regime in ['BEARISH_STRONG', 'BEARISH_MODERATE'] else "NEUTRAL"
        
        # Oil: Growth sensitive
        oil_bias = "OVERWEIGHT" if score > 0.2 else "UNDERWEIGHT"
        
        return {
            'Gold': gold_bias,
            'Oil': oil_bias,
            'Industrial': "OVERWEIGHT" if score > 0.3 else "NEUTRAL"
        }
    
    def _generate_market_recommendation(self, signals: Dict, confidence: float) -> str:
        """Generate overall market recommendation"""
        equity_bias = signals.get('equity_bias', 'NEUTRAL')
        risk_adjustment = signals.get('risk_adjustment', 1.0)
        
        if confidence < 0.5:
            return "LOW_CONFIDENCE - HOLD"
        
        if equity_bias == "LONG" and risk_adjustment > 1.0:
            return "STRONG BUY - ECONOMIC EXPANSION"
        elif equity_bias == "LONG":
            return "MODERATE BUY - GROWTH EXPECTED"
        elif equity_bias == "SHORT":
            return "CAUTION - ECONOMIC CONTRACTION"
        else:
            return "NEUTRAL - MIXED SIGNALS"
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """
        Get current economic summary
        
        Returns:
            Dictionary with economic overview
        """
        if self.economic_data is None or (hasattr(self.economic_data, 'empty') and self.economic_data.empty):
            return {"status": "No data available"}
        
        try:
            latest = self.economic_data.iloc[-1]
            
            summary = {
                'last_update': self.last_data_fetch.isoformat() if self.last_data_fetch else None,
                'data_points': len(self.economic_data),
                'indicators': {}
            }
            
            # Key economic indicators
            for indicator in ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'DGS10', 'VIXCLS']:
                if indicator in latest:
                    summary['indicators'][indicator] = {
                        'current': float(latest[indicator]),
                        'change_1m': self._calculate_change(latest[indicator], indicator, 1),
                        'change_1y': self._calculate_change(latest[indicator], indicator, 12)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate economic summary: {e}")
            return {"status": "Error", "error": str(e)}
    
    def _calculate_change(self, current_value: float, indicator: str, periods: int) -> Optional[float]:
        """Calculate percentage change over specified periods"""
        try:
            if indicator not in self.economic_data.columns:
                return None
            
            series = self.economic_data[indicator].dropna()
            if len(series) <= periods:
                return None
            
            past_value = series.iloc[-(periods + 1)]
            if past_value == 0:
                return None
            
            change = (current_value / past_value - 1) * 100
            return change
            
        except Exception:
            return None
