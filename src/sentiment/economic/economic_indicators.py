#!/usr/bin/env python3
"""
Economic Indicators Configuration
Defines key economic indicators and their properties for sentiment analysis
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class IndicatorType(Enum):
    """Types of economic indicators"""
    GROWTH = "growth"
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    MONETARY = "monetary"
    CONSUMER = "consumer"
    FINANCIAL = "financial"
    CURRENCY = "currency"
    HOUSING = "housing"

class DataFrequency(Enum):
    """Data frequency for indicators"""
    DAILY = "d"
    MONTHLY = "m"
    QUARTERLY = "q"
    ANNUAL = "a"

@dataclass
class EconomicIndicator:
    """Economic indicator definition"""
    series_id: str
    name: str
    description: str
    indicator_type: IndicatorType
    frequency: DataFrequency
    weight: float
    inverse: bool = False  # Whether lower values are better
    optimal_range: Optional[tuple] = None  # Optimal range (min, max)
    
class EconomicIndicators:
    """Repository of key economic indicators for sentiment analysis"""
    
    # Core economic indicators
    INDICATORS = {
        'GDP': EconomicIndicator(
            series_id='GDP',
            name='Real Gross Domestic Product',
            description='Real GDP adjusted for inflation',
            indicator_type=IndicatorType.GROWTH,
            frequency=DataFrequency.QUARTERLY,
            weight=0.15
        ),
        
        'UNRATE': EconomicIndicator(
            series_id='UNRATE',
            name='Unemployment Rate',
            description='Seasonally adjusted unemployment rate',
            indicator_type=IndicatorType.EMPLOYMENT,
            frequency=DataFrequency.MONTHLY,
            weight=0.10,
            inverse=True
        ),
        
        'CPIAUCSL': EconomicIndicator(
            series_id='CPIAUCSL',
            name='Consumer Price Index',
            description='Consumer Price Index for All Urban Consumers',
            indicator_type=IndicatorType.INFLATION,
            frequency=DataFrequency.MONTHLY,
            weight=0.10,
            optimal_range=(1.5, 2.5)  # 1.5-2.5% inflation optimal
        ),
        
        'FEDFUNDS': EconomicIndicator(
            series_id='FEDFUNDS',
            name='Federal Funds Rate',
            description='Effective federal funds rate',
            indicator_type=IndicatorType.MONETARY,
            frequency=DataFrequency.MONTHLY,
            weight=0.08
        ),
        
        'DGS10': EconomicIndicator(
            series_id='DGS10',
            name='10-Year Treasury Rate',
            description='10-year treasury constant maturity rate',
            indicator_type=IndicatorType.MONETARY,
            frequency=DataFrequency.DAILY,
            weight=0.07
        ),
        
        'DGS2': EconomicIndicator(
            series_id='DGS2',
            name='2-Year Treasury Rate',
            description='2-year treasury constant maturity rate',
            indicator_type=IndicatorType.MONETARY,
            frequency=DataFrequency.DAILY,
            weight=0.05
        ),
        
        'PAYEMS': EconomicIndicator(
            series_id='PAYEMS',
            name='Nonfarm Payrolls',
            description='All employees: total nonfarm payrolls',
            indicator_type=IndicatorType.EMPLOYMENT,
            frequency=DataFrequency.MONTHLY,
            weight=0.08
        ),
        
        'INDPRO': EconomicIndicator(
            series_id='INDPRO',
            name='Industrial Production',
            description='Industrial production index',
            indicator_type=IndicatorType.GROWTH,
            frequency=DataFrequency.MONTHLY,
            weight=0.07
        ),
        
        'UMCSENT': EconomicIndicator(
            series_id='UMCSENT',
            name='Consumer Sentiment',
            description='University of Michigan consumer sentiment',
            indicator_type=IndicatorType.CONSUMER,
            frequency=DataFrequency.MONTHLY,
            weight=0.06
        ),
        
        'HOUST': EconomicIndicator(
            series_id='HOUST',
            name='Housing Starts',
            description='Housing starts: total',
            indicator_type=IndicatorType.HOUSING,
            frequency=DataFrequency.MONTHLY,
            weight=0.05
        ),
        
        'RSXFS': EconomicIndicator(
            series_id='RSXFS',
            name='Retail Sales',
            description='Retail and food services sales',
            indicator_type=IndicatorType.CONSUMER,
            frequency=DataFrequency.MONTHLY,
            weight=0.06
        ),
        
        'DSPIC96': EconomicIndicator(
            series_id='DSPIC96',
            name='Real Disposable Income',
            description='Real disposable personal income',
            indicator_type=IndicatorType.CONSUMER,
            frequency=DataFrequency.MONTHLY,
            weight=0.05
        ),
        
        'M2SL': EconomicIndicator(
            series_id='M2SL',
            name='M2 Money Supply',
            description='M2 money supply',
            indicator_type=IndicatorType.MONETARY,
            frequency=DataFrequency.MONTHLY,
            weight=0.04
        ),
        
        'VIXCLS': EconomicIndicator(
            series_id='VIXCLS',
            name='VIX Index',
            description='VIX volatility index',
            indicator_type=IndicatorType.FINANCIAL,
            frequency=DataFrequency.DAILY,
            weight=0.08,
            inverse=True
        ),
        
        'T10Y2Y': EconomicIndicator(
            series_id='T10Y2Y',
            name='10-Year minus 2-Year Spread',
            description='10-year minus 2-year treasury constant maturity rate',
            indicator_type=IndicatorType.FINANCIAL,
            frequency=DataFrequency.DAILY,
            weight=0.10
        ),
        
        'T10Y3M': EconomicIndicator(
            series_id='T10Y3M',
            name='10-Year minus 3-Month Spread',
            description='10-year minus 3-month treasury constant maturity rate',
            indicator_type=IndicatorType.FINANCIAL,
            frequency=DataFrequency.DAILY,
            weight=0.05
        ),
        
        # Currency indicators
        'DEXUSEU': EconomicIndicator(
            series_id='DEXUSEU',
            name='US/Euro Exchange Rate',
            description='U.S. / Euro exchange rate',
            indicator_type=IndicatorType.CURRENCY,
            frequency=DataFrequency.DAILY,
            weight=0.03
        ),
        
        'DEXUSUK': EconomicIndicator(
            series_id='DEXUSUK',
            name='US/UK Exchange Rate',
            description='U.S. / U.K. exchange rate',
            indicator_type=IndicatorType.CURRENCY,
            frequency=DataFrequency.DAILY,
            weight=0.02
        ),
        
        'DEXJPUS': EconomicIndicator(
            series_id='DEXJPUS',
            name='Japan/US Exchange Rate',
            description='Japan / U.S. exchange rate',
            indicator_type=IndicatorType.CURRENCY,
            frequency=DataFrequency.DAILY,
            weight=0.02
        ),
        
        'DEXCHUS': EconomicIndicator(
            series_id='DEXCHUS',
            name='China/US Exchange Rate',
            description='China / U.S. exchange rate',
            indicator_type=IndicatorType.CURRENCY,
            frequency=DataFrequency.DAILY,
            weight=0.03
        ),
        
        # Additional financial indicators
        'BAMLCC0A0CMNATRIV': EconomicIndicator(
            series_id='BAMLCC0A0CMNATRIV',
            name='Corporate Index',
            description='ICE BofA US Corporate Index',
            indicator_type=IndicatorType.FINANCIAL,
            frequency=DataFrequency.DAILY,
            weight=0.04
        ),
        
        'DFF': EconomicIndicator(
            series_id='DFF',
            name='Daily Federal Funds Rate',
            description='Daily effective federal funds rate',
            indicator_type=IndicatorType.MONETARY,
            frequency=DataFrequency.DAILY,
            weight=0.03
        )
    }
    
    # Indicator categories for analysis
    CATEGORIES = {
        'growth': ['GDP', 'INDPRO', 'RSXFS', 'HOUST'],
        'employment': ['UNRATE', 'PAYEMS'],
        'inflation': ['CPIAUCSL', 'M2SL'],
        'monetary': ['FEDFUNDS', 'DFF', 'DGS10', 'DGS2', 'T10Y2Y', 'T10Y3M'],
        'consumer': ['UMCSENT', 'DSPIC96'],
        'financial': ['VIXCLS', 'BAMLCC0A0CMNATRIV'],
        'currency': ['DEXUSEU', 'DEXUSUK', 'DEXJPUS', 'DEXCHUS', 'DEXUSCA'],
        'housing': ['HOUST']
    }
    
    # Sentiment thresholds for different indicator types
    SENTIMENT_THRESHOLDS = {
        IndicatorType.GROWTH: {
            'strong_bullish': 0.7,
            'bullish': 0.3,
            'neutral': 0.0,
            'bearish': -0.3,
            'strong_bearish': -0.7
        },
        IndicatorType.EMPLOYMENT: {
            'strong_bullish': 0.6,
            'bullish': 0.2,
            'neutral': 0.0,
            'bearish': -0.2,
            'strong_bearish': -0.6
        },
        IndicatorType.INFLATION: {
            'strong_bullish': 0.3,  # Moderate inflation
            'bullish': 0.1,
            'neutral': 0.0,
            'bearish': -0.3,  # High or low inflation
            'strong_bearish': -0.6
        },
        IndicatorType.MONETARY: {
            'strong_bullish': 0.5,
            'bullish': 0.2,
            'neutral': 0.0,
            'bearish': -0.2,
            'strong_bearish': -0.5
        },
        IndicatorType.CONSUMER: {
            'strong_bullish': 0.6,
            'bullish': 0.3,
            'neutral': 0.0,
            'bearish': -0.3,
            'strong_bearish': -0.6
        },
        IndicatorType.FINANCIAL: {
            'strong_bullish': 0.5,
            'bullish': 0.2,
            'neutral': 0.0,
            'bearish': -0.2,
            'strong_bearish': -0.5
        }
    }
    
    @classmethod
    def get_indicator(cls, series_id: str) -> Optional[EconomicIndicator]:
        """Get indicator by series ID"""
        return cls.INDICATORS.get(series_id)
    
    @classmethod
    def get_indicators_by_type(cls, indicator_type: IndicatorType) -> List[EconomicIndicator]:
        """Get all indicators of a specific type"""
        return [ind for ind in cls.INDICATORS.values() if ind.indicator_type == indicator_type]
    
    @classmethod
    def get_indicators_by_category(cls, category: str) -> List[EconomicIndicator]:
        """Get all indicators in a specific category"""
        series_ids = cls.CATEGORIES.get(category, [])
        return [cls.INDICATORS[series_id] for series_id in series_ids if series_id in cls.INDICATORS]
    
    @classmethod
    def get_core_indicators(cls) -> List[EconomicIndicator]:
        """Get core indicators with highest weights"""
        return sorted(cls.INDICATORS.values(), key=lambda x: x.weight, reverse=True)[:10]
    
    @classmethod
    def get_daily_indicators(cls) -> List[EconomicIndicator]:
        """Get indicators with daily frequency"""
        return [ind for ind in cls.INDICATORS.values() if ind.frequency == DataFrequency.DAILY]
    
    @classmethod
    def validate_indicator_config(cls) -> bool:
        """Validate indicator configuration"""
        total_weight = sum(ind.weight for ind in cls.INDICATORS.values())
        
        # Check if weights sum to reasonable value (should be close to 1.0)
        if abs(total_weight - 1.0) > 0.1:
            print(f"Warning: Indicator weights sum to {total_weight}, expected ~1.0")
        
        # Check for missing categories
        for category, indicators in cls.CATEGORIES.items():
            missing = [ind for ind in indicators if ind not in cls.INDICATORS]
            if missing:
                print(f"Warning: Missing indicators in {category}: {missing}")
        
        return True
    
    @classmethod
    def get_sentiment_threshold(cls, indicator_type: IndicatorType, sentiment_level: str) -> float:
        """Get sentiment threshold for indicator type and level"""
        thresholds = cls.SENTIMENT_THRESHOLDS.get(indicator_type, {})
        return thresholds.get(sentiment_level, 0.0)
