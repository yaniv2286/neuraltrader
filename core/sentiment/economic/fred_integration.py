#!/usr/bin/env python3
"""
FRED API Integration - Federal Reserve Economic Data
Fetches and processes economic indicators for sentiment analysis
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import json
import os

logger = logging.getLogger(__name__)

class FREDIntegration:
    """
    FRED API integration for fetching economic data
    
    Provides access to Federal Reserve Economic Data with proper
    rate limiting, caching, and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/fred_cache"):
        """
        Initialize FRED API integration
        
        Args:
            api_key: FRED API key (can be set in environment variable FRED_API_KEY)
            cache_dir: Directory for caching FRED data
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            logger.warning("FRED_API_KEY not found. Set environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.stlouisfed.org/fred"
        self.cache_dir = cache_dir
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.session = requests.Session()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Key economic indicators
        self.key_indicators = {
            'GDP': 'GDP',                    # Real GDP
            'UNRATE': 'UNRATE',              # Unemployment Rate
            'CPIAUCSL': 'CPIAUCSL',          # Consumer Price Index
            'FEDFUNDS': 'FEDFUNDS',          # Federal Funds Rate
            'DGS10': 'DGS10',                # 10-Year Treasury Constant Maturity Rate
            'DGS2': 'DGS2',                  # 2-Year Treasury Constant Maturity Rate
            'PAYEMS': 'PAYEMS',              # All Employees: Total Nonfarm Payrolls
            'INDPRO': 'INDPRO',              # Industrial Production Index
            'UMCSENT': 'UMCSENT',            # University of Michigan Consumer Sentiment
            'HOUST': 'HOUST',                # Housing Starts: Total
            'RSXFS': 'RSXFS',                # Retail and Food Services Sales
            'DSPIC96': 'DSPIC96',            # Real Disposable Personal Income
            'M2SL': 'M2SL',                  # M2 Money Supply
            'DEXUSEU': 'DEXUSEU',            # US/Euro Exchange Rate
            'DEXUSUK': 'DEXUSUK',            # US/UK Exchange Rate
            'DEXCHUS': 'DEXCHUS',            # China/US Exchange Rate
            'DEXUSAL': 'DEXUSAL',            # US/Australia Exchange Rate
            'DEXUSCA': 'DEXUSCA',            # US/Canada Exchange Rate
            'VIXCLS': 'VIXCLS',              # VIX Volatility Index
            'BAMLCC0A0CMNATRIV': 'BAMLCC0A0CMNATRIV',  # ICE BofA US Corporate Index
            'T10Y2Y': 'T10Y2Y',              # 10-Year minus 2-Year Treasury Spread
            'T10Y3M': 'T10Y3M',              # 10-Year minus 3-Month Treasury Spread
            'DFF': 'DFF',                    # Daily Federal Funds Rate
            'DEXJPUS': 'DEXJPUS',            # Japan/US Exchange Rate
            'DEXKUS': 'DEXKUS',              # Korea/US Exchange Rate
            'DEXMXUS': 'DEXMXUS',            # Mexico/US Exchange Rate
            'DEXSIUS': 'DEXSIUS',            # Singapore/US Exchange Rate
            'DEXINUS': 'DEXINUS',            # India/US Exchange Rate
            'DEXBZUS': 'DEXBZUS',            # Brazil/US Exchange Rate
            'DEXUSNZ': 'DEXUSNZ',            # US/New Zealand Exchange Rate
            'DEXTHUS': 'DEXTHUS',            # Thailand/US Exchange Rate
            'DEXSZUS': 'DEXSZUS',            # South Africa/US Exchange Rate
            'DEXNOUS': 'DEXNOUS',            # Norway/US Exchange Rate
            'DEXSEUS': 'DEXSEUS',            # Sweden/US Exchange Rate
            'DEXHKUS': 'DEXHKUS',            # Hong Kong/US Exchange Rate
        }
        
        # Indicator categories and weights
        self.indicator_categories = {
            'growth': ['GDP', 'INDPRO', 'RSXFS', 'HOUST'],
            'employment': ['UNRATE', 'PAYEMS'],
            'inflation': ['CPIAUCSL', 'M2SL'],
            'monetary': ['FEDFUNDS', 'DFF', 'DGS10', 'DGS2', 'T10Y2Y', 'T10Y3M'],
            'consumer': ['UMCSENT', 'DSPIC96'],
            'financial': ['VIXCLS', 'BAMLCC0A0CMNATRIV'],
            'currency': ['DEXUSEU', 'DEXUSUK', 'DEXJPUS', 'DEXCHUS', 'DEXUSCA']
        }
        
        logger.info(f"FRED Integration initialized with {len(self.key_indicators)} indicators")
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response data
        """
        if not self.api_key:
            raise ValueError("FRED API key required")
        
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'error_code' in data:
                raise Exception(f"FRED API Error: {data.get('error_message', 'Unknown error')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode FRED API response: {e}")
            raise
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get information about a specific series
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Series information dictionary
        """
        cache_file = os.path.join(self.cache_dir, f"info_{series_id}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                # Check if cache is recent (less than 24 hours)
                cache_time = datetime.fromisoformat(data.get('cached_at', '1970-01-01'))
                if (datetime.now() - cache_time).total_seconds() < 86400:
                    logger.debug(f"Using cached info for {series_id}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cached info for {series_id}: {e}")
        
        # Fetch from API
        params = {'series_id': series_id}
        data = self._make_request('series', params)
        
        if 'seriess' not in data:
            raise Exception(f"No data found for series {series_id}")
        
        series_info = data['seriess'][0]
        series_info['cached_at'] = datetime.now().isoformat()
        
        # Cache the result
        try:
            with open(cache_file, 'w') as f:
                json.dump(series_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache info for {series_id}: {e}")
        
        return series_info
    
    def get_series_data(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get time series data for a specific indicator
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with date and value columns
        """
        # Default date range
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        cache_file = os.path.join(self.cache_dir, f"data_{series_id}_{start_date}_{end_date}.csv")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # Cache for 1 hour for recent data, 24 hours for historical data
                cache_duration = 3600 if datetime.now().strftime('%Y-%m-%d') == end_date else 86400
                
                if (datetime.now() - cache_time).total_seconds() < cache_duration:
                    logger.debug(f"Using cached data for {series_id}")
                    return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
            except Exception as e:
                logger.warning(f"Failed to load cached data for {series_id}: {e}")
        
        # Fetch from API
        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        data = self._make_request('series/observations', params)
        
        if 'observations' not in data:
            raise Exception(f"No observations found for series {series_id}")
        
        # Convert to DataFrame
        observations = data['observations']
        df = pd.DataFrame(observations)
        
        if df.empty:
            logger.warning(f"No data returned for series {series_id}")
            return pd.DataFrame()
        
        # Convert date column and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Convert value column to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove null values
        df = df.dropna(subset=['value'])
        
        # Cache the result
        try:
            df.to_csv(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache data for {series_id}: {e}")
        
        logger.info(f"Fetched {len(df)} observations for {series_id}")
        return df
    
    def get_multiple_series(self, series_ids: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple series
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Dictionary mapping series IDs to DataFrames
        """
        results = {}
        
        logger.info(f"Fetching data for {len(series_ids)} series from FRED")
        
        for series_id in series_ids:
            try:
                df = self.get_series_data(series_id, start_date, end_date)
                if not df.empty:
                    results[series_id] = df
                else:
                    logger.warning(f"No data available for {series_id}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {series_id}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(series_ids)} series")
        return results
    
    def get_key_indicators_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all key economic indicators
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        return self.get_multiple_series(list(self.key_indicators.values()), start_date, end_date)
    
    def get_latest_releases(self) -> List[Dict]:
        """
        Get latest data releases from FRED
        
        Returns:
            List of recent releases
        """
        try:
            data = self._make_request('releases', {'limit': 100})
            return data.get('releases', [])
        except Exception as e:
            logger.error(f"Failed to fetch latest releases: {e}")
            return []
    
    def search_series(self, search_text: str, limit: int = 1000) -> List[Dict]:
        """
        Search for series by text
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching series
        """
        try:
            params = {
                'search_text': search_text,
                'limit': limit
            }
            data = self._make_request('series/search', params)
            return data.get('seriess', [])
        except Exception as e:
            logger.error(f"Failed to search series: {e}")
            return []
    
    def get_category_data(self, category_id: int) -> Dict:
        """
        Get data for a specific category
        
        Args:
            category_id: FRED category ID
            
        Returns:
            Category information and series
        """
        try:
            # Get category info
            category_data = self._make_request('category', {'category_id': category_id})
            
            # Get series in category
            series_data = self._make_request('category/series', {'category_id': category_id})
            
            return {
                'category': category_data.get('categories', [{}])[0],
                'series': series_data.get('seriess', [])
            }
        except Exception as e:
            logger.error(f"Failed to get category data: {e}")
            return {}
    
    def validate_api_key(self) -> bool:
        """
        Validate API key by making a test request
        
        Returns:
            True if API key is valid
        """
        try:
            self.get_series_info('GDP')
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'info_files': 0,
            'data_files': 0,
            'total_size_mb': 0
        }
        
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    stats['total_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
                    
                    if file.startswith('info_'):
                        stats['info_files'] += 1
                    elif file.startswith('data_'):
                        stats['data_files'] += 1
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
        
        return stats
    
    def clear_cache(self, older_than_days: int = 30) -> None:
        """
        Clear cached files older than specified days
        
        Args:
            older_than_days: Remove files older than this many days
        """
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        removed_files = 0
        
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        removed_files += 1
            
            logger.info(f"Removed {removed_files} cached files older than {older_than_days} days")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'session'):
            self.session.close()
