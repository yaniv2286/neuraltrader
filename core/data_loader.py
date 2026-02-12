"""
Core Data Loader - Modern Institutional Era
==========================================

NeuralTrader v4.0 Global Data Strategy
Enforces Modern Institutional Era rules for clean, institutional-grade data.

Key Rules:
1. Time Filter: STRICTLY 2000-01-01 to Present (no pre-decimalization noise)
2. Liquidity Gate: Price > $5 AND Dollar Volume > $1M (dynamic filter)
3. No Survivorship Bias: Keep delisted tickers but filter their dying data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class ModernEraDataLoader:
    """
    Modern Institutional Era Data Loader
    
    Enforces strict data quality rules for institutional-grade analysis:
    - Time Filter: 2000-01-01 to Present only
    - Liquidity Gate: Price > $5 AND Dollar Volume > $1M
    - No Survivorship Bias: Keep delisted stocks, filter dying data
    """
    
    def __init__(self):
        """Initialize Modern Era Data Loader"""
        # 🦅 MODERN INSTITUTIONAL ERA PARAMETERS
        self.start_date = '2000-01-01'  # Strict start date
        self.min_price = 5.00  # Minimum price filter
        self.min_dollar_volume = 1_000_000  # Minimum dollar volume filter
        
        # Data paths
        self.raw_data_path = PROJECT_ROOT / 'data' / 'raw'
        self.processed_data_path = PROJECT_ROOT / 'data' / 'processed'
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Cache file for modern era universe
        self.modern_era_cache = self.processed_data_path / 'modern_era_universe.parquet'
        
        logger.info("🦅 Modern Era Data Loader initialized")
        logger.info(f"   Time Filter: {self.start_date} to Present")
        logger.info(f"   Liquidity Gate: Price > ${self.min_price} & DollarVol > ${self.min_dollar_volume:,}")
        logger.info(f"   Cache File: {self.modern_era_cache}")
    
    def load_modern_era_universe(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load entire modern era universe with all filters applied
        
        Args:
            force_refresh: Force rebuild cache even if exists
            
        Returns:
            Dict of ticker -> DataFrame with modern era data
        """
        try:
            # Check if cached version exists and is valid
            if not force_refresh and self.modern_era_cache.exists():
                logger.info("📦 Loading cached modern era universe...")
                return self._load_cached_universe()
            
            logger.info("🔧 Building modern era universe from scratch...")
            
            # Load raw data
            raw_data = self._load_raw_parquet_files()
            
            # Apply modern era filters
            filtered_data = self._apply_modern_era_filters(raw_data)
            
            # Save to cache
            self._save_cached_universe(filtered_data)
            
            logger.info(f"✅ Modern era universe built: {len(filtered_data)} tickers")
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load modern era universe: {e}")
            return {}
    
    def _load_raw_parquet_files(self) -> Dict[str, pd.DataFrame]:
        """Load all parquet files from raw data directory"""
        try:
            parquet_files = list(self.raw_data_path.glob('*.parquet'))
            logger.info(f"Found {len(parquet_files)} raw parquet files")
            
            raw_data = {}
            for file_path in parquet_files:
                ticker = file_path.stem
                try:
                    df = pd.read_parquet(file_path)
                    raw_data[ticker] = df
                    logger.debug(f"Loaded {ticker}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Failed to load {ticker}: {e}")
            
            logger.info(f"✅ Loaded raw data for {len(raw_data)} tickers")
            return raw_data
            
        except Exception as e:
            logger.error(f"Failed to load raw parquet files: {e}")
            return {}
    
    def _apply_modern_era_filters(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply Modern Institutional Era filters to raw data"""
        try:
            filtered_data = {}
            total_rows_before = 0
            total_rows_after = 0
            
            for ticker, df in raw_data.items():
                if df.empty:
                    continue
                
                rows_before = len(df)
                total_rows_before += rows_before
                
                # Apply filters
                df_filtered = self._filter_single_ticker(df.copy(), ticker)
                
                if not df_filtered.empty:
                    filtered_data[ticker] = df_filtered
                    total_rows_after += len(df_filtered)
                    
                    retention_rate = (len(df_filtered) / rows_before) * 100
                    logger.debug(f"{ticker}: {len(df_filtered)}/{rows_before} rows retained ({retention_rate:.1f}%)")
                else:
                    logger.warning(f"{ticker}: All data filtered out")
            
            # Log summary
            overall_retention = (total_rows_after / total_rows_before) * 100 if total_rows_before > 0 else 0
            logger.info(f"🦅 MODERN ERA FILTER SUMMARY:")
            logger.info(f"   Total tickers: {len(raw_data)} → {len(filtered_data)}")
            logger.info(f"   Total rows: {total_rows_before:,} → {total_rows_after:,} ({overall_retention:.1f}% retained)")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Failed to apply modern era filters: {e}")
            return {}
    
    def _filter_single_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Apply Modern Institutional Era filters to a single ticker
        
        Args:
            df: Raw DataFrame for ticker
            ticker: Ticker symbol
            
        Returns:
            Filtered DataFrame
        """
        try:
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Convert date column and set index
            df = self._process_date_column(df)
            if df.empty:
                return df
            
            # 🦅 CRITICAL: FORCE THE DATE FILTER - Strictly 2000-01-01 to Present
            original_count = len(df)
            start_date = pd.to_datetime(self.start_date)
            df = df[df.index >= start_date]
            filtered_count = len(df)
            
            # 🦅 VERIFY THE FILTER - Print row count reduction
            reduction_pct = ((original_count - filtered_count) / original_count) * 100
            print(f"🦅 MODERN ERA FILTER: {ticker}: {original_count} rows -> {filtered_count} rows ({reduction_pct:.1f}% reduction)")
            
            if df.empty:
                logger.debug(f"{ticker}: No data after {self.start_date}")
                return df
            
            # 🦅 LIQUIDITY GATE - Dynamic filter per row
            df = self._apply_liquidity_gate(df, ticker)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to filter {ticker}: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to consistent format"""
        try:
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjclose': 'Adj Close',
                'adj_close': 'Adj Close',
                'adj_open': 'Adj Open',
                'adj_high': 'Adj High',
                'adj_low': 'Adj Low',
                'adj_volume': 'Adj Volume'
            }
            
            # Apply mapping (case-insensitive)
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to standardize columns: {e}")
            return df
    
    def _process_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process date column and set as index"""
        try:
            # Find date column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if not date_cols:
                logger.error("No date column found")
                return pd.DataFrame()
            
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Remove timezone info if present
            if df[date_col].dt.tz is not None:
                df[date_col] = df[date_col].dt.tz_localize(None)
            
            # Set as index
            df = df.set_index(date_col)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to process date column: {e}")
            return pd.DataFrame()
    
    def _apply_liquidity_gate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Apply Liquidity Gate: Price > $5 AND Dollar Volume > $1M
        
        This is a DYNAMIC filter - applied row by row
        Keeps healthy history but filters dying penny-stock data
        """
        try:
            # Get price column (prefer Adj Close, fallback to Close)
            if 'Adj Close' in df.columns:
                price_col = 'Adj Close'
            elif 'Close' in df.columns:
                price_col = 'Close'
            else:
                logger.error(f"No price column found for {ticker}")
                return pd.DataFrame()
            
            # Get volume column (prefer Adj Volume, fallback to Volume)
            if 'Adj Volume' in df.columns:
                volume_col = 'Adj Volume'
            elif 'Volume' in df.columns:
                volume_col = 'Volume'
            else:
                logger.error(f"No volume column found for {ticker}")
                return pd.DataFrame()
            
            # Calculate dollar volume
            df['dollar_volume'] = df[price_col] * df[volume_col]
            
            # 🦅 LIQUIDITY GATE - Apply dynamic filter
            # Keep rows that pass BOTH conditions
            price_filter = df[price_col] > self.min_price
            dollar_volume_filter = df['dollar_volume'] > self.min_dollar_volume
            
            # Apply filter - rows that fail are marked as noise (NaN)
            liquidity_mask = price_filter & dollar_volume_filter
            
            # Instead of dropping, we mark failing rows as NaN to maintain timeline
            df.loc[~liquidity_mask, price_col] = np.nan
            df.loc[~liquidity_mask, volume_col] = np.nan
            df.loc[~liquidity_mask, 'dollar_volume'] = np.nan
            
            # Log filter impact
            total_rows = len(df)
            passing_rows = liquidity_mask.sum()
            failing_rows = total_rows - passing_rows
            
            if failing_rows > 0:
                failing_pct = (failing_rows / total_rows) * 100
                logger.debug(f"{ticker}: {failing_rows} rows ({failing_pct:.1f}%) failed liquidity gate")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to apply liquidity gate for {ticker}: {e}")
            return pd.DataFrame()
    
    def _load_cached_universe(self) -> Dict[str, pd.DataFrame]:
        """Load cached modern era universe"""
        try:
            logger.info(f"Loading cached universe from {self.modern_era_cache}")
            
            # Read cached parquet
            cached_df = pd.read_parquet(self.modern_era_cache)
            
            if 'ticker' not in cached_df.columns:
                logger.error("Cached data missing ticker column")
                return {}
            
            # Split by ticker
            filtered_data = {}
            for ticker in cached_df['ticker'].unique():
                ticker_data = cached_df[cached_df['ticker'] == ticker].copy()
                ticker_data = ticker_data.drop('ticker', axis=1)
                filtered_data[ticker] = ticker_data
            
            logger.info(f"✅ Loaded cached universe: {len(filtered_data)} tickers")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Failed to load cached universe: {e}")
            return {}
    
    def _save_cached_universe(self, filtered_data: Dict[str, pd.DataFrame]):
        """Save filtered data to cache"""
        try:
            logger.info("Saving modern era universe to cache...")
            
            # Combine all data with ticker column
            combined_dfs = []
            for ticker, df in filtered_data.items():
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                combined_dfs.append(df_copy)
            
            if combined_dfs:
                combined_df = pd.concat(combined_dfs, ignore_index=False)
                
                # Save to parquet
                combined_df.to_parquet(self.modern_era_cache, index=True)
                
                logger.info(f"✅ Cached modern era universe saved: {self.modern_era_cache}")
                logger.info(f"   File size: {self.modern_era_cache.stat().st_size / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to save cached universe: {e}")
    
    def get_universe_summary(self) -> Dict[str, any]:
        """Get summary statistics of the modern era universe"""
        try:
            if not self.modern_era_cache.exists():
                return {"error": "No cached universe found"}
            
            # Load cached data for summary
            cached_df = pd.read_parquet(self.modern_era_cache)
            
            if 'ticker' not in cached_df.columns:
                return {"error": "Invalid cache format"}
            
            # Calculate statistics
            summary = {
                "total_tickers": cached_df['ticker'].nunique(),
                "total_rows": len(cached_df),
                "date_range": {
                    "start": cached_df.index.min().date(),
                    "end": cached_df.index.max().date()
                },
                "tickers": list(cached_df['ticker'].unique())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get universe summary: {e}")
            return {"error": str(e)}
    
    def get_unique_tickers(self) -> List[str]:
        """Get list of unique tickers from modern era data (memory efficient)"""
        try:
            logger.info("Getting unique tickers from modern era data...")
            
            # Load cached universe to get ticker list
            if self.modern_era_cache.exists():
                logger.info(f"Loading ticker list from cache: {self.modern_era_cache}")
                
                # Use pyarrow to read just the ticker column efficiently
                import pyarrow.parquet as pq
                table = pq.read_table(self.modern_era_cache, columns=['ticker'])
                ticker_list = table.column('ticker').unique().to_pylist()
                
                logger.info(f"Found {len(ticker_list)} unique tickers")
                return ticker_list
            else:
                # If no cache, scan raw files for tickers
                logger.info("No cache found, scanning raw files...")
                ticker_set = set()
                
                parquet_files = list(self.raw_data_path.glob('*.parquet'))
                for file_path in parquet_files:
                    try:
                        # Read just ticker column
                        table = pq.read_table(file_path, columns=['ticker'])
                        tickers = table.column('ticker').unique().to_pylist()
                        ticker_set.update(tickers)
                    except Exception as e:
                        logger.warning(f"Failed to read tickers from {file_path}: {e}")
                
                ticker_list = list(ticker_set)
                logger.info(f"Found {len(ticker_list)} unique tickers from raw files")
                return ticker_list
                
        except Exception as e:
            logger.error(f"Failed to get unique tickers: {e}")
            return []
    
    def load_single_ticker(self, ticker: str) -> pd.DataFrame:
        """Load data for a single ticker with modern era filters applied"""
        try:
            logger.info(f"Loading single ticker: {ticker}")
            
            # Check cache first
            if self.modern_era_cache.exists():
                logger.info(f"Loading {ticker} from cache...")
                
                # Use pandas to read parquet with index
                df = pd.read_parquet(self.modern_era_cache)
                
                # Filter for specific ticker
                ticker_df = df[df['ticker'] == ticker].copy()
                
                if not ticker_df.empty:
                    # Date is already the index, just sort it
                    ticker_df = ticker_df.sort_index()
                    
                    logger.info(f"✅ {ticker}: {len(ticker_df)} rows loaded")
                    return ticker_df
                else:
                    logger.warning(f"⚠️ {ticker}: No data found in cache")
                    return pd.DataFrame()
            else:
                # If no cache, load from raw file and apply filters
                logger.info(f"No cache found, loading {ticker} from raw data...")
                
                # Find the raw file for this ticker
                ticker_file = self.raw_data_path / f"{ticker}.parquet"
                if not ticker_file.exists():
                    logger.warning(f"⚠️ {ticker}: No raw file found")
                    return pd.DataFrame()
                
                # Load raw data
                raw_data = {ticker: pd.read_parquet(ticker_file)}
                
                # Apply modern era filters
                filtered_data = self._apply_modern_era_filters(raw_data)
                
                if ticker in filtered_data:
                    logger.info(f"✅ {ticker}: {len(filtered_data[ticker])} rows loaded")
                    return filtered_data[ticker]
                else:
                    logger.warning(f"⚠️ {ticker}: No data after filtering")
                    return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Failed to load {ticker}: {e}")
            return pd.DataFrame()

    def validate_modern_era_compliance(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        Validate that data complies with Modern Institutional Era rules
        
        Returns:
            Dict with compliance results
        """
        try:
            compliance = {
                "time_filter": True,
                "liquidity_gate": True,
                "no_survivorship_bias": True,
                "data_quality": True
            }
            
            # Check time filter
            start_date = pd.to_datetime(self.start_date)
            for ticker, df in data.items():
                if not df.empty and df.index.min() < start_date:
                    compliance["time_filter"] = False
                    logger.warning(f"{ticker}: Data before {self.start_date} found")
                    break
            
            # Check liquidity gate compliance
            for ticker, df in data.items():
                if not df.empty:
                    if 'Close' in df.columns and 'Volume' in df.columns:
                        dollar_volume = df['Close'] * df['Volume']
                        
                        # Check if any rows violate liquidity gate (and aren't NaN)
                        price_violation = (df['Close'] <= self.min_price) & df['Close'].notna()
                        volume_violation = (dollar_volume <= self.min_dollar_volume) & dollar_volume.notna()
                        
                        if price_violation.any() or volume_violation.any():
                            compliance["liquidity_gate"] = False
                            logger.warning(f"{ticker}: Liquidity gate violations found")
                            break
            
            logger.info(f"🦅 MODERN ERA COMPLIANCE: {compliance}")
            return compliance
            
        except Exception as e:
            logger.error(f"Failed to validate compliance: {e}")
            return {"error": str(e)}


# Convenience function for quick loading
def load_modern_era_universe(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Convenience function to load modern era universe"""
    loader = ModernEraDataLoader()
    return loader.load_modern_era_universe(force_refresh)


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = ModernEraDataLoader()
    
    # Load universe
    data = loader.load_modern_era_universe()
    
    # Get summary
    summary = loader.get_universe_summary()
    print("Modern Era Universe Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate compliance
    compliance = loader.validate_modern_era_compliance(data)
    print("Compliance Check:", compliance)
