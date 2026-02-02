"""
NeuralTrader 2.0 - Alpha Factory
=================================

Transforms raw Parquet files into a 42-feature ML matrix.

Features:
- RSI(14)
- ATR(14) 
- SMA Distances (20, 50, 200)
- Volume Z-Scores
- Relative Strength (Ticker Return minus SPY Return)

Target: 5-day forward log return
Normalization: Cross-Sectional Z-Scoring
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralTrader.AlphaFactory")

class AlphaFactory:
    """
    Creates alpha features from raw market data.
    """
    
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed"):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Feature parameters
        self.RSI_PERIOD = 14
        self.ATR_PERIOD = 14
        self.SMA_PERIODS = [20, 50, 200]
        self.VOLUME_Z_PERIOD = 20
        self.FORWARD_DAYS = 5
        
        # SPY ticker for market benchmark
        self.SPY_TICKER = "SPY"
        
        logger.info("Alpha Factory initialized")
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_log_returns(self, prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate log returns for multiple periods."""
        features = pd.DataFrame(index=prices.index)
        
        for period in periods:
            log_ret = np.log(prices / prices.shift(period))
            features[f'log_return_{period}d'] = log_ret
        
        return features
    
    def calculate_bollinger_band_width(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Band Width."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        band_width = (upper_band - lower_band) / sma
        return band_width
    
    def calculate_daily_dollar_volume_z_score(self, price: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Z-Score of Daily Dollar Volume."""
        dollar_volume = price * volume
        rolling_mean = dollar_volume.rolling(window=period).mean()
        rolling_std = dollar_volume.rolling(window=period).std()
        z_scores = (dollar_volume - rolling_mean) / rolling_std
        return z_scores
    
    def calculate_sma_distances(self, prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate SMA distance features."""
        features = pd.DataFrame(index=prices.index)
        
        for period in periods:
            sma = prices.rolling(window=period).mean()
            features[f'sma_{period}_distance'] = (prices - sma) / sma
            features[f'sma_{period}_ratio'] = prices / sma
        
        return features
    
    def calculate_volume_z_scores(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate volume Z-scores."""
        rolling_mean = volume.rolling(window=period).mean()
        rolling_std = volume.rolling(window=period).std()
        z_scores = (volume - rolling_mean) / rolling_std
        return z_scores
    
    def calculate_relative_strength(self, returns: pd.Series, spy_returns: pd.Series) -> pd.Series:
        """Calculate relative strength vs SPY."""
        return returns - spy_returns
    
    def create_features_for_ticker(self, ticker: str, spy_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create features for a single ticker.
        
        Args:
            ticker: Ticker symbol
            spy_data: SPY data for relative strength
            
        Returns:
            DataFrame with features
        """
        file_path = self.raw_path / f"{ticker}.parquet"
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {ticker}")
            return pd.DataFrame()
        
        try:
            # Load data
            df = pd.read_parquet(file_path)
            
            # Basic validation
            if len(df) < 100:
                logger.warning(f"Insufficient data for {ticker}: {len(df)} days")
                return pd.DataFrame()
            
            # Calculate returns
            df['returns'] = df['adjClose'].pct_change()
            df['log_returns'] = np.log(df['adjClose'] / df['adjClose'].shift(1))
            
            # Calculate target variable (5-day forward log return)
            df['target'] = df['log_returns'].shift(-self.FORWARD_DAYS)
            
            # Calculate features
            features = pd.DataFrame(index=df.index)
            
            # Momentum Features
            # 1. RSI(14)
            features['rsi_14'] = self.calculate_rsi(df['adjClose'], self.RSI_PERIOD)
            
            # 2. Log Returns (10, 20, 60 day)
            log_return_features = self.calculate_log_returns(df['adjClose'], [10, 20, 60])
            features = pd.concat([features, log_return_features], axis=1)
            
            # Volatility Features
            # 3. ATR(14)
            features['atr_14'] = self.calculate_atr(df['high'], df['low'], df['adjClose'], self.ATR_PERIOD)
            features['atr_pct'] = features['atr_14'] / df['adjClose']
            
            # 4. Bollinger Band Width
            features['bb_width'] = self.calculate_bollinger_band_width(df['adjClose'])
            
            # Relative Strength Features
            # 5. Ticker performance vs. SPY (Benchmark)
            if spy_data is not None:
                # Align dates
                common_index = df.index.intersection(spy_data.index)
                if len(common_index) > 0:
                    spy_returns_aligned = spy_data.loc[common_index, 'returns']
                    ticker_returns_aligned = df.loc[common_index, 'returns']
                    
                    rel_strength = self.calculate_relative_strength(ticker_returns_aligned, spy_returns_aligned)
                    features.loc[common_index, 'relative_strength'] = rel_strength
                else:
                    features['relative_strength'] = 0.0
            else:
                features['relative_strength'] = 0.0
            
            # Volume Features
            # 6. Z-Score of Daily Dollar Volume
            features['dollar_volume_z'] = self.calculate_daily_dollar_volume_z_score(df['adjClose'], df['volume'])
            
            # Target Variable
            # 7. 5-day forward log return
            features['target'] = df['log_returns'].shift(-self.FORWARD_DAYS)
            
            # Add ticker and date
            features['ticker'] = ticker
            features['date'] = pd.to_datetime(df['date'])  # Convert to datetime
            
            # Keep recent data even without targets (for inference)
            # Don't drop NaNs from target column for the most recent 5 days
            features_with_target = features.dropna(subset=['target'])
            features_without_target = features[features['target'].isna()].copy()
            
            # For inference, we need features even without targets
            # Keep the most recent 5 days even if target is NaN
            if len(features_without_target) > 0:
                # Sort by date and keep last 5
                features_without_target = features_without_target.nlargest(5, 'date')
                # Set target to 0 for these rows (won't be used in training)
                features_without_target['target'] = 0.0
                # Combine
                features = pd.concat([features_with_target, features_without_target], ignore_index=True)
            else:
                features = features_with_target
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features for {ticker}: {e}")
            return pd.DataFrame()
    
    def load_spy_data(self) -> pd.DataFrame:
        """Load SPY data for market benchmark."""
        spy_file = self.raw_path / f"{self.SPY_TICKER}.parquet"
        
        if not spy_file.exists():
            logger.warning(f"SPY data not found: {spy_file}")
            return pd.DataFrame()
        
        try:
            spy_df = pd.read_parquet(spy_file)
            spy_df['returns'] = spy_df['adjClose'].pct_change()
            return spy_df
        except Exception as e:
            logger.error(f"Error loading SPY data: {e}")
            return pd.DataFrame()
    
    def create_master_feature_matrix(self, max_tickers: int = None) -> pd.DataFrame:
        """
        Create master feature matrix for all tickers.
        
        Args:
            max_tickers: Maximum number of tickers to process (for testing)
            
        Returns:
            Master feature DataFrame
        """
        logger.info("Creating master feature matrix...")
        
        # Get list of tickers
        ticker_files = list(self.raw_path.glob("*.parquet"))
        tickers = [f.stem for f in ticker_files if f.stem != self.SPY_TICKER]
        
        if max_tickers:
            tickers = tickers[:max_tickers]
            logger.info(f"Limited to {max_tickers} tickers for testing")
        
        # Load SPY data
        spy_data = self.load_spy_data()
        
        # Create features for each ticker
        all_features = []
        
        for ticker in tqdm(tickers, desc="Creating features"):
            features = self.create_features_for_ticker(ticker, spy_data)
            
            if not features.empty:
                all_features.append(features)
            else:
                logger.warning(f"No features created for {ticker}")
        
        if not all_features:
            logger.error("No features created for any ticker")
            return pd.DataFrame()
        
        # Combine all features
        master_df = pd.concat(all_features, ignore_index=False)
        
        logger.info(f"Master matrix created: {len(master_df)} rows, {len(master_df.columns)} columns")
        
        return master_df
    
    def apply_cross_sectional_z_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional Z-scoring for each date.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Normalized DataFrame
        """
        logger.info("Applying cross-sectional Z-scoring...")
        
        # Get feature columns (exclude ticker, target, and date)
        feature_cols = [col for col in df.columns if col not in ['ticker', 'target', 'date']]
        
        # Group by date and apply Z-scoring
        df_normalized = df.copy()
        
        for date in tqdm(df.index.unique(), desc="Z-scoring dates"):
            date_mask = df.index == date
            date_data = df.loc[date_mask, feature_cols]
            
            if len(date_data) > 1:  # Need at least 2 stocks for Z-scoring
                # Calculate mean and std
                mean_vals = date_data.mean()
                std_vals = date_data.std()
                
                # Avoid division by zero
                std_vals = std_vals.replace(0, 1)
                
                # Apply Z-scoring
                normalized = (date_data - mean_vals) / std_vals
                
                # Update dataframe
                df_normalized.loc[date_mask, feature_cols] = normalized
        
        logger.info("Cross-sectional Z-scoring completed")
        
        return df_normalized
    
    def save_feature_matrix(self, df: pd.DataFrame, filename: str = "master_feature_matrix.parquet") -> None:
        """
        Save feature matrix to Parquet.
        
        Args:
            df: Feature DataFrame
            filename: Output filename
        """
        output_path = self.processed_path / filename
        
        # Convert to float32 for memory efficiency (exclude date column)
        feature_cols = [col for col in df.columns if col not in ['ticker', 'target', 'date']]
        df[feature_cols] = df[feature_cols].astype(np.float32)
        
        # Save to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')
        
        logger.info(f"Feature matrix saved: {output_path}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def save_feature_metadata(self, df: pd.DataFrame) -> None:
        """Save feature metadata."""
        metadata = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max())
            },
            'tickers': df['ticker'].nunique(),
            'feature_count': len([col for col in df.columns if col not in ['ticker', 'target']]),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = self.processed_path / "feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature metadata saved: {metadata_path}")
    
    def run_continuous(self, max_tickers: int = None, check_interval: int = 30) -> pd.DataFrame:
        """
        Run continuous alpha factory processing with graceful file handling.
        
        Args:
            max_tickers: Maximum number of tickers to process
            check_interval: Seconds to wait between processing cycles
            
        Returns:
            Master feature DataFrame
        """
        logger.info("Starting continuous Alpha Factory pipeline...")
        logger.info(f"Check interval: {check_interval} seconds")
        
        while True:
            try:
                # Get current parquet files
                parquet_files = list(self.raw_path.glob("*.parquet"))
                
                if not parquet_files:
                    logger.info("No parquet files found. Waiting...")
                    time.sleep(check_interval)
                    continue
                
                logger.info(f"Found {len(parquet_files)} parquet files")
                
                # Process files with graceful handling
                processed_count = 0
                skipped_count = 0
                
                for file_path in tqdm(parquet_files, desc="Processing files"):
                    try:
                        # Check if file is locked (being written by downloader)
                        if self.is_file_locked(file_path):
                            logger.debug(f"Skipping locked file: {file_path.name}")
                            skipped_count += 1
                            continue
                        
                        # Process the file
                        ticker = file_path.stem
                        if ticker == self.SPY_TICKER:
                            continue  # Skip SPY (used as benchmark)
                        
                        # Load and process
                        features = self.create_features_for_ticker(ticker, None)
                        
                        if not features.empty:
                            processed_count += 1
                        else:
                            logger.warning(f"No features created for {ticker}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {file_path.name}: {e}")
                        skipped_count += 1
                        continue
                
                logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
                
                # If we processed new files, update the master matrix
                if processed_count > 0:
                    logger.info("Updating master feature matrix...")
                    master_df = self.create_master_feature_matrix(max_tickers=max_tickers)
                    
                    if not master_df.empty:
                        # Apply cross-sectional Z-scoring
                        normalized_df = self.apply_cross_sectional_z_scoring(master_df)
                        
                        # Save feature matrix
                        self.save_feature_matrix(normalized_df)
                        
                        # Save metadata
                        self.save_feature_metadata(normalized_df)
                        
                        logger.info(f"Updated matrix: {len(normalized_df)} rows, {len(normalized_df.columns)} columns")
                    else:
                        logger.warning("No feature data to save")
                else:
                    logger.info("No new files processed")
                
                # Wait before next cycle
                logger.info(f"Waiting {check_interval} seconds...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Continuous processing stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                time.sleep(check_interval)
                continue
        
        # Return final matrix
        try:
            final_matrix = pd.read_parquet(self.processed_path / "master_feature_matrix.parquet")
            return final_matrix
        except FileNotFoundError:
            logger.error("No feature matrix found")
            return pd.DataFrame()
    
    def is_file_locked(self, file_path: Path) -> bool:
        """
        Check if a file is locked (being written to).
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is locked, False otherwise
        """
        try:
            # Try to open the file in read mode
            with open(file_path, 'rb') as f:
                # Try to read first byte
                f.read(1)
            return False
        except (IOError, OSError, PermissionError):
            return True
    
    def run(self, max_tickers: int = None) -> pd.DataFrame:
        """
        Run the complete alpha factory pipeline.
        
        Args:
            max_tickers: Maximum number of tickers to process
            
        Returns:
            Master feature DataFrame
        """
        logger.info("Starting Alpha Factory pipeline...")
        
        # Create master feature matrix
        master_df = self.create_master_feature_matrix(max_tickers=max_tickers)
        
        if master_df.empty:
            logger.error("No feature data created")
            return pd.DataFrame()
        
        # Apply cross-sectional Z-scoring
        normalized_df = self.apply_cross_sectional_z_scoring(master_df)
        
        # Save feature matrix
        self.save_feature_matrix(normalized_df)
        
        # Save metadata
        self.save_feature_metadata(normalized_df)
        
        logger.info("Alpha Factory pipeline completed successfully")
        
        return normalized_df

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create alpha features from raw market data")
    parser.add_argument("--raw-path", default="data/raw", help="Path to raw data")
    parser.add_argument("--processed-path", default="data/processed", help="Path to processed data")
    parser.add_argument("--max-tickers", type=int, help="Maximum tickers to process (for testing)")
    parser.add_argument("--continuous", action="store_true", help="Run continuous processing mode")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval in seconds for continuous mode")
    
    args = parser.parse_args()
    
    # Initialize alpha factory
    factory = AlphaFactory(raw_path=args.raw_path, processed_path=args.processed_path)
    
    # Run pipeline
    if args.continuous:
        feature_matrix = factory.run_continuous(max_tickers=args.max_tickers, check_interval=args.check_interval)
    else:
        feature_matrix = factory.run(max_tickers=args.max_tickers)
    
    if not feature_matrix.empty:
        print("\n" + "="*50)
        print("ALPHA FACTORY SUMMARY")
        print("="*50)
        print(f"Shape: {feature_matrix.shape}")
        print(f"Date Range: {feature_matrix.index.min()} to {feature_matrix.index.max()}")
        print(f"Tickers: {feature_matrix['ticker'].nunique()}")
        print(f"Features: {len([col for col in feature_matrix.columns if col not in ['ticker', 'target']])}")
        print("="*50)
    else:
        print("No feature matrix created")

if __name__ == "__main__":
    main()
