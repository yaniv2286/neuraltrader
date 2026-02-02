#!/usr/bin/env python3
"""
NeuralTrader 2.0 - Live Ranker Execution Bridge
==============================================

Core Intelligence Engine for 25% ARR mission.
Transforms trained model into actionable daily trading signals.

Features:
- Real-time data synchronization
- Feature calculation with exact training methodology
- Cross-sectional Z-scoring normalization
- Model inference with XGBoost Ranker
- Daily action report with stop losses
- One-click trading recommendations
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Import our alpha factory for feature calculation
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ml.features.alpha_factory import AlphaFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralTrader.LiveRanker")

class LiveRanker:
    """
    Live ranking system for daily trading signals.
    """
    
    def __init__(self, raw_path: str = "data/raw", models_path: str = "models", logs_path: str = "logs"):
        self.raw_path = Path(raw_path)
        self.models_path = Path(models_path)
        self.logs_path = Path(logs_path)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Alpha Factory for feature calculation
        self.alpha_factory = AlphaFactory(raw_path=raw_path, processed_path="data/temp_processed")
        self.temp_processed_path = Path("data/temp_processed")
        self.temp_processed_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters (must match training)
        self.feature_cols = [
            'rsi_14', 'log_return_10d', 'log_return_20d', 'log_return_60d',
            'atr_14', 'atr_pct', 'bb_width', 'relative_strength', 'dollar_volume_z'
        ]
        
        logger.info("LiveRanker initialized")
        logger.info(f"Raw data path: {self.raw_path}")
        logger.info(f"Models path: {self.models_path}")
        logger.info(f"Logs path: {self.logs_path}")
    
    def load_model(self) -> xgb.XGBRanker:
        """Load the trained XGBoost Ranker model."""
        logger.info("Loading trained model...")
        
        model_path = self.models_path / "neural_ranker_v1.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = xgb.XGBRanker()
        model.load_model(str(model_path))
        
        logger.info("Model loaded successfully")
        return model
    
    def sync_latest_data(self) -> pd.DataFrame:
        """
        Synchronize latest data from raw parquet files.
        Pulls last 100 trading days for every ticker.
        
        Returns:
            Combined DataFrame with latest data for all tickers
        """
        logger.info("Syncing latest data with 100-day context...")
        
        # Get all parquet files
        parquet_files = list(self.raw_path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data/raw/")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Load and combine all data with 100-day window
        all_data = []
        
        for file_path in parquet_files:
            try:
                ticker = file_path.stem
                df = pd.read_parquet(file_path)
                
                # Get last 100 trading days
                if len(df) >= 100:
                    df_100 = df.tail(100).copy()
                else:
                    logger.warning(f"Insufficient data for {ticker}: {len(df)} days (< 100)")
                    continue
                
                df_100['ticker'] = ticker
                all_data.append(df_100)
                logger.debug(f"Loaded {ticker}: {len(df_100)} rows (100-day window)")
            except Exception as e:
                logger.warning(f"Error loading {file_path.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.set_index('date')
        combined_df = combined_df.sort_index()
        
        logger.info(f"Combined data: {combined_df.shape}")
        logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        logger.info(f"Tickers: {combined_df['ticker'].nunique()}")
        
        return combined_df
    
    def calculate_latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for the latest available date using exact training methodology.
        Uses full 100-day window for indicator pre-calculation.
        
        Args:
            df: Raw price data (100-day window for each ticker)
            
        Returns:
            DataFrame with calculated features for latest date
        """
        logger.info("Calculating latest features with 100-day context...")
        
        # Get the latest date with data
        latest_date = df.index.max()
        logger.info(f"Processing data for {latest_date.date()}")
        
        # Calculate features using full 100-day window
        features_list = []
        
        for ticker in df['ticker'].unique():
            # Get full 100-day historical data for this ticker
            ticker_historical = df[df['ticker'] == ticker].copy()
            
            if len(ticker_historical) < 20:  # Need minimum data for indicators
                logger.warning(f"Insufficient data for {ticker}: {len(ticker_historical)} days")
                continue
            
            try:
                # Indicator Pre-Calculation: Calculate on full 100-day window
                features = self._calculate_ticker_features_full_window(ticker_historical)
                features['ticker'] = ticker
                features['date'] = latest_date
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error calculating features for {ticker}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No features calculated successfully")
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.set_index('date')
        
        # NaN Shield: Check for NaN values
        if features_df[self.feature_cols].isnull().values.any():
            nan_cols = features_df[self.feature_cols].isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            logger.warning(f"⚠️  NaN VALUES DETECTED: {nan_cols.to_dict()}")
        
        logger.info(f"Calculated features: {features_df.shape}")
        return features_df
    
    def _calculate_ticker_features_full_window(self, df: pd.DataFrame) -> dict:
        """
        Calculate features using full 100-day window, then select final row for prediction.
        
        Args:
            df: 100-day historical data for one ticker
            
        Returns:
            Features dict with final row values
        """
        features = {}
        
        try:
            # Indicator Pre-Calculation on full 100-day window
            rsi_14_full = self._calculate_rsi(df['adjClose'], 14)
            atr_14_full = self._calculate_atr(df['high'], df['low'], df['adjClose'], 14)
            
            # Moving averages for additional context
            sma_20 = df['adjClose'].rolling(window=20).mean()
            sma_50 = df['adjClose'].rolling(window=50).mean()
            
            # Select final row values for prediction
            current_price = df['adjClose'].iloc[-1]
            
            # Final features (using latest values)
            features['rsi_14'] = rsi_14_full
            features['log_return_10d'] = np.log(current_price / df['adjClose'].iloc[-10]) if len(df) >= 10 else 0.0
            features['log_return_20d'] = np.log(current_price / df['adjClose'].iloc[-20]) if len(df) >= 20 else 0.0
            features['log_return_60d'] = np.log(current_price / df['adjClose'].iloc[-60]) if len(df) >= 60 else 0.0
            features['atr_14'] = atr_14_full
            features['atr_pct'] = atr_14_full / current_price if current_price > 0 else 0.0
            features['bb_width'] = self._calculate_bb_width(df['adjClose'])
            features['relative_strength'] = 0.0  # Simplified
            features['dollar_volume_z'] = self._calculate_volume_z(df['adjClose'], df['volume'])
            
            # Indicator Validation - ensure RSI and ATR are valid
            if pd.isna(features['rsi_14']):
                logger.warning(f"⚠️  RSI_14 is NaN for ticker")
                features['rsi_14'] = 50.0  # Default RSI
            
            if pd.isna(features['atr_14']):
                logger.warning(f"⚠️  ATR_14 is NaN for ticker")
                features['atr_14'] = 0.0  # Default ATR
                features['atr_pct'] = 0.0
            
            # Convert to float32
            for key in features:
                features[key] = float(features[key]) if not pd.isna(features[key]) else 0.0
            
        except Exception as e:
            logger.warning(f"Error in feature calculation: {e}")
            # Return default values
            features = {col: 0.0 for col in self.feature_cols}
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0  # Default RSI if insufficient data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(high) < period + 1:
            return 0.0  # Default ATR if insufficient data
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_bb_width(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> float:
        """Calculate Bollinger Band Width."""
        if len(prices) < period + 1:
            return 0.0  # Default BB width if insufficient data
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        band_width = (upper_band - lower_band) / sma
        return band_width.iloc[-1] if not pd.isna(band_width.iloc[-1]) else 0.0
    
    def _calculate_volume_z(self, price: pd.Series, volume: pd.Series, period: int = 20) -> float:
        """Calculate Volume Z-Score."""
        if len(price) < period + 1:
            return 0.0  # Default Z-score if insufficient data
        
        dollar_volume = price * volume
        rolling_mean = dollar_volume.rolling(window=period).mean()
        rolling_std = dollar_volume.rolling(window=period).std()
        z_score = (dollar_volume - rolling_mean) / rolling_std
        return z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0.0
    
    def apply_z_scoring(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional Z-scoring (exact same as training).
        Ensures horizontal application across all tickers to break 'Tie' bug.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Normalized DataFrame
        """
        logger.info("Applying cross-sectional Z-scoring...")
        
        # Get feature columns (exclude ticker)
        feature_cols = [col for col in features_df.columns if col not in ['ticker']]
        
        # Apply Z-scoring horizontally across all tickers for latest date
        normalized_df = features_df.copy()
        
        # Process each date (should be just one date for latest snapshot)
        for date in features_df.index.unique():
            date_mask = features_df.index == date
            date_data = features_df.loc[date_mask, feature_cols]
            
            logger.info(f"Z-scoring {len(date_data)} stocks for date {date}")
            
            if len(date_data) > 1:  # Need multiple stocks for Z-scoring
                # Calculate mean and std horizontally across all tickers
                mean_vals = date_data.mean()
                std_vals = date_data.std()
                
                logger.info(f"Feature means: {mean_vals.to_dict()}")
                logger.info(f"Feature stds: {std_vals.to_dict()}")
                
                # Z-Score Correction: Apply horizontal Z-scoring to break ties
                std_vals = std_vals.replace(0, 1)  # Avoid division by zero
                normalized_data = (date_data - mean_vals) / std_vals
                
                normalized_df.loc[date_mask, feature_cols] = normalized_data
                
                # Log sample values to verify differentiation
                sample_tickers = normalized_df.loc[date_mask].head(5)['ticker'].tolist()
                sample_scores = normalized_df.loc[date_mask].head(5)[feature_cols[0]].tolist()
                logger.info(f"Sample {feature_cols[0]} values: {list(zip(sample_tickers, sample_scores))}")
                
                # Verify no ties in first feature
                unique_scores = len(set(normalized_df.loc[date_mask, feature_cols[0]]))
                logger.info(f"Unique {feature_cols[0]} scores: {unique_scores}/{len(date_data)} (should be equal)")
            else:
                logger.warning(f"Only 1 stock available for Z-scoring on {date}")
        
        return normalized_df
    
    def predict_rankings(self, model: xgb.XGBRanker, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict rankings using the trained model.
        Drops NaN values before inference.
        
        Args:
            model: Trained XGBoost Ranker
            features_df: Normalized features
            
        Returns:
            DataFrame with rankings and scores
        """
        logger.info("Predicting rankings...")
        
        # Get feature columns (exclude ticker)
        feature_cols = [col for col in features_df.columns if col not in ['ticker']]
        
        # Check for NaN values and drop them
        nan_mask = features_df[feature_cols].isnull().any(axis=1)
        if nan_mask.any():
            dropped_tickers = features_df[nan_mask]['ticker'].tolist()
            logger.warning(f"Dropping {len(dropped_tickers)} tickers with NaN features: {dropped_tickers[:5]}...")
            features_df = features_df[~nan_mask]
        
        if features_df.empty:
            raise ValueError("No valid features after dropping NaN values")
        
        # Prepare data for prediction
        X = features_df[feature_cols].values
        
        # Make predictions
        scores = model.predict(X)
        
        # Create results DataFrame
        results_df = features_df[['ticker']].copy()
        results_df['model_score'] = scores
        results_df['rank'] = results_df['model_score'].rank(ascending=False)
        
        # Sort by rank
        results_df = results_df.sort_values('rank')
        
        logger.info(f"Generated rankings for {len(results_df)} stocks")
        return results_df
    
    def calculate_stop_losses(self, rankings_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR-based stop losses for top stocks.
        Stop Loss Enforcement: $Current Price - 2.5 × ATR$
        
        Args:
            rankings_df: DataFrame with rankings
            features_df: DataFrame with features
            
        Returns:
            DataFrame with stop losses
        """
        logger.info("Calculating stop losses...")
        
        # Merge rankings with features to get ATR values and current prices
        # Reset index to make ticker a column for merging
        features_for_merge = features_df.reset_index()
        
        # Get current prices from the original data
        current_prices = {}
        for ticker in features_for_merge['ticker'].unique():
            try:
                ticker_file = self.raw_path / f"{ticker}.parquet"
                if ticker_file.exists():
                    df_price = pd.read_parquet(ticker_file)
                    current_prices[ticker] = df_price['adjClose'].iloc[-1]
                else:
                    current_prices[ticker] = 0.0
            except:
                current_prices[ticker] = 0.0
        
        # Add current price to features
        features_for_merge['current_price'] = features_for_merge['ticker'].map(current_prices)
        
        merged_df = rankings_df.merge(
            features_for_merge[['ticker', 'atr_14', 'atr_pct', 'current_price']].drop_duplicates('ticker'),
            on='ticker',
            how='left'
        )
        
        # Stop Loss Enforcement: $Current Price - 2.5 × ATR$
        merged_df['stop_loss_atr'] = merged_df['current_price'] - (merged_df['atr_14'] * 2.5)
        merged_df['stop_loss_pct'] = merged_df['atr_pct'] * 2.5
        
        # Ensure stop losses are valid float values
        merged_df['stop_loss_atr'] = merged_df['stop_loss_atr'].fillna(0.0)
        merged_df['stop_loss_pct'] = merged_df['stop_loss_pct'].fillna(0.0)
        
        # Validate stop losses
        invalid_stops = merged_df[merged_df['stop_loss_atr'] <= 0]
        if not invalid_stops.empty:
            logger.warning(f"⚠️  Invalid stop losses for: {invalid_stops['ticker'].tolist()}")
            # Set default stop loss at 5% below current price
            merged_df.loc[merged_df['stop_loss_atr'] <= 0, 'stop_loss_atr'] = merged_df.loc[merged_df['stop_loss_atr'] <= 0, 'current_price'] * 0.95
        
        logger.info(f"Stop loss range: ${merged_df['stop_loss_atr'].min():.2f} to ${merged_df['stop_loss_atr'].max():.2f}")
        
        return merged_df
    
    def generate_daily_report(self, rankings_df: pd.DataFrame, stop_losses_df: pd.DataFrame) -> str:
        """
        Generate daily action report.
        
        Args:
            rankings_df: DataFrame with rankings
            stop_losses_df: DataFrame with stop losses
            
        Returns:
            Formatted report string
        """
        logger.info("Generating daily report...")
        
        # Debug: Print available columns
        logger.info(f"stop_losses_df columns: {list(stop_losses_df.columns)}")
        logger.info(f"rankings_df columns: {list(rankings_df.columns)}")
        
        # Get top 10 stocks
        top_10 = rankings_df.head(10)
        
        # Merge with features for additional info
        # Use only available columns
        available_cols = ['ticker', 'stop_loss_atr', 'stop_loss_pct']
        if 'rsi_14' in stop_losses_df.columns:
            available_cols.insert(1, 'rsi_14')
        
        features_for_report = stop_losses_df[available_cols].drop_duplicates('ticker')
        top_10_with_features = top_10.merge(
            features_for_report,
            on='ticker',
            how='left'
        )
        
        # Get current prices (approximate)
        current_prices = {}
        for ticker in top_10_with_features['ticker']:
            try:
                # Get latest price from raw data (simplified)
                ticker_file = self.raw_path / f"{ticker}.parquet"
                if ticker_file.exists():
                    df = pd.read_parquet(ticker_file)
                    current_prices[ticker] = df['adjClose'].iloc[-1]
                else:
                    current_prices[ticker] = 0.0
            except:
                current_prices[ticker] = 0.0
        
        # Build report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"NEURALTRADER 2.0 - DAILY RANKING REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Top 10 Stock Recommendations")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Adjust header based on available columns
        if 'rsi_14' in available_cols:
            report_lines.append(f"{'Rank':<6} {'Ticker':<8} {'Score':<10} {'RSI':<8} {'ATR Stop':<12} {'Current Price':<12}")
        else:
            report_lines.append(f"{'Rank':<6} {'Ticker':<8} {'Score':<10} {'ATR Stop':<12} {'Current Price':<12}")
        report_lines.append("-" * 80)
        
        for _, row in top_10_with_features.iterrows():
            rank = int(row['rank'])
            ticker = row['ticker']
            score = row['model_score']
            stop_loss = row['stop_loss_atr']
            price = current_prices.get(ticker, 0.0)
            
            if 'rsi_14' in available_cols:
                rsi = row.get('rsi_14', 0.0)
                report_lines.append(f"{rank:<6} {ticker:<8} {score:<10.4f} {rsi:<8.2f} ${stop_loss:<10.2f} ${price:<10.2f}")
            else:
                report_lines.append(f"{rank:<6} {ticker:<8} {score:<10.4f} ${stop_loss:<10.2f} ${price:<10.2f}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("RISK MANAGEMENT NOTES:")
        report_lines.append("- Use 2x ATR stop losses as shown")
        report_lines.append("- Position size: Risk 1-2% per trade")
        report_lines.append("- Maximum portfolio heat: 20% drawdown")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, report: str) -> None:
        """Save daily report to logs."""
        report_path = self.logs_path / "daily_signals.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Daily report saved: {report_path}")
    
    def run(self) -> None:
        """
        Run the complete live ranking pipeline.
        """
        logger.info("Starting Live Ranker pipeline...")
        
        try:
            # Load model
            model = self.load_model()
            
            # Sync latest data
            raw_data = self.sync_latest_data()
            
            # Calculate latest features
            features_df = self.calculate_latest_features(raw_data)
            
            # Apply Z-scoring
            normalized_df = self.apply_z_scoring(features_df)
            
            # Predict rankings
            rankings_df = self.predict_rankings(model, normalized_df)
            
            # Calculate stop losses
            stop_losses_df = self.calculate_stop_losses(rankings_df, features_df)
            
            # Generate and save report
            report = self.generate_daily_report(rankings_df, stop_losses_df)
            
            # Print to console
            print(report)
            
            # Save to file
            self.save_report(report)
            
            logger.info("Live Ranker pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Live Ranker failed: {e}")
            print(f"\n❌ ERROR: {e}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live ranking system for daily trading signals")
    parser.add_argument("--raw-path", default="data/raw", help="Path to raw data")
    parser.add_argument("--models-path", default="models", help="Path to models")
    parser.add_argument("--logs-path", default="logs", help="Path to logs")
    
    args = parser.parse_args()
    
    # Initialize and run live ranker
    ranker = LiveRanker(
        raw_path=args.raw_path,
        models_path=args.models_path,
        logs_path=args.logs_path
    )
    
    ranker.run()

if __name__ == "__main__":
    main()
