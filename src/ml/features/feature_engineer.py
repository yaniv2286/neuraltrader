"""
ML Feature Engineer - Advanced Feature Generation
===============================================

Creates comprehensive features for ML models:
- Price-based features (returns, momentum, volatility)
- Volume-based features (volume profile, money flow)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market regime features
- Cross-sectional features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from ta import add_all_ta_features
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("NeuralTrader.FeatureEngineer")

class FeatureEngineer:
    """
    Advanced feature engineering for ML models.
    
    Features:
    - Price momentum and reversal
    - Volume and money flow
    - Technical indicators
    - Market regime integration
    - Cross-sectional ranking
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Feature scalers
        self.price_scaler = RobustScaler()
        self.volume_scaler = StandardScaler()
        self.scalers_fitted = False
        
        # Feature list for reproducibility
        self.feature_list: List[str] = []
        
    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Load ticker data from Parquet file.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_path / f"{ticker}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {ticker}.parquet")
        
        df = pd.read_parquet(file_path)
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with price features added
        """
        features = df.copy()
        
        # Returns
        features['returns_1d'] = features['adjClose'].pct_change()
        features['returns_5d'] = features['adjClose'].pct_change(5)
        features['returns_10d'] = features['adjClose'].pct_change(10)
        features['returns_20d'] = features['adjClose'].pct_change(20)
        features['returns_60d'] = features['adjClose'].pct_change(60)
        
        # Log returns
        features['log_returns_1d'] = np.log(features['adjClose'] / features['adjClose'].shift(1))
        features['log_returns_5d'] = np.log(features['adjClose'] / features['adjClose'].shift(5))
        features['log_returns_20d'] = np.log(features['adjClose'] / features['adjClose'].shift(20))
        
        # Momentum
        features['momentum_5d'] = features['adjClose'] / features['adjClose'].shift(5) - 1
        features['momentum_10d'] = features['adjClose'] / features['adjClose'].shift(10) - 1
        features['momentum_20d'] = features['adjClose'] / features['adjClose'].shift(20) - 1
        features['momentum_60d'] = features['adjClose'] / features['adjClose'].shift(60) - 1
        
        # Price position
        features['price_position_20d'] = (features['adjClose'] - features['adjClose'].rolling(20).min()) / (features['adjClose'].rolling(20).max() - features['adjClose'].rolling(20).min())
        features['price_position_60d'] = (features['adjClose'] - features['adjClose'].rolling(60).min()) / (features['adjClose'].rolling(60).max() - features['adjClose'].rolling(60).min())
        features['price_position_252d'] = (features['adjClose'] - features['adjClose'].rolling(252).min()) / (features['adjClose'].rolling(252).max() - features['adjClose'].rolling(252).min())
        
        # Moving averages
        features['sma_5'] = features['adjClose'].rolling(5).mean()
        features['sma_10'] = features['adjClose'].rolling(10).mean()
        features['sma_20'] = features['adjClose'].rolling(20).mean()
        features['sma_50'] = features['adjClose'].rolling(50).mean()
        features['sma_200'] = features['adjClose'].rolling(200).mean()
        
        # Moving average ratios
        features['price_sma_5'] = features['adjClose'] / features['sma_5']
        features['price_sma_20'] = features['adjClose'] / features['sma_20']
        features['price_sma_50'] = features['adjClose'] / features['sma_50']
        features['price_sma_200'] = features['adjClose'] / features['sma_200']
        
        # Exponential moving averages
        features['ema_12'] = features['adjClose'].ewm(span=12).mean()
        features['ema_26'] = features['adjClose'].ewm(span=26).mean()
        features['ema_50'] = features['adjClose'].ewm(span=50).mean()
        
        # EMA ratios
        features['price_ema_12'] = features['adjClose'] / features['ema_12']
        features['price_ema_26'] = features['adjClose'] / features['ema_26']
        
        return features
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with volatility features added
        """
        features = df.copy()
        
        # Price volatility
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_10d'] = features['returns_1d'].rolling(10).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        features['volatility_60d'] = features['returns_1d'].rolling(60).std()
        
        # ATR (Average True Range)
        features['tr'] = np.maximum(
            features['high'] - features['low'],
            np.maximum(
                abs(features['high'] - features['adjClose'].shift(1)),
                abs(features['low'] - features['adjClose'].shift(1))
            )
        )
        features['atr_14'] = features['tr'].rolling(14).mean()
        features['atr_20'] = features['tr'].rolling(20).mean()
        features['atr_60'] = features['tr'].rolling(60).mean()
        
        # ATR percentage
        features['atr_pct_14'] = features['atr_14'] / features['adjClose']
        features['atr_pct_20'] = features['atr_pct_14'] / features['adjClose']
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
        features['vol_ratio_10_60'] = features['volatility_10d'] / features['volatility_60d']
        
        # Realized volatility (annualized)
        features['realized_vol_20d'] = features['returns_1d'].rolling(20).std() * np.sqrt(252)
        features['realized_vol_60d'] = features['returns_1d'].rolling(60).std() * np.sqrt(252)
        
        return features
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with volume features added
        """
        features = df.copy()
        
        # Volume moving averages
        features['volume_sma_5'] = features['volume'].rolling(5).mean()
        features['volume_sma_20'] = features['volume'].rolling(20).mean()
        features['volume_sma_50'] = features['volume'].rolling(50).mean()
        
        # Volume ratios
        features['volume_ratio_5'] = features['volume'] / features['volume_sma_5']
        features['volume_ratio_20'] = features['volume'] / features['volume_sma_20']
        features['volume_ratio_50'] = features['volume'] / features['volume_sma_50']
        
        # Dollar volume
        features['dollar_volume'] = features['adjClose'] * features['volume']
        features['dollar_volume_sma_20'] = features['dollar_volume'].rolling(20).mean()
        features['dollar_volume_ratio'] = features['dollar_volume'] / features['dollar_volume_sma_20']
        
        # Volume price trend
        features['vpt'] = (features['volume'] * np.sign(features['returns_1d'])).cumsum()
        features['vpt_sma_20'] = features['vpt'].rolling(20).mean()
        
        # On-balance volume
        features['obv'] = (features['volume'] * np.sign(features['adjClose'].diff())).cumsum()
        features['obv_sma_20'] = features['obv'].rolling(20).mean()
        
        # Volume weighted average price (VWAP)
        features['vwap_20'] = (features['dollar_volume'].rolling(20).sum() / features['volume'].rolling(20).sum())
        features['price_vwap_20'] = features['adjClose'] / features['vwap_20']
        
        # Money flow index
        features['mfi_14'] = self._calculate_mfi(features, 14)
        
        return features
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['adjClose']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using TA library.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            # Use TA library to add all indicators
            features = add_all_ta_features(
                df, 
                open="open", 
                high="high", 
                low="low", 
                close="adjClose", 
                volume="volume",
                fillna=True
            )
            
            # Select most important indicators
            important_indicators = [
                'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
                'momentum_rsi', 'momentum_stoch', 'momentum_stoch_signal',
                'momentum_macd', 'momentum_macd_signal', 'momentum_macd_diff',
                'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw',
                'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_kcw',
                'volume_obv', 'volume_ad', 'volume_cmf'
            ]
            
            # Keep only important indicators that exist
            existing_indicators = [col for col in important_indicators if col in features.columns]
            features = features[existing_indicators + ['adjClose', 'volume']]
            
        except Exception as e:
            logger.warning(f"TA library failed: {e}, using manual indicators")
            features = df.copy()
            
            # Manual RSI
            delta = features['adjClose'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['momentum_rsi'] = 100 - (100 / (1 + rs))
            
            # Manual MACD
            ema_12 = features['adjClose'].ewm(span=12).mean()
            ema_26 = features['adjClose'].ewm(span=26).mean()
            features['momentum_macd'] = ema_12 - ema_26
            features['momentum_macd_signal'] = features['momentum_macd'].ewm(span=9).mean()
            
            # Manual Bollinger Bands
            sma_20 = features['adjClose'].rolling(20).mean()
            std_20 = features['adjClose'].rolling(20).std()
            features['volatility_bbm'] = sma_20
            features['volatility_bbh'] = sma_20 + 2 * std_20
            features['volatility_bbl'] = sma_20 - 2 * std_20
            features['volatility_bbw'] = (features['volatility_bbh'] - features['volatility_bbl']) / features['volatility_bbm']
        
        return features
    
    def calculate_regime_features(self, df: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime-aware features.
        
        Args:
            df: OHLCV DataFrame
            regime_data: Market regime DataFrame
            
        Returns:
            DataFrame with regime features added
        """
        features = df.copy()
        
        # Merge regime data
        features = features.merge(regime_data[['regime']], left_index=True, right_index=True, how='left')
        
        # Regime encoding
        features['regime_bull'] = (features['regime'] == 'BULL').astype(int)
        features['regime_bear'] = (features['regime'] == 'BEAR').astype(int)
        features['regime_caution'] = (features['regime'] == 'CAUTION').astype(int)
        
        # Regime interaction with returns
        features['returns_bull'] = features['returns_1d'] * features['regime_bull']
        features['returns_bear'] = features['returns_1d'] * features['regime_bear']
        
        # Volatility by regime
        features['volatility_bull'] = features['volatility_20d'] * features['regime_bull']
        features['volatility_bear'] = features['volatility_20d'] * features['regime_bear']
        
        return features
    
    def calculate_cross_sectional_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate cross-sectional features (ranking across all tickers).
        
        Args:
            data_dict: Dictionary of ticker to DataFrame
            
        Returns:
            Dictionary with cross-sectional features added
        """
        # Create combined DataFrame for cross-sectional calculations
        all_data = []
        for ticker, df in data_dict.items():
            df_copy = df.copy()
            df_copy['ticker'] = ticker
            all_data.append(df_copy[['adjClose', 'volume', 'ticker']])
        
        combined_df = pd.concat(all_data)
        
        # Calculate cross-sectional rankings by date
        for date in combined_df.index.unique():
            date_data = combined_df.loc[date]
            
            # Market cap ranking (using dollar volume as proxy)
            date_data['market_cap_rank'] = date_data['adjClose'] * date_data['volume']
            date_data['market_cap_rank'] = date_data['market_cap_rank'].rank(pct=True)
            
            # Volume ranking
            date_data['volume_rank'] = date_data['volume'].rank(pct=True)
            
            # Update combined DataFrame
            combined_df.loc[date, ['market_cap_rank', 'volume_rank']] = date_data[['market_cap_rank', 'volume_rank']]
        
        # Add rankings back to individual DataFrames
        enhanced_data = {}
        for ticker, df in data_dict.items():
            df_copy = df.copy()
            ticker_data = combined_df[combined_df['ticker'] == ticker]
            
            df_copy['market_cap_rank'] = ticker_data['market_cap_rank']
            df_copy['volume_rank'] = ticker_data['volume_rank']
            
            enhanced_data[ticker] = df_copy
        
        return enhanced_data
    
    def create_features(self, ticker: str, regime_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create all features for a ticker.
        
        Args:
            ticker: Ticker symbol
            regime_data: Market regime DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Creating features for {ticker}")
        
        # Load data
        df = self.load_ticker_data(ticker)
        
        # Calculate feature groups
        df = self.calculate_price_features(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_volume_features(df)
        df = self.calculate_technical_indicators(df)
        
        # Add regime features if available
        if regime_data is not None:
            df = self.calculate_regime_features(df, regime_data)
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill', limit=5)
        df = df.dropna()
        
        # Store feature list
        if not self.feature_list:
            self.feature_list = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'adjClose', 'volume']]
        
        logger.info(f"Created {len(self.feature_list)} features for {ticker}")
        
        return df
    
    def create_features_batch(self, tickers: List[str], regime_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Create features for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            regime_data: Market regime DataFrame
            
        Returns:
            Dictionary of ticker to feature DataFrame
        """
        logger.info(f"Creating features for {len(tickers)} tickers")
        
        feature_data = {}
        
        for ticker in tickers:
            try:
                features = self.create_features(ticker, regime_data)
                feature_data[ticker] = features
            except Exception as e:
                logger.error(f"Failed to create features for {ticker}: {e}")
        
        # Add cross-sectional features
        feature_data = self.calculate_cross_sectional_features(feature_data)
        
        logger.info(f"Successfully created features for {len(feature_data)} tickers")
        
        return feature_data
    
    def save_features(self, feature_data: Dict[str, pd.DataFrame], suffix: str = "") -> None:
        """
        Save features to Parquet files.
        
        Args:
            feature_data: Dictionary of ticker to feature DataFrame
            suffix: Optional suffix for filenames
        """
        for ticker, df in feature_data.items():
            filename = f"{ticker}_features{suffix}.parquet"
            filepath = self.processed_path / filename
            
            df.to_parquet(filepath, compression='snappy')
        
        # Save feature list
        feature_list_path = self.processed_path / f"feature_list{suffix}.json"
        import json
        with open(feature_list_path, 'w') as f:
            json.dump(self.feature_list, f, indent=2)
        
        logger.info(f"Saved features for {len(feature_data)} tickers")

# Usage Example
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load regime data
    regime_data = pd.read_csv("data/processed/market_regime.csv", index_col=0, parse_dates=True)
    
    # Create features for a single ticker
    features = engineer.create_features("AAPL", regime_data)
    print(f"Features for AAPL: {features.shape}")
    print(f"Feature columns: {len(engineer.feature_list)}")
    
    # Create features for multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    feature_data = engineer.create_features_batch(tickers, regime_data)
    
    # Save features
    engineer.save_features(feature_data)
    print("Features saved successfully")
