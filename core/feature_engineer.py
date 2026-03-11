"""
Feature Engineer - Consolidated feature engineering for NeuralTrader Core
Advanced feature creation for stock prediction with CPU optimization
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .indicators import TechnicalIndicators


class FeatureEngineer:
    """Advanced feature engineering for stock prediction - CPU optimized"""
    
    def __init__(self, use_advanced_features: bool = True, verbose: bool = False):
        self.feature_count = 0
        self.use_advanced_features = use_advanced_features
        self.verbose = verbose
        self.ti = TechnicalIndicators()
    
    def create_features(self, data: pd.DataFrame, target_type: str = 'log_returns') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create advanced features for model training
        
        Args:
            data: OHLCV data
            target_type: Type of target variable ('log_returns', 'returns', 'direction', 'multi_horizon')
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # 🦅 TYPE ALIGNMENT - Standardize date format
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
            
            # 🦅 AGGRESSIVE DATETIME REMOVAL - Remove ALL datetime columns before any processing
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                elif df[col].dtype == 'object':
                    # Check if it's a date string
                    try:
                        pd.to_datetime(df[col].iloc[0])
                        datetime_cols.append(col)
                    except:
                        pass
            
            if datetime_cols:
                df = df.drop(columns=datetime_cols)
            
            # 🦅 NUMERIC ONLY FILTER - Ensure only numeric data is processed
            numeric_cols = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
                    numeric_cols.append(col)
            
            df = df[numeric_cols]
            
            # Standardize column names to lowercase
            required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
            required_cols_upper = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if all(col in df.columns for col in required_cols_upper):
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
            elif not all(col in df.columns for col in required_cols_lower):
                raise ValueError(f"Missing required columns. Need either {required_cols_lower} or {required_cols_upper}")
            
            # Add basic technical indicators
            df = self._apply_basic_indicators(df)
            
            # Add advanced features if enabled
            if self.use_advanced_features:
                df = self._add_advanced_features(df)
            
            # Add market regime features
            df = self._add_regime_features(df)
            
            # Create target variable BEFORE dropping NaNs
            target = self._create_target(df, target_type)
            
            # Combine features and target, then drop NaNs together
            combined_df = df.copy()
            combined_df['target'] = target
            
            # Drop rows where any feature or target is NaN
            combined_df = combined_df.dropna()
            
            # Remove non-feature columns (comprehensive exclusion)
            exclude_cols = [
                'target', 'date', 'Date', 'datetime', 'Datetime', 'ticker',
                'timestamp', 'Timestamp', 'time', 'Time'
            ]
            # 🦅 COMPREHENSIVE DATETIME EXCLUSION - Prevent all datetime comparison errors
            for col in combined_df.columns:
                # Exclude by name patterns
                if any(pattern in col.lower() for pattern in ['date', 'time', 'timestamp']):
                    exclude_cols.append(col)
                # Exclude by dtype
                elif pd.api.types.is_datetime64_any_dtype(combined_df[col]):
                    exclude_cols.append(col)
                # Exclude object columns that might contain dates
                elif combined_df[col].dtype == 'object':
                    try:
                        pd.to_datetime(combined_df[col].iloc[0])
                        exclude_cols.append(col)
                    except:
                        pass  # Not a date column
            
            # Remove duplicates while preserving order
            exclude_cols = list(dict.fromkeys(exclude_cols))
            
            features = combined_df.drop([c for c in exclude_cols if c in combined_df.columns], axis=1)
            target = combined_df['target']
            
            # 🦅 NEURAL RESTORATION - Clip ONLY normalized features, NOT raw prices
            # Define raw price columns that should NEVER be clipped
            raw_price_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume',
                'divCash', 'splitFactor'
            ]
            
            # Get normalized feature columns (everything except raw prices)
            normalized_cols = [col for col in features.columns if col not in raw_price_cols]
            
            # Clip only normalized features to prevent extreme values
            if normalized_cols:
                features[normalized_cols] = features[normalized_cols].clip(lower=-10, upper=10)
            
            # ENSEMBLE VISION RESTORATION - Ensure all required features are present
            # Map expected column names to actual data based on model metadata (RENAMED VERSIONS)
            required_features = [
                "Price", "high", "low", "open", "volume",
                "Adj Close", "Adj High", "Adj Low", "Adj Open", "Adj Volume",
                "Dividend", "Split Factor", "sma_20", "ema_20", "rsi",
                "obv", "vwap", "rolling_volatility", "atr_14", "momentum", "roc",
                "body_size", "upper_wick", "lower_wick", "drawdown_pct",
                "close_lag_1", "volume_lag_1", "rsi_lag_1", "momentum_5",
                "momentum_10", "momentum_20", "roc_5", "roc_10", "roc_20",
                "volume_ratio", "volume_log", "macd", "macd_signal", "macd_histogram",
                "obv_ratio", "volatility_20", "volatility_50", "atr_ratio",
                "price_efficiency", "bb_upper", "bb_lower", "bb_middle",
                "bb_width", "bb_position", "bb_overbought", "bb_oversold",
                "vol_regime", "high_vol", "low_vol", "trend_regime",
                "strong_uptrend", "strong_downtrend", "sma10_sma50_cross",
                "sma50_sma200_cross", "macd_bullish", "macd_bearish",
                "price_sma10_ratio", "price_sma50_ratio", "price_sma200_ratio",
                "rs_vs_spy", "roc_63", "high_52w_prox", "volume_breakout"
            ]
            
            # Add any missing required features with zeros
            for feat in required_features:
                if feat not in features.columns:
                    if feat == 'Split Factor':
                        features[feat] = 1.0
                    elif feat in ['Dividend']:
                        features[feat] = 0.0
                    elif feat in ['Adj Close', 'Adj High', 'Adj Low', 'Adj Open', 'Adj Volume']:
                        # Map to available lowercase columns if possible
                        mapping = {
                            'Adj Close': 'adjClose' if 'adjClose' in features.columns else 'close',
                            'Adj High': 'adjHigh' if 'adjHigh' in features.columns else 'high',
                            'Adj Low': 'adjLow' if 'adjLow' in features.columns else 'low',
                            'Adj Open': 'adjOpen' if 'adjOpen' in features.columns else 'open',
                            'Adj Volume': 'adjVolume' if 'adjVolume' in features.columns else 'volume'
                        }
                        if feat in mapping and mapping[feat] in features.columns:
                            features[feat] = features[mapping[feat]]
                        else:
                            features[feat] = 0.0
                    else:
                        features[feat] = 0.0
            
            # 🦅 FEATURE ORDER ALIGNMENT - Ensure exact order as model expects
            features = features[required_features]
            
            # 🦅 NOMENCLATURE REPAIR - Strict renaming map to match model expectations
            final_rename_map = {
                'adjClose': 'Adj Close',
                'adjHigh': 'Adj High', 
                'adjLow': 'Adj Low',
                'adjOpen': 'Adj Open',
                'adjVolume': 'Adj Volume',
                'divCash': 'Dividend',
                'splitFactor': 'Split Factor',
                'close': 'Price'
            }
            
            # Apply renaming only to columns that exist
            for old_name, new_name in final_rename_map.items():
                if old_name in features.columns:
                    features = features.rename(columns={old_name: new_name})
            
            # 🦅 DATAFRAME INTEGRITY - Ensure we return a DataFrame, never numpy array
            if not isinstance(features, pd.DataFrame):
                raise ValueError("Features must be a pandas DataFrame")
            
            self.feature_count = len(features.columns)
            
            return features, target
            
        except Exception as e:
            if self.verbose:
                print(f"Error in feature creation: {e}")
            raise
    
    def _apply_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic technical indicators with feature normalization"""
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # 🦅 FEATURE GAP FIX - Add missing features that model expects
        # If adjusted columns don't exist, use regular prices as fallback
        if 'adjClose' not in df.columns:
            df['adjClose'] = close
        if 'adjHigh' not in df.columns:
            df['adjHigh'] = high
        if 'adjLow' not in df.columns:
            df['adjLow'] = low
        if 'adjOpen' not in df.columns:
            df['adjOpen'] = open_price
        if 'adjVolume' not in df.columns:
            df['adjVolume'] = volume
        
        # Add dividend and split factor as zeros (common for most stocks)
        if 'divCash' not in df.columns:
            df['divCash'] = 0
        if 'splitFactor' not in df.columns:
            df['splitFactor'] = 1.0
        
        # 🦅 FEATURE NORMALIZATION - Replace raw prices with ratios and returns
        # Price returns instead of raw prices
        df['close_5d_return'] = close.pct_change(5)
        df['close_10d_return'] = close.pct_change(10)
        df['close_20d_return'] = close.pct_change(20)
        
        # Moving averages as ratios (normalized)
        df['sma_20'] = self.ti.sma(close, 20)
        df['sma_20_ratio'] = close.div(df['sma_20']).replace([np.inf, -np.inf], 0).fillna(0) - 1
        
        df['ema_20'] = self.ti.ema(close, 20)
        df['ema_20_ratio'] = close.div(df['ema_20']).replace([np.inf, -np.inf], 0).fillna(0) - 1
        
        # Volume normalization with nuclear math sanitation
        volume_ma = volume.rolling(20).mean()
        df['volume_ratio'] = volume.div(volume_ma.replace(0, np.nan)).fillna(0) - 1
        # 🦅 MATH SANITIZATION - Safe log calculation
        df['volume_log'] = np.log1p(volume.clip(lower=0))
        
        # RSI (already normalized 0-100)
        df['rsi'] = self.ti.rsi(close, 14)
        
        # MACD (normalized)
        macd, signal, histogram = self.ti.macd(close)
        # 🦅 NUCLEAR MATH SANITIZATION - Handle zero price
        df['macd'] = macd.div(close.replace(0, np.nan)).fillna(0)
        df['macd_signal'] = signal.div(close.replace(0, np.nan)).fillna(0)
        df['macd_histogram'] = histogram.div(close.replace(0, np.nan)).fillna(0)
        
        # OBV (On-Balance Volume)
        df['obv'] = self._calculate_obv(df)
        obv_ma = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'].div(obv_ma.replace(0, np.nan)).fillna(0)
        
        # VWAP
        df['vwap'] = self._calculate_vwap(df)
        
        # Rolling volatility
        df['rolling_volatility'] = close.pct_change().rolling(20).std()
        
        # ATR
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Basic momentum
        df['momentum'] = close - close.shift(10)
        df['roc'] = close.pct_change(10)
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        close = df['close']
        volume = df['volume']
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=df.index).cumsum()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume_cumsum = df['volume'].cumsum()
        # 🦅 NUCLEAR MATH SANITIZATION - Handle zero volume
        vwap = ((typical_price * df['volume']).cumsum()).div(volume_cumsum.replace(0, np.nan)).fillna(typical_price)
        return pd.Series(vwap, index=df.index)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features with normalization for maximum performance"""
        close = df['close']
        
        # 🦅 ADVANCED FEATURE NORMALIZATION
        # Price action features (already normalized ratios)
        df['body_size'] = (abs(close - df['open'])).div((df['high'] - df['low']).replace(0, np.nan)).fillna(0)
        df['upper_wick'] = (df['high'] - np.maximum(df['open'], close)).div(close.replace(0, np.nan)).fillna(0)  # Normalized by price
        df['lower_wick'] = (np.minimum(df['open'], close) - df['low']).div(close.replace(0, np.nan)).fillna(0)  # Normalized by price
        
        # Drawdown features (already percentage)
        df['drawdown_pct'] = (close - close.expanding().max()).div(close.expanding().max().replace(0, np.nan)).fillna(0)
        
        # Lag features (returns instead of raw values)
        df['close_lag_1'] = close.pct_change(1)
        df['volume_lag_1'] = df['volume'].pct_change(1)
        df['rsi_lag_1'] = df['rsi'].diff()  # RSI change
        
        # Momentum features (normalized returns)
        df['momentum_5'] = close.pct_change(5)
        df['momentum_10'] = close.pct_change(10)
        df['momentum_20'] = close.pct_change(20)
        
        # Rate of change (already returns)
        df['roc_5'] = close.pct_change(5)
        df['roc_10'] = close.pct_change(10)
        df['roc_20'] = close.pct_change(20)
        df['roc_63'] = close.pct_change(63)  # 63-day rate of change
        
        # Price ratio features (normalized)
        df['price_sma10_ratio'] = close.div(close.rolling(10).mean().replace(0, np.nan)).fillna(0) - 1
        df['price_sma50_ratio'] = close.div(close.rolling(50).mean().replace(0, np.nan)).fillna(0) - 1
        df['price_sma200_ratio'] = close.div(close.rolling(200).mean().replace(0, np.nan)).fillna(0) - 1
        
        # Moving average crossover features
        sma10 = close.rolling(10).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        df['sma10_sma50_cross'] = (sma10 > sma50).astype(int)
        df['sma50_sma200_cross'] = (sma50 > sma200).astype(int)
        
        # MACD features (already normalized in basic indicators)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)
        
        # Bollinger Bands (normalized) - 🦅 ERROR HANDLING
        try:
            bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(close, 20, 2)
            if bb_upper is not None and bb_middle is not None and bb_lower is not None:
                df['bb_upper'] = (bb_upper - close).div(close.replace(0, np.nan)).fillna(0)  # Normalized distance
                df['bb_lower'] = (close - bb_lower).div(close.replace(0, np.nan)).fillna(0)  # Normalized distance
                df['bb_middle'] = (bb_middle - close).div(close.replace(0, np.nan)).fillna(0)  # Normalized distance
                df['bb_width'] = (bb_upper - bb_lower).div(close.replace(0, np.nan)).fillna(0)  # Normalized width
                df['bb_position'] = (close - bb_lower).div((bb_upper - bb_lower).replace(0, np.nan)).fillna(0)  # Position within bands
                df['bb_overbought'] = (df['bb_position'] > 0.8).astype(int)
                df['bb_oversold'] = (df['bb_position'] < 0.2).astype(int)
            else:
                # Initialize with zeros if Bollinger Bands calculation fails
                df['bb_upper'] = 0
                df['bb_lower'] = 0
                df['bb_middle'] = 0
                df['bb_width'] = 0
                df['bb_position'] = 0
                df['bb_overbought'] = 0
                df['bb_oversold'] = 0
        except Exception as e:
            # Initialize with zeros if Bollinger Bands calculation fails
            df['bb_upper'] = 0
            df['bb_lower'] = 0
            df['bb_middle'] = 0
            df['bb_width'] = 0
            df['bb_position'] = 0
            df['bb_overbought'] = 0
            df['bb_oversold'] = 0
        
        # Volatility regime features - 🦅 ERROR HANDLING
        try:
            vol_mean = df['rolling_volatility'].rolling(252).mean()
            df['vol_regime'] = (df['rolling_volatility'] > vol_mean).astype(int)
            df['high_vol'] = (df['rolling_volatility'] > df['rolling_volatility'].quantile(0.75)).astype(int)
            df['low_vol'] = (df['rolling_volatility'] < df['rolling_volatility'].quantile(0.25)).astype(int)
        except Exception as e:
            df['vol_regime'] = 0
            df['high_vol'] = 0
            df['low_vol'] = 0
        
        # Trend regime features
        df['trend_regime'] = (close > close.rolling(50).mean()).astype(int)
        df['strong_uptrend'] = (close > close.rolling(200).mean()).astype(int)
        df['strong_downtrend'] = (close < close.rolling(200).mean()).astype(int)
        
        # Price efficiency (how much price moves vs. noise)
        df['price_efficiency'] = abs(close.pct_change()).div(df['rolling_volatility'].replace(0, np.nan)).fillna(0)
        
        # Relative strength vs SPY (requires SPY data - placeholder for now)
        # This will be calculated at signal generation time when SPY data is available
        df['rs_vs_spy'] = 0.0  # Placeholder - will be filled during signal generation
        
        # 52-week high proximity
        high_52w = df['high'].rolling(252).max()  # 252 trading days ≈ 1 year
        df['high_52w_prox'] = close.div(high_52w.replace(0, np.nan)).fillna(0)
        
        # Volume breakout (volume vs 50-day average)
        volume_50d_avg = df['volume'].rolling(50).mean()
        df['volume_breakout'] = df['volume'].div(volume_50d_avg.replace(0, np.nan)).fillna(1.0)
        
        return df
    
    def _add_extra_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extra momentum indicators"""
        close = df['close']
        
        # Multiple timeframe momentum
        df['momentum_5'] = close - close.shift(5)
        df['momentum_10'] = close - close.shift(10)
        df['momentum_20'] = close - close.shift(20)
        
        # Rate of change
        df['roc_5'] = close.pct_change(5)
        df['roc_10'] = close.pct_change(10)
        df['roc_20'] = close.pct_change(20)
        
        # Price relative to moving averages
        sma_10 = close.rolling(10).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        df['price_sma10_ratio'] = close / sma_10
        df['price_sma50_ratio'] = close / sma_50
        df['price_sma200_ratio'] = close / sma_200
        
        # Moving average crossovers
        df['sma10_sma50_cross'] = (sma_10 > sma_50).astype(int)
        df['sma50_sma200_cross'] = (sma_50 > sma_200).astype(int)
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        close = df['close']
        
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_middle'] = sma_20
        
        # Bollinger Band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band signals
        df['bb_overbought'] = (close > df['bb_upper']).astype(int)
        df['bb_oversold'] = (close < df['bb_lower']).astype(int)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        # Volatility regime
        returns = df['close'].pct_change()
        volatility_20 = returns.rolling(20).std()
        volatility_50 = returns.rolling(50).std()
        
        df['vol_regime'] = volatility_20 / volatility_50
        df['high_vol'] = (df['vol_regime'] > 1.5).astype(int)
        df['low_vol'] = (df['vol_regime'] < 0.7).astype(int)
        
        # Trend regime
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        df['trend_regime'] = (sma_20 > sma_50).astype(int)
        df['strong_uptrend'] = (df['close'] > sma_20 * 1.05).astype(int)
        df['strong_downtrend'] = (df['close'] < sma_20 * 0.95).astype(int)
        
        # Market efficiency
        df['price_efficiency'] = df['close'] / sma_20
        
        return df
    
    def _create_target(self, df: pd.DataFrame, target_type: str = 'log_returns') -> pd.Series:
        """Create advanced target variables"""
        if target_type == 'multi_horizon':
            # Multi-horizon direction prediction
            target_1d = np.sign(df['close'].shift(-1) / df['close'] - 1)
            target_5d = np.sign(df['close'].shift(-5) / df['close'] - 1)
            target_10d = np.sign(df['close'].shift(-10) / df['close'] - 1)
            
            # Weighted combination (more weight to near-term)
            target = (0.5 * target_1d + 0.3 * target_5d + 0.2 * target_10d)
            
        elif target_type == 'volatility_adjusted':
            # Volatility-adjusted returns
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            target = (returns.shift(-1) / volatility).fillna(0)
            
        elif target_type == 'regime_aware':
            # Regime-aware targets
            high_vol_mask = df['high_vol'] == 1
            low_vol_mask = df['low_vol'] == 1
            
            # In high volatility, predict smaller moves
            base_target = np.sign(df['close'].shift(-1) / df['close'] - 1)
            target = base_target.copy()
            target[high_vol_mask] *= 0.5  # Reduce target magnitude in high vol
            target[low_vol_mask] *= 1.5   # Increase target magnitude in low vol
            
        elif target_type == 'log_returns':
            # Log returns of next day
            # 🦅 SAFE-GUARD PATTERN - Prevent log-zero crashes
            returns = np.log1p(df['close'].pct_change().shift(-1)).fillna(0)
            target = returns  # Don't drop NaNs here - handled in main method
        elif target_type == 'returns':
            # Simple returns
            returns = df['close'].pct_change().shift(-1)
            target = returns  # Don't drop NaNs here - handled in main method
        elif target_type == 'direction':
            # Direction (up/down)
            returns = df['close'].pct_change().shift(-1)
            direction = (returns > 0).astype(int)
            target = direction  # Don't drop NaNs here - handled in main method
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return target
    
    def get_feature_count(self) -> int:
        """Get number of features created"""
        return self.feature_count
    
    def get_feature_names(self, data: pd.DataFrame) -> list:
        """Get list of feature names"""
        X, _ = self.create_features(data)
        if X is not None:
            return X.columns.tolist()
        return []
