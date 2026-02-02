"""
SPY Benchmark Feature Loader - Deterministic Join Contract
==========================================================

Implements the mandatory SPY benchmark contract:
1. Benchmark Ticker: SPY (Adjusted Prices, Daily)
2. Required Market Features:
   - mkt_atr: ATR(14)
   - mkt_rsi: RSI(14)
   - mkt_macd: MACD(12/26/9) histogram only
   - mkt_momentum: 20-day log return (frozen method)
3. Join Rule: Left join SPY features into every ticker by Date
4. Fill Rule: ffill benchmark columns only, max 5 trading days
5. Hard Fail: If SPY missing or join fails -> ABORT RUN

This module is CRITICAL for fixing the "0 signals" bug caused by
feature mismatch between training and inference.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import hashlib

from src.core.data_store import get_data_store, DataStoreError


class SPYBenchmarkError(Exception):
    """Raised when SPY benchmark contract is violated."""
    pass


@dataclass
class SPYBenchmarkConfig:
    """Configuration for SPY benchmark features."""
    benchmark_ticker: str = 'SPY'
    atr_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_period: int = 20
    max_ffill_days: int = 5
    
    def to_dict(self) -> dict:
        return {
            'benchmark_ticker': self.benchmark_ticker,
            'atr_period': self.atr_period,
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'momentum_period': self.momentum_period,
            'max_ffill_days': self.max_ffill_days
        }


# Required market feature columns - FROZEN
REQUIRED_MKT_FEATURES = ['mkt_atr', 'mkt_rsi', 'mkt_macd', 'mkt_momentum']


class SPYBenchmarkLoader:
    """
    Loads SPY data and generates market benchmark features.
    
    Contract:
    - All features are deterministic and reproducible
    - Feature definitions are FROZEN (do not modify)
    - Hard fail on any error (no silent skipping)
    """
    
    def __init__(self, config: Optional[SPYBenchmarkConfig] = None):
        self.config = config or SPYBenchmarkConfig()
        self.data_store = get_data_store()
        self._spy_features: Optional[pd.DataFrame] = None
        self._feature_hash: Optional[str] = None
        
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd_histogram(self, close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        """Calculate MACD Histogram (frozen as mkt_macd)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return histogram
    
    def _calculate_momentum(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate momentum as log return (FROZEN method)."""
        return np.log(close / close.shift(period))
    
    def load_spy_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load SPY data and generate all required market features.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with Date index and mkt_* columns
            
        Raises:
            SPYBenchmarkError: If SPY data cannot be loaded or features fail
        """
        try:
            spy_df = self.data_store.get_ticker_data(
                self.config.benchmark_ticker,
                start_date,
                end_date
            )
        except DataStoreError as e:
            raise SPYBenchmarkError(
                f"HARD FAIL: Cannot load SPY benchmark data. "
                f"SPY is REQUIRED for all backtests. Error: {e}"
            )
        
        if spy_df is None or spy_df.empty:
            raise SPYBenchmarkError(
                "HARD FAIL: SPY data is empty. Cannot proceed without benchmark."
            )
        
        # Generate features
        features = pd.DataFrame(index=spy_df.index)
        
        # mkt_atr: ATR(14)
        features['mkt_atr'] = self._calculate_atr(
            spy_df['high'], spy_df['low'], spy_df['close'],
            self.config.atr_period
        )
        
        # mkt_rsi: RSI(14)
        features['mkt_rsi'] = self._calculate_rsi(
            spy_df['close'],
            self.config.rsi_period
        )
        
        # mkt_macd: MACD histogram only (FROZEN)
        features['mkt_macd'] = self._calculate_macd_histogram(
            spy_df['close'],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        
        # mkt_momentum: 20-day log return (FROZEN method)
        features['mkt_momentum'] = self._calculate_momentum(
            spy_df['close'],
            self.config.momentum_period
        )
        
        # Validate all required features exist
        missing = [f for f in REQUIRED_MKT_FEATURES if f not in features.columns]
        if missing:
            raise SPYBenchmarkError(
                f"HARD FAIL: Missing required market features: {missing}"
            )
        
        # Check for excessive NaN (more than 10% after warmup)
        warmup_period = max(
            self.config.atr_period,
            self.config.rsi_period,
            self.config.macd_slow + self.config.macd_signal,
            self.config.momentum_period
        )
        
        features_after_warmup = features.iloc[warmup_period:]
        nan_pct = features_after_warmup.isna().sum() / len(features_after_warmup) * 100
        
        for col, pct in nan_pct.items():
            if pct > 10:
                raise SPYBenchmarkError(
                    f"HARD FAIL: Feature {col} has {pct:.1f}% NaN after warmup. "
                    f"Data quality issue detected."
                )
        
        # Cache features
        self._spy_features = features
        self._feature_hash = self._compute_feature_hash(features)
        
        return features
    
    def _compute_feature_hash(self, features: pd.DataFrame) -> str:
        """Compute deterministic hash of feature columns."""
        col_str = ','.join(sorted(features.columns.tolist()))
        return hashlib.md5(col_str.encode()).hexdigest()[:16]
    
    def get_feature_hash(self) -> str:
        """Get hash of loaded features for manifest validation."""
        if self._feature_hash is None:
            raise SPYBenchmarkError("Features not loaded. Call load_spy_features first.")
        return self._feature_hash
    
    def join_to_ticker(
        self,
        ticker_df: pd.DataFrame,
        ticker_symbol: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Left join SPY benchmark features to a ticker DataFrame.
        
        Args:
            ticker_df: DataFrame with DatetimeIndex
            ticker_symbol: Ticker symbol (for error messages)
            
        Returns:
            Tuple of (joined DataFrame, list of warnings)
            
        Raises:
            SPYBenchmarkError: If join fails or required features missing
        """
        if self._spy_features is None:
            raise SPYBenchmarkError(
                "HARD FAIL: SPY features not loaded. Call load_spy_features first."
            )
        
        warnings = []
        
        # Perform left join
        result = ticker_df.copy()
        
        for col in REQUIRED_MKT_FEATURES:
            if col in self._spy_features.columns:
                result[col] = self._spy_features[col].reindex(result.index)
        
        # Forward fill benchmark columns only (max 5 days)
        for col in REQUIRED_MKT_FEATURES:
            if col in result.columns:
                result[col] = result[col].ffill(limit=self.config.max_ffill_days)
        
        # Check for missing features after join
        missing_after_join = []
        for col in REQUIRED_MKT_FEATURES:
            if col not in result.columns:
                missing_after_join.append(col)
            elif result[col].isna().all():
                missing_after_join.append(f"{col} (all NaN)")
        
        if missing_after_join:
            raise SPYBenchmarkError(
                f"HARD FAIL: Ticker {ticker_symbol} missing market features after join: "
                f"{missing_after_join}. Cannot proceed."
            )
        
        # Warn if significant NaN remains
        nan_counts = result[REQUIRED_MKT_FEATURES].isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                pct = count / len(result) * 100
                if pct > 5:
                    warnings.append(
                        f"{ticker_symbol}: {col} has {count} NaN ({pct:.1f}%) after ffill"
                    )
        
        return result, warnings


# Global instance
_spy_loader: Optional[SPYBenchmarkLoader] = None


def get_spy_benchmark_loader() -> SPYBenchmarkLoader:
    """Get global SPY benchmark loader instance."""
    global _spy_loader
    if _spy_loader is None:
        _spy_loader = SPYBenchmarkLoader()
    return _spy_loader


def reset_spy_benchmark_loader():
    """Reset global instance (for testing)."""
    global _spy_loader
    _spy_loader = None
