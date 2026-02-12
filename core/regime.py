"""
Regime Filter - The Shield
=========================

Quad-Core Market Regime Detection System
Implements the "Golden Rule": If the Shield (Regime) is RED, the Fund is CASH.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Quad-Core Regime Filter (The Shield)
    
    State 0 (GREEN): SPY > 200 SMA & Weekly > 20 EMA & Vol < 20. (Full Deployment).
    State 1 (YELLOW): Daily Trend UP, but internal breadth/vol is weak. (Cautious Deployment).
    State 2 (RED): Daily Trend DOWN (SPY < 200 SMA). Hard Stop. All signals blocked.
    """
    
    def __init__(self):
        """Initialize Regime Filter with default parameters"""
        self.sma_period = 200  # 200-day SMA for primary trend
        self.weekly_ema_period = 20  # Weekly 20 EMA for secondary trend
        self.volatility_threshold = 20  # Volatility threshold
        self.lookback_days = 60  # Lookback for volatility calculation
        
        logger.info("🛡️ Regime Filter (The Shield) initialized")
        logger.info(f"   SMA Period: {self.sma_period} days")
        logger.info(f"   Weekly EMA Period: {self.weekly_ema_period} weeks")
        logger.info(f"   Volatility Threshold: {self.volatility_threshold}")
    
    def calculate_regime_state(self, spy_data: pd.DataFrame) -> pd.Series:
        """
        Calculate regime state for each day using Quad-Core logic
        
        Args:
            spy_data: DataFrame with SPY OHLCV data, indexed by date
            
        Returns:
            Series with regime state (0=GREEN, 1=YELLOW, 2=RED) for each date
        """
        try:
            if len(spy_data) < self.sma_period:
                logger.warning(f"Insufficient data for regime calculation: {len(spy_data)} < {self.sma_period}")
                return pd.Series(1, index=spy_data.index)  # Default to YELLOW
            
            # Calculate primary indicators
            spy_data = spy_data.copy()
            
            # 200-day SMA (Daily Trend)
            spy_data['sma_200'] = spy_data['Close'].rolling(window=self.sma_period).mean()
            
            # Weekly data (resample to weekly)
            weekly_data = spy_data['Close'].resample('W').last()
            weekly_ema = weekly_data.ewm(span=self.weekly_ema_period).mean()
            
            # Map weekly EMA back to daily dates
            spy_data['weekly_ema'] = weekly_ema.reindex(spy_data.index, method='ffill')
            
            # Volatility calculation (20-day rolling standard deviation of returns)
            spy_data['returns'] = spy_data['Close'].pct_change()
            spy_data['volatility'] = spy_data['returns'].rolling(window=20).std() * np.sqrt(252) * 100
            
            # Initialize regime series
            regime = pd.Series(index=spy_data.index, dtype=int)
            
            # Apply Quad-Core logic
            for i in range(len(spy_data)):
                current_date = spy_data.index[i]
                current_price = spy_data['Close'].iloc[i]
                sma_200 = spy_data['sma_200'].iloc[i]
                weekly_ema_val = spy_data['weekly_ema'].iloc[i]
                volatility = spy_data['volatility'].iloc[i]
                
                # Skip if indicators are not available
                if pd.isna(sma_200) or pd.isna(weekly_ema_val) or pd.isna(volatility):
                    regime.iloc[i] = 1  # Default to YELLOW
                    continue
                
                # 🦅 QUAD-CORE REGIME LOGIC
                # State 2 (RED): Daily Trend DOWN - Hard Stop
                if current_price < sma_200:
                    regime.iloc[i] = 2
                # State 0 (GREEN): All conditions met - Full Deployment
                elif (current_price > sma_200 and 
                      current_price > weekly_ema_val and 
                      volatility < self.volatility_threshold):
                    regime.iloc[i] = 0
                # State 1 (YELLOW): Everything else - Cautious Deployment
                else:
                    regime.iloc[i] = 1
            
            logger.info(f"🛡️ Regime calculation complete: {len(regime)} days")
            self._log_regime_summary(regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"❌ Regime calculation failed: {e}")
            return pd.Series(1, index=spy_data.index)  # Default to YELLOW
    
    def _log_regime_summary(self, regime: pd.Series):
        """Log summary statistics of regime states"""
        try:
            total_days = len(regime)
            green_days = (regime == 0).sum()
            yellow_days = (regime == 1).sum()
            red_days = (regime == 2).sum()
            
            green_pct = (green_days / total_days) * 100
            yellow_pct = (yellow_days / total_days) * 100
            red_pct = (red_days / total_days) * 100
            
            logger.info("🛡️ REGIME SUMMARY:")
            logger.info(f"   GREEN (Aggressive): {green_days} days ({green_pct:.1f}%)")
            logger.info(f"   YELLOW (Caution): {yellow_days} days ({yellow_pct:.1f}%)")
            logger.info(f"   RED (Defensive): {red_days} days ({red_pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to log regime summary: {e}")
    
    def get_current_regime(self, spy_data: pd.DataFrame) -> int:
        """
        Get current regime state for latest data
        
        Args:
            spy_data: DataFrame with SPY OHLCV data
            
        Returns:
            Current regime state (0=GREEN, 1=YELLOW, 2=RED)
        """
        try:
            if len(spy_data) < self.sma_period:
                return 1  # Default to YELLOW
            
            regime_series = self.calculate_regime_state(spy_data)
            return int(regime_series.iloc[-1])
            
        except Exception as e:
            logger.error(f"Failed to get current regime: {e}")
            return 1  # Default to YELLOW
    
    def validate_regime_data(self, regime_series: pd.Series) -> Dict[str, bool]:
        """
        Validate regime series for consistency and correctness
        
        Args:
            regime_series: Series with regime states
            
        Returns:
            Dict with validation results
        """
        try:
            validation_results = {
                'has_data': len(regime_series) > 0,
                'valid_states': regime_series.isin([0, 1, 2]).all(),
                'no_gaps': not regime_series.isnull().any(),
                'string_law_compliant': True  # Will check date format
            }
            
            # Check String Law compliance (YYYY-MM-DD format)
            if hasattr(regime_series.index, 'strftime'):
                try:
                    # Test if dates can be formatted as YYYY-MM-DD
                    test_date = regime_series.index[0]
                    test_date.strftime('%Y-%m-%d')
                except:
                    validation_results['string_law_compliant'] = False
            
            logger.info(f"🛡️ Regime validation: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Regime validation failed: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def get_regime_color(state: int) -> str:
        """Get color for regime state"""
        colors = {
            0: 'lightgreen',    # GREEN - Aggressive
            1: 'lightyellow',   # YELLOW - Caution
            2: 'lightcoral'     # RED - Defensive
        }
        return colors.get(state, 'lightgray')
    
    @staticmethod
    def get_regime_label(state: int) -> str:
        """Get label for regime state"""
        labels = {
            0: 'GREEN (Aggressive)',
            1: 'YELLOW (Caution)',
            2: 'RED (Defensive)'
        }
        return labels.get(state, 'UNKNOWN')
