"""
VXX Volatility Gate - Market Stress Protection
============================================

Implements Bollinger Band logic to prevent entries during extreme volatility.
VXX (Volatility Index ETF) spikes during market stress and crashes.

Logic:
- Calculate 20-day Bollinger Bands on VXX
- Gate triggers when VXX breaks above upper band (2 standard deviations)
- When gate is triggered, no new positions are allowed
- Existing positions can still exit (risk management)

Usage:
    from src.execution.vxx_gate import check_vxx_bollinger_gate
    
    if check_vxx_bollinger_gate(vxx_data):
        # Proceed with new positions
        execute_new_trades()
    else:
        # Block new entries, allow exits only
        execute_exit_trades_only()
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class VXXVolatilityGate:
    """VXX volatility gate using Bollinger Bands"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize VXX volatility gate
        
        Args:
            period: Period for moving average and standard deviation (default: 20)
            std_dev: Number of standard deviations for bands (default: 2.0)
        """
        self.period = period
        self.std_dev = std_dev
        self.logger = logging.getLogger(__name__)
        
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands on VXX data
        
        Args:
            data: DataFrame with 'close' column and datetime index
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        try:
            if len(data) < self.period:
                raise ValueError(f"Insufficient data: need {self.period} bars, got {len(data)}")
            
            df = data.copy()
            
            # Calculate moving average
            df[f'sma_{self.period}'] = df['close'].rolling(window=self.period).mean()
            
            # Calculate standard deviation
            df[f'std_{self.period}'] = df['close'].rolling(window=self.period).std()
            
            # Calculate upper and lower bands
            df['upper_band'] = df[f'sma_{self.period}'] + (df[f'std_{self.period}'] * self.std_dev)
            df['lower_band'] = df[f'sma_{self.period}'] - (df[f'std_{self.period}'] * self.std_dev)
            
            # Calculate bandwidth and position
            df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df[f'sma_{self.period}']
            df['position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"[VXX_GATE] Bollinger Band calculation failed: {e}")
            raise
    
    def check_gate_status(self, vxx_data: Optional[pd.DataFrame]) -> bool:
        """
        Check if VXX volatility gate allows new positions
        
        Args:
            vxx_data: VXX price data with 'close' column
            
        Returns:
            True if gate is open (allow new positions), False if closed (block new positions)
        """
        try:
            if vxx_data is None or len(vxx_data) == 0:
                self.logger.warning("[VXX_GATE] No VXX data available, gate open by default")
                return True
            
            if len(vxx_data) < self.period:
                self.logger.warning(f"[VXX_GATE] Insufficient VXX data ({len(vxx_data)} < {self.period}), gate open")
                return True
            
            # Calculate Bollinger Bands
            df_with_bands = self.calculate_bollinger_bands(vxx_data)
            
            # Get latest values
            latest_close = df_with_bands['close'].iloc[-1]
            latest_upper = df_with_bands['upper_band'].iloc[-1]
            latest_sma = df_with_bands[f'sma_{self.period}'].iloc[-1]
            latest_position = df_with_bands['position'].iloc[-1]
            
            # Gate logic: close when VXX breaks above upper band
            if latest_close > latest_upper:
                self.logger.error(f"[VXX_GATE] VOLATILITY SPIKE DETECTED")
                self.logger.error(f"[VXX_GATE] VXX: {latest_close:.2f} > Upper Band: {latest_upper:.2f}")
                self.logger.error(f"[VXX_GATE] Position: {latest_position:.3f} (1.0 = upper band)")
                self.logger.error("[VXX_GATE] GATE CLOSED - No new positions allowed")
                return False
            
            # Log normal conditions
            self.logger.info(f"[VXX_GATE] Volatility normal - Gate Open")
            self.logger.info(f"[VXX_GATE] VXX: {latest_close:.2f}, SMA: {latest_sma:.2f}, Upper: {latest_upper:.2f}")
            self.logger.info(f"[VXX_GATE] Position: {latest_position:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[VXX_GATE] Gate check failed: {e}")
            return True  # Fail open - allow trading if gate fails
    
    def get_volatility_metrics(self, vxx_data: Optional[pd.DataFrame]) -> dict:
        """
        Get detailed volatility metrics for reporting
        
        Args:
            vxx_data: VXX price data
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            if vxx_data is None or len(vxx_data) < self.period:
                return {'error': 'Insufficient data'}
            
            df_with_bands = self.calculate_bollinger_bands(vxx_data)
            
            latest = df_with_bands.iloc[-1]
            
            return {
                'vxx_close': latest['close'],
                'sma_20': latest[f'sma_{self.period}'],
                'upper_band': latest['upper_band'],
                'lower_band': latest['lower_band'],
                'position': latest['position'],
                'bandwidth': latest['bandwidth'],
                'distance_to_upper': latest['upper_band'] - latest['close'],
                'distance_to_lower': latest['close'] - latest['lower_band'],
                'gate_status': 'OPEN' if latest['close'] <= latest['upper_band'] else 'CLOSED'
            }
            
        except Exception as e:
            self.logger.error(f"[VXX_GATE] Metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def analyze_volatility_regime(self, vxx_data: Optional[pd.DataFrame], lookback_days: int = 60) -> dict:
        """
        Analyze VXX volatility regime over recent period
        
        Args:
            vxx_data: VXX price data
            lookback_days: Number of days to analyze (default: 60)
            
        Returns:
            Dictionary with regime analysis
        """
        try:
            if vxx_data is None or len(vxx_data) < lookback_days:
                return {'error': 'Insufficient data for regime analysis'}
            
            df_with_bands = self.calculate_bollinger_bands(vxx_data)
            recent_data = df_with_bands.tail(lookback_days)
            
            # Calculate regime metrics
            avg_position = recent_data['position'].mean()
            max_position = recent_data['position'].max()
            min_position = recent_data['position'].min()
            
            # Count gate violations
            violations = (recent_data['close'] > recent_data['upper_band']).sum()
            violation_rate = violations / len(recent_data)
            
            # Current regime classification
            if avg_position > 0.8:
                regime = 'HIGH_VOLATILITY'
            elif avg_position < 0.2:
                regime = 'LOW_VOLATILITY'
            else:
                regime = 'NORMAL'
            
            return {
                'regime': regime,
                'avg_position': avg_position,
                'max_position': max_position,
                'min_position': min_position,
                'violations': violations,
                'violation_rate': violation_rate,
                'lookback_days': lookback_days,
                'current_position': recent_data['position'].iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"[VXX_GATE] Regime analysis failed: {e}")
            return {'error': str(e)}

# Global instance for easy access
vxx_gate = VXXVolatilityGate()

def check_vxx_bollinger_gate(vxx_data: Optional[pd.DataFrame]) -> bool:
    """
    Convenience function to check VXX volatility gate
    
    Args:
        vxx_data: VXX price data with 'close' column
        
    Returns:
        True if gate is open (allow new positions), False if closed
    """
    return vxx_gate.check_gate_status(vxx_data)

def get_vxx_volatility_metrics(vxx_data: Optional[pd.DataFrame]) -> dict:
    """
    Convenience function to get VXX volatility metrics
    
    Args:
        vxx_data: VXX price data
        
    Returns:
        Dictionary with volatility metrics
    """
    return vxx_gate.get_volatility_metrics(vxx_data)

def analyze_vxx_volatility_regime(vxx_data: Optional[pd.DataFrame], lookback_days: int = 60) -> dict:
    """
    Convenience function to analyze VXX volatility regime
    
    Args:
        vxx_data: VXX price data
        lookback_days: Number of days to analyze
        
    Returns:
        Dictionary with regime analysis
    """
    return vxx_gate.analyze_volatility_regime(vxx_data, lookback_days)

# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Test with sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.random.randn(100).cumsum()  # Random walk
    sample_data = pd.DataFrame({'close': prices}, index=dates)
    
    gate = VXXVolatilityGate()
    
    # Test gate status
    is_open = gate.check_gate_status(sample_data)
    print(f"Gate Status: {'OPEN' if is_open else 'CLOSED'}")
    
    # Test metrics
    metrics = gate.get_volatility_metrics(sample_data)
    print(f"Metrics: {metrics}")
    
    # Test regime analysis
    regime = gate.analyze_volatility_regime(sample_data)
    print(f"Regime: {regime}")
