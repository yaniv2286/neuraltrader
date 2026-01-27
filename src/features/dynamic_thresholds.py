import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class DynamicThresholds:
    """Dynamic threshold adjustment based on market conditions"""
    
    def __init__(self):
        self.volatility_window = 21  # 1 month
        self.trend_window = 63      # 3 months
        self.volume_window = 20      # 20 days
        
    def calculate_volatility_adjusted_thresholds(self, df: pd.DataFrame, 
                                               base_buy: float = 0.01, 
                                               base_sell: float = 0.01,
                                               price_col: str = 'close') -> Tuple[float, float]:
        """
        Adjust thresholds based on recent volatility
        Higher volatility = wider thresholds
        """
        # Calculate recent volatility
        returns = df[price_col].pct_change().dropna()
        recent_vol = returns.tail(self.volatility_window).std()
        historical_vol = returns.std()
        
        # Volatility ratio
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Adjust thresholds
        if vol_ratio > 1.5:  # High volatility period
            multiplier = 1.5
        elif vol_ratio > 1.2:  # Moderately high
            multiplier = 1.2
        elif vol_ratio < 0.7:  # Low volatility
            multiplier = 0.8
        else:  # Normal
            multiplier = 1.0
            
        return base_buy * multiplier, base_sell * multiplier
    
    def calculate_trend_adjusted_thresholds(self, df: pd.DataFrame,
                                            base_buy: float = 0.01,
                                            base_sell: float = 0.01,
                                            price_col: str = 'close') -> Tuple[float, float]:
        """
        Adjust thresholds based on trend strength
        Strong trend = tighter thresholds for counter-trend trades
        """
        # Calculate trend strength
        sma_short = df[price_col].rolling(window=20).mean()
        sma_long = df[price_col].rolling(window=self.trend_window).mean()
        
        # Trend direction and strength
        trend_direction = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else -1
        trend_strength = abs((sma_short.iloc[-1] / sma_long.iloc[-1]) - 1)
        
        # Adjust based on trend
        if trend_direction > 0 and trend_strength > 0.05:  # Strong uptrend
            # Tighten sell thresholds (take profits), loosen buy (add to positions)
            buy_adj = base_buy * 1.2
            sell_adj = base_sell * 0.8
        elif trend_direction < 0 and trend_strength > 0.05:  # Strong downtrend
            # Tighten buy thresholds (be selective), loosen sell (cut losses)
            buy_adj = base_buy * 0.8
            sell_adj = base_sell * 1.2
        else:  # Weak or no trend
            buy_adj = base_buy
            sell_adj = base_sell
            
        return buy_adj, sell_adj
    
    def calculate_volume_adjusted_thresholds(self, df: pd.DataFrame,
                                            base_buy: float = 0.01,
                                            base_sell: float = 0.01,
                                            price_col: str = 'close',
                                            volume_col: str = 'volume') -> Tuple[float, float]:
        """
        Adjust thresholds based on volume patterns
        High volume = more reliable signals = tighter thresholds
        """
        if volume_col not in df.columns:
            return base_buy, base_sell
            
        # Volume analysis
        recent_volume = df[volume_col].tail(self.volume_window)
        avg_volume = df[volume_col].rolling(window=50).mean().iloc[-1]
        
        volume_ratio = recent_volume.mean() / avg_volume if avg_volume > 0 else 1.0
        
        # Adjust thresholds
        if volume_ratio > 2.0:  # Very high volume
            multiplier = 0.8  # Tighter thresholds
        elif volume_ratio > 1.5:  # High volume
            multiplier = 0.9
        elif volume_ratio < 0.5:  # Low volume
            multiplier = 1.2  # Wider thresholds
        else:
            multiplier = 1.0
            
        return base_buy * multiplier, base_sell * multiplier
    
    def calculate_regime_adjusted_thresholds(self, df: pd.DataFrame,
                                            base_buy: float = 0.01,
                                            base_sell: float = 0.01,
                                            price_col: str = 'close',
                                            regime: str = 'NEUTRAL') -> Tuple[float, float]:
        """
        Adjust thresholds based on market regime
        """
        regime_multipliers = {
            'BULL': {'buy': 0.8, 'sell': 1.2},      # Aggressive in bull
            'BEAR': {'buy': 1.5, 'sell': 0.8},      # Conservative in bear
            'SIDEWAYS': {'buy': 1.0, 'sell': 1.0},  # Neutral
            'NEUTRAL': {'buy': 1.0, 'sell': 1.0}
        }
        
        mult = regime_multipliers.get(regime, regime_multipliers['NEUTRAL'])
        
        return base_buy * mult['buy'], base_sell * mult['sell']
    
    def get_dynamic_thresholds(self, df: pd.DataFrame,
                             base_buy: float = 0.01,
                             base_sell: float = 0.01,
                             price_col: str = 'close',
                             volume_col: str = 'volume',
                             regime: str = 'NEUTRAL') -> Dict[str, Any]:
        """
        Calculate all dynamic thresholds and return comprehensive info
        """
        # Calculate different adjustments
        vol_buy, vol_sell = self.calculate_volatility_adjusted_thresholds(df, base_buy, base_sell, price_col)
        trend_buy, trend_sell = self.calculate_trend_adjusted_thresholds(df, base_buy, base_sell, price_col)
        vol_buy_adj, vol_sell_adj = self.calculate_volume_adjusted_thresholds(df, base_buy, base_sell, price_col, volume_col)
        regime_buy, regime_sell = self.calculate_regime_adjusted_thresholds(df, base_buy, base_sell, price_col, regime)
        
        # Combine adjustments (weighted average)
        buy_threshold = (vol_buy * 0.3 + trend_buy * 0.3 + vol_buy_adj * 0.2 + regime_buy * 0.2)
        sell_threshold = (vol_sell * 0.3 + trend_sell * 0.3 + vol_sell_adj * 0.2 + regime_sell * 0.2)
        
        # Calculate adjustment factors
        buy_adjustment = buy_threshold / base_buy
        sell_adjustment = sell_threshold / base_sell
        
        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'buy_adjustment': buy_adjustment,
            'sell_adjustment': sell_adjustment,
            'volatility_adjusted': (vol_buy, vol_sell),
            'trend_adjusted': (trend_buy, trend_sell),
            'volume_adjusted': (vol_buy_adj, vol_sell_adj),
            'regime_adjusted': (regime_buy, regime_sell),
            'adjustment_factors': {
                'volatility': vol_buy / base_buy,
                'trend': trend_buy / base_buy,
                'volume': vol_buy_adj / base_buy,
                'regime': regime_buy / base_buy
            }
        }
    
    def get_adaptive_thresholds_series(self, df: pd.DataFrame,
                                     base_buy: float = 0.01,
                                     base_sell: float = 0.01,
                                     price_col: str = 'close',
                                     volume_col: str = 'volume',
                                     regime_series: pd.Series = None) -> pd.DataFrame:
        """
        Calculate time series of dynamic thresholds
        """
        thresholds = []
        
        for i in range(max(self.volatility_window, self.trend_window), len(df)):
            window_df = df.iloc[:i+1]
            
            # Get regime for this window
            regime = 'NEUTRAL'
            if regime_series is not None and i < len(regime_series):
                regime = regime_series.iloc[i]
            
            # Calculate thresholds
            thresh_dict = self.get_dynamic_thresholds(
                window_df, base_buy, base_sell, price_col, volume_col, regime
            )
            
            thresholds.append({
                'date': df.index[i],
                'buy_threshold': thresh_dict['buy_threshold'],
                'sell_threshold': thresh_dict['sell_threshold'],
                'buy_adjustment': thresh_dict['buy_adjustment'],
                'sell_adjustment': thresh_dict['sell_adjustment']
            })
        
        return pd.DataFrame(thresholds).set_index('date')
