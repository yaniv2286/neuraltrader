"""
Core Trading Strategy - Entry/Exit Decision Rules
Protected module containing all trading strategy logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class TradingStrategy:
    """
    Core trading strategy with entry/exit decision rules
    """
    
    @staticmethod
    def generate_signal_from_momentum(df: pd.DataFrame, 
                                      buy_threshold: float = 0.02, 
                                      sell_threshold: float = -0.02) -> float:
        """
        Generate trading signal based on momentum
        
        Args:
            df: DataFrame with OHLCV data
            buy_threshold: Momentum threshold for buy signal (default 2%)
            sell_threshold: Momentum threshold for sell signal (default -2%)
            
        Returns:
            Signal confidence: 0.0-1.0 (0.7 = buy, 0.3 = sell, 0.0 = hold)
        """
        try:
            if len(df) < 20:
                return 0.0
            
            # Simple momentum signal
            recent_return = df['close'].pct_change(5).iloc[-1]
            
            if recent_return > buy_threshold:
                return 0.7
            elif recent_return < sell_threshold:
                return 0.3
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    @staticmethod
    def check_entry_conditions(df: pd.DataFrame, 
                               rsi_oversold: float = 30,
                               rsi_overbought: float = 70,
                               volume_threshold: float = 1.5) -> Dict[str, bool]:
        """
        Check multiple entry conditions
        
        Args:
            df: DataFrame with OHLCV and indicator data
            rsi_oversold: RSI level for oversold condition
            rsi_overbought: RSI level for overbought condition
            volume_threshold: Volume spike threshold (multiplier of average)
            
        Returns:
            Dict with boolean flags for each condition
        """
        try:
            if len(df) < 20:
                return {
                    'trend_up': False,
                    'rsi_oversold': False,
                    'rsi_overbought': False,
                    'volume_spike': False,
                    'price_above_ma': False
                }
            
            latest = df.iloc[-1]
            
            # Calculate indicators if not present
            if 'rsi_14' not in df.columns:
                from core.indicators import calculate_rsi
                df['rsi_14'] = calculate_rsi(df['close'], period=14)
            
            if 'sma_20' not in df.columns:
                df['sma_20'] = df['close'].rolling(window=20).mean()
            
            if 'volume_avg' not in df.columns:
                df['volume_avg'] = df['volume'].rolling(window=20).mean()
            
            # Check conditions
            conditions = {
                'trend_up': latest['close'] > df['close'].iloc[-5],
                'rsi_oversold': latest['rsi_14'] < rsi_oversold if 'rsi_14' in latest else False,
                'rsi_overbought': latest['rsi_14'] > rsi_overbought if 'rsi_14' in latest else False,
                'volume_spike': latest['volume'] > (latest['volume_avg'] * volume_threshold) if 'volume_avg' in latest else False,
                'price_above_ma': latest['close'] > latest['sma_20'] if 'sma_20' in latest else False
            }
            
            return conditions
            
        except Exception as e:
            return {
                'trend_up': False,
                'rsi_oversold': False,
                'rsi_overbought': False,
                'volume_spike': False,
                'price_above_ma': False
            }
    
    @staticmethod
    def check_exit_conditions(df: pd.DataFrame,
                              entry_price: float,
                              stop_loss_pct: float = 0.02,
                              take_profit_pct: float = 0.05,
                              trailing_stop_pct: float = 0.03) -> Dict[str, bool]:
        """
        Check exit conditions for an open position
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Price at which position was entered
            stop_loss_pct: Stop loss percentage (default 2%)
            take_profit_pct: Take profit percentage (default 5%)
            trailing_stop_pct: Trailing stop percentage (default 3%)
            
        Returns:
            Dict with exit signals and reasons
        """
        try:
            if len(df) == 0:
                return {'should_exit': False, 'reason': None}
            
            current_price = df['close'].iloc[-1]
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Calculate highest price since entry (for trailing stop)
            highest_price = df['close'].max()
            drawdown_from_high = (current_price - highest_price) / highest_price
            
            # Check exit conditions
            if pnl_pct <= -stop_loss_pct:
                return {'should_exit': True, 'reason': 'stop_loss', 'pnl_pct': pnl_pct}
            
            if pnl_pct >= take_profit_pct:
                return {'should_exit': True, 'reason': 'take_profit', 'pnl_pct': pnl_pct}
            
            if drawdown_from_high <= -trailing_stop_pct:
                return {'should_exit': True, 'reason': 'trailing_stop', 'pnl_pct': pnl_pct}
            
            return {'should_exit': False, 'reason': None, 'pnl_pct': pnl_pct}
            
        except Exception as e:
            return {'should_exit': False, 'reason': None}
    
    @staticmethod
    def calculate_position_size(capital: float,
                                risk_per_trade: float,
                                entry_price: float,
                                stop_loss_price: float,
                                max_position_pct: float = 0.20) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            capital: Total available capital
            risk_per_trade: Risk percentage per trade (e.g., 0.01 for 1%)
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            max_position_pct: Maximum position size as % of capital (default 20%)
            
        Returns:
            Number of shares to buy
        """
        try:
            # Risk-based position sizing
            risk_amount = capital * risk_per_trade
            price_risk = abs(entry_price - stop_loss_price)
            
            if price_risk == 0:
                return 0
            
            shares_by_risk = int(risk_amount / price_risk)
            
            # Apply maximum position size constraint
            max_shares = int((capital * max_position_pct) / entry_price)
            
            # Return the smaller of the two
            return min(shares_by_risk, max_shares)
            
        except Exception as e:
            return 0
    
    @staticmethod
    def evaluate_market_regime(df: pd.DataFrame, 
                               lookback: int = 60) -> str:
        """
        Determine current market regime
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period for regime detection
            
        Returns:
            Market regime: 'uptrend', 'downtrend', 'sideways', 'high_volatility'
        """
        try:
            if len(df) < lookback:
                return 'unknown'
            
            recent_data = df.tail(lookback)
            
            # Calculate trend
            returns = recent_data['close'].pct_change()
            avg_return = returns.mean()
            
            # Calculate volatility
            volatility = returns.std()
            
            # Determine regime
            if volatility > 0.03:  # High volatility threshold (3% daily)
                return 'high_volatility'
            elif avg_return > 0.001:  # Positive trend
                return 'uptrend'
            elif avg_return < -0.001:  # Negative trend
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            return 'unknown'
    
    @staticmethod
    def combine_signals(ml_signal: float,
                       momentum_signal: float,
                       technical_signal: float,
                       weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
        """
        Combine multiple signals with weights
        
        Args:
            ml_signal: ML model signal (0.0-1.0)
            momentum_signal: Momentum-based signal (0.0-1.0)
            technical_signal: Technical indicator signal (0.0-1.0)
            weights: Weights for each signal (must sum to 1.0)
            
        Returns:
            Combined signal (0.0-1.0)
        """
        try:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = tuple(w / total_weight for w in weights)
            
            # Weighted average
            combined = (
                ml_signal * normalized_weights[0] +
                momentum_signal * normalized_weights[1] +
                technical_signal * normalized_weights[2]
            )
            
            return combined
            
        except Exception as e:
            return 0.5  # Neutral signal on error


class RiskManager:
    """Risk management rules and constraints"""
    
    @staticmethod
    def check_risk_limits(portfolio_value: float,
                         position_value: float,
                         daily_loss: float,
                         max_position_pct: float = 0.20,
                         max_daily_loss_pct: float = 0.05) -> Dict[str, bool]:
        """
        Check if trade violates risk limits
        
        Args:
            portfolio_value: Total portfolio value
            position_value: Value of proposed position
            daily_loss: Current daily loss amount
            max_position_pct: Maximum position size (default 20%)
            max_daily_loss_pct: Maximum daily loss (default 5%)
            
        Returns:
            Dict with risk check results
        """
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        daily_loss_pct = abs(daily_loss) / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'position_size_ok': position_pct <= max_position_pct,
            'daily_loss_ok': daily_loss_pct <= max_daily_loss_pct,
            'can_trade': position_pct <= max_position_pct and daily_loss_pct <= max_daily_loss_pct
        }
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float,
                                  avg_win: float,
                                  avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Historical win rate (0.0-1.0)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size
            
        Returns:
            Optimal position size as fraction of capital
        """
        try:
            if avg_loss == 0:
                return 0.0
            
            win_loss_ratio = avg_win / abs(avg_loss)
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply half-Kelly for safety
            return max(0.0, min(kelly * 0.5, 0.25))  # Cap at 25%
            
        except Exception as e:
            return 0.0
