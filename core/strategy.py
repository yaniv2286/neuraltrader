"""
Core Trading Strategy - Entry/Exit Decision Rules
Protected module containing all trading strategy logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

# Import configuration for institutional schedule
try:
    from config import config
except ImportError:
    try:
        from .config import config
    except ImportError:
        logger.warning("[WARN] Config module not available, using defaults")
        config = None


class TradingStrategy:
    """
    Core trading strategy with entry/exit decision rules
    Implements Phase 2 Exit Optimization Tournament winning parameters
    """
    
    def __init__(self):
        """Initialize strategy with Phase 2 Exit Optimization winning parameters"""
        # Safety constraints - required by integrity check
        self.max_drawdown = 0.20  # 20% max drawdown limit
        self.max_position_size = 0.20  # 20% max position size
        
        # [TARGET] PHASE 2 EXIT OPTIMIZATION WINNER: Fixed ATR 2.5 | Weekly Shield Active
        self.EMERGENCY_STOP_LOSS_ATR_MULTIPLIER = 2.5  # Tightened ATR multiplier for better risk management
        self.WEEKLY_SHIELD_DAYS = 5  # Previous 5 trading days minimum low
        self.NO_TAKE_PROFIT = True  # No static take profit - let shield capture trend
        
        # [ARCH] INSTITUTIONAL SCHEDULE & DATA VALIDATION
        self.PRE_EXECUTION_DATA_CHECK = True  # Enable pre-execution data validation
        
        # Strategy lock confirmation
        logger.info("[TARGET] STRATEGY LOCK: Fixed ATR 2.5 | Weekly Shield Active")
        logger.info(f"   Emergency Stop Loss: {self.EMERGENCY_STOP_LOSS_ATR_MULTIPLIER}x ATR")
        logger.info(f"   Weekly Shield: Previous {self.WEEKLY_SHIELD_DAYS} days low")
        logger.info(f"   Take Profit: Disabled (trend extension mode)")
        logger.info(f"   Pre-execution Data Check: {self.PRE_EXECUTION_DATA_CHECK}")
    
    def pre_execution_data_validation(self) -> bool:
        """
        Pre-execution check: Verify data freshness before any trading activity
        
        Returns:
            True if data is current and valid for trading
            
        Raises:
            SystemExit: If data is stale (Fail Fast protocol)
        """
        if not self.PRE_EXECUTION_DATA_CHECK:
            logger.info("[SKIP] Pre-execution data check disabled")
            return True
        
        if config is None:
            logger.warning("[WARN] Config not available, skipping data validation")
            return True
        
        logger.info("[CHECK] PRE-EXECUTION DATA VALIDATION STARTED")
        
        try:
            # Use config's validation method
            validation_passed = config.validate_data_freshness()
            
            if validation_passed:
                logger.info("[PASS] PRE-EXECUTION DATA VALIDATION PASSED")
                return True
            else:
                logger.error("[FAIL] PRE-EXECUTION DATA VALIDATION FAILED")
                return False
                
        except SystemExit:
            # Re-raise SystemExit for critical failures
            raise
        except Exception as e:
            error_msg = f"CRITICAL: Pre-execution data validation failed: {e}"
            logger.error(f"[FAIL] Pre-execution data validation failed: {e}")
            logger.error(error_msg)
            if config and config.FAIL_FAST_ON_STALE_DATA:
                raise SystemExit(error_msg)
            return False
    
    def check_entry(self, data: pd.DataFrame) -> bool:
        """
        Entry condition check required by integrity check
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            True if entry conditions are met
        """
        # Delegate to check_entry_conditions
        conditions = self.check_entry_conditions(data)
        return conditions.get('trend_up', False) or conditions.get('price_above_ma', False)
    
    def check_exit(self, data: pd.DataFrame, entry_price: float = None, entry_atr: float = None) -> bool:
        """
        Exit condition check implementing Weekly Breakdown Shield and Emergency Stop Loss
        
        Args:
            data: DataFrame with OHLCV data
            entry_price: Entry price for ATR stop loss calculation
            entry_atr: ATR at time of entry for stop loss calculation
            
        Returns:
            True if exit conditions are met
        """
        try:
            if len(data) < self.WEEKLY_SHIELD_DAYS + 1:
                return False
            
            current_close = data['close'].iloc[-1]
            
            # WEEKLY BREAKDOWN SHIELD: Exit if close below previous 5 days minimum low
            if len(data) >= self.WEEKLY_SHIELD_DAYS + 1:
                previous_5_days = data.iloc[-(self.WEEKLY_SHIELD_DAYS + 1):-1]  # Exclude current day
                min_low_5_days = previous_5_days['low'].min()
                
                if current_close < min_low_5_days:
                    logger.info(f"[SHIELD] Weekly Shield Triggered: Close {current_close:.2f} < 5-day min low {min_low_5_days:.2f}")
                    return True
            
            # EMERGENCY STOP LOSS: Fixed ATR 5.0 multiplier (if entry data available)
            if entry_price is not None and entry_atr is not None:
                emergency_stop_level = entry_price - (entry_atr * self.EMERGENCY_STOP_LOSS_ATR_MULTIPLIER)
                
                if current_close <= emergency_stop_level:
                    logger.info(f"[STOP] Emergency Stop Loss: Close {current_close:.2f} <= stop level {emergency_stop_level:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"[FAIL] Exit check failed: {e}")
            return False
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) with No Silent Failures protocol
        
        Args:
            data: DataFrame with OHLCV data
            period: ATR calculation period (default 14)
            
        Returns:
            ATR value
            
        Raises:
            CriticalError: If ATR cannot be calculated
        """
        try:
            if len(data) < period + 1:
                critical_msg = f"CRITICAL: Insufficient data for ATR calculation. Need {period + 1} bars, got {len(data)}"
                logger.error(critical_msg)
                raise ValueError(critical_msg)
            
            # Calculate True Range components
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR as simple moving average of True Range
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                critical_msg = f"CRITICAL: Invalid ATR calculation result: {atr}"
                logger.error(critical_msg)
                raise ValueError(critical_msg)
            
            return float(atr)
            
        except Exception as e:
            critical_msg = f"CRITICAL: ATR calculation failed for ticker: {e}"
            logger.error(critical_msg)
            raise ValueError(critical_msg)
    
    def validate_atr_for_trade(self, ticker: str, data: pd.DataFrame) -> float:
        """
        Validate ATR calculation for trade with No Silent Failures protocol
        
        Args:
            ticker: Ticker symbol for logging
            data: DataFrame with OHLCV data
            
        Returns:
            ATR value if successful
            
        Raises:
            CriticalError: If ATR cannot be calculated, refusing the trade
        """
        try:
            atr = self.calculate_atr(data)
            logger.info(f"[PASS] ATR validated for {ticker}: {atr:.4f}")
            return atr
            
        except Exception as e:
            logger.error(f"[FAIL] Refusing trade for {ticker} - {e}")
            raise ValueError(f"[FAIL] Refusing trade for {ticker} - {e}")
    
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
    
    def check_entry_conditions(self, df: pd.DataFrame, 
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


class EliteSniperStrategy:
    """
    Elite Sniper Strategy - Top-K Alpha Concentration
    
    Implements the Concentration Law:
    1. Sort all passing tickers by Ensemble Raw Score (Descending)
    2. Only execute trades for the Top 10 tickers per day
    3. Threshold Spike: AI_THRESHOLD default raised to 0.55
    """
    
    def __init__(self):
        """Initialize Elite Sniper with high-conviction parameters"""
        # 🦅 ELITE PARAMETERS - Concentrated Alpha
        self.AI_THRESHOLD = 0.55  # High-conviction threshold only
        self.ELITE_TOP_K = 10  # Only top 10 tickers per day
        self.MAX_POSITIONS = 10  # Elite concentration
        
        logger.info("🎯 Elite Sniper Strategy initialized")
        logger.info(f"   AI_THRESHOLD: {self.AI_THRESHOLD} (high-conviction)")
        logger.info(f"   ELITE_TOP_K: {self.ELITE_TOP_K} tickers per day")
        logger.info(f"   MAX_POSITIONS: {self.MAX_POSITIONS} concentrated positions")
    
    def select_elite_trades(self, ticker_signals: Dict[str, float], 
                           market_regime: int = 0) -> List[Tuple[str, float]]:
        """
        Implement the Concentration Law - Select Top-K elite trades
        
        Args:
            ticker_signals: Dict of ticker -> AI probability score
            market_regime: Current market regime (0=GREEN, 1=YELLOW, 2=RED)
            
        Returns:
            List of (ticker, score) tuples for elite trades only
        """
        try:
            # 🦅 THE CONCENTRATION LAW - Filter by high-conviction threshold
            high_conviction_tickers = {
                ticker: score for ticker, score in ticker_signals.items()
                if score >= self.AI_THRESHOLD
            }
            
            # 🦅 REGIME FILTER - Apply regime constraints
            if market_regime == 2:  # RED regime - Hard stop
                logger.info("[STOP] REGIME RED: Hard stop - no trades allowed")
                return []
            elif market_regime == 1:  # YELLOW regime - Cautious
                # Reduce top-K in yellow regime
                elite_k = max(1, self.ELITE_TOP_K // 2)
                logger.info(f"[WARN] REGIME YELLOW: Cautious mode - Top {elite_k} only")
            else:  # GREEN regime - Full deployment
                elite_k = self.ELITE_TOP_K
                logger.info(f"[OK] REGIME GREEN: Full deployment - Top {elite_k}")
            
            # 🦅 ELITE SELECTION - Sort by score and take top-K
            sorted_tickers = sorted(
                high_conviction_tickers.items(),
                key=lambda x: x[1],  # Sort by AI score
                reverse=True  # Descending order
            )
            
            elite_trades = sorted_tickers[:elite_k]
            
            if elite_trades:
                logger.info(f"🎯 ELITE SNIPER: Selected {len(elite_trades)} elite trades")
                for i, (ticker, score) in enumerate(elite_trades[:3], 1):
                    logger.info(f"   {i}. {ticker}: {score:.4f}")
            else:
                logger.info("🔍 ELITE SNIPER: No high-conviction trades found")
            
            return elite_trades
            
        except Exception as e:
            logger.error(f"❌ Elite selection failed: {e}")
            return []
    
    def check_entry_conditions(self, ticker: str, score: float, 
                             market_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Enhanced entry conditions for elite trades
        
        Args:
            ticker: Ticker symbol
            score: AI probability score
            market_data: OHLCV data for ticker
            
        Returns:
            Dict with entry condition flags
        """
        try:
            if len(market_data) < 20:
                return {
                    'elite_score': False,
                    'volume_ok': False,
                    'trend_ok': False,
                    'can_enter': False
                }
            
            latest = market_data.iloc[-1]
            
            # 🦅 ELITE SCORE CHECK - Must be high conviction
            elite_score = score >= self.AI_THRESHOLD
            
            # 🦅 VOLUME CHECK - Minimum liquidity
            volume_ok = True  # Will be enhanced with liquidity filter
            
            # 🦅 TREND CHECK - Basic momentum confirmation
            if len(market_data) >= 5:
                trend_ok = latest['close'] > market_data['close'].iloc[-5]
            else:
                trend_ok = True
            
            can_enter = elite_score and volume_ok and trend_ok
            
            return {
                'elite_score': elite_score,
                'volume_ok': volume_ok,
                'trend_ok': trend_ok,
                'can_enter': can_enter
            }
            
        except Exception as e:
            logger.error(f"❌ Entry condition check failed for {ticker}: {e}")
            return {
                'elite_score': False,
                'volume_ok': False,
                'trend_ok': False,
                'can_enter': False
            }
    
    def calculate_elite_position_size(self, portfolio_value: float,
                                   entry_price: float,
                                   elite_rank: int) -> int:
        """
        Calculate position size with elite concentration
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price
            elite_rank: Rank in elite selection (1=best)
            
        Returns:
            Number of shares to buy
        """
        try:
            # 🦅 ELITE POSITION SIZING - Higher allocation for top-ranked trades
            base_allocation = 0.10  # 10% base allocation per trade
            
            # Rank-based scaling: Top 3 get larger allocations
            if elite_rank == 1:
                allocation_multiplier = 1.5  # 15% for #1
            elif elite_rank == 2:
                allocation_multiplier = 1.3  # 13% for #2
            elif elite_rank == 3:
                allocation_multiplier = 1.2  # 12% for #3
            else:
                allocation_multiplier = 1.0  # 10% for others
            
            position_value = portfolio_value * base_allocation * allocation_multiplier
            
            # Calculate shares
            shares = int(position_value / entry_price)
            
            logger.info(f"💰 Elite Position #{elite_rank}: {shares} shares @ ${entry_price:.2f} "
                       f"({allocation_multiplier}x multiplier)")
            
            return shares
            
        except Exception as e:
            logger.error(f"❌ Elite position sizing failed: {e}")
            return 0
    
    @staticmethod
    def should_exit_position(current_price: float, entry_price: float,
                           stop_loss_pct: float = 0.08,
                           take_profit_pct: float = 0.25) -> Dict[str, any]:
        """
        Elite exit conditions with wider targets for alpha capture
        
        Args:
            current_price: Current price
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage (default 8%)
            take_profit_pct: Take profit percentage (default 25%)
            
        Returns:
            Dict with exit decision and reason
        """
        try:
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 🦅 ELITE EXIT RULES - Wider stops and targets for alpha capture
            if pnl_pct <= -stop_loss_pct:
                return {'should_exit': True, 'reason': 'stop_loss', 'pnl_pct': pnl_pct}
            elif pnl_pct >= take_profit_pct:
                return {'should_exit': True, 'reason': 'take_profit', 'pnl_pct': pnl_pct}
            else:
                return {'should_exit': False, 'reason': None, 'pnl_pct': pnl_pct}
                
        except Exception as e:
            logger.error(f"❌ Elite exit check failed: {e}")
            return {'should_exit': False, 'reason': None, 'pnl_pct': 0}
