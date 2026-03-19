"""
Core Trading Strategy - Entry/Exit Decision Rules
Protected module containing all trading strategy logic
"""

import pandas as pd
import numpy as np
import traceback
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
        
        # [v5] PHASE 12 VALIDATED PARAMETERS (CAGR 15.92%, DD 22.99% over 26yr)
        self.STOP_LOSS_ATR_MULT   = 2.5    # 2.5x ATR20 stop — validated in 26yr backtest
        self.STOP_LOSS_MAX_PCT    = 0.20   # ATR stop cap: never wider than 20%
        self.STOP_LOSS_MIN_PCT    = 0.08   # ATR stop floor: never tighter than 8%
        self.TRAIL_STOP_PCT       = 0.12   # 12% trailing stop from peak
        self.TRAIL_ACTIVATE_PCT   = 0.05   # Trailing stop activates once position >=5% in profit
        self.TAKE_PROFIT_PCT      = 0.40   # 40% take-profit
        self.MAX_HOLD_DAYS        = 25     # 25-day timeout
        self.MAX_RISK_PER_TRADE   = 0.020  # 2.0% portfolio risk per trade
        self.MIN_POSITION_PCT     = 0.05   # 5% portfolio floor per position
        self.MAX_POSITION_PCT     = 0.10   # 10% portfolio cap per position
        self.MAX_POSITIONS        = 15     # Max concurrent positions
        self.UNCLE_POINT_DD       = 0.20   # 20% drawdown circuit breaker
        self.COOLDOWN_DAYS        = 10     # 10-day cooldown after uncle point
        self.MIN_PRICE            = 10.0   # $10 minimum price filter
        self.MIN_AVG_VOLUME       = 500_000 # 500k minimum average volume
        
        # 🚀 PHASE 13 REGIME-ADAPTIVE THRESHOLDS (Optimized)
        self.CRISIS_THRESHOLD     = 0.80   # CRISIS: Very strict (no entries in crisis)
        self.BEAR_THRESHOLD       = 0.72   # BEAR: Strict threshold
        self.BULL_THRESHOLD       = 0.65   # BULL: Standard threshold
        self.DEFAULT_THRESHOLD    = 0.70   # DEFAULT: Conservative fallback
        
        # Legacy thresholds (kept for compatibility)
        self.CONFIDENCE_THRESHOLD = 0.65   # Standard entry threshold
        
        # [ARCH] INSTITUTIONAL SCHEDULE & DATA VALIDATION
        self.PRE_EXECUTION_DATA_CHECK = True  # Enable pre-execution data validation
        
        # Strategy lock confirmation
        logger.info("[v5] STRATEGY LOCK: Phase 12 v5 Parameters (CAGR 15.92% validated)")
        logger.info(f"   ATR Stop: {self.STOP_LOSS_ATR_MULT}x ATR20 [{self.STOP_LOSS_MIN_PCT*100:.0f}%-{self.STOP_LOSS_MAX_PCT*100:.0f}%]")
        logger.info(f"   Trailing Stop: {self.TRAIL_STOP_PCT*100:.0f}% from peak (activates at +{self.TRAIL_ACTIVATE_PCT*100:.0f}%)")
        logger.info(f"   Take Profit: {self.TAKE_PROFIT_PCT*100:.0f}% | Max Hold: {self.MAX_HOLD_DAYS}d")
        logger.info(f"   Risk/Trade: {self.MAX_RISK_PER_TRADE*100:.1f}% | Pos Floor: {self.MIN_POSITION_PCT*100:.0f}%")
        logger.info(f"   Uncle Point: {self.UNCLE_POINT_DD*100:.0f}% DD | Cooldown: {self.COOLDOWN_DAYS}d")
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
    
    def calc_atr_stop_pct(self, data: pd.DataFrame) -> float:
        """
        Calculate ATR-based stop-loss percentage: 2.5x ATR20 / price, clamped [8%, 20%].
        Returns the stop percentage to use for this position.
        """
        try:
            d = data.tail(22)
            if len(d) < 14:
                return self.STOP_LOSS_MIN_PCT
            high  = d['high']
            low   = d['low']
            close = d['close']
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.mean()
            atr_pct = (atr * self.STOP_LOSS_ATR_MULT) / close.iloc[-1]
            return float(np.clip(atr_pct, self.STOP_LOSS_MIN_PCT, self.STOP_LOSS_MAX_PCT))
        except Exception as e:
            logger.warning(f"[WARN] ATR stop calc failed: {e} — using floor {self.STOP_LOSS_MIN_PCT:.0%}")
            return self.STOP_LOSS_MIN_PCT

    def check_exit(self, data: pd.DataFrame, entry_price: float = None, entry_atr: float = None,
                   stop_loss_pct: float = None, peak_price: float = None, hold_days: int = 0) -> dict:
        """
        Exit condition check implementing Phase 12 v5 exit hierarchy:
        1. ATR-based stop-loss (8-20%)
        2. Trailing stop (12% from peak, activates once >=5% profit)
        3. Take-profit (40%)
        4. Timeout (25 days)
        
        Returns:
            dict: {'should_exit': bool, 'reason': str or None}
        """
        try:
            if len(data) == 0 or entry_price is None:
                return {'should_exit': False, 'reason': None}

            current_price = data['close'].iloc[-1]
            pnl_pct = (current_price - entry_price) / entry_price

            # Use provided stop or calculate ATR stop
            _stop_pct = stop_loss_pct if stop_loss_pct is not None else self.calc_atr_stop_pct(data)

            # 1. ATR-based stop-loss
            if pnl_pct <= -_stop_pct:
                logger.info(f"[STOP] ATR stop-loss: pnl={pnl_pct:.1%} <= -{_stop_pct:.1%}")
                return {'should_exit': True, 'reason': 'STOP_LOSS'}

            # 2. Trailing stop (only activates once position is >=5% in profit)
            _peak = peak_price if peak_price is not None else current_price
            if _peak > entry_price * (1 + self.TRAIL_ACTIVATE_PCT):
                trail_pct = (_peak - current_price) / _peak
                if trail_pct >= self.TRAIL_STOP_PCT:
                    logger.info(f"[TRAIL] Trailing stop: {trail_pct:.1%} from peak ${_peak:.2f}")
                    return {'should_exit': True, 'reason': 'TRAIL_STOP'}

            # 3. Take-profit
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                logger.info(f"[TP] Take-profit: pnl={pnl_pct:.1%} >= {self.TAKE_PROFIT_PCT:.0%}")
                return {'should_exit': True, 'reason': 'TAKE_PROFIT'}

            # 4. Timeout
            if hold_days >= self.MAX_HOLD_DAYS:
                logger.info(f"[TIMEOUT] Hold days {hold_days} >= {self.MAX_HOLD_DAYS}")
                return {'should_exit': True, 'reason': 'TIMEOUT'}

            return {'should_exit': False, 'reason': None}

        except Exception as e:
            logger.error(f"[FAIL] Exit check failed: {e}\n{traceback.format_exc()}")
            return {'should_exit': False, 'reason': None}
    
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
    
    def check_exit_conditions(self, df: pd.DataFrame,
                              entry_price: float,
                              stop_loss_pct: float = None,
                              peak_price: float = None,
                              hold_days: int = 0) -> Dict[str, bool]:
        """
        Check exit conditions for an open position using v5 parameters.
        Delegates to check_exit() — single source of truth for exit logic.
        """
        return self.check_exit(
            data=df,
            entry_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            peak_price=peak_price,
            hold_days=hold_days
        )
    
    def calculate_position_size(self, capital: float,
                                entry_price: float,
                                volatility_20d: float,
                                confidence: float = 0.50) -> int:
        """
        Phase 12 v5 inverse-volatility position sizing.
        size = (2% risk / annualized_vol) * conf_mult
        Floor: 5% of capital | Cap: 10% of capital
        """
        try:
            if entry_price <= 0 or volatility_20d <= 0:
                return 0
            risk_amount    = capital * self.MAX_RISK_PER_TRADE
            position_value = risk_amount / volatility_20d
            conf_mult      = (confidence / self.CONFIDENCE_THRESHOLD) ** 2.0
            position_value = position_value * conf_mult
            # Floor: minimum 5% of portfolio
            position_value = max(position_value, capital * self.MIN_POSITION_PCT)
            # Cap: maximum 10% of portfolio
            position_value = min(position_value, capital * self.MAX_POSITION_PCT)
            return max(0, int(position_value / entry_price))
        except Exception as e:
            logger.error(f"[FAIL] Position sizing failed: {e}")
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
