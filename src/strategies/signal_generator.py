"""
NeuralTrader Signal Generator
Defines when and why to open/close trades with clear logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class TradingSignal:
    """Individual trading signal with detailed information"""
    
    def __init__(self, 
                 signal_type: SignalType,
                 price: float,
                 timestamp: pd.Timestamp,
                 confidence: float,
                 reason: str,
                 indicators: Dict[str, float],
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 position_size: float = 1.0):
        
        self.signal_type = signal_type
        self.price = price
        self.timestamp = timestamp
        self.confidence = confidence
        self.reason = reason
        self.indicators = indicators
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
    
    def __str__(self):
        return f"{self.signal_type.value} @ ${self.price:.2f} - {self.reason} (Conf: {self.confidence:.2f})"

class SignalGenerator:
    """
    Generates trading signals based on technical indicators
    Clear, explainable logic for every trade decision
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.position = None  # Current position: None, 'long', 'short'
        self.entry_price = None
        self.signals_history = []
    
    def _get_default_config(self) -> Dict:
        """Default trading configuration"""
        return {
            # Moving Average Crossover
            'ma_short_period': 20,
            'ma_long_period': 50,
            'ma_threshold': 0.001,  # 0.1% threshold
            
            # RSI
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_low': 40,
            'rsi_neutral_high': 60,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_threshold': 0.02,  # 2% threshold
            
            # Support/Resistance
            'support_resistance_period': 20,
            'support_threshold': 0.005,  # 0.5% threshold
            'resistance_threshold': 0.005,
            
            # Volume
            'volume_ma_period': 20,
            'volume_threshold': 1.5,  # 1.5x average volume
            
            # Risk Management
            'stop_loss_pct': 0.02,  # 2% stop loss
            'take_profit_pct': 0.06,  # 6% take profit
            'max_position_size': 1.0,
            
            # Trend Confirmation
            'trend_period': 10,
            'trend_threshold': 0.002,  # 0.2% trend threshold
        }
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate all trading signals for the given data
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        for i in range(len(df)):
            try:
                # Get current data
                current_data = df.iloc[i]
                current_price = current_data['close']
                timestamp = df.index[i]
                
                # Get indicators
                indicators = self._get_indicators(df, i)
                
                # Generate signal
                signal = self._analyze_candle(current_data, indicators, timestamp)
                
                if signal:
                    signals.append(signal)
                    self.signals_history.append(signal)
                    
            except Exception as e:
                print(f"⚠️ Error generating signal at index {i}: {e}")
                continue
        
        return signals
    
    def _get_indicators(self, df: pd.DataFrame, index: int) -> Dict[str, float]:
        """Extract all relevant indicators for current candle"""
        indicators = {}
        
        current = df.iloc[index]
        
        # Moving Averages
        if 'sma_20' in df.columns:
            indicators['sma_20'] = current['sma_20']
        if 'sma_50' in df.columns:
            indicators['sma_50'] = current['sma_50']
        
        # RSI
        if 'rsi' in df.columns:
            indicators['rsi'] = current['rsi']
        
        # Bollinger Bands
        if 'bb_upper' in df.columns:
            indicators['bb_upper'] = current['bb_upper']
        if 'bb_lower' in df.columns:
            indicators['bb_lower'] = current['bb_lower']
        if 'bb_middle' in df.columns:
            indicators['bb_middle'] = current['bb_middle']
        
        # Volume
        if 'volume' in df.columns and 'volume_sma' in df.columns:
            indicators['volume'] = current['volume']
            indicators['volume_sma'] = current['volume_sma']
        
        # Support/Resistance
        if 'support_level' in df.columns:
            indicators['support'] = current['support_level']
        if 'resistance_level' in df.columns:
            indicators['resistance'] = current['resistance_level']
        
        # Returns
        if 'returns' in df.columns:
            indicators['returns'] = current['returns']
        if 'returns_5d' in df.columns:
            indicators['returns_5d'] = current['returns_5d']
        
        return indicators
    
    def _analyze_candle(self, current_data: pd.Series, indicators: Dict[str, float], timestamp: pd.Timestamp) -> Optional[TradingSignal]:
        """Analyze current candle and generate trading signal"""
        
        current_price = current_data['close']
        
        # Skip if insufficient data
        if not indicators:
            return None
        
        # Initialize signal variables
        signal_type = SignalType.HOLD
        confidence = 0.0
        reason = "No clear signal"
        stop_loss = None
        take_profit = None
        
        # 1. MOVING AVERAGE CROSSOVER STRATEGY
        ma_signal, ma_confidence, ma_reason = self._check_ma_crossover(indicators, current_price)
        if ma_confidence > confidence:
            signal_type = ma_signal
            confidence = ma_confidence
            reason = ma_reason
        
        # 2. RSI STRATEGY
        rsi_signal, rsi_confidence, rsi_reason = self._check_rsi_signals(indicators, current_price)
        if rsi_confidence > confidence:
            signal_type = rsi_signal
            confidence = rsi_confidence
            reason = rsi_reason
        
        # 3. BOLLINGER BANDS STRATEGY
        bb_signal, bb_confidence, bb_reason = self._check_bollinger_bands(indicators, current_price)
        if bb_confidence > confidence:
            signal_type = bb_signal
            confidence = bb_confidence
            reason = bb_reason
        
        # 4. SUPPORT/RESISTANCE STRATEGY
        sr_signal, sr_confidence, sr_reason = self._check_support_resistance(indicators, current_price)
        if sr_confidence > confidence:
            signal_type = sr_signal
            confidence = sr_confidence
            reason = sr_reason
        
        # 5. VOLUME CONFIRMATION
        volume_confidence = self._check_volume_confirmation(indicators)
        confidence *= volume_confidence
        
        # 6. POSITION MANAGEMENT
        if self.position:
            # Check for exit signals
            exit_signal, exit_confidence, exit_reason = self._check_exit_conditions(indicators, current_price)
            if exit_confidence > 0.7:  # High confidence exit
                signal_type = exit_signal
                confidence = exit_confidence
                reason = f"Exit: {exit_reason}"
        
        # Generate final signal if confidence is high enough
        if confidence > 0.3 and signal_type != SignalType.HOLD:
            # Calculate stop loss and take profit
            if signal_type in [SignalType.BUY, SignalType.SELL]:
                stop_loss = self._calculate_stop_loss(current_price, signal_type)
                take_profit = self._calculate_take_profit(current_price, signal_type)
            
            return TradingSignal(
                signal_type=signal_type,
                price=current_price,
                timestamp=timestamp,
                confidence=confidence,
                reason=reason,
                indicators=indicators,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(confidence)
            )
        
        return None
    
    def _check_ma_crossover(self, indicators: Dict[str, float], current_price: float) -> Tuple[SignalType, float, str]:
        """Check moving average crossover signals"""
        
        if 'sma_20' not in indicators or 'sma_50' not in indicators:
            return SignalType.HOLD, 0.0, "MA data not available"
        
        sma_short = indicators['sma_20']
        sma_long = indicators['sma_50']
        
        # Golden Cross: Short MA crosses above Long MA
        if sma_short > sma_long:
            distance = (sma_short - sma_long) / sma_long
            if distance > self.config['ma_threshold']:
                return SignalType.BUY, 0.7, f"Golden Cross: SMA20 ({sma_short:.2f}) > SMA50 ({sma_long:.2f})"
        
        # Death Cross: Short MA crosses below Long MA
        elif sma_short < sma_long:
            distance = (sma_long - sma_short) / sma_long
            if distance > self.config['ma_threshold']:
                return SignalType.SELL, 0.7, f"Death Cross: SMA20 ({sma_short:.2f}) < SMA50 ({sma_long:.2f})"
        
        return SignalType.HOLD, 0.0, "No MA crossover"
    
    def _check_rsi_signals(self, indicators: Dict[str, float], current_price: float) -> Tuple[SignalType, float, str]:
        """Check RSI-based signals"""
        
        if 'rsi' not in indicators:
            return SignalType.HOLD, 0.0, "RSI data not available"
        
        rsi = indicators['rsi']
        
        # Oversold: RSI below 30
        if rsi < self.config['rsi_oversold']:
            confidence = (self.config['rsi_oversold'] - rsi) / self.config['rsi_oversold']
            return SignalType.BUY, confidence, f"RSI Oversold: {rsi:.1f} < {self.config['rsi_oversold']}"
        
        # Overbought: RSI above 70
        elif rsi > self.config['rsi_overbought']:
            confidence = (rsi - self.config['rsi_overbought']) / (100 - self.config['rsi_overbought'])
            return SignalType.SELL, confidence, f"RSI Overbought: {rsi:.1f} > {self.config['rsi_overbought']}"
        
        # Neutral but trending
        elif self.config['rsi_neutral_low'] <= rsi <= self.config['rsi_neutral_high']:
            if 'returns' in indicators:
                trend = indicators['returns']
                if trend > self.config['trend_threshold']:
                    return SignalType.BUY, 0.3, f"RSI Neutral ({rsi:.1f}) with uptrend"
                elif trend < -self.config['trend_threshold']:
                    return SignalType.SELL, 0.3, f"RSI Neutral ({rsi:.1f}) with downtrend"
        
        return SignalType.HOLD, 0.0, "RSI Neutral"
    
    def _check_bollinger_bands(self, indicators: Dict[str, float], current_price: float) -> Tuple[SignalType, float, str]:
        """Check Bollinger Bands signals"""
        
        if not all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            return SignalType.HOLD, 0.0, "Bollinger Bands data not available"
        
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        
        bb_width = bb_upper - bb_lower
        
        # Price above upper band (overbought)
        if current_price > bb_upper:
            distance = (current_price - bb_upper) / bb_width
            if distance > self.config['bb_threshold']:
                return SignalType.SELL, 0.6, f"Price above BB Upper: ${current_price:.2f} > ${bb_upper:.2f}"
        
        # Price below lower band (oversold)
        elif current_price < bb_lower:
            distance = (bb_lower - current_price) / bb_width
            if distance > self.config['bb_threshold']:
                return SignalType.BUY, 0.6, f"Price below BB Lower: ${current_price:.2f} < ${bb_lower:.2f}"
        
        # Price near middle band (potential reversal)
        elif abs(current_price - bb_middle) / bb_width < 0.1:
            if 'returns' in indicators:
                trend = indicators['returns']
                if trend > self.config['trend_threshold']:
                    return SignalType.BUY, 0.4, f"Price near BB Middle with uptrend"
                elif trend < -self.config['trend_threshold']:
                    return SignalType.SELL, 0.4, f"Price near BB Middle with downtrend"
        
        return SignalType.HOLD, 0.0, "Price within Bollinger Bands"
    
    def _check_support_resistance(self, indicators: Dict[str, float], current_price: float) -> Tuple[SignalType, float, str]:
        """Check support and resistance levels"""
        
        if 'support' not in indicators or 'resistance' not in indicators:
            return SignalType.HOLD, 0.0, "Support/Resistance data not available"
        
        support = indicators['support']
        resistance = indicators['resistance']
        
        # Near resistance (potential sell)
        if resistance > 0:
            distance_to_resistance = (resistance - current_price) / resistance
            if distance_to_resistance < self.config['resistance_threshold']:
                confidence = 1 - distance_to_resistance / self.config['resistance_threshold']
                return SignalType.SELL, confidence, f"Near Resistance: ${current_price:.2f} ~ ${resistance:.2f}"
        
        # Near support (potential buy)
        if support > 0:
            distance_to_support = (current_price - support) / support
            if distance_to_support < self.config['support_threshold']:
                confidence = 1 - distance_to_support / self.config['support_threshold']
                return SignalType.BUY, confidence, f"Near Support: ${current_price:.2f} ~ ${support:.2f}"
        
        return SignalType.HOLD, 0.0, "Price between support and resistance"
    
    def _check_volume_confirmation(self, indicators: Dict[str, float]) -> float:
        """Check if volume confirms the signal"""
        
        if 'volume' not in indicators or 'volume_sma' not in indicators:
            return 0.8  # Default confidence if no volume data
        
        volume = indicators['volume']
        volume_sma = indicators['volume_sma']
        
        if volume_sma > 0:
            volume_ratio = volume / volume_sma
            if volume_ratio >= self.config['volume_threshold']:
                return 1.0  # Strong volume confirmation
            elif volume_ratio >= 1.0:
                return 0.9  # Moderate volume confirmation
            else:
                return 0.7  # Weak volume confirmation
        
        return 0.8
    
    def _check_exit_conditions(self, indicators: Dict[str, float], current_price: float) -> Tuple[SignalType, float, str]:
        """Check for exit conditions when in a position"""
        
        if self.position is None:
            return SignalType.HOLD, 0.0, "No position to exit"
        
        entry_price = self.entry_price
        if entry_price is None:
            return SignalType.HOLD, 0.0, "No entry price recorded"
        
        # Calculate P&L
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Stop Loss
        if self.position == 'long' and pnl_pct <= -self.config['stop_loss_pct']:
            return SignalType.CLOSE_LONG, 0.9, f"Stop Loss: {pnl_pct:.2%}"
        elif self.position == 'short' and pnl_pct >= self.config['stop_loss_pct']:
            return SignalType.CLOSE_SHORT, 0.9, f"Stop Loss: {pnl_pct:.2%}"
        
        # Take Profit
        if self.position == 'long' and pnl_pct >= self.config['take_profit_pct']:
            return SignalType.CLOSE_LONG, 0.8, f"Take Profit: {pnl_pct:.2%}"
        elif self.position == 'short' and pnl_pct <= -self.config['take_profit_pct']:
            return SignalType.CLOSE_SHORT, 0.8, f"Take Profit: {pnl_pct:.2%}"
        
        # Reversal signals
        if self.position == 'long':
            # Check for bearish signals
            if 'rsi' in indicators and indicators['rsi'] > self.config['rsi_overbought']:
                return SignalType.CLOSE_LONG, 0.6, f"RSI Overbought: {indicators['rsi']:.1f}"
            elif 'returns' in indicators and indicators['returns'] < -self.config['trend_threshold']:
                return SignalType.CLOSE_LONG, 0.5, f"Downtrend detected: {indicators['returns']:.4f}"
        
        elif self.position == 'short':
            # Check for bullish signals
            if 'rsi' in indicators and indicators['rsi'] < self.config['rsi_oversold']:
                return SignalType.CLOSE_SHORT, 0.6, f"RSI Oversold: {indicators['rsi']:.1f}"
            elif 'returns' in indicators and indicators['returns'] > self.config['trend_threshold']:
                return SignalType.CLOSE_SHORT, 0.5, f"Uptrend detected: {indicators['returns']:.4f}"
        
        return SignalType.HOLD, 0.0, "No exit signal"
    
    def _calculate_stop_loss(self, current_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss price"""
        if signal_type == SignalType.BUY:
            return current_price * (1 - self.config['stop_loss_pct'])
        elif signal_type == SignalType.SELL:
            return current_price * (1 + self.config['stop_loss_pct'])
        return None
    
    def _calculate_take_profit(self, current_price: float, signal_type: SignalType) -> float:
        """Calculate take profit price"""
        if signal_type == SignalType.BUY:
            return current_price * (1 + self.config['take_profit_pct'])
        elif signal_type == SignalType.SELL:
            return current_price * (1 - self.config['take_profit_pct'])
        return None
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        return min(confidence * self.config['max_position_size'], self.config['max_position_size'])
    
    def update_position(self, signal: TradingSignal):
        """Update position based on signal"""
        if signal.signal_type in [SignalType.BUY]:
            self.position = 'long'
            self.entry_price = signal.price
        elif signal.signal_type in [SignalType.SELL]:
            self.position = 'short'
            self.entry_price = signal.price
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            self.position = None
            self.entry_price = None
    
    def get_signals_summary(self) -> pd.DataFrame:
        """Get summary of all generated signals"""
        if not self.signals_history:
            return pd.DataFrame()
        
        data = []
        for signal in self.signals_history:
            data.append({
                'timestamp': signal.timestamp,
                'signal': signal.signal_type.value,
                'price': signal.price,
                'confidence': signal.confidence,
                'reason': signal.reason,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size
            })
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the signal generator
    print("🧪 Testing Signal Generator")
    print("=" * 40)
    
    generator = SignalGenerator()
    print(f"✅ Signal Generator initialized with config:")
    for key, value in generator.config.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Signal Generator ready for Phase 4!")
