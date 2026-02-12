"""
NeuralTrader Golden Shield - Advanced Market Regime Filter
========================================================

Implements the Hybrid Regime Filter to protect against -85% drawdowns.
Combines institutional trend filter with momentum filter and VIX kill-switch.

Features:
- Dual-SMA Filter (200-day trend + 20-day momentum)
- VIX Kill-Switch (volatility spike detection)
- AI Breadth Gate (ensemble confidence threshold)
- Cash preservation during market stress
- Tiingo data integration for VIX/VXX

Usage:
    from src.trading.golden_shield import GoldenShield
    
    shield = GoldenShield()
    decision = shield.evaluate_market_conditions(market_data, ai_signals)
"""

import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tiingo_loader import TiingoDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoldenShield:
    """
    Golden Shield - Advanced Market Regime Filter
    Protects portfolio from severe drawdowns through multi-layer filtering
    """
    
    def __init__(self):
        """Initialize Golden Shield with protection parameters"""
        
        # Dual-SMA Filter Parameters
        self.trend_sma_period = 200  # Institutional trend filter
        self.momentum_sma_period = 20  # Speed/momentum filter
        self.sma_tolerance = 0.015  # 1.5% tolerance for re-entry
        
        # VIX Kill-Switch Parameters (Relaxed)
        self.vix_threshold = 32.0  # VIX level threshold (relaxed from 28)
        self.vix_spike_threshold = 0.20  # 20% daily spike threshold
        
        # AI Breadth Gate Parameters (Relaxed)
        self.ai_confidence_threshold = 0.49  # Minimum average confidence (relaxed from 0.52)
        self.top_signals_count = 20  # Top signals to average
        
        # Data loader
        self.data_loader = TiingoDataLoader(cache_dir=str(PROJECT_ROOT / 'data' / 'cache' / 'tiingo'))
        
        # Cache for market data
        self.market_cache = {}
        self.cache_expiry = timedelta(hours=1)
        
        logger.info("Golden Shield initialized")
        logger.info(f"Trend SMA: {self.trend_sma_period} days")
        logger.info(f"Momentum SMA: {self.momentum_sma_period} days")
        logger.info(f"SMA Tolerance: {self.sma_tolerance:.1%}")
        logger.info(f"VIX Threshold: {self.vix_threshold} (relaxed)")
        logger.info(f"VIX Spike Threshold: {self.vix_spike_threshold:.1%}")
        logger.info(f"AI Confidence Threshold: {self.ai_confidence_threshold} (relaxed)")
    
    def evaluate_market_conditions(self, market_data: Dict, ai_signals: Dict) -> Dict:
        """
        Evaluate market conditions using all three filters
        
        Args:
            market_data: Dictionary with market data (SPY, VIX, etc.)
            ai_signals: Dictionary with AI ensemble signals
            
        Returns:
            Dictionary with filter results and trading decision
        """
        try:
            logger.info("🛡️ Golden Shield: Evaluating market conditions")
            
            results = {
                'timestamp': datetime.now(),
                'trading_allowed': False,
                'filters': {},
                'market_state': 'UNKNOWN',
                'protection_level': 'HIGH'
            }
            
            # 1. Dual-SMA Filter
            sma_decision = self._evaluate_dual_sma_filter(market_data)
            results['filters']['dual_sma'] = sma_decision
            
            # 2. VIX Kill-Switch
            vix_decision = self._evaluate_vix_kill_switch(market_data)
            results['filters']['vix_kill_switch'] = vix_decision
            
            # 3. AI Breadth Gate
            ai_decision = self._evaluate_ai_breadth_gate(ai_signals)
            results['filters']['ai_breadth_gate'] = ai_decision
            
            # 4. Combined Decision
            results['trading_allowed'] = self._make_final_decision(sma_decision, vix_decision, ai_decision)
            
            # 5. Market State Assessment
            results['market_state'] = self._assess_market_state(results)
            results['protection_level'] = self._calculate_protection_level(results)
            
            # 6. Log Results
            self._log_evaluation_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating market conditions: {e}")
            return {
                'timestamp': datetime.now(),
                'trading_allowed': False,
                'filters': {'error': str(e)},
                'market_state': 'ERROR',
                'protection_level': 'MAXIMUM'
            }
    
    def _evaluate_dual_sma_filter(self, market_data: Dict) -> Dict:
        """Evaluate Dual-SMA Filter (Trend + Momentum)"""
        try:
            logger.info("📊 Evaluating Dual-SMA Filter")
            
            # Get SPY data
            spy_data = self._get_market_data('SPY')
            if spy_data is None or len(spy_data) < 200:
                return {
                    'allowed': False,
                    'reason': 'Insufficient SPY data',
                    'trend_signal': None,
                    'momentum_signal': None
                }
            
            # Calculate SMAs
            if 'price' in spy_data.columns:
                spy_data['sma_20'] = spy_data['price'].rolling(window=self.momentum_sma_period, min_periods=1).mean()
                spy_data['sma_200'] = spy_data['price'].rolling(window=self.trend_sma_period, min_periods=1).mean()
            elif 'close' in spy_data.columns:
                spy_data['sma_20'] = spy_data['close'].rolling(window=self.momentum_sma_period, min_periods=1).mean()
                spy_data['sma_200'] = spy_data['close'].rolling(window=self.trend_sma_period, min_periods=1).mean()
            else:
                raise ValueError("No price column found in SPY data")
            
            # Get latest values
            if 'price' in spy_data.columns:
                latest_price = spy_data['price'].iloc[-1]
                latest_sma_20 = spy_data['sma_20'].iloc[-1]
                latest_sma_200 = spy_data['sma_200'].iloc[-1]
            else:
                latest_price = spy_data['close'].iloc[-1]
                latest_sma_20 = spy_data['sma_20'].iloc[-1]
                latest_sma_200 = spy_data['sma_200'].iloc[-1]
            
            # Generate signals
            trend_signal = latest_price > latest_sma_200
            
            # Momentum signal with tolerance for re-entry
            momentum_distance = (latest_price - latest_sma_20) / latest_sma_20
            momentum_signal = latest_price > latest_sma_20 or abs(momentum_distance) <= self.sma_tolerance
            
            # Both must be True for trading (but momentum has tolerance)
            allowed = trend_signal and momentum_signal
            
            result = {
                'allowed': allowed,
                'reason': self._get_sma_filter_reason(trend_signal, momentum_signal),
                'trend_signal': trend_signal,
                'momentum_signal': momentum_signal,
                'latest_price': latest_price,
                'sma_200': latest_sma_200,
                'sma_20': latest_sma_20
            }
            
            logger.info(f"📊 Dual-SMA: Trend={trend_signal}, Momentum={momentum_signal}, Allowed={allowed}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Dual-SMA filter: {e}")
            return {
                'allowed': False,
                'reason': f'Dual-SMA filter error: {str(e)}',
                'trend_signal': None,
                'momentum_signal': None
            }
    
    def _evaluate_vix_kill_switch(self, market_data: Dict) -> Dict:
        """Evaluate VIX Kill-Switch"""
        try:
            logger.info("⚡ Evaluating VIX Kill-Switch")
            
            # Try VIX first, then VXX as fallback
            vix_data = self._get_market_data('VIX')
            if vix_data is None:
                vix_data = self._get_market_data('VXX')
            
            if vix_data is None or len(vix_data) < 2:
                return {
                    'allowed': False,
                    'reason': 'Insufficient VIX/VXX data',
                    'vix_level': None,
                    'vix_spike': None
                }
            
            # Get latest VIX level
            if 'vix_value' in vix_data.columns:
                latest_vix = vix_data['vix_value'].iloc[-1]
            elif 'close' in vix_data.columns:
                latest_vix = vix_data['close'].iloc[-1]
            else:
                raise ValueError("No VIX column found in VIX data")
            
            # Calculate daily spike
            if len(vix_data) >= 2:
                if 'vix_1d_change' in vix_data.columns:
                    vix_spike = vix_data['vix_1d_change'].iloc[-1]
                else:
                    # Calculate from price changes
                    if 'vix_value' in vix_data.columns:
                        prev_vix = vix_data['vix_value'].iloc[-2]
                    else:
                        prev_vix = vix_data['close'].iloc[-2]
                    vix_spike = (latest_vix - prev_vix) / prev_vix if prev_vix > 0 else 0
            else:
                vix_spike = 0
            
            # Check kill-switch conditions
            vix_triggered = latest_vix > self.vix_threshold
            spike_triggered = abs(vix_spike) > self.vix_spike_threshold
            
            allowed = not (vix_triggered or spike_triggered)
            
            result = {
                'allowed': allowed,
                'reason': self._get_vix_kill_switch_reason(vix_triggered, spike_triggered, latest_vix, vix_spike),
                'vix_level': latest_vix,
                'vix_spike': vix_spike,
                'vix_threshold_triggered': vix_triggered,
                'vix_spike_triggered': spike_triggered
            }
            
            logger.info(f"⚡ VIX Kill-Switch: Level={latest_vix:.1f}, Spike={vix_spike:.1%}, Allowed={allowed}")
            return result
            
        except Exception as e:
            logger.error(f"Error in VIX Kill-Switch: {e}")
            return {
                'allowed': False,
                'reason': f'VIX Kill-Switch error: {str(e)}',
                'vix_level': None,
                'vix_spike': None
            }
    
    def _evaluate_ai_breadth_gate(self, ai_signals: Dict) -> Dict:
        """Evaluate AI Breadth Gate"""
        try:
            logger.info("🧠 Evaluating AI Breadth Gate")
            
            if not ai_signals:
                return {
                    'allowed': False,
                    'reason': 'No AI signals available',
                    'avg_confidence': 0.0,
                    'signal_count': 0
                }
            
            # Extract confidence scores from AI signals
            confidences = []
            for ticker, signal_data in ai_signals.items():
                if isinstance(signal_data, dict) and 'confidence' in signal_data:
                    confidences.append(signal_data['confidence'])
                elif isinstance(signal_data, (int, float)):
                    confidences.append(float(signal_data))
            
            if not confidences:
                return {
                    'allowed': False,
                    'reason': 'No confidence scores in AI signals',
                    'avg_confidence': 0.0,
                    'signal_count': 0
                }
            
            # Get top signals
            confidences.sort(reverse=True)
            top_confidences = confidences[:self.top_signals_count]
            
            # Calculate average confidence
            avg_confidence = np.mean(top_confidences) if top_confidences else 0.0
            
            # Check threshold
            allowed = avg_confidence >= self.ai_confidence_threshold
            
            result = {
                'allowed': allowed,
                'reason': self._get_ai_breadth_reason(avg_confidence, allowed),
                'avg_confidence': avg_confidence,
                'signal_count': len(confidences),
                'top_signals_count': len(top_confidences),
                'threshold': self.ai_confidence_threshold
            }
            
            logger.info(f"🧠 AI Breadth Gate: Avg Confidence={avg_confidence:.3f}, Allowed={allowed}")
            return result
            
        except Exception as e:
            logger.error(f"Error in AI Breadth Gate: {e}")
            return {
                'allowed': False,
                'reason': f'AI Breadth Gate error: {str(e)}',
                'avg_confidence': 0.0,
                'signal_count': 0
            }
    
    def _make_final_decision(self, sma_decision: Dict, vix_decision: Dict, ai_decision: Dict) -> bool:
        """Make final trading decision based on all filters"""
        
        # All filters must allow trading
        filters_allowed = [
            sma_decision.get('allowed', False),
            vix_decision.get('allowed', False),
            ai_decision.get('allowed', False)
        ]
        
        # Log individual filter status
        logger.info(f"🛡️ Filter Status: SMA={filters_allowed[0]}, VIX={filters_allowed[1]}, AI={filters_allowed[2]}")
        
        # Final decision - all must be True
        final_decision = all(filters_allowed)
        
        if final_decision:
            logger.info("🛡️ Golden Shield: ✅ TRADING ALLOWED (All filters passed)")
        else:
            failed_filters = []
            if not filters_allowed[0]:
                failed_filters.append("Dual-SMA")
            if not filters_allowed[1]:
                failed_filters.append("VIX Kill-Switch")
            if not filters_allowed[2]:
                failed_filters.append("AI Breadth Gate")
            
            logger.warning(f"🛡️ Golden Shield: ❌ TRADING BLOCKED (Failed: {', '.join(failed_filters)})")
        
        return final_decision
    
    def _assess_market_state(self, results: Dict) -> str:
        """Assess overall market state"""
        try:
            filters = results['filters']
            
            # Check for danger signals
            vix_triggered = filters.get('vix_kill_switch', {}).get('vix_threshold_triggered', False)
            vix_spike = filters.get('vix_kill_switch', {}).get('vix_spike_triggered', False)
            
            if vix_triggered:
                return "CRISIS - HIGH VOLATILITY"
            elif vix_spike:
                return "STRESS - VOLATILITY SPIKE"
            elif not filters.get('dual_sma', {}).get('allowed', False):
                return "BEAR MARKET"
            elif not filters.get('ai_breadth_gate', {}).get('allowed', False):
                return "UNCERTAIN - LOW AI CONFIDENCE"
            elif results['trading_allowed']:
                return "BULL MARKET - ALL CLEAR"
            else:
                return "NEUTRAL - CAUTION ADVISED"
                
        except Exception as e:
            logger.error(f"Error assessing market state: {e}")
            return "UNKNOWN"
    
    def _calculate_protection_level(self, results: Dict) -> str:
        """Calculate protection level based on filter results"""
        try:
            if not results['trading_allowed']:
                return "MAXIMUM"
            
            filters = results['filters']
            
            # Check individual filter strength
            sma_strength = self._calculate_sma_strength(filters.get('dual_sma', {}))
            vix_risk = self._calculate_vix_risk(filters.get('vix_kill_switch', {}))
            ai_strength = filters.get('ai_breadth_gate', {}).get('avg_confidence', 0)
            
            # Overall protection assessment
            if vix_risk > 0.7:
                return "HIGH"
            elif sma_strength < 0.3 or ai_strength < 0.6:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"Error calculating protection level: {e}")
            return "UNKNOWN"
    
    def _get_market_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get market data for ticker from market_filters.json"""
        try:
            # Check cache first
            cache_key = f"{ticker}_market_data"
            current_time = datetime.now()
            
            if cache_key in self.market_cache:
                cached_data, cached_time = self.market_cache[cache_key]
                if current_time - cached_time < self.cache_expiry:
                    logger.debug(f"Using cached data for {ticker}")
                    return cached_data
            
            # Try to load from market_filters.json first
            market_filters_file = PROJECT_ROOT / 'data' / 'market_filters.json'
            if market_filters_file.exists():
                logger.debug(f"Loading {ticker} from market_filters.json")
                
                with open(market_filters_file, 'r') as f:
                    data = json.load(f)
                
                if ticker in data:
                    records = data[ticker]
                    df = pd.DataFrame(records)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Cache the data
                    self.market_cache[cache_key] = (df, current_time)
                    logger.info(f"✅ Loaded {ticker} from market_filters.json: {len(df)} rows")
                    return df
            
            # Fallback to Tiingo cache
            logger.debug(f"Loading {ticker} from Tiingo cache")
            data = self.data_loader.load_ticker_data(ticker, '2000-01-01', datetime.now().strftime('%Y-%m-%d'))
            
            if data is not None and len(data) > 0:
                # Cache the data
                self.market_cache[cache_key] = (data, current_time)
                logger.info(f"✅ Loaded {ticker} from Tiingo cache: {len(data)} rows")
                return data
            else:
                logger.warning(f"No data available for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {e}")
            return None
    
    def _get_sma_filter_reason(self, trend_signal: bool, momentum_signal: bool) -> str:
        """Get reason for SMA filter decision"""
        if not trend_signal and not momentum_signal:
            return "Both trend and momentum filters failed (BEAR market)"
        elif not trend_signal:
            return "Trend filter failed (below 200-day SMA)"
        elif not momentum_signal:
            return "Momentum filter failed (below 20-day SMA - outside tolerance)"
        else:
            return f"Both filters passed (BULL market) - Trend: {trend_signal}, Momentum: {momentum_signal}"
    
    def _get_vix_kill_switch_reason(self, vix_triggered: bool, spike_triggered: bool, 
                                       vix_level: float, vix_spike: float) -> str:
        """Get reason for VIX kill-switch decision"""
        if vix_triggered and spike_triggered:
            return f"VIX level ({vix_level:.1f}) AND spike ({vix_spike:.1%}) exceeded thresholds"
        elif vix_triggered:
            return f"VIX level ({vix_level:.1f}) exceeded threshold ({self.vix_threshold})"
        elif spike_triggered:
            return f"VIX spike ({vix_spike:.1%}) exceeded threshold ({self.vix_spike_threshold:.1%})"
        else:
            return "VIX levels normal"
    
    def _get_ai_breadth_reason(self, avg_confidence: float, allowed: bool) -> str:
        """Get reason for AI breadth gate decision"""
        if allowed:
            return f"AI confidence ({avg_confidence:.3f}) above threshold ({self.ai_confidence_threshold})"
        else:
            return f"AI confidence ({avg_confidence:.3f}) below threshold ({self.ai_confidence_threshold})"
    
    def _calculate_sma_strength(self, sma_decision: Dict) -> float:
        """Calculate SMA filter strength (0-1)"""
        try:
            if not sma_decision.get('allowed', False):
                return 0.0
            
            # Calculate distance from SMAs
            price = sma_decision.get('latest_price', 0)
            sma_200 = sma_decision.get('sma_200', 0)
            sma_20 = sma_decision.get('sma_20', 0)
            
            if sma_200 > 0 and sma_20 > 0:
                strength_200 = (price / sma_200 - 1)  # Distance from 200-day SMA
                strength_20 = (price / sma_20 - 1)   # Distance from 20-day SMA
                return min(1.0, (strength_200 + strength_20) / 0.2)  # Normalize to 0-1
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_vix_risk(self, vix_decision: Dict) -> float:
        """Calculate VIX risk level (0-1)"""
        try:
            vix_level = vix_decision.get('vix_level', 0)
            vix_spike = abs(vix_decision.get('vix_spike', 0))
            
            # Risk based on VIX level and spike
            level_risk = min(1.0, vix_level / 40.0)  # Normalize VIX level (40 is very high)
            spike_risk = min(1.0, vix_spike / 0.5)   # Normalize spike (50% is extreme)
            
            return max(level_risk, spike_risk)
            
        except Exception:
            return 0.0
    
    def _log_evaluation_results(self, results: Dict):
        """Log evaluation results"""
        logger.info("=" * 60)
        logger.info("🛡️ GOLDEN SHIELD EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {results['timestamp']}")
        logger.info(f"Trading Allowed: {'✅ YES' if results['trading_allowed'] else '❌ NO'}")
        logger.info(f"Market State: {results['market_state']}")
        logger.info(f"Protection Level: {results['protection_level']}")
        
        logger.info("Filter Details:")
        for filter_name, filter_result in results['filters'].items():
            status = '✅ PASS' if filter_result.get('allowed', False) else '❌ BLOCK'
            reason = filter_result.get('reason', 'No reason')
            logger.info(f"  {filter_name}: {status} - {reason}")
        
        logger.info("=" * 60)

# Convenience function for quick usage
def evaluate_market_conditions(market_data: Dict = None, ai_signals: Dict = None) -> Dict:
    """
    Quick evaluation of market conditions using Golden Shield
    
    Args:
        market_data: Market data dictionary (optional)
        ai_signals: AI signals dictionary (optional)
        
    Returns:
        Golden Shield evaluation results
    """
    shield = GoldenShield()
    return shield.evaluate_market_conditions(market_data or {}, ai_signals or {})

if __name__ == "__main__":
    # Test the Golden Shield
    shield = GoldenShield()
    
    # Example usage
    market_data = {}  # Would contain SPY, VIX data
    ai_signals = {}    # Would contain ensemble predictions
    
    results = shield.evaluate_market_conditions(market_data, ai_signals)
    print(f"Trading Allowed: {results['trading_allowed']}")
    print(f"Market State: {results['market_state']}")
