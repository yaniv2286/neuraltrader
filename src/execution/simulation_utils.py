"""
NeuralTrader Simulation Utilities
=================================

Shared utilities for simulation processing used by both sequential and parallel simulators.
Avoids circular import issues.

Functions:
- evaluate_exit_conditions
- check_risk_shields  
- generate_ai_scores
- apply_risk_machine
- calculate_portfolio_value
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import required components
from core.strategy import TradingStrategy
from core.ai_models import EnsemblePredictor
from src.execution.risk_manager import RiskManager

# Import UnifiedDataManager class definition to avoid circular import
class UnifiedDataManager:
    """Unified data interface for all operational modes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_market_data(self, mode: str, target_date: Optional[str] = None, 
                        date_range: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load market data based on operational mode
        
        Args:
            mode: 'live', 'paper', 'backtest', 'simulation'
            target_date: Specific date for backtest (YYYY-MM-DD)
            date_range: Date range for simulation (YYYY-MM-DD,YYYY-MM-DD)
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        try:
            if mode in ['live', 'paper']:
                return self._load_tiingo_data()
            elif mode in ['backtest', 'simulation']:
                return self._load_parquet_data(target_date, date_range)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            self.logger.error(f"[FATAL] Data loading failed: {e}")
            raise
    
    def _load_tiingo_data(self) -> Dict[str, pd.DataFrame]:
        """Load real-time data from Tiingo API"""
        # Placeholder for Tiingo integration
        self.logger.warning("[DATA] Tiingo data loading not implemented")
        return {}
    
    def _load_parquet_data(self, target_date: Optional[str] = None, 
                          date_range: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load historical data from parquet files"""
        try:
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
            parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
            
            market_data = {}
            target_dt = None
            
            if target_date:
                target_dt = pd.to_datetime(target_date)
                self.logger.info(f"[DATA] Loading parquet data for date: {target_date}")
            elif date_range:
                start_date, end_date = date_range.split(',')
                self.logger.info(f"[DATA] Loading parquet data for range: {start_date} to {end_date}")
            
            self.logger.info(f"[DATA] Found {len(parquet_files)} parquet files")
            
            for file_path in parquet_files:
                try:
                    ticker = file_path.replace('.parquet', '')
                    file_full_path = os.path.join(data_dir, file_path)
                    
                    df = pd.read_parquet(file_full_path)
                    
                    # Ensure proper datetime index
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    # Filter by date if specified
                    if target_dt:
                        # Get data up to and including target date
                        df = df[df.index <= target_dt]
                    elif date_range:
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    if not df.empty:
                        market_data[ticker] = df
                        
                except Exception as e:
                    self.logger.error(f"[DATA] Failed to load {ticker}: {e}")
                    continue
            
            self.logger.info(f"[DATA] Successfully loaded {len(market_data)} tickers from parquet")
            return market_data
            
        except Exception as e:
            self.logger.error(f"[FATAL] Parquet data loading failed: {e}")
            raise
    
    def get_trading_dates(self, date_range: str) -> List[pd.Timestamp]:
        """Get all valid trading dates within the date range"""
        try:
            start_date, end_date = date_range.split(',')
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Get sample ticker to determine trading calendar
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
            parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
            
            if not parquet_files:
                raise ValueError("No parquet files found")
            
            # Load first file to get trading dates
            sample_file = os.path.join(data_dir, parquet_files[0])
            sample_df = pd.read_parquet(sample_file)
            
            # Ensure proper datetime index
            if 'Date' in sample_df.columns:
                sample_df['Date'] = pd.to_datetime(sample_df['Date'])
                sample_df.set_index('Date', inplace=True)
            elif 'date' in sample_df.columns:
                sample_df['date'] = pd.to_datetime(sample_df['date'])
                sample_df.set_index('date', inplace=True)
            
            # Filter by date range and get valid trading days
            trading_dates = sample_df[(sample_df.index >= start_dt) & (sample_df.index <= end_dt)].index
            
            # Filter to weekdays only (trading days)
            trading_dates = [date for date in trading_dates if date.weekday() < 5]
            
            self.logger.info(f"[DATA] Found {len(trading_dates)} trading days in range")
            return trading_dates
            
        except Exception as e:
            self.logger.error(f"[FATAL] Failed to get trading dates: {e}")
            raise
    
    def load_ticker_data_for_date(self, tickers: List[str], target_date: pd.Timestamp, 
                                 lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Lazy-load ticker data for a specific date with lookback period
        
        Args:
            tickers: List of ticker symbols to load
            target_date: Target trading date
            lookback_days: Number of days of historical data to include
            
        Returns:
            Dictionary mapping tickers to DataFrames with historical data
        """
        try:
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
            market_data = {}
            
            # Calculate date range for lookback
            start_date = target_date - pd.Timedelta(days=lookback_days)
            
            for ticker in tickers:
                try:
                    file_path = f"{ticker}.parquet"
                    file_full_path = os.path.join(data_dir, file_path)
                    
                    if not os.path.exists(file_full_path):
                        continue
                    
                    # Load only the required date range
                    df = pd.read_parquet(file_full_path)
                    
                    # Ensure proper datetime index
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    # Filter to lookback period
                    df = df[(df.index >= start_date) & (df.index <= target_date)]
                    
                    if not df.empty and target_date in df.index:
                        market_data[ticker] = df
                        
                except Exception as e:
                    self.logger.debug(f"[DATA] Failed to load {ticker} for {target_date}: {e}")
                    continue
            
            self.logger.debug(f"[DATA] Loaded {len(market_data)} tickers for {target_date}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"[FATAL] Failed to load ticker data for date {target_date}: {e}")
            raise
    
    def get_available_tickers_for_date(self, target_date: pd.Timestamp) -> List[str]:
        """Get list of tickers that have data available for the target date"""
        try:
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
            parquet_files = [f.replace('.parquet', '') for f in os.listdir(data_dir) if f.endswith('.parquet')]
            
            available_tickers = []
            
            for ticker in parquet_files:
                try:
                    file_path = os.path.join(data_dir, f"{ticker}.parquet")
                    
                    # Quick check - read just the index to see if date exists
                    df = pd.read_parquet(file_path, columns=['Date'] if 'Date' in pd.read_parquet(file_path).columns else ['date'])
                    
                    # Ensure proper datetime index
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    if target_date in df.index:
                        available_tickers.append(ticker)
                        
                except Exception:
                    continue
            
            return available_tickers
            
        except Exception as e:
            self.logger.error(f"[FATAL] Failed to get available tickers for {target_date}: {e}")
            raise

logger = logging.getLogger(__name__)

def evaluate_exit_conditions(portfolio: Dict, market_data: Dict[str, pd.DataFrame], current_date: datetime = None) -> List[Dict]:
    """Evaluate exit conditions for current positions with Dynamic Sector-Based Exits (Phase 10)"""
    logger = logging.getLogger(__name__)
    exits = []
    
    try:
        strategy = TradingStrategy()
        
        # PHASE 10: Load sector authority for dynamic exits
        try:
            from scripts.sector_rotation import SectorAuthority
            sector_auth = SectorAuthority()
            sector_ranks = sector_auth.get_sector_momentum()
            logger.info(f"[PHASE10] Sector authority loaded for dynamic exits: {len(sector_ranks)} sectors ranked")
        except Exception as e:
            logger.warning(f"[PHASE10] Sector authority not available for dynamic exits: {e}")
            sector_ranks = []
            sector_auth = None
        
        # Create sector mapping for quick lookup
        ticker_to_sector = {}
        if sector_auth and sector_ranks:
            sector_auth.load_sector_map()
            for sector, tickers in sector_auth.sector_map.items():
                for ticker in tickers:
                    ticker_to_sector[ticker] = sector
        
        for ticker, position in portfolio.get('positions', {}).items():
            if ticker not in market_data:
                logger.warning(f"[EXIT] No market data for {ticker}")
                continue
            
            ticker_data = market_data[ticker]
            current_price = ticker_data['close'].iloc[-1]
            
            # 1. Weekly Shield (existing logic)
            if strategy.check_exit(ticker_data):
                exits.append({
                    'ticker': ticker,
                    'reason': 'Weekly Shield',
                    'action': 'SELL',
                    'quantity': position.get('shares', 0),
                    'price': current_price
                })
                continue
            
            # 2. Emergency Stop Loss (ATR x 2.5 + 15% backup)
            try:
                entry_price = position.get('cost_basis', current_price)
                atr = strategy.calculate_atr(ticker_data)
                
                # ATR-based stop loss
                atr_stop_level = entry_price - (atr * strategy.EMERGENCY_STOP_LOSS_ATR_MULTIPLIER)
                
                # Percentage-based backup stop loss (15% max loss)
                pct_stop_level = entry_price * 0.85
                
                # Use the higher (more conservative) stop level
                stop_level = max(atr_stop_level, pct_stop_level)
                
                if current_price <= stop_level:
                    stop_type = "ATR" if current_price <= atr_stop_level else "Percentage"
                    exits.append({
                        'ticker': ticker,
                        'reason': f'Emergency Stop ({stop_type}: {strategy.EMERGENCY_STOP_LOSS_ATR_MULTIPLIER}x ATR / 15%)',
                        'action': 'SELL',
                        'quantity': position.get('shares', 0),
                        'price': current_price
                    })
                    continue
            except Exception as e:
                logger.warning(f"[EXIT] Stop loss check failed for {ticker}: {e}")
            
            # 3. PHASE 10: Dynamic Sector-Based Exits
            if sector_auth and sector_ranks and ticker in ticker_to_sector:
                try:
                    ticker_sector = ticker_to_sector[ticker]
                    
                    # Find sector rank (lower index = stronger sector)
                    sector_rank_index = None
                    for i, (sector, roc) in enumerate(sector_ranks):
                        if sector == ticker_sector:
                            sector_rank_index = i
                            sector_roc = roc
                            break
                    
                    if sector_rank_index is not None:
                        # Dynamic exit thresholds based on sector performance
                        total_sectors = len(sector_ranks)
                        sector_percentile = (total_sectors - sector_rank_index - 1) / total_sectors
                        
                        # Calculate position performance
                        entry_price = position.get('cost_basis', current_price)
                        position_return = (current_price - entry_price) / entry_price
                        
                        logger.info(f"[PHASE10] {ticker} sector: {ticker_sector} (rank {sector_rank_index+1}/{total_sectors}, ROC: {sector_roc:.2f}%, return: {position_return:.2%})")
                        
                        # Dynamic exit rules based on sector strength
                        exit_triggered = False
                        exit_reason = ""
                        
                        # Rule 1: Weakest sectors (bottom 25%) - stricter exits
                        if sector_percentile <= 0.25:
                            # Exit on any loss or minimal gain (< 1%)
                            if position_return < 0.01:
                                exit_triggered = True
                                exit_reason = f'Dynamic Sector Exit (Weak Sector: {ticker_sector} rank {sector_rank_index+1})'
                        
                        # Rule 2: Weak sectors (bottom 50%) - moderate exits
                        elif sector_percentile <= 0.50:
                            # Exit on losses > 2% or gains < 0.5%
                            if position_return < -0.02 or position_return < 0.005:
                                exit_triggered = True
                                exit_reason = f'Dynamic Sector Exit (Weak Sector: {ticker_sector} rank {sector_rank_index+1})'
                        
                        # Rule 3: Strong sectors (top 25%) - relaxed exits
                        elif sector_percentile >= 0.75:
                            # Only exit on significant losses (> 5%)
                            if position_return < -0.05:
                                exit_triggered = True
                                exit_reason = f'Dynamic Sector Exit (Strong Sector Loss: {ticker_sector} rank {sector_rank_index+1})'
                        
                        # Rule 4: Very weak sectors with negative momentum - aggressive exits
                        if sector_roc < -2.0:  # Sector ROC < -2%
                            if position_return < 0.02:  # Any gain less than 2%
                                exit_triggered = True
                                exit_reason = f'Dynamic Sector Exit (Negative Momentum: {ticker_sector} ROC {sector_roc:.1f}%)'
                        
                        if exit_triggered:
                            exits.append({
                                'ticker': ticker,
                                'reason': exit_reason,
                                'action': 'SELL',
                                'quantity': position.get('shares', 0),
                                'price': current_price
                            })
                            logger.info(f"[PHASE10] Dynamic sector exit triggered for {ticker}: {exit_reason}")
                
                except Exception as e:
                    logger.warning(f"[PHASE10] Dynamic sector exit check failed for {ticker}: {e}")
        
        logger.info(f"[EXIT] Found {len(exits)} exit conditions")
        return exits
        
    except Exception as e:
        logger.error(f"[EXIT] Exit evaluation failed: {e}")
        return []

def check_risk_shields(market_data: Dict[str, pd.DataFrame], current_date: datetime = None) -> bool:
    """Check all risk shields before allowing new positions using historical simulation data"""
    logger = logging.getLogger(__name__)
    
    try:
        shields_passed = True
        
        # 1. VXX Volatility Gate (using historical data)
        vxx_data = market_data.get('VXX')
        if vxx_data is not None and len(vxx_data) >= 20:
            if not check_vxx_bollinger_gate(vxx_data):
                shields_passed = False
                logger.error("[SHIELD] VXX volatility gate triggered - extreme market stress")
        else:
            logger.warning("[SHIELD] VXX data not available for volatility check")
        
        # 2. Market Regime Filter (SPY trend using historical data)
        spy_data = market_data.get('SPY')
        if spy_data is not None and len(spy_data) >= 20:
            spy_ma20 = spy_data['close'].rolling(20).mean().iloc[-1]
            spy_latest = spy_data['close'].iloc[-1]
            
            if spy_latest < spy_ma20:
                shields_passed = False
                logger.error("[SHIELD] SPY below 20-day MA - bearish regime")
            else:
                logger.info("[SHIELD] SPY above 20-day MA - bullish regime")
        else:
            logger.warning("[SHIELD] SPY data not available for regime check")
        
        # 3. Sector Momentum (simplified check using available sector ETFs)
        sector_etfs = ['XLE', 'XLB', 'XLP', 'XLI', 'XLRE', 'XLU', 'XLC', 'XLV', 'XLK', 'XLF', 'XLY']
        sector_count = 0
        strong_sectors = 0
        
        for etf in sector_etfs:
            etf_data = market_data.get(etf)
            if etf_data is not None and len(etf_data) >= 20:
                sector_count += 1
                # Calculate 20-day ROC
                current_price = etf_data['close'].iloc[-1]
                price_20_days_ago = etf_data['close'].iloc[-20]
                roc = ((current_price - price_20_days_ago) / price_20_days_ago) * 100
                
                if roc > 2.0:  # Strong momentum
                    strong_sectors += 1
                
                logger.debug(f"[SHIELD] {etf}: {roc:.2f}% (20-day ROC)")
        
        if sector_count > 0:
            strong_ratio = strong_sectors / sector_count
            logger.info(f"[SHIELD] Sector momentum: {strong_sectors}/{sector_count} sectors strong ({strong_ratio:.1%})")
            
            # If too few sectors are strong, block new positions
            if strong_ratio < 0.3:  # Less than 30% strong sectors
                shields_passed = False
                logger.error("[SHIELD] Weak sector momentum - blocking new positions")
        else:
            logger.warning("[SHIELD] No sector data available for momentum check")
        
        if shields_passed:
            logger.info("[SHIELD] All risk shields passed - new positions allowed")
        else:
            logger.error("[SHIELD] Risk shields failed - no new positions allowed")
        
        return shields_passed
        
    except Exception as e:
        logger.error(f"[SHIELD] Risk shield check failed: {e}")
        return False

def check_vxx_bollinger_gate(vxx_data: Optional[pd.DataFrame]) -> bool:
    """Check VXX volatility gate using Bollinger Bands"""
    logger = logging.getLogger(__name__)
    
    if vxx_data is None or len(vxx_data) < 20:
        logger.warning("[VXX_GATE] Insufficient VXX data, passing through")
        return True
    
    try:
        # Calculate 20-day Bollinger Bands
        vxx_data = vxx_data.copy()
        vxx_data['sma_20'] = vxx_data['close'].rolling(20).mean()
        vxx_data['std_20'] = vxx_data['close'].rolling(20).std()
        vxx_data['upper_band'] = vxx_data['sma_20'] + (vxx_data['std_20'] * 2)
        vxx_data['lower_band'] = vxx_data['sma_20'] - (vxx_data['std_20'] * 2)
        
        latest_close = vxx_data['close'].iloc[-1]
        latest_upper = vxx_data['upper_band'].iloc[-1]
        
        # Gate triggers when VXX breaks above upper band (extreme volatility)
        if latest_close > latest_upper:
            logger.error(f"[VXX_GATE] Volatility spike: {latest_close:.2f} > {latest_upper:.2f}")
            return False
        
        logger.info(f"[VXX_GATE] Volatility normal: {latest_close:.2f} < {latest_upper:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"[VXX_GATE] Bollinger Band calculation failed: {e}")
        return True  # Pass through on error

def generate_ai_scores(market_data: Dict[str, pd.DataFrame], min_confidence: float = 0.60) -> Dict[str, Dict]:
    """Generate AI Council scores for all tickers"""
    logger = logging.getLogger(__name__)
    
    try:
        ensemble = EnsemblePredictor()
        scores = {}
        
        logger.info(f"[AI] Scoring {len(market_data)} tickers with min confidence {min_confidence}")
        
        for ticker, data in market_data.items():
            try:
                # Generate features and predict
                signal, confidence, details = ensemble.predict_from_ohlcv(data)
                
                # Filter by confidence threshold and BUY signal
                if confidence >= min_confidence and signal == 'BUY':
                    scores[ticker] = {
                        'signal': signal,
                        'confidence': confidence,
                        'details': details,
                        'price': data['close'].iloc[-1] if 'close' in data.columns else None
                    }
                    
            except Exception as e:
                logger.warning(f"[AI] Failed to score {ticker}: {e}")
                continue
        
        logger.info(f"[AI] Generated {len(scores)} qualified scores")
        return scores
        
    except Exception as e:
        logger.error(f"[AI] AI scoring failed: {e}")
        return {}

def calculate_real_volatilities(tickers: List[str], market_data: Dict[str, pd.DataFrame]) -> List[float]:
    """Calculate real volatilities from market data"""
    logger = logging.getLogger(__name__)
    volatilities = []
    
    for ticker in tickers:
        try:
            data = market_data.get(ticker)
            if data is None or len(data) < 20:
                volatilities.append(0.25)  # Default volatility
                continue
            
            # Calculate 20-day returns and annualize
            data['returns'] = data['close'].pct_change()
            recent_returns = data['returns'].dropna().tail(20)
            
            if len(recent_returns) > 0:
                volatility = recent_returns.std() * np.sqrt(252)  # Annualized
                volatilities.append(max(volatility, 0.01))  # Minimum 1% volatility
            else:
                volatilities.append(0.25)
                
        except Exception as e:
            logger.warning(f"[RISK] Volatility calculation failed for {ticker}: {e}")
            volatilities.append(0.25)
    
    return volatilities

def calculate_portfolio_value(portfolio: Dict, market_data: Dict[str, pd.DataFrame]) -> float:
    """Calculate current portfolio value"""
    logger = logging.getLogger(__name__)
    
    try:
        cash = portfolio.get('cash', 0.0)
        positions_value = 0.0
        
        for ticker, position in portfolio.get('positions', {}).items():
            if ticker in market_data:
                current_price = market_data[ticker]['close'].iloc[-1]
                shares = position.get('shares', 0)
                positions_value += shares * current_price
        
        return cash + positions_value
        
    except Exception as e:
        logger.error(f"[RISK] Portfolio value calculation failed: {e}")
        return portfolio.get('cash', 0.0)

def apply_risk_machine(ai_scores: Dict, portfolio: Dict, market_data: Dict[str, pd.DataFrame], current_date: datetime = None) -> Tuple[List[Dict], bool]:
    """Apply Risk Machine Phase 9.5 logic"""
    logger = logging.getLogger(__name__)
    
    try:
        risk_manager = RiskManager()
        
        # 1. Circuit Breaker Check (12% drawdown)
        current_value = calculate_portfolio_value(portfolio, market_data)
        peak_value = portfolio.get('peak_portfolio_value', 100000.0)
        
        # Use provided current_date or default to now for live/paper modes
        if current_date is None:
            current_date = datetime.now()
            
        cooldown_str = portfolio.get('circuit_breaker_cooldown_until')
        
        circuit_triggered, cooldown_until = risk_manager.check_portfolio_circuit_breaker(
            current_value=current_value,
            peak_value=peak_value,
            current_date=current_date,
            cooldown_date_str=cooldown_str
        )
        
        if circuit_triggered:
            logger.error(f"[CIRCUIT] 12% drawdown triggered - trading halted")
            return [], True  # No trades, circuit active
        
        # 2. Hysteresis Rule (15% premium for position swaps)
        current_positions = list(portfolio.get('positions', {}).keys())
        max_positions = 10  # From architecture
        
        if len(current_positions) >= max_positions and ai_scores:
            # Calculate current average score
            current_scores = []
            for ticker in current_positions:
                if ticker in ai_scores:
                    current_scores.append(ai_scores[ticker]['confidence'])
            
            if current_scores:
                current_avg_score = sum(current_scores) / len(current_scores)
                required_premium = current_avg_score * 1.15
                
                # Filter candidates that don't meet premium
                filtered_scores = {k: v for k, v in ai_scores.items() 
                                if v['confidence'] >= required_premium}
                
                logger.info(f"[HYSTERESIS] Required premium: {required_premium:.3f}, "
                          f"Current avg: {current_avg_score:.3f}, "
                          f"Filtered candidates: {len(filtered_scores)}")
                
                ai_scores = filtered_scores
        
        # 3. Infinite Loop Guard - Skip tickers already in portfolio
        ai_scores = {k: v for k, v in ai_scores.items() 
                   if k not in current_positions}
        
        if ai_scores:
            logger.info(f"[INFINITE_LOOP] Guard: {len(current_positions)} positions, "
                       f"{len(ai_scores)} new candidates")
        
        # 4. Inverse Volatility Sizing
        trades = []
        if ai_scores:
            tickers = list(ai_scores.keys())
            volatilities = calculate_real_volatilities(tickers, market_data)
            available_cash = portfolio.get('cash', 0)
            risk_capital = available_cash * 0.20  # 20% of cash for new positions
            
            allocations = risk_manager.calculate_inverse_vol_sizing(
                tickers=tickers,
                volatilities=volatilities,
                total_risk_capital=risk_capital
            )
            
            # Create trade orders
            for ticker, allocation in allocations.items():
                if allocation > 0:
                    price = ai_scores[ticker]['price']
                    if price and price > 0:
                        quantity = int(allocation / price)
                        if quantity > 0:
                            trades.append({
                                'ticker': ticker,
                                'action': 'BUY',
                                'quantity': quantity,
                                'price': price,
                                'reason': f"AI Score: {ai_scores[ticker]['confidence']:.3f}",
                                'allocation': allocation,
                                'ai_score': ai_scores[ticker]['confidence']
                            })
        
        logger.info(f"[RISK] Generated {len(trades)} trades, circuit_active: {circuit_triggered}")
        return trades, circuit_triggered
        
    except Exception as e:
        logger.error(f"[RISK] Risk machine failed: {e}")
        return [], False
