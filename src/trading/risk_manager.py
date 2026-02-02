"""
NeuralTrader Risk Manager - The Constitution
============================================

Enforces the 0.9% Risk Per Trade logic with hard caps and safety checks.
Implements the defensive constitution that protects capital.

Features:
- 0.9% risk per trade based on live account equity
- 30% Tech Sector cap enforcement
- 15% VXX Black Swan Exit protection
- Duplicate position protection
- Comprehensive risk validation

Usage:
    from src.trading.risk_manager import RiskManager
    
    rm = RiskManager()
    decision = rm.evaluate_trade('AAPL', 150.0, account_info, current_positions)
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from ib_insync import IB, Stock, Order, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskDecision:
    """Risk decision constants"""
    APPROVED = "APPROVED"
    REJECT_RISK = "REJECT_RISK"
    REJECT_SECTOR = "REJECT_SECTOR"
    REJECT_BLACK_SWAN = "REJECT_BLACK_SWAN"
    REJECT_DUPLICATE = "REJECT_DUPLICATE"
    REJECT_DATA = "REJECT_DATA"
    REJECT_CAPITAL = "REJECT_CAPITAL"

class RiskManager:
    """
    Risk Manager - The Constitution
    Enforces all risk management rules and capital protection
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 2):
        """Initialize Risk Manager with IBKR connection"""
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Risk parameters (Phase 5 optimized)
        self.risk_per_trade = 0.009  # 0.9% risk per trade
        self.max_sector_exposure = 0.30  # 30% max per sector
        self.black_swan_threshold = 0.15  # 15% VXX surge
        
        # Initialize IB connection
        self.ib = None
        self._connect_ibkr()
        
        # Sector mappings for S&P 100 universe
        self.sector_mappings = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
            'GOOG': 'Technology', 'META': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology',
            'ORCL': 'Technology', 'INTU': 'Technology', 'AMD': 'Technology', 'CSCO': 'Technology',
            'TXN': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology', 'ADI': 'Technology',
            'PYPL': 'Technology', 'FISV': 'Technology', 'NOW': 'Technology', 'ADP': 'Technology',
            'BLK': 'Technology', 'SPGI': 'Technology', 'CB': 'Technology', 'ICE': 'Technology',
            'CME': 'Technology', 'V': 'Technology', 'MA': 'Technology', 'AVGO': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
            'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
            'MDT': 'Healthcare', 'ISRG': 'Healthcare', 'GILD': 'Healthcare', 'REGN': 'Healthcare',
            'VRTX': 'Healthcare', 'ZTS': 'Healthcare', 'SYK': 'Healthcare', 'AMGN': 'Healthcare',
            'CI': 'Healthcare', 'ANTM': 'Healthcare', 'ELV': 'Healthcare', 'CVS': 'Healthcare',
            'BKNG': 'Healthcare',
            
            # Consumer Discretionary
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
            'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
            'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary', 'DIS': 'Consumer Discretionary',
            
            # Consumer Staples
            'PG': 'Consumer Staples', 'PEP': 'Consumer Staples', 'COST': 'Consumer Staples',
            'WMT': 'Consumer Staples', 'KO': 'Consumer Staples', 'CL': 'Consumer Staples',
            'MDLZ': 'Consumer Staples', 'MO': 'Consumer Staples',
            
            # Financials
            'BRK.B': 'Financials', 'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials', 'AXP': 'Financials',
            'AIG': 'Financials', 'MET': 'Financials', 'TRV': 'Financials', 'SCHW': 'Financials',
            
            # Industrials
            'GE': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials', 'RTX': 'Industrials',
            'LMT': 'Industrials', 'CAT': 'Industrials', 'DE': 'Industrials', 'MMM': 'Industrials',
            'UNP': 'Industrials', 'BA': 'Industrials', 'CSX': 'Industrials', 'NSC': 'Industrials',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
            'EOG': 'Energy', 'OXY': 'Energy', 'PSX': 'Energy', 'MPC': 'Energy',
            
            # Utilities
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'AEP': 'Utilities',
            'XEL': 'Utilities', 'SRE': 'Utilities', 'D': 'Utilities', 'PEG': 'Utilities',
            
            # Real Estate
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'EQIX': 'Real Estate', 'CCI': 'Real Estate',
            'PSA': 'Real Estate', 'EXR': 'Real Estate', 'SPG': 'Real Estate',
            
            # Materials
            'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'DD': 'Materials',
            'FCX': 'Materials', 'NEM': 'Materials', 'RIO': 'Materials', 'BHP': 'Materials',
            
            # Communication Services
            'VZ': 'Communication Services', 'T': 'Communication Services', 'CMCSA': 'Communication Services',
            'NFLX': 'Communication Services', 'DIS': 'Communication Services', 'CHTR': 'Communication Services',
            
            # Other
            'PM': 'Other', 'MO': 'Other', 'T': 'Other'
        }
        
        logger.info("Risk Manager initialized")
        logger.info(f"Risk per trade: {self.risk_per_trade:.1%}")
        logger.info(f"Max sector exposure: {self.max_sector_exposure:.1%}")
        logger.info(f"Black Swan threshold: {self.black_swan_threshold:.1%}")
    
    def _connect_ibkr(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib = IB()
            self.ib.connect(host=self.host, port=self.port, clientId=self.client_id)
            
            if self.ib.isConnected():
                logger.info("✅ Connected to Interactive Brokers")
            else:
                logger.error("❌ Failed to connect to Interactive Brokers")
                raise ConnectionError("Could not connect to IBKR")
                
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            raise
    
    def evaluate_trade(self, ticker: str, current_price: float, account_info: Dict, 
                      current_positions: List[Dict], market_data: Dict = None) -> Tuple[str, Dict]:
        """
        Evaluate a trade request against all risk rules
        
        Args:
            ticker: Ticker symbol to trade
            current_price: Current market price
            account_info: Account information from Alpaca
            current_positions: List of current positions
            market_data: Market data for calculations
            
        Returns:
            Tuple of (decision, details)
        """
        try:
            logger.info(f"Evaluating trade for {ticker} at ${current_price:.2f}")
            
            # 1. Check Black Swan condition first
            black_swan_decision = self._check_black_swan()
            if black_swan_decision != RiskDecision.APPROVED:
                return black_swan_decision, {'reason': 'Black Swan event detected'}
            
            # 2. Check for duplicate positions
            duplicate_decision = self._check_duplicate_position(ticker, current_positions)
            if duplicate_decision != RiskDecision.APPROVED:
                return duplicate_decision, {'reason': 'Position already exists'}
            
            # 3. Check capital requirements
            capital_decision = self._check_capital_requirements(current_price, account_info)
            if capital_decision != RiskDecision.APPROVED:
                return capital_decision, {'reason': 'Insufficient capital'}
            
            # 4. Check sector exposure
            sector_decision = self._check_sector_exposure(ticker, current_price, account_info, current_positions)
            if sector_decision != RiskDecision.APPROVED:
                return sector_decision, {'reason': 'Sector exposure limit exceeded'}
            
            # 5. Calculate position size
            position_size = self._calculate_position_size(ticker, current_price, account_info, market_data)
            
            # 6. Final validation
            if position_size <= 0:
                return RiskDecision.REJECT_RISK, {'reason': 'Invalid position size calculation'}
            
            # Trade approved
            details = {
                'position_size': position_size,
                'position_value': position_size * current_price,
                'risk_amount': account_info['portfolio_value'] * self.risk_per_trade,
                'sector': self.sector_mappings.get(ticker, 'Unknown'),
                'current_sector_exposure': self._get_sector_exposure(self.sector_mappings.get(ticker), current_positions, account_info)
            }
            
            logger.info(f"✅ Trade APPROVED: {ticker} - {position_size} shares @ ${current_price:.2f}")
            return RiskDecision.APPROVED, details
            
        except Exception as e:
            logger.error(f"Error evaluating trade for {ticker}: {e}")
            return RiskDecision.REJECT_DATA, {'reason': f'Evaluation error: {str(e)}'}
    
    def _check_black_swan(self) -> str:
        """
        Check for Black Swan event (VXX surge > 15%) using IBKR
        
        Returns:
            RiskDecision constant
        """
        try:
            # Ensure IBKR connection
            if not self.ib.isConnected():
                self._connect_ibkr()
            
            # Get VXX data
            vxx_contract = Stock('VXX', 'SMART', 'USD')
            
            # Calculate date range (last 5 trading days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Extra buffer for weekends
            
            # Request historical data
            vxx_bars = self.ib.reqHistoricalData(
                vxx_contract,
                start=start_date,
                end=end_date,
                barSize='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            
            if not vxx_bars or len(vxx_bars) < 5:
                logger.warning("Insufficient VXX data for Black Swan check")
                return RiskDecision.APPROVED  # Allow if can't check
            
            # Calculate 5-day return
            recent_close = vxx_bars[-1].close
            five_days_ago_close = vxx_bars[0].close
            vxx_return = (recent_close - five_days_ago_close) / five_days_ago_close
            
            logger.info(f"VXX 5-day return: {vxx_return:.2%}")
            
            if vxx_return > self.black_swan_threshold:
                logger.warning(f"🚨 BLACK SWAN EVENT: VXX surged {vxx_return:.1%} > {self.black_swan_threshold:.1%}")
                return RiskDecision.REJECT_BLACK_SWAN
            
            return RiskDecision.APPROVED
            
        except Exception as e:
            logger.error(f"Error checking Black Swan: {e}")
            return RiskDecision.APPROVED  # Allow if error in check
    
    def _check_duplicate_position(self, ticker: str, current_positions: List[Dict]) -> str:
        """
        Check if we already have a position in this ticker
        
        Args:
            ticker: Ticker symbol
            current_positions: List of current positions
            
        Returns:
            RiskDecision constant
        """
        try:
            for position in current_positions:
                if position.get('symbol') == ticker and float(position.get('qty', 0)) > 0:
                    logger.warning(f"Duplicate position check: Already own {ticker}")
                    return RiskDecision.REJECT_DUPLICATE
            
            return RiskDecision.APPROVED
            
        except Exception as e:
            logger.error(f"Error checking duplicate position: {e}")
            return RiskDecision.REJECT_DATA
    
    def _check_capital_requirements(self, current_price: float, account_info: Dict) -> str:
        """
        Check if we have sufficient capital for the trade
        
        Args:
            current_price: Current market price
            account_info: Account information
            
        Returns:
            RiskDecision constant
        """
        try:
            portfolio_value = account_info.get('portfolio_value', 0)
            buying_power = account_info.get('buying_power', 0)
            
            # Minimum position value (0.5% of portfolio)
            min_position_value = portfolio_value * 0.005
            
            if buying_power < min_position_value:
                logger.warning(f"Insufficient buying power: ${buying_power:.2f} < ${min_position_value:.2f}")
                return RiskDecision.REJECT_CAPITAL
            
            return RiskDecision.APPROVED
            
        except Exception as e:
            logger.error(f"Error checking capital requirements: {e}")
            return RiskDecision.REJECT_DATA
    
    def _check_sector_exposure(self, ticker: str, current_price: float, account_info: Dict, 
                             current_positions: List[Dict]) -> str:
        """
        Check if adding this position would exceed sector exposure limits
        
        Args:
            ticker: Ticker symbol
            current_price: Current market price
            account_info: Account information
            current_positions: List of current positions
            
        Returns:
            RiskDecision constant
        """
        try:
            sector = self.sector_mappings.get(ticker, 'Unknown')
            portfolio_value = account_info.get('portfolio_value', 0)
            
            # Calculate current sector exposure
            current_exposure = self._get_sector_exposure(sector, current_positions, account_info)
            
            # Calculate new position value (estimated)
            new_position_value = portfolio_value * self.risk_per_trade * 10  # Rough estimate
            new_exposure = current_exposure + new_position_value
            
            max_sector_value = portfolio_value * self.max_sector_exposure
            
            if new_exposure > max_sector_value:
                logger.warning(f"Sector cap exceeded: {sector} - Current: ${current_exposure:.2f}, "
                              f"New: ${new_exposure:.2f} > ${max_sector_value:.2f}")
                return RiskDecision.REJECT_SECTOR
            
            return RiskDecision.APPROVED
            
        except Exception as e:
            logger.error(f"Error checking sector exposure: {e}")
            return RiskDecision.REJECT_DATA
    
    def _get_sector_exposure(self, sector: str, current_positions: List[Dict], account_info: Dict) -> float:
        """
        Calculate current exposure to a sector
        
        Args:
            sector: Sector name
            current_positions: List of current positions
            account_info: Account information
            
        Returns:
            Current sector exposure in dollars
        """
        try:
            sector_exposure = 0.0
            
            for position in current_positions:
                pos_ticker = position.get('symbol', '')
                pos_qty = float(position.get('qty', 0))
                pos_value = float(position.get('market_value', 0))
                
                if self.sector_mappings.get(pos_ticker) == sector and pos_qty > 0:
                    sector_exposure += pos_value
            
            return sector_exposure
            
        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return 0.0
    
    def _calculate_position_size(self, ticker: str, current_price: float, account_info: Dict, 
                               market_data: Dict = None) -> int:
        """
        Calculate position size based on 0.9% risk per trade and Chandelier Exit distance
        
        Args:
            ticker: Ticker symbol
            current_price: Current market price
            account_info: Account information
            market_data: Market data for calculations
            
        Returns:
            Number of shares to trade
        """
        try:
            portfolio_value = account_info.get('portfolio_value', 0)
            risk_amount = portfolio_value * self.risk_per_trade
            
            # Get Chandelier Exit distance (prefer over ATR)
            chandelier_exit = self._calculate_chandelier_exit(ticker, market_data)
            
            if chandelier_exit > 0:
                stop_distance = abs(current_price - chandelier_exit)
                logger.info(f"Using Chandelier Exit for {ticker}: ${chandelier_exit:.2f}, distance: ${stop_distance:.2f}")
            else:
                # Fallback to ATR
                atr = self._get_atr_value(ticker, market_data)
                if atr <= 0:
                    logger.warning(f"Invalid ATR for {ticker}, using fallback calculation")
                    atr = current_price * 0.02  # 2% of price as fallback
                
                stop_distance = atr * 2.0  # 2x ATR stop
                logger.info(f"Using ATR for {ticker}: {atr:.4f}, distance: ${stop_distance:.2f}")
            
            # Calculate position size: Risk Amount / Stop Distance
            position_value = risk_amount / stop_distance
            shares = int(position_value / current_price)
            
            # Apply position bounds (0.5% to 25% of portfolio)
            min_position_value = portfolio_value * 0.005
            max_position_value = portfolio_value * 0.25
            
            if shares * current_price < min_position_value:
                shares = int(min_position_value / current_price)
            elif shares * current_price > max_position_value:
                shares = int(max_position_value / current_price)
            
            # Ensure minimum trade size
            if shares < 1:
                shares = 1
            
            logger.info(f"Position size calculation for {ticker}: {shares} shares "
                       f"(${shares * current_price:.2f} value, risk: ${risk_amount:.2f})")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _calculate_chandelier_exit(self, ticker: str, market_data: Dict = None) -> float:
        """
        Calculate Chandelier Exit price for position sizing
        
        Args:
            ticker: Ticker symbol
            market_data: Market data dictionary
            
        Returns:
            Chandelier Exit price
        """
        try:
            if market_data and ticker in market_data:
                df = market_data[ticker]
                if not df.empty and len(df) >= 22:
                    # Calculate Chandelier Exit (22-period highest high - 3x ATR)
                    highest_high = df['high'].rolling(window=22).max().iloc[-1]
                    atr = self._calculate_atr(df, 22).iloc[-1]
                    
                    chandelier_exit = highest_high - (3 * atr)
                    return chandelier_exit
            
            # Fallback: fetch from API
            bars = self.api.get_bars(
                symbol=ticker,
                timeframe=tradeapi.TimeFrame.Day,
                limit=30
            ).df
            
            if not bars.empty and len(bars) >= 22:
                highest_high = bars['high'].rolling(window=22).max().iloc[-1]
                
                # Calculate ATR
                high_low = bars['high'] - bars['low']
                high_close = abs(bars['high'] - bars['close'].shift())
                low_close = abs(bars['low'] - bars['close'].shift())
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=22).mean().iloc[-1]
                
                chandelier_exit = highest_high - (3 * atr)
                return chandelier_exit
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit for {ticker}: {e}")
            return 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR for given period"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    def suggest_alternative_sectors(self, current_sector: str, current_positions: List[Dict], 
                                   account_info: Dict) -> List[str]:
        """
        Suggest alternative sectors when current sector cap is hit
        
        Args:
            current_sector: Sector that's at capacity
            current_positions: Current positions
            account_info: Account information
            
        Returns:
            List of alternative sectors sorted by available capacity
        """
        try:
            # Calculate current sector exposures
            sector_exposures = {}
            portfolio_value = account_info.get('portfolio_value', 0)
            
            for position in current_positions:
                pos_ticker = position.get('symbol', '')
                pos_sector = self.sector_mappings.get(pos_ticker, 'Unknown')
                pos_value = float(position.get('market_value', 0))
                
                if pos_sector not in sector_exposures:
                    sector_exposures[pos_sector] = 0
                sector_exposures[pos_sector] += pos_value
            
            # Calculate available capacity for each sector
            sector_capacity = {}
            for sector in self.sector_mappings.values():
                if sector == 'Unknown':
                    continue
                
                current_exposure = sector_exposures.get(sector, 0)
                max_exposure = portfolio_value * self.max_sector_exposure
                available_capacity = max_exposure - current_exposure
                
                sector_capacity[sectors] = {
                    'current_exposure': current_exposure,
                    'max_exposure': max_exposure,
                    'available_capacity': available_capacity,
                    'utilization': (current_exposure / max_exposure) * 100 if max_exposure > 0 else 0
                }
            
            # Sort by available capacity (descending)
            sorted_sectors = sorted(sector_capacity.items(), 
                                   key=lambda x: x[1]['available_capacity'], 
                                   reverse=True)
            
            # Filter out current sector and sectors with no capacity
            alternative_sectors = []
            for sector, capacity in sorted_sectors:
                if sector != current_sector and capacity['available_capacity'] > 0:
                    alternative_sectors.append(sector)
            
            logger.info(f"Alternative sectors to {current_sector}: {alternative_sectors[:3]}")
            return alternative_sectors[:3]  # Return top 3 alternatives
            
        except Exception as e:
            logger.error(f"Error suggesting alternative sectors: {e}")
            return []
    
    def _get_atr_value(self, ticker: str, market_data: Dict = None) -> float:
        """
        Get ATR value for ticker
        
        Args:
            ticker: Ticker symbol
            market_data: Market data dictionary
            
        Returns:
            ATR value
        """
        try:
            if market_data and ticker in market_data:
                df = market_data[ticker]
                if not df.empty and len(df) >= 14:
                    # Calculate ATR
                    high_low = df['high'] - df['low']
                    high_close = abs(df['high'] - df['close'].shift())
                    low_close = abs(df['low'] - df['close'].shift())
                    
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    
                    return atr
            
            # Fallback: fetch from API
            bars = self.api.get_bars(
                symbol=ticker,
                timeframe=tradeapi.TimeFrame.Day,
                limit=20
            ).df
            
            if not bars.empty and len(bars) >= 14:
                high_low = bars['high'] - bars['low']
                high_close = abs(bars['high'] - bars['close'].shift())
                low_close = abs(bars['low'] - bars['close'].shift())
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1]
                
                return atr
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting ATR for {ticker}: {e}")
            return 0.0
    
    def get_account_info(self) -> Dict:
        """Get current account information from IBKR"""
        try:
            if not self.ib.isConnected():
                self._connect_ibkr()
            
            # Get account summary
            account_summary = self.ib.accountSummary()
            
            account_info = {}
            for item in account_summary:
                account_info[item.tag] = item.value
            
            # Convert to proper types
            return {
                'account_id': account_info.get('AccountId', ''),
                'equity': float(account_info.get('NetLiquidation', 0)),
                'cash': float(account_info.get('CashBalance', 0)),
                'portfolio_value': float(account_info.get('NetLiquidation', 0)),
                'buying_power': float(account_info.get('BuyingPower', 0)),
                'maint_margin_req': float(account_info.get('MaintMarginReq', 0)),
                'available_funds': float(account_info.get('AvailableFunds', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_current_positions(self) -> List[Dict]:
        """Get current positions from IBKR"""
        try:
            if not self.ib.isConnected():
                self._connect_ibkr()
            
            positions = self.ib.positions()
            
            return [
                {
                    'symbol': pos.contract.symbol,
                    'sec_type': pos.contract.secType,
                    'exchange': pos.contract.exchange,
                    'currency': pos.contract.currency,
                    'position': float(pos.position),
                    'market_price': float(pos.marketPrice),
                    'market_value': float(pos.marketValue),
                    'average_cost': float(pos.averageCost),
                    'unrealized_pl': float(pos.unrealizedPNL),
                    'realized_pl': float(pos.realizedPNL),
                    'account': pos.account
                }
                for pos in positions
                if pos.position != 0  # Only include positions with non-zero quantity
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def log_risk_summary(self, account_info: Dict, current_positions: List[Dict]):
        """Log risk management summary"""
        logger.info("=== Risk Management Summary ===")
        logger.info(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        logger.info(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"Current Positions: {len(current_positions)}")
        
        # Sector exposure
        sector_exposure = {}
        for position in current_positions:
            ticker = position.get('symbol', '')
            sector = self.sector_mappings.get(ticker, 'Unknown')
            value = float(position.get('market_value', 0))
            
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += value
        
        logger.info("Sector Exposure:")
        for sector, exposure in sector_exposure.items():
            pct = (exposure / account_info.get('portfolio_value', 1)) * 100
            status = "⚠️" if pct > 30 else "✅"
            logger.info(f"  {sector}: ${exposure:,.2f} ({pct:.1f}%) {status}")

# Usage example
if __name__ == "__main__":
    # Test the risk manager
    rm = RiskManager()
    
    # Get account info
    account = rm.get_account_info()
    positions = rm.get_current_positions()
    
    # Log summary
    rm.log_risk_summary(account, positions)
    
    # Test a trade evaluation
    decision, details = rm.evaluate_trade('AAPL', 150.0, account, positions)
    logger.info(f"Trade decision: {decision}")
    logger.info(f"Details: {details}")
