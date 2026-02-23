#!/usr/bin/env python3
"""
Sector Rotation - Sector-based risk management and momentum analysis
Implements SectorAuthority class for sector diversification and volatility controls
"""

import json
import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add scripts directory to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts import data_manager

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a default handler if none exists
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SectorAuthority:
    """
    Sector-based risk management authority
    Handles sector rotation, momentum analysis, and volatility controls
    """
    
    def __init__(self):
        """Initialize SectorAuthority with logging and data paths"""
        self.logger = logging.getLogger(__name__)
        self.sector_map = None
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def load_sector_map(self) -> bool:
        """
        Load the sector mapping from config/sector_map.json
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            sector_file = os.path.join(self.project_root, 'config', 'sector_map.json')
            
            if not os.path.exists(sector_file):
                self.logger.error(f"[ERROR] Sector map file not found: {sector_file}")
                return False
            
            with open(sector_file, 'r') as f:
                self.sector_map = json.load(f)
            
            self.logger.info(f"[OK] Sector map loaded: {len(self.sector_map)} sectors")
            
            # Log sector summary
            total_tickers = sum(len(tickers) for tickers in self.sector_map.values())
            self.logger.info(f"[INFO] Total tickers in sector map: {total_tickers}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load sector map: {e}")
            return False
    
    def get_sector_momentum(self) -> List[Tuple[str, float]]:
        """
        Calculate 20-day Rate of Change for all sector ETFs
        
        Returns:
            List[Tuple[str, float]]: Ranked list of sectors from Strongest to Weakest
        """
        if not self.sector_map:
            if not self.load_sector_map():
                return []
        
        try:
            dm = data_manager.DataManager()
            
            sector_momentum = {}
            sector_etfs = list(self.sector_map.keys())
            
            self.logger.info(f"[ANALYSIS] Calculating momentum for {len(sector_etfs)} sector ETFs")
            
            for etf in sector_etfs:
                try:
                    # Load ETF data
                    etf_data = dm._load_ticker_data(etf)
                    
                    if etf_data is None or etf_data.empty:
                        self.logger.warning(f"[WARN] No data found for {etf}")
                        continue
                    
                    # Ensure we have at least 20 days of data
                    if len(etf_data) < 20:
                        self.logger.warning(f"[WARN] Insufficient data for {etf}: {len(etf_data)} days")
                        continue
                    
                    # Calculate 20-day Rate of Change (ROC)
                    # ROC = ((Current Price - Price 20 days ago) / Price 20 days ago) * 100
                    current_price = etf_data['close'].iloc[-1]
                    price_20_days_ago = etf_data['close'].iloc[-20]
                    
                    roc = ((current_price - price_20_days_ago) / price_20_days_ago) * 100
                    sector_momentum[etf] = roc
                    
                    self.logger.info(f"[MOMENTUM] {etf}: {roc:.2f}% (20-day ROC)")
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to calculate momentum for {etf}: {e}")
                    continue
            
            # Sort sectors by momentum (Strongest to Weakest)
            ranked_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"[RANKING] Sector momentum ranking (Strongest to Weakest):")
            for i, (etf, roc) in enumerate(ranked_sectors, 1):
                strength = "STRONG" if roc > 2 else "MODERATE" if roc > 0 else "WEAK"
                self.logger.info(f"  {i}. {etf}: {roc:.2f}% ({strength})")
            
            return ranked_sectors
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to calculate sector momentum: {e}")
            return []
    
    def apply_sector_tax(self, ticker_scores: Dict[str, float], sector_ranks: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Apply sector tax to tickers in bottom 3 weakest sectors
        
        Args:
            ticker_scores: Dictionary of ticker scores
            sector_ranks: Ranked list of sectors from get_sector_momentum()
            
        Returns:
            Dict[str, float]: Updated ticker scores with sector tax applied
        """
        if not self.sector_map:
            if not self.load_sector_map():
                return ticker_scores
        
        if not sector_ranks or len(sector_ranks) < 3:
            self.logger.warning("[WARN] Insufficient sector data for tax application")
            return ticker_scores
        
        # Identify bottom 3 sectors (weakest)
        bottom_sectors = [sector for sector, _ in sector_ranks[-3:]]
        
        self.logger.warning(f"[RISK] Bottom 3 sectors identified: {bottom_sectors}")
        
        # Create reverse mapping: ticker -> sector
        ticker_to_sector = {}
        for sector, tickers in self.sector_map.items():
            for ticker in tickers:
                ticker_to_sector[ticker] = sector
        
        # Apply 15% tax to tickers in bottom sectors
        updated_scores = ticker_scores.copy()
        tax_count = 0
        
        for ticker, score in ticker_scores.items():
            if ticker in ticker_to_sector:
                ticker_sector = ticker_to_sector[ticker]
                
                if ticker_sector in bottom_sectors:
                    # Apply 15% tax (reduce score by 15%)
                    original_score = score
                    updated_scores[ticker] = score * 0.85
                    tax_count += 1
                    
                    self.logger.warning(f"[RISK] Applied Sector Tax to {ticker} (Sector: {ticker_sector}). "
                                      f"Score: {original_score:.3f} -> {updated_scores[ticker]:.3f} (-15%)")
        
        self.logger.info(f"[SUMMARY] Sector tax applied to {tax_count} tickers in bottom 3 sectors")
        
        return updated_scores
    
    def check_global_stop(self) -> bool:
        """
        Check global volatility using VXX Bollinger Bands
        
        Returns:
            bool: True if VXX breaks upper Bollinger Band (STOP BUYING), False otherwise
        """
        try:
            dm = data_manager.DataManager()
            
            # Load VXX data
            vxx_data = dm._load_ticker_data('VXX')
            
            if vxx_data is None or vxx_data.empty:
                self.logger.error("[ERROR] No VXX data available for volatility check")
                return False
            
            # Ensure we have sufficient data
            if len(vxx_data) < 20:
                self.logger.error(f"[ERROR] Insufficient VXX data: {len(vxx_data)} days (need 20+)")
                return False
            
            # Calculate 20-day Moving Average and Standard Deviation
            vxx_data['MA20'] = vxx_data['close'].rolling(window=20).mean()
            vxx_data['STD20'] = vxx_data['close'].rolling(window=20).std()
            
            # Calculate Bollinger Bands
            vxx_data['Upper_BB'] = vxx_data['MA20'] + (2 * vxx_data['STD20'])
            
            # Get latest values
            latest_close = vxx_data['close'].iloc[-1]
            latest_ma = vxx_data['MA20'].iloc[-1]
            latest_upper_bb = vxx_data['Upper_BB'].iloc[-1]
            latest_std = vxx_data['STD20'].iloc[-1]
            
            self.logger.info(f"[VOLATILITY] VXX Analysis:")
            self.logger.info(f"  Close: ${latest_close:.2f}")
            self.logger.info(f"  MA20: ${latest_ma:.2f}")
            self.logger.info(f"  Upper BB: ${latest_upper_bb:.2f}")
            self.logger.info(f"  StdDev: ${latest_std:.2f}")
            
            # Check if VXX broke upper Bollinger Band
            if latest_close > latest_upper_bb:
                self.logger.error(f"[STOP] Global Volatility Alert: VXX broke upper Bollinger Band!")
                self.logger.error(f"[STOP] VXX Close (${latest_close:.2f}) > Upper BB (${latest_upper_bb:.2f})")
                self.logger.error(f"[ACTION] STOP BUYING - High volatility detected!")
                return True
            else:
                self.logger.info(f"[OK] VXX within normal volatility range")
                self.logger.info(f"[OK] VXX Close (${latest_close:.2f}) <= Upper BB (${latest_upper_bb:.2f})")
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to check global volatility: {e}")
            return False
    
    def get_sector_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get the sector for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Optional[str]: Sector ETF symbol or None if not found
        """
        if not self.sector_map:
            if not self.load_sector_map():
                return None
        
        for sector, tickers in self.sector_map.items():
            if ticker in tickers:
                return sector
        
        return None
    
    def get_sector_summary(self) -> Dict[str, Dict]:
        """
        Get comprehensive sector analysis summary
        
        Returns:
            Dict with sector statistics and analysis
        """
        if not self.sector_map:
            if not self.load_sector_map():
                return {}
        
        sector_momentum = self.get_sector_momentum()
        global_stop = self.check_global_stop()
        
        summary = {
            'sector_count': len(self.sector_map),
            'total_tickers': sum(len(tickers) for tickers in self.sector_map.values()),
            'momentum_ranking': sector_momentum,
            'bottom_sectors': [sector for sector, _ in sector_momentum[-3:]] if sector_momentum else [],
            'global_stop_signal': global_stop,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return summary


def main():
    """Main function for testing SectorAuthority"""
    logger.info("[START] Sector Authority Test")
    
    try:
        # Initialize Sector Authority
        authority = SectorAuthority()
        
        # Load sector map
        if not authority.load_sector_map():
            logger.error("[FAIL] Could not load sector map")
            return 1
        
        # Get sector momentum
        momentum = authority.get_sector_momentum()
        
        # Test sector tax application
        test_scores = {
            'AAPL': 0.85, 'MSFT': 0.78, 'JPM': 0.72, 'XOM': 0.65,
            'UNH': 0.70, 'AMZN': 0.82, 'NEE': 0.68, 'CAT': 0.75
        }
        
        updated_scores = authority.apply_sector_tax(test_scores, momentum)
        
        # Check global volatility
        global_stop = authority.check_global_stop()
        
        # Get summary
        summary = authority.get_sector_summary()
        
        logger.info("[SUCCESS] Sector Authority test completed")
        logger.info(f"[INFO] Global Stop Signal: {'ACTIVE' if global_stop else 'INACTIVE'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Sector Authority test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
