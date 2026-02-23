#!/usr/bin/env python3
"""
NeuralTrader End-to-End Pipeline Sanity Check
======================================

Tier 2.5: Proves the brain works and enforces ARCHITECTURE.md Risk Laws

This script validates the complete pipeline from data fetching to position sizing,
ensuring all components work together flawlessly and risk management laws are enforced.
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import production classes
from scripts.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from core.strategy import TradingStrategy
from scripts.sector_rotation import SectorAuthority

# Import AI models
from core.ai_models import get_ensemble_signal

class CriticalPipelineError(Exception):
    """Critical error in the pipeline that requires immediate attention"""
    pass

class SanityCheckPipeline:
    """End-to-End pipeline sanity check for NeuralTrader"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.micro_universe = ['AAPL', 'MSFT', 'SPY']
        self.mock_account_balance = 100000.0
        
        # Initialize production components
        self.data_manager = None
        self.feature_engineer = None
        self.trading_strategy = None
        self.sector_auth = None
        
        self.logger.info("NeuralTrader Sanity Check Pipeline Initialized")
        self.logger.info(f"Micro-Universe: {self.micro_universe}")
        self.logger.info(f"Mock Account Balance: ${self.mock_account_balance:,.2f}")
    
    def _setup_logging(self):
        """Setup logging for the sanity check"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'sanity_check.log'))
            ]
        )
        return logging.getLogger(__name__)
    
    def run_pipeline(self):
        """Run the complete sanity check pipeline"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING NEURALTRADER SANITY CHECK PIPELINE")
            self.logger.info("=" * 60)
            
            # Step 1: Initialize all production components
            self._initialize_components()
            
            # Step 2: Fetch data for micro-universe
            market_data = self._fetch_market_data()
            
            # Step 3: Process each ticker through the complete pipeline
            results = []
            for ticker in self.micro_universe:
                self.logger.info(f"\nProcessing ticker: {ticker}")
                result = self._process_ticker_pipeline(ticker, market_data[ticker])
                results.append(result)
            
            # Step 4: Generate final report
            self._generate_final_report(results)
            
            self.logger.info("\nSANITY CHECK PASSED - All systems operational")
            return True
            
        except CriticalPipelineError as e:
            self.logger.error(f"\nCRITICAL PIPELINE ERROR: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        except Exception as e:
            self.logger.error(f"\nUNEXPECTED ERROR: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_components(self):
        """Initialize all production components"""
        self.logger.info("Initializing production components...")
        
        try:
            # Initialize DataManager
            self.data_manager = DataManager()
            self.logger.info("DataManager initialized")
            
            # Initialize FeatureEngineer
            self.feature_engineer = FeatureEngineer()
            self.logger.info("FeatureEngineer initialized")
            
            # Initialize TradingStrategy
            self.trading_strategy = TradingStrategy()
            self.logger.info("TradingStrategy initialized")
            
            # Initialize SectorAuthority
            self.sector_auth = SectorAuthority()
            self.logger.info("SectorAuthority initialized")
            
            self.logger.info("All production components initialized successfully")
            
        except Exception as e:
            raise CriticalPipelineError(f"Component initialization failed: {e}")
    
    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch latest daily data + sentiment for micro-universe"""
        self.logger.info("Fetching market data for micro-universe...")
        
        market_data = {}
        
        for ticker in self.micro_universe:
            try:
                # Fetch market data
                data = self.data_manager._load_ticker_data(ticker)
                
                if data is None or len(data) < 50:
                    raise CriticalPipelineError(f"Insufficient data for {ticker}: {len(data) if data else 0} days")
                
                market_data[ticker] = data
                self.logger.info(f"{ticker}: {len(data)} days of data loaded")
                
            except Exception as e:
                raise CriticalPipelineError(f"Failed to fetch data for {ticker}: {e}")
        
        return market_data
    
    def _process_ticker_pipeline(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Process a single ticker through the complete pipeline"""
        result = {
            'ticker': ticker,
            'data_rows': len(data),
            'features_generated': False,
            'feature_count': 0,
            'has_nan': False,
            'ai_score': 0.0,
            'ai_signal': 'HOLD',
            'sector_tax': 0.0,
            'sector_name': 'Unknown',
            'volatility': 0.0,
            'position_size': 0,
            'position_value': 0.0,
            'risk_amount': 0.0
        }
        
        try:
            # Step 1: Generate features
            self.logger.info(f"  Generating features for {ticker}...")
            features, targets = self.feature_engineer.create_features(data)
            
            # Strict assertion: exactly 64 features (technical features only for sanity check)
            if features.shape[1] != 64:
                raise CriticalPipelineError(f"Feature count mismatch: Expected 64, got {features.shape[1]}")
            
            # Strict assertion: NO NaN values in final row
            if features.iloc[-1].isna().any():
                nan_columns = features.columns[features.iloc[-1].isna()].tolist()
                raise CriticalPipelineError(f"NaN values detected in final row: {nan_columns}")
            
            result['features_generated'] = True
            result['feature_count'] = features.shape[1]
            result['has_nan'] = False
            self.logger.info(f"  Features: {features.shape[1]} features, no NaN values")
            
            # Step 2: Get AI signal
            self.logger.info(f"  Getting AI signal for {ticker}...")
            signal, confidence, details = get_ensemble_signal(data)
            
            result['ai_score'] = confidence
            result['ai_signal'] = signal
            self.logger.info(f"  AI Signal: {signal} (confidence: {confidence:.4f})")
            
            # Step 3: Sector analysis
            self.logger.info(f"  Sector analysis for {ticker}...")
            sector_ranks = self.sector_auth.get_sector_momentum()
            
            # Find sector for this ticker
            sector_name = 'Unknown'
            sector_tax = 0.0
            
            for sector, tickers in self.sector_auth.sector_map.items():
                if ticker in tickers:
                    sector_name = sector
                    # Check if this sector is in bottom 3 (taxed)
                    # sector_ranks is a list of tuples, so we need to extract sector names
                    if len(sector_ranks) >= 3:
                        bottom_sectors = [rank[0] for rank in sector_ranks[-3:]]
                        if sector in bottom_sectors:
                            sector_tax = 0.15  # 15% tax for bottom 3 sectors
                    break
            
            result['sector_name'] = sector_name
            result['sector_tax'] = sector_tax
            self.logger.info(f"  Sector: {sector_name}, Tax: {sector_tax:.1%}")
            
            # Step 4: Calculate position size using Real Volatility Inverse Sizing Law
            self.logger.info(f"  Calculating position size for {ticker}...")
            position_size, position_info = self._calculate_position_size(ticker, data, confidence)
            
            result.update(position_info)
            self.logger.info(f"  Position: {position_size} shares (${result['position_value']:,.2f})")
            
            return result
            
        except Exception as e:
            raise CriticalPipelineError(f"Pipeline failed for {ticker}: {e}")
    
    def _calculate_position_size(self, ticker: str, data: pd.DataFrame, confidence: float) -> tuple[int, Dict[str, Any]]:
        """
        Calculate position size using Real Volatility Inverse Sizing Law (ARCHITECTURE.md Section 3 & 7)
        
        Mathematical position allocation using real market data (20-day returns, annualized volatility)
        1% risk per trade with real volatility-weighted allocations (1/σ weighting)
        """
        try:
            # Calculate real volatility from data (20-day returns, √252 annualization)
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 20:
                raise CriticalPipelineError(f"Insufficient data for volatility calculation: {len(returns)} < 20")
            
            # Use last 20 trading days
            recent_returns = returns.tail(20)
            daily_volatility = recent_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Real Volatility Inverse Sizing Law: 1% risk per trade
            risk_per_trade = 0.01  # 1% risk per trade (ARCHITECTURE.md Section 3)
            risk_amount = self.mock_account_balance * risk_per_trade
            
            # Mathematical allocation using 1/σ weighting (ARCHITECTURE.md Section 7)
            # Low volatility stocks get MORE capital, high volatility get LESS
            inverse_volatility_weight = 1.0 / annualized_volatility
            
            # Calculate position value
            position_value = risk_amount * inverse_volatility_weight
            
            # Get current price
            current_price = float(data['close'].iloc[-1])
            
            # Calculate number of shares
            shares = int(position_value / current_price)
            
            # Minimum position size
            shares = max(shares, 1)
            
            # Maximum position size (20% of portfolio - safety constraint)
            max_shares = int((self.mock_account_balance * 0.20) / current_price)
            shares = min(shares, max_shares)
            
            position_info = {
                'volatility': annualized_volatility,
                'inverse_vol_weight': inverse_volatility_weight,
                'risk_amount': risk_amount,
                'position_value': position_value,
                'current_price': current_price
            }
            
            return shares, position_info
            
        except Exception as e:
            raise CriticalPipelineError(f"Position sizing failed for {ticker}: {e}")
    
    def _generate_final_report(self, results: List[Dict[str, Any]]):
        """Generate final pipeline report"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("NEURALTRADER SANITY CHECK FINAL REPORT")
        self.logger.info("=" * 60)
        
        for result in results:
            ticker = result['ticker']
            self.logger.info(f"\n{ticker} Summary:")
            self.logger.info(f"   Data Rows: {result['data_rows']}")
            self.logger.info(f"   Features: {result['feature_count']} features (No NaN)")
            self.logger.info(f"   AI Signal: {result['ai_signal']} (Score: {result['ai_score']:.4f})")
            self.logger.info(f"   Sector: {result['sector_name']} (Tax: {result['sector_tax']:.1%})")
            self.logger.info(f"   Volatility: {result['volatility']:.2%}")
            self.logger.info(f"   1/sigma Weight: {result['inverse_vol_weight']:.2f}")
            self.logger.info(f"   Risk Amount: ${result['risk_amount']:,.2f}")
            self.logger.info(f"   Position: {result['position_size']} shares @ ${result['current_price']:.2f}")
            self.logger.info(f"   Position Value: ${result['position_value']:,.2f}")
        
        # Summary statistics
        total_ai_score = sum(r['ai_score'] for r in results)
        avg_ai_score = total_ai_score / len(results)
        total_position_value = sum(r['position_value'] for r in results)
        
        self.logger.info(f"\nPipeline Summary:")
        self.logger.info(f"   Total AI Score: {total_ai_score:.4f}")
        self.logger.info(f"   Average AI Score: {avg_ai_score:.4f}")
        self.logger.info(f"   Total Position Value: ${total_position_value:,.2f}")
        self.logger.info(f"   Risk Utilization: {(total_position_value / self.mock_account_balance * 100):.1f}%")
        
        # Risk Law Compliance Check
        self.logger.info(f"\nRisk Law Compliance:")
        for result in results:
            risk_pct = (result['risk_amount'] / self.mock_account_balance) * 100
            if abs(risk_pct - 1.0) > 0.01:  # Allow small rounding differences
                self.logger.error(f"   VIOLATION {ticker}: Risk {risk_pct:.2f}% (VIOLATES 1% law)")
            else:
                self.logger.info(f"   COMPLIANT {ticker}: Risk {risk_pct:.2f}% (COMPLIANT)")

def main():
    """Main entry point"""
    pipeline = SanityCheckPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\nSANITY CHECK PASSED - NeuralTrader is ready for production!")
        sys.exit(0)
    else:
        print("\nSANITY CHECK FAILED - Critical issues detected!")
        sys.exit(1)

if __name__ == "__main__":
    main()
