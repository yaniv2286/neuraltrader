#!/usr/bin/env python3
"""
NeuralTrader Short-Side Alpha Research
======================================

R&D script to analyze AI's potential for identifying short opportunities
during bearish market conditions. This is NON-DESTRUCTIVE and does not
affect any live trading code or portfolio.

Research Question: Does our AI Council have "Short-Side Alpha"?
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.ai_models import EnsemblePredictor
    from src.trading.virtual_engine import VirtualEngine
    from src.data.yfinance_manager import YFinanceManager
    from src.trading.risk_manager import RiskManager
    
    print("🔬 NeuralTrader Short-Side Alpha Research")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Mode: RESEARCH ONLY - NON-DESTRUCTIVE")
    print()
    
    class ShortSideAnalyzer:
        """Analyzer for short-side alpha potential"""
        
        def __init__(self):
            """Initialize the analyzer"""
            print("🚀 Initializing Short-Side Analyzer...")
            
            # Load AI models (same as production)
            self.ensemble = EnsemblePredictor()
            print("✅ Ensemble AI Council loaded")
            
            # Initialize data manager
            self.yf_manager = YFinanceManager()
            print("✅ YFinance Manager initialized")
            
            # Initialize virtual engine for market filter
            self.virtual_engine = VirtualEngine()
            print("✅ Virtual Engine initialized (for market filter)")
            
            # Research parameters
            self.SHORT_STOP_LOSS = 0.05  # 5% stop loss for shorts (tighter than longs)
            self.SLIPPAGE_RATE = 0.001   # 0.1% slippage
            self.BOTTOM_N_STOCKS = 5      # Analyze bottom 5 stocks
            self.LOOKBACK_DAYS = 30       # 30-day analysis period
            
            print(f"📊 Research Parameters:")
            print(f"   Short Stop Loss: {self.SHORT_STOP_LOSS:.1%}")
            print(f"   Slippage Rate: {self.SLIPPAGE_RATE:.1%}")
            print(f"   Bottom N Stocks: {self.BOTTOM_N_STOCKS}")
            print(f"   Lookback Period: {self.LOOKBACK_DAYS} days")
            print()
        
        def get_bearish_periods(self, days_back: int = 90) -> List[datetime]:
            """Identify bearish market periods using SPY filter"""
            print("🐻 Identifying Bearish Market Periods...")
            
            bearish_dates = []
            end_date = datetime.now()
            
            for i in range(days_back):
                test_date = end_date - timedelta(days=i)
                
                # Simple bearish detection (SPY below 20-day SMA)
                try:
                    spy_data = yf.download('SPY', period="60d", progress=False)
                    if len(spy_data) >= 20:
                        current_price = spy_data['Close'].iloc[-1]
                        sma_20 = spy_data['Close'].rolling(20).mean().iloc[-1]
                        
                        if current_price < sma_20:
                            bearish_dates.append(test_date)
                            print(f"   📅 {test_date.strftime('%Y-%m-%d')}: BEARISH (SPY: ${current_price:.2f} vs SMA: ${sma_20:.2f})")
                
                except Exception as e:
                    print(f"   ⚠️  Error checking {test_date.strftime('%Y-%m-%d')}: {e}")
                    continue
            
            print(f"🐻 Found {len(bearish_dates)} bearish periods in last {days_back} days")
            print()
            return bearish_dates
        
        def analyze_short_potential(self, date: datetime) -> Dict:
            """Analyze short potential for a specific date"""
            print(f"🔬 Analyzing Short Potential for {date.strftime('%Y-%m-%d')}...")
            
            try:
                # Get S&P 100 tickers
                tickers = self.yf_manager.sp100_tickers[:20]  # Limit for research speed
                
                # Get AI scores for all tickers
                ai_scores = {}
                price_data = {}
                
                print("   📊 Getting AI scores and price data...")
                for ticker in tickers:
                    try:
                        # Get historical data for the date
                        end_date = date + timedelta(days=1)
                        start_date = date - timedelta(days=self.LOOKBACK_DAYS)
                        
                        hist_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                        if hist_data.empty or len(hist_data) < 20:
                            continue
                        
                        # Get price at analysis date
                        if date in hist_data.index:
                            analysis_price = hist_data['Close'].loc[date]
                        else:
                            # Find closest date
                            closest_date = hist_data.index[hist_data.index <= date].max()
                            if pd.isna(closest_date):
                                continue
                            analysis_price = hist_data['Close'].loc[closest_date]
                        
                        # Get future price (5 days later for short profit calculation)
                        future_date = date + timedelta(days=5)
                        if future_date in hist_data.index:
                            future_price = hist_data['Close'].loc[future_date]
                        else:
                            # Find closest future date
                            closest_future = hist_data.index[hist_data.index >= date].min()
                            if pd.isna(closest_future):
                                continue
                            future_price = hist_data['Close'].loc[closest_future]
                        
                        # Generate AI signal
                        features = self._generate_features(ticker, hist_data)
                        if features is not None:
                            ai_score = self.ensemble.predict_proba(features)[0, 1]  # Probability of "1" (buy signal)
                            ai_scores[ticker] = ai_score
                            price_data[ticker] = {
                                'entry_price': analysis_price,
                                'exit_price': future_price,
                                'date': date,
                                'future_date': closest_future if 'closest_future' in locals() else future_date
                            }
                    
                    except Exception as e:
                        continue
                
                if not ai_scores:
                    return {'error': 'No valid data for analysis'}
                
                # Identify bottom N stocks (most negative/lowest buy probability)
                bottom_stocks = sorted(ai_scores.items(), key=lambda x: x[1])[:self.BOTTOM_N_STOCKS]
                
                print(f"   🎯 Bottom {self.BOTTOM_N_STOCKS} Stocks (Lowest Buy Probability):")
                for ticker, score in bottom_stocks:
                    print(f"      {ticker}: {score:.3f} buy probability")
                
                # Simulate short positions
                short_results = []
                for ticker, ai_score in bottom_stocks:
                    if ticker not in price_data:
                        continue
                    
                    entry_price = price_data[ticker]['entry_price']
                    exit_price = price_data[ticker]['exit_price']
                    
                    # Calculate short profit (price decrease is profit)
                    price_change_pct = (entry_price - exit_price) / entry_price
                    
                    # Apply slippage
                    slippage_cost = self.SLIPPAGE_RATE
                    net_profit_pct = price_change_pct - slippage_cost
                    
                    # Check stop loss
                    stop_loss_hit = abs(price_change_pct) > self.SHORT_STOP_LOSS
                    if stop_loss_hit:
                        net_profit_pct = -self.SHORT_STOP_LOSS - slippage_cost
                    
                    short_results.append({
                        'ticker': ticker,
                        'ai_score': ai_score,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'price_change_pct': price_change_pct,
                        'net_profit_pct': net_profit_pct,
                        'stop_loss_hit': stop_loss_hit
                    })
                
                # Calculate aggregate performance
                total_return = sum([r['net_profit_pct'] for r in short_results])
                avg_return = total_return / len(short_results) if short_results else 0
                win_rate = sum([1 for r in short_results if r['net_profit_pct'] > 0]) / len(short_results) if short_results else 0
                
                return {
                    'date': date,
                    'bottom_stocks': bottom_stocks,
                    'short_results': short_results,
                    'total_return': total_return,
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'num_stocks': len(short_results)
                }
                
            except Exception as e:
                return {'error': str(e)}
        
        def _generate_features(self, ticker: str, hist_data: pd.DataFrame) -> np.ndarray:
            """Generate features for AI prediction (simplified version)"""
            try:
                if len(hist_data) < 20:
                    return None
                
                # Simple feature engineering
                features = []
                
                # Price-based features
                close_prices = hist_data['Close']
                features.append(close_prices.pct_change(5).iloc[-1])  # 5-day return
                features.append(close_prices.pct_change(10).iloc[-1])  # 10-day return
                features.append(close_prices.pct_change(20).iloc[-1])  # 20-day return
                
                # Volatility
                features.append(close_prices.pct_change().rolling(10).std().iloc[-1])  # 10-day volatility
                
                # Volume features (if available)
                if 'Volume' in hist_data.columns:
                    volume = hist_data['Volume']
                    features.append(volume.pct_change(5).iloc[-1])  # 5-day volume change
                else:
                    features.append(0)
                
                # Price position
                features.append((close_prices.iloc[-1] - close_prices.min()) / (close_prices.max() - close_prices.min()))
                
                return np.array(features).reshape(1, -1)
                
            except Exception as e:
                return None
        
        def run_research(self, num_periods: int = 10) -> Dict:
            """Run the complete short-side alpha research"""
            print("🔬 Starting Short-Side Alpha Research...")
            print("=" * 60)
            
            # Get bearish periods
            bearish_dates = self.get_bearish_periods(days_back=90)
            
            if not bearish_dates:
                print("❌ No bearish periods found for analysis")
                return {'error': 'No bearish periods found'}
            
            # Limit analysis to recent periods
            analysis_dates = bearish_dates[-num_periods:]
            
            print(f"📊 Analyzing {len(analysis_dates)} bearish periods...")
            print()
            
            results = []
            for date in analysis_dates:
                result = self.analyze_short_potential(date)
                if 'error' not in result:
                    results.append(result)
                    print(f"   ✅ {date.strftime('%Y-%m-%d')}: {result['avg_return']:.2%} avg return, {result['win_rate']:.1%} win rate")
                else:
                    print(f"   ❌ {date.strftime('%Y-%m-%d')}: {result['error']}")
            
            if not results:
                print("❌ No successful analyses completed")
                return {'error': 'No successful analyses'}
            
            # Aggregate research results
            total_periods = len(results)
            total_return = sum([r['total_return'] for r in results])
            avg_return_per_period = total_return / total_periods
            overall_win_rate = sum([r['win_rate'] for r in results]) / total_periods
            
            # Cash comparison (0% return)
            cash_return = 0.0
            short_alpha = avg_return_per_period - cash_return
            
            print()
            print("=" * 60)
            print("📊 SHORT-SIDE ALPHA RESEARCH RESULTS")
            print("=" * 60)
            print(f"📅 Analysis Periods: {total_periods} bearish periods")
            print(f"📈 Average Short Return: {avg_return_per_period:.2%} per period")
            print(f"💰 Cash Return (Benchmark): {cash_return:.2%}")
            print(f"🎯 Short Alpha vs Cash: {short_alpha:.2%}")
            print(f"🏆 Overall Win Rate: {overall_win_rate:.1%}")
            print()
            
            # Detailed breakdown
            print("📋 Period Breakdown:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['date'].strftime('%Y-%m-%d')}: "
                      f"{result['avg_return']:.2%} return, "
                      f"{result['win_rate']:.1%} win rate, "
                      f"{result['num_stocks']} stocks")
            
            print()
            print("🎯 CONCLUSION:")
            if short_alpha > 0:
                print(f"✅ POSITIVE SHORT ALPHA DETECTED!")
                print(f"   AI shows {short_alpha:.2%} advantage over cash during bearish periods")
                print(f"   Recommendation: Consider short-side strategy development")
            else:
                print(f"❌ NO SHORT ALPHA DETECTED")
                print(f"   AI underperforms cash by {abs(short_alpha):.2%} during bearish periods")
                print(f"   Recommendation: Focus on long-side and capital preservation")
            
            print()
            print("=" * 60)
            print("🔬 Research Complete - NON-DESTRUCTIVE")
            print("💡 This analysis did not affect any live trading systems")
            print("=" * 60)
            
            return {
                'total_periods': total_periods,
                'avg_return_per_period': avg_return_per_period,
                'short_alpha': short_alpha,
                'overall_win_rate': overall_win_rate,
                'detailed_results': results
            }
    
    # Run the research
    analyzer = ShortSideAnalyzer()
    research_results = analyzer.run_research(num_periods=5)  # Analyze 5 recent bearish periods
    
    print("\n🎯 Short-Side Alpha Research Complete!")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("🔧 Make sure the virtual environment is activated")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("🔍 Check system configuration and data availability")
    sys.exit(1)
