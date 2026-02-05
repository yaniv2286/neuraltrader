"""
NeuralTrader Trade Verification System
=====================================

Verifies if trades should have been taken but weren't executed.
Provides detailed analysis of missed opportunities and execution gaps.

Features:
- Signal generation verification
- Trade opportunity analysis
- Risk assessment validation
- Execution gap identification
- Missed opportunity reporting
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
import logging

class TradeVerifier:
    """
    Trade verification system for NeuralTrader
    """
    
    def __init__(self, logs_dir: str = None):
        """Initialize trade verifier"""
        self.project_root = Path(__file__).parent.parent.parent
        self.logs_dir = Path(logs_dir) if logs_dir else self.project_root / "logs"
        self.verification_dir = self.logs_dir / "verification"
        self.verification_dir.mkdir(exist_ok=True)
        
        # Timezone handling
        self.eastern = pytz.timezone('US/Eastern')
        self.israel = pytz.timezone('Asia/Jerusalem')
        
        # Configure logging
        self.logger = logging.getLogger('TradeVerifier')
        
        # Load trading modules
        self._load_trading_modules()
    
    def _load_trading_modules(self):
        """Load trading modules for verification"""
        try:
            from ..data.yfinance_manager import YFinanceManager
            from ..trading.risk_manager import RiskManager
            from ..trading.virtual_engine import VirtualEngine
            
            self.yfinance_manager = YFinanceManager()
            self.risk_manager = RiskManager()
            self.virtual_engine = VirtualEngine()
            
            self.logger.info("✅ Trading modules loaded for verification")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load trading modules: {e}")
            raise
    
    def verify_trading_session(self, date: datetime = None) -> Dict:
        """
        Verify a trading session for missed opportunities
        
        Args:
            date: Date to verify (default: today)
            
        Returns:
            Verification results with missed opportunities
        """
        if date is None:
            date = datetime.now(self.eastern)
        
        self.logger.info(f"🔍 Verifying trading session for {date.strftime('%Y-%m-%d')}")
        
        verification_results = {
            "date": date.strftime('%Y-%m-%d'),
            "verification_time": datetime.now(self.israel).isoformat(),
            "market_status": self._get_market_status(date),
            "signals_generated": [],
            "opportunities_analyzed": [],
            "trades_executed": [],
            "missed_opportunities": [],
            "execution_gaps": [],
            "summary": {}
        }
        
        try:
            # Step 1: Generate signals for the day
            signals = self._generate_daily_signals(date)
            verification_results["signals_generated"] = signals
            
            # Step 2: Analyze each signal for trading opportunity
            opportunities = self._analyze_trading_opportunities(signals, date)
            verification_results["opportunities_analyzed"] = opportunities
            
            # Step 3: Check actual trades executed
            actual_trades = self._get_actual_trades(date)
            verification_results["trades_executed"] = actual_trades
            
            # Step 4: Identify missed opportunities
            missed_opportunities = self._identify_missed_opportunities(opportunities, actual_trades)
            verification_results["missed_opportunities"] = missed_opportunities
            
            # Step 5: Analyze execution gaps
            execution_gaps = self._analyze_execution_gaps(opportunities, actual_trades)
            verification_results["execution_gaps"] = execution_gaps
            
            # Step 6: Generate summary
            verification_results["summary"] = self._generate_verification_summary(
                signals, opportunities, actual_trades, missed_opportunities
            )
            
            # Save verification results
            self._save_verification_results(verification_results)
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"❌ Error during verification: {e}")
            verification_results["error"] = str(e)
            return verification_results
    
    def _generate_daily_signals(self, date: datetime) -> List[Dict]:
        """Generate trading signals for verification"""
        signals = []
        
        try:
            # Get market data for the date
            end_date = date + timedelta(days=1)
            market_data = self.yfinance_manager.fetch_daily_data(
                period="5d",
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not market_data:
                self.logger.warning("⚠️ No market data available for signal generation")
                return signals
            
            # Generate signals for top tickers
            for ticker in self.yfinance_manager.sp100_tickers[:20]:  # Limit for verification
                if ticker not in market_data:
                    continue
                
                df = market_data[ticker]
                if df.empty or len(df) < 50:
                    continue
                
                try:
                    # Generate signal using existing logic
                    signal_strength = self._generate_signal_strength(df, ticker)
                    
                    if signal_strength > 0.3:  # Lower threshold for verification
                        current_price = self.yfinance_manager.get_latest_prices([ticker]).get(ticker)
                        
                        signal = {
                            "ticker": ticker,
                            "signal_strength": signal_strength,
                            "signal_type": "BUY" if signal_strength > 0.5 else "HOLD",
                            "current_price": current_price,
                            "timestamp": date.strftime('%Y-%m-%d %H:%M:%S EST'),
                            "confidence": self._calculate_signal_confidence(df, signal_strength)
                        }
                        
                        signals.append(signal)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error generating signal for {ticker}: {e}")
                    continue
            
            self.logger.info(f"📊 Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signals: {e}")
            return signals
    
    def _generate_signal_strength(self, df: pd.DataFrame, ticker: str) -> float:
        """Generate signal strength for verification"""
        try:
            # Simple momentum-based signal for verification
            if len(df) < 20:
                return 0.0
            
            # Calculate technical indicators
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            latest = df.iloc[-1]
            
            # Signal components
            momentum_signal = 0.0
            trend_signal = 0.0
            rsi_signal = 0.0
            
            # Momentum: Price above SMAs
            if latest['Close'] > latest['sma_20']:
                momentum_signal += 0.3
            if latest['Close'] > latest['sma_50']:
                momentum_signal += 0.2
            
            # Trend: SMA alignment
            if latest['sma_20'] > latest['sma_50']:
                trend_signal += 0.3
            
            # RSI: Not overbought
            if 30 < latest['rsi'] < 70:
                rsi_signal += 0.2
            
            # Combine signals
            total_signal = momentum_signal + trend_signal + rsi_signal
            
            return min(total_signal, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating signal strength: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_signal_confidence(self, df: pd.DataFrame, signal_strength: float) -> float:
        """Calculate confidence level for signal"""
        try:
            # Volume-based confidence
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_confidence = min(current_volume / avg_volume, 1.0) if avg_volume > 0 else 0.5
            
            # Price stability
            price_volatility = df['Close'].pct_change().rolling(10).std().iloc[-1]
            stability_confidence = max(0.5, 1.0 - price_volatility * 10)
            
            # Combine confidence factors
            overall_confidence = (signal_strength * 0.4 + volume_confidence * 0.3 + stability_confidence * 0.3)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence: {e}")
            return 0.5
    
    def _analyze_trading_opportunities(self, signals: List[Dict], date: datetime) -> List[Dict]:
        """Analyze signals for trading opportunities"""
        opportunities = []
        
        try:
            # Get account info for risk calculations
            account_info = self.risk_manager.get_account_info()
            current_positions = self.risk_manager.get_current_positions()
            
            for signal in signals:
                if signal['signal_strength'] < 0.5:  # Only strong signals
                    continue
                
                # Evaluate trade with risk manager
                decision, details = self.risk_manager.evaluate_trade(
                    signal['ticker'],
                    signal['current_price'],
                    account_info,
                    current_positions
                )
                
                opportunity = {
                    "ticker": signal['ticker'],
                    "signal_strength": signal['signal_strength'],
                    "signal_type": signal['signal_type'],
                    "current_price": signal['current_price'],
                    "confidence": signal['confidence'],
                    "risk_decision": decision.value if hasattr(decision, 'value') else str(decision),
                    "risk_details": details,
                    "recommended_position_size": details.get('position_size', 0),
                    "reason_for_rejection": details.get('rejection_reason', ''),
                    "timestamp": signal['timestamp']
                }
                
                opportunities.append(opportunity)
            
            self.logger.info(f"📊 Analyzed {len(opportunities)} trading opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing opportunities: {e}")
            return opportunities
    
    def _get_actual_trades(self, date: datetime) -> List[Dict]:
        """Get actual trades executed on the date"""
        try:
            # Load portfolio and trade history
            portfolio = self.virtual_engine.portfolio
            trade_history = portfolio.get('trade_history', [])
            
            # Filter trades for the specific date
            date_str = date.strftime('%Y-%m-%d')
            actual_trades = []
            
            for trade in trade_history:
                trade_date = datetime.fromisoformat(trade['timestamp']).strftime('%Y-%m-%d')
                if trade_date == date_str:
                    actual_trades.append(trade)
            
            self.logger.info(f"📊 Found {len(actual_trades)} actual trades")
            return actual_trades
            
        except Exception as e:
            self.logger.error(f"❌ Error getting actual trades: {e}")
            return []
    
    def _identify_missed_opportunities(self, opportunities: List[Dict], actual_trades: List[Dict]) -> List[Dict]:
        """Identify missed trading opportunities"""
        missed_opportunities = []
        
        try:
            # Create set of actually traded tickers
            traded_tickers = {trade['ticker'] for trade in actual_trades}
            
            # Find opportunities that weren't traded
            for opportunity in opportunities:
                if opportunity['ticker'] not in traded_tickers:
                    missed_opportunity = {
                        "ticker": opportunity['ticker'],
                        "signal_strength": opportunity['signal_strength'],
                        "confidence": opportunity['confidence'],
                        "current_price": opportunity['current_price'],
                        "recommended_position_size": opportunity['recommended_position_size'],
                        "risk_decision": opportunity['risk_decision'],
                        "reason_for_rejection": opportunity['reason_for_rejection'],
                        "missed_reason": self._determine_miss_reason(opportunity, actual_trades),
                        "potential_pnl": self._estimate_missed_pnl(opportunity),
                        "timestamp": opportunity['timestamp']
                    }
                    
                    missed_opportunities.append(missed_opportunity)
            
            self.logger.info(f"📊 Identified {len(missed_opportunities)} missed opportunities")
            return missed_opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error identifying missed opportunities: {e}")
            return missed_opportunities
    
    def _determine_miss_reason(self, opportunity: Dict, actual_trades: List[Dict]) -> str:
        """Determine why an opportunity was missed"""
        if opportunity['risk_decision'] != 'APPROVED':
            return f"Risk rejected: {opportunity['reason_for_rejection']}"
        
        # Check if it was a timing issue
        if len(actual_trades) == 0:
            return "No trades executed - possible system issue"
        
        # Check if position size was too small
        if opportunity['recommended_position_size'] < 10:
            return "Position size too small"
        
        return "Unknown - requires investigation"
    
    def _estimate_missed_pnl(self, opportunity: Dict) -> Dict:
        """Estimate potential P&L from missed opportunity"""
        try:
            # Simple estimation based on signal strength
            potential_return = opportunity['signal_strength'] * 0.05  # 5% max return
            position_value = opportunity['recommended_position_size'] * opportunity['current_price']
            potential_pnl = position_value * potential_return
            
            return {
                "potential_return_pct": potential_return * 100,
                "position_value": position_value,
                "potential_pnl": potential_pnl,
                "confidence_level": opportunity['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error estimating missed P&L: {e}")
            return {"potential_pnl": 0, "confidence_level": 0}
    
    def _analyze_execution_gaps(self, opportunities: List[Dict], actual_trades: List[Dict]) -> List[Dict]:
        """Analyze execution gaps and patterns"""
        execution_gaps = []
        
        try:
            # Gap 1: Approved trades not executed
            approved_not_executed = [
                opp for opp in opportunities 
                if opp['risk_decision'] == 'APPROVED' and opp['ticker'] not in [t['ticker'] for t in actual_trades]
            ]
            
            if approved_not_executed:
                execution_gaps.append({
                    "gap_type": "APPROVED_NOT_EXECUTED",
                    "count": len(approved_not_executed),
                    "details": approved_not_executed,
                    "potential_impact": sum([self._estimate_missed_pnl(opp)['potential_pnl'] for opp in approved_not_executed])
                })
            
            # Gap 2: Low execution rate
            if len(opportunities) > 0:
                execution_rate = len(actual_trades) / len(opportunities)
                if execution_rate < 0.5:  # Less than 50% execution rate
                    execution_gaps.append({
                        "gap_type": "LOW_EXECUTION_RATE",
                        "execution_rate": execution_rate,
                        "opportunities": len(opportunities),
                        "executed": len(actual_trades)
                    })
            
            # Gap 3: No trades despite signals
            if len(opportunities) > 0 and len(actual_trades) == 0:
                execution_gaps.append({
                    "gap_type": "NO_TRADES_DESPITE_SIGNALS",
                    "signal_count": len(opportunities),
                    "strong_signals": len([opp for opp in opportunities if opp['signal_strength'] > 0.7])
                })
            
            self.logger.info(f"📊 Found {len(execution_gaps)} execution gaps")
            return execution_gaps
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing execution gaps: {e}")
            return execution_gaps
    
    def _generate_verification_summary(self, signals: List[Dict], opportunities: List[Dict], 
                                     actual_trades: List[Dict], missed_opportunities: List[Dict]) -> Dict:
        """Generate verification summary"""
        summary = {
            "total_signals": len(signals),
            "strong_signals": len([s for s in signals if s['signal_strength'] > 0.5]),
            "opportunities_analyzed": len(opportunities),
            "approved_opportunities": len([o for o in opportunities if o['risk_decision'] == 'APPROVED']),
            "trades_executed": len(actual_trades),
            "missed_opportunities": len(missed_opportunities),
            "execution_rate": len(actual_trades) / len(opportunities) if opportunities else 0,
            "approval_rate": len([o for o in opportunities if o['risk_decision'] == 'APPROVED']) / len(opportunities) if opportunities else 0,
            "missed_pnl_estimate": sum([mo['potential_pnl']['potential_pnl'] for mo in missed_opportunities]),
            "verification_status": "PASSED" if len(missed_opportunities) == 0 else "ATTENTION_REQUIRED"
        }
        
        return summary
    
    def _get_market_status(self, date: datetime) -> Dict:
        """Get market status for the date"""
        try:
            # Simple market status check
            is_weekday = date.weekday() < 5
            is_market_open = is_weekday and 9 <= date.hour <= 16
            
            return {
                "is_weekday": is_weekday,
                "is_market_open": is_market_open,
                "market_hours": "09:30-16:00 EST",
                "date": date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market status: {e}")
            return {"error": str(e)}
    
    def _save_verification_results(self, results: Dict):
        """Save verification results to file"""
        try:
            date_str = results["date"]
            filename = f"trade_verification_{date_str}.json"
            filepath = self.verification_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Verification results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving verification results: {e}")
    
    def create_verification_report(self, date: datetime = None) -> str:
        """Create a formatted verification report"""
        if date is None:
            date = datetime.now(self.eastern)
        
        try:
            # Load or generate verification results
            verification_file = self.verification_dir / f"trade_verification_{date.strftime('%Y-%m-%d')}.json"
            
            if verification_file.exists():
                with open(verification_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                results = self.verify_trading_session(date)
            
            # Generate formatted report
            summary = results.get("summary", {})
            missed_opps = results.get("missed_opportunities", [])
            execution_gaps = results.get("execution_gaps", [])
            
            report = f"""
[SUMMARY] NEURALTRADER TRADE VERIFICATION REPORT
=====================================
[DATE] Date: {results.get('date', date.strftime('%Y-%m-%d'))}
[TIME] Generated: {results.get('verification_time', datetime.now(self.israel).strftime('%Y-%m-%d %H:%M:%S IST'))}

[SUMMARY] VERIFICATION SUMMARY:
------------------------
Total Signals Generated: {summary.get('total_signals', 0)}
Strong Signals (>0.5): {summary.get('strong_signals', 0)}
Opportunities Analyzed: {summary.get('opportunities_analyzed', 0)}
Approved Opportunities: {summary.get('approved_opportunities', 0)}
Trades Executed: {summary.get('trades_executed', 0)}
Missed Opportunities: {summary.get('missed_opportunities', 0)}

[METRICS] EXECUTION METRICS:
-------------------
Execution Rate: {summary.get('execution_rate', 0):.1%}
Approval Rate: {summary.get('approval_rate', 0):.1%}
Missed P&L Estimate: ${summary.get('missed_pnl_estimate', 0):,.2f}
Verification Status: {summary.get('verification_status', 'UNKNOWN')}

[ALERT] MISSED OPPORTUNITIES:
-----------------------"""
            
            if missed_opps:
                for i, missed in enumerate(missed_opps[:5], 1):  # Show top 5
                    report += f"""
{i}. {missed.get('ticker', 'Unknown')}
   Signal Strength: {missed.get('signal_strength', 0):.3f}
   Confidence: {missed.get('confidence', 0):.1%}
   Current Price: ${missed.get('current_price', 0):.2f}
   Position Size: {missed.get('recommended_position_size', 0)} shares
   Missed P&L: ${missed.get('potential_pnl', {}).get('potential_pnl', 0):.2f}
   Reason: {missed.get('missed_reason', 'Unknown')}"""
            else:
                report += "\n✅ No missed opportunities detected!"
            
            if execution_gaps:
                report += f"""

⚠️ EXECUTION GAPS:
-----------------"""
                for gap in execution_gaps:
                    report += f"""
{gap.get('gap_type', 'Unknown')}: {gap.get('count', 0)} instances"""
                    if 'potential_impact' in gap:
                        report += f" (Impact: ${gap.get('potential_impact', 0):,.2f})"
            
            report += f"""

📧 RECOMMENDATIONS:
------------------
"""
            
            if summary.get('verification_status') == 'ATTENTION_REQUIRED':
                report += """
⚠️ ATTENTION REQUIRED:
- Review missed opportunities immediately
- Check execution system functionality
- Verify risk management settings
- Consider manual intervention if needed"""
            else:
                report += """
✅ VERIFICATION PASSED:
- All approved opportunities were executed
- No execution gaps detected
- System operating normally"""
            
            report += f"""

🔍 DETAILED ANALYSIS:
------------------
Full verification data available at: logs/verification/trade_verification_{date.strftime('%Y-%m-%d')}.json

---
NeuralTrader Trade Verification System
Phase 6.2: Task Scheduler Integration
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error creating verification report: {e}")
            return f"❌ Error generating verification report: {e}"
    
    def send_verification_report(self, date: datetime = None) -> bool:
        """Send verification report via email"""
        try:
            from .notifier import EmailNotifier
            
            # Create report
            report = self.create_verification_report(date)
            
            # Send email
            notifier = EmailNotifier()
            subject = f"🔍 NeuralTrader Trade Verification Report - {date.strftime('%Y-%m-%d') if date else datetime.now(self.israel).strftime('%Y-%m-%d')}"
            
            success = notifier.send_email(
                to_email=notifier.recipient_email,
                subject=subject,
                body=report
            )
            
            if success:
                self.logger.info("📧 Trade verification report sent successfully")
            else:
                self.logger.error("❌ Failed to send trade verification report")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error sending verification report: {e}")
            return False
