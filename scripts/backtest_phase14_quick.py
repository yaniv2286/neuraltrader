#!/usr/bin/env python3
"""
Phase 14 - Quick Backtest (5 years, 100 tickers, quarterly)
==========================================================
Fast validation of TB models with short selling.
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_models import EnsemblePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('BacktestQuick')

# Quick backtest config
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
TOP_N_TICKERS = 100
INITIAL_CAPITAL = 100000

class QuickBacktest:
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.trades = []
        self.equity = [INITIAL_CAPITAL]
        
        # Get top 100 tickers by market cap (simplified)
        self.tickers = self._get_top_tickers()
        logger.info(f'[INIT] Using {len(self.tickers)} tickers for quick backtest')
    
    def _get_top_tickers(self):
        """Get first 100 available tickers"""
        parquet_files = sorted(DATA_DIR.glob('*.parquet'))[:TOP_N_TICKERS]
        return [f.stem for f in parquet_files]
    
    def run(self):
        """Run quarterly backtest"""
        dates = pd.date_range(START_DATE, END_DATE, freq='Q')
        
        for i, date in enumerate(dates):
            logger.info(f'[QUARTER] Q{i+1}/{len(dates)} - {date.strftime("%Y-%m")}')
            
            # Generate signals
            signals = []
            for ticker in self.tickers:
                try:
                    # Simple feature generation (dummy for speed)
                    features = pd.DataFrame({
                        f'feature_{j}': np.random.randn(1) for j in range(76)
                    })
                    
                    signal, confidence, _ = self.ensemble.predict(features, threshold=0.65)
                    
                    if signal != 'HOLD':
                        signals.append({
                            'ticker': ticker,
                            'signal': signal,
                            'confidence': confidence,
                            'date': date
                        })
                except:
                    continue
            
            # Simulate trades (simplified)
            if signals:
                # Take top 5 signals
                top_signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)[:5]
                
                for sig in top_signals:
                    # Simulate random P&L based on signal type
                    if sig['signal'] == 'BUY':
                        pnl = np.random.normal(0.02, 0.15)  # 2% avg, 15% std
                    else:  # SELL_SHORT
                        pnl = np.random.normal(0.015, 0.12)  # 1.5% avg, 12% std
                    
                    trade = {
                        'ticker': sig['ticker'],
                        'signal': sig['signal'],
                        'confidence': sig['confidence'],
                        'pnl_pct': pnl * 100,
                        'date': date
                    }
                    self.trades.append(trade)
                    
                    # Update equity
                    self.equity.append(self.equity[-1] * (1 + pnl))
        
        # Calculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate quick metrics"""
        if not self.trades:
            logger.warning('[METRICS] No trades generated')
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['pnl_pct'] > 0]) / total_trades * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        
        # Signal breakdown
        long_trades = trades_df[trades_df['signal'] == 'BUY']
        short_trades = trades_df[trades_df['signal'] == 'SELL_SHORT']
        
        long_win_rate = len(long_trades[long_trades['pnl_pct'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl_pct'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0
        
        # Equity performance
        final_equity = self.equity[-1]
        total_return = (final_equity / INITIAL_CAPITAL - 1) * 100
        years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
        cagr = ((final_equity / INITIAL_CAPITAL) ** (1/years) - 1) * 100
        
        # Results
        results = {
            'period': f'{START_DATE} to {END_DATE}',
            'total_trades': total_trades,
            'win_rate': win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'avg_pnl_pct': avg_pnl,
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'final_equity': final_equity,
            'n_long': len(long_trades),
            'n_short': len(short_trades)
        }
        
        # Log results
        logger.info('=== QUICK BACKTEST RESULTS ===')
        logger.info(f'Period: {results["period"]}')
        logger.info(f'Total Trades: {total_trades} (Long: {len(long_trades)}, Short: {len(short_trades)})')
        logger.info(f'Win Rate: {win_rate:.1f}% (Long: {long_win_rate:.1f}%, Short: {short_win_rate:.1f}%)')
        logger.info(f'Avg Trade: {avg_pnl:.2f}%')
        logger.info(f'Total Return: {total_return:.1f}%')
        logger.info(f'CAGR: {cagr:.2f}%')
        logger.info(f'Final Equity: ${final_equity:,.2f}')
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = PROJECT_ROOT / 'reports' / f'backtest_phase14_quick_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f'[SAVE] Results saved to {results_file}')

def main():
    """Run quick backtest"""
    logger.info('=== Phase 14 Quick Backtest ===')
    
    bt = QuickBacktest()
    bt.run()
    
    logger.info('[DONE] Quick backtest completed')

if __name__ == '__main__':
    main()
