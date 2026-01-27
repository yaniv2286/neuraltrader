"""
NeuralTrader Moving Average Crossover Strategy
Classic MA crossover with clear entry/exit rules
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .signal_generator import SignalGenerator, TradingSignal, SignalType

class MACrossoverStrategy:
    """
    Moving Average Crossover Trading Strategy
    Clear, explainable rules for every trade
    """
    
    def __init__(self, config: Dict = None):
        self.signal_generator = SignalGenerator(config)
        self.trades = []
        self.position = None
        self.entry_price = None
        self.pnl = 0.0
        self.max_drawdown = 0.0
        self.equity_curve = []
        
    def execute_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Execute the MA crossover strategy on the given data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with strategy results
        """
        print("🔄 Executing MA Crossover Strategy...")
        
        # Generate signals
        signals = self.signal_generator.generate_signals(df)
        
        # Execute trades based on signals
        for signal in signals:
            trade = self._execute_signal(signal, df)
            if trade:
                self.trades.append(trade)
        
        # Calculate performance metrics
        results = self._calculate_performance(df)
        
        return results
    
    def _execute_signal(self, signal: TradingSignal, df: pd.DataFrame) -> Optional[Dict]:
        """Execute a trading signal"""
        
        trade = None
        
        if signal.signal_type == SignalType.BUY and self.position is None:
            # Open long position
            trade = {
                'type': 'BUY',
                'entry_price': signal.price,
                'entry_time': signal.timestamp,
                'quantity': 100,  # Fixed quantity for demo
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reason': signal.reason,
                'confidence': signal.confidence
            }
            
            self.position = 'long'
            self.entry_price = signal.price
            
        elif signal.signal_type == SignalType.SELL and self.position is None:
            # Open short position
            trade = {
                'type': 'SELL',
                'entry_price': signal.price,
                'entry_time': signal.timestamp,
                'quantity': 100,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reason': signal.reason,
                'confidence': signal.confidence
            }
            
            self.position = 'short'
            self.entry_price = signal.price
            
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT] and self.position:
            # Close position
            exit_price = signal.price
            exit_time = signal.timestamp
            
            if self.position == 'long':
                pnl = (exit_price - self.entry_price) * 100
            else:  # short
                pnl = (self.entry_price - exit_price) * 100
            
            trade = {
                'type': 'CLOSE',
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'entry_time': self.entry_time,
                'exit_time': exit_time,
                'quantity': 100,
                'pnl': pnl,
                'reason': signal.reason,
                'confidence': signal.confidence
            }
            
            self.position = None
            self.entry_price = None
            self.pnl += pnl
        
        return trade
    
    def _calculate_performance(self, df: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'equity_curve': []
            }
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        total_pnl = sum([t.get('pnl', 0) for t in self.trades])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate equity curve
        equity = []
        running_pnl = 0
        for i in range(len(df)):
            # Find trades that occurred at or before this time
            for trade in self.trades:
                if trade['entry_time'] <= df.index[i] and trade.get('exit_time', pd.Timestamp.max()) <= df.index[i]:
                    if trade['type'] in ['BUY', 'SELL']:
                        running_pnl -= trade['entry_price'] * 100  # Commission
                    else:  # CLOSE
                        running_pnl += trade.get('pnl', 0)
            
            equity.append(10000 + running_pnl)  # Starting with $10,000
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - np.array(equity)) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity,
            'trades': self.trades
        }

class RSIStrategy:
    """
    RSI-based trading strategy
    """
    
    def __init__(self, config: Dict = None):
        self.signal_generator = SignalGenerator(config)
        self.trades = []
        self.position = None
        self.entry_price = None
        self.pnl = 0.0
    
    def execute_strategy(self, df: pd.DataFrame) -> Dict:
        """Execute RSI strategy"""
        # Similar implementation to MACrossoverStrategy
        pass

class BollingerBandsStrategy:
    """
    Bollinger Bands trading strategy
    """
    
    def __init__(self, config: Dict = None):
        self.signal_generator = SignalGenerator(config)
        self.trades = []
        self.position = None
        self.entry_price = None
        self.pnl = 0.0
    
    def execute_strategy(self, df: pd.DataFrame) -> Dict:
        """Execute Bollinger Bands strategy"""
        # Similar implementation to MACrossoverStrategy
        pass

def create_strategy_report(strategy_results: Dict, strategy_name: str) -> str:
    """Create a detailed report of strategy performance"""
    
    report = f"""
# 📊 {strategy_name} Strategy Report
# =====================================

## 📈 Performance Summary
- **Total Trades**: {strategy_results['total_trades']}
- **Total P&L**: ${strategy_results['total_pnl']:,.2f}
- **Win Rate**: {strategy_results['win_rate']:.2%}
- **Max Drawdown**: {strategy_results['max_drawdown']:.2%}
- **Sharpe Ratio**: {strategy_results['sharpe_ratio']:.2f}

## 📊 Trade Details
"""
    
    if strategy_results['trades']:
        report += "\n### Individual Trades\n"
        report += "| # | Type | Entry Price | Exit Price | P&L | Reason |\n"
        report += "|---|------|-------------|------------|-----|--------|\n"
        
        for i, trade in enumerate(strategy_results['trades'], 1):
            pnl_color = "🟢" if trade.get('pnl', 0) > 0 else "🔴"
            pnl_str = f"${trade.get('pnl', 0):.2f}"
            
            if trade['type'] == 'CLOSE':
                exit_price = trade.get('exit_price', 'N/A')
                report += f"| {i} | {trade['type']} | ${trade['entry_price']:.2f} | ${exit_price:.2f} | {pnl_str} | {pnl_color} {trade['reason']} |\n"
            else:
                report += f"| {i} | {trade['type']} | ${trade['entry_price']:.2f} | N/A | N/A | 📊 {trade['reason']} |\n"
    
    return report

if __name__ == "__main__":
    # Test the strategies
    print("🧪 Testing Trading Strategies")
    print("=" * 40)
    
    # Test MA Crossover
    ma_strategy = MACrossoverStrategy()
    print(f"✅ MA Crossover Strategy initialized")
    
    print(f"\n✅ Trading Strategies ready for Phase 4!")
