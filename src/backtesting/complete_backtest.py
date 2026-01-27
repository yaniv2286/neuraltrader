"""
Phase 5: Complete Backtesting System (Steps 1-7)
Full trading system with risk management, portfolio optimization, and performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data.enhanced_preprocess import build_enhanced_model_input
from models.model_trainer import ModelTrainer
from models.model_selector import ModelSelector

class CompleteBacktester:
    """
    Complete backtesting system with all Phase 5 features
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 1.0,
        slippage_pct: float = 0.001,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.02,
        max_positions: int = 10
    ):
        """
        Initialize complete backtester
        
        Args:
            initial_capital: Starting capital ($100,000)
            commission: Commission per trade ($1)
            slippage_pct: Slippage percentage (0.1%)
            position_size_pct: Position size as % of portfolio (10%)
            stop_loss_pct: Stop loss percentage (2%)
            max_positions: Maximum concurrent positions (10)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
        
        # Performance tracking
        self.daily_returns = []
        self.peak_value = initial_capital
        self.max_drawdown = 0
        
    def run_complete_backtest(
        self,
        tickers: List[str],
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31'
    ) -> Dict:
        """
        Run complete backtest with all features
        
        Returns comprehensive performance report
        """
        print("="*70)
        print("PHASE 5: COMPLETE BACKTESTING SYSTEM")
        print("="*70)
        print(f"\n🎯 Testing {len(tickers)} tickers: {', '.join(tickers)}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"\n⚙️ Configuration:")
        print(f"   Position Size: {self.position_size_pct*100:.0f}% per trade")
        print(f"   Stop Loss: {self.stop_loss_pct*100:.0f}%")
        print(f"   Max Positions: {self.max_positions}")
        print(f"   Commission: ${self.commission} per trade")
        print(f"   Slippage: {self.slippage_pct*100:.2f}%")
        
        # Train models and generate predictions
        print(f"\n{'='*70}")
        print("TRAINING MODELS")
        print(f"{'='*70}")
        
        models_data = {}
        
        for ticker in tickers:
            try:
                print(f"\n📊 {ticker}...")
                
                # Load data
                df = build_enhanced_model_input(
                    ticker=ticker,
                    timeframes=['1d'],
                    start=start_date,
                    end=end_date,
                    validate_data=True,
                    create_features=True
                )
                
                # Train model
                trainer = ModelTrainer(use_log_returns=True, n_features=25)
                X, y = trainer.prepare_data(df, target_col='close')
                
                # Split
                n_samples = len(X)
                train_size = int(0.70 * n_samples)
                
                X_train = X.iloc[:train_size]
                y_train = y.iloc[:train_size]
                X_all = X
                
                # Feature selection
                selected_features = trainer.select_features(X_train, y_train, method='correlation')
                X_train_selected = X_train[selected_features]
                X_all_selected = X_all[selected_features]
                
                # Train
                selector = ModelSelector()
                recommendations = selector.get_recommended_models(task_type='stock_prediction')
                model_class = recommendations['primary']['model']
                
                model_params = {
                    'n_estimators': 300,
                    'max_depth': 3,
                    'learning_rate': 0.02,
                    'tree_method': 'hist',
                    'reg_alpha': 0.5,
                    'reg_lambda': 1.0,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3
                }
                
                model = model_class(**model_params)
                model.fit(X_train_selected.values, y_train.values)
                
                # Generate predictions for all data
                predictions = model.predict(X_all_selected.values)
                
                # Store
                models_data[ticker] = {
                    'data': df,
                    'predictions': predictions,
                    'dates': [str(idx)[:10] if 'Period' not in str(idx) else str(idx).split("'")[1] for idx in X_all.index]
                }
                
                print(f"   ✅ Trained ({len(df)} days)")
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
                continue
        
        # Simulate trading
        print(f"\n{'='*70}")
        print("SIMULATING TRADING")
        print(f"{'='*70}")
        
        # Get all unique dates
        all_dates = set()
        for data in models_data.values():
            all_dates.update(data['dates'])
        all_dates = sorted(list(all_dates))
        
        print(f"\n📅 Simulating {len(all_dates)} trading days...")
        
        # Day-by-day simulation
        for i, date in enumerate(all_dates):
            # Get current prices and predictions for all tickers
            current_data = {}
            for ticker, data in models_data.items():
                if date in data['dates']:
                    idx = data['dates'].index(date)
                    current_data[ticker] = {
                        'price': data['data']['close'].iloc[idx],
                        'prediction': data['predictions'][idx]
                    }
            
            # Check stop losses
            for ticker in list(self.positions.keys()):
                if ticker in current_data:
                    current_price = current_data[ticker]['price']
                    entry_price = self.positions[ticker]['entry_price']
                    loss_pct = (current_price - entry_price) / entry_price
                    
                    if loss_pct <= -self.stop_loss_pct:
                        # Stop loss triggered
                        self._execute_sell(ticker, current_price, date, 0, 'STOP_LOSS')
            
            # Generate signals
            for ticker, info in current_data.items():
                prediction = info['prediction']
                price = info['price']
                
                # BUY signal: prediction > 0.5%
                if prediction > 0.005 and ticker not in self.positions and len(self.positions) < self.max_positions:
                    self._execute_buy(ticker, price, date, prediction)
                
                # SELL signal: prediction < -0.5% and we hold the stock
                elif prediction < -0.005 and ticker in self.positions:
                    self._execute_sell(ticker, price, date, prediction, 'SIGNAL')
            
            # Track portfolio value
            portfolio_value = self._calculate_portfolio_value(current_data)
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Track drawdown
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Calculate final results
        results = self._calculate_performance()
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _execute_buy(self, ticker: str, price: float, date: str, signal: float) -> bool:
        """Execute buy order"""
        position_value = self.cash * self.position_size_pct
        execution_price = price * (1 + self.slippage_pct)
        shares = int(position_value / execution_price)
        
        if shares == 0:
            return False
        
        cost = shares * execution_price + self.commission
        
        if cost > self.cash:
            return False
        
        self.cash -= cost
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': execution_price,
            'entry_date': date
        }
        
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': execution_price,
            'cost': cost,
            'signal': signal
        })
        
        return True
    
    def _execute_sell(self, ticker: str, price: float, date: str, signal: float, reason: str) -> bool:
        """Execute sell order"""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        shares = position['shares']
        entry_price = position['entry_price']
        
        execution_price = price * (1 - self.slippage_pct)
        proceeds = shares * execution_price - self.commission
        
        pnl = proceeds - (shares * entry_price)
        pnl_pct = (execution_price - entry_price) / entry_price * 100
        
        self.cash += proceeds
        del self.positions[ticker]
        
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': execution_price,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'signal': signal,
            'reason': reason
        })
        
        return True
    
    def _calculate_portfolio_value(self, current_data: Dict) -> float:
        """Calculate current portfolio value"""
        position_value = 0
        for ticker, position in self.positions.items():
            if ticker in current_data:
                position_value += position['shares'] * current_data[ticker]['price']
        
        return self.cash + position_value
    
    def _calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        df_portfolio = pd.DataFrame(self.portfolio_values)
        df_trades = pd.DataFrame(self.trades)
        
        # Basic metrics
        final_value = df_portfolio['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate daily returns
        df_portfolio['daily_return'] = df_portfolio['value'].pct_change()
        
        # Annual return
        days = len(df_portfolio)
        years = days / 252
        annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = df_portfolio['daily_return'] - (0.02/252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Win rate
        sell_trades = df_trades[df_trades['action'] == 'SELL']
        if len(sell_trades) > 0:
            winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
            win_rate = winning_trades / len(sell_trades) * 100
            avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'].mean()) if len(sell_trades) - winning_trades > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': len(df_trades),
            'winning_trades': winning_trades if len(sell_trades) > 0 else 0,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_history': df_portfolio,
            'trades_history': df_trades
        }
    
    def _print_summary(self, results: Dict):
        """Print performance summary"""
        print(f"\n{'='*70}")
        print("BACKTEST RESULTS")
        print(f"{'='*70}")
        
        print(f"\n💰 RETURNS:")
        print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"   Final Value: ${results['final_value']:,.2f}")
        print(f"   Total Return: {results['total_return_pct']:.2f}%")
        print(f"   Annual Return: {results['annual_return_pct']:.2f}%")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        
        print(f"\n📈 TRADING STATS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   Win Rate: {results['win_rate_pct']:.2f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Avg Win: ${results['avg_win']:,.2f}")
        print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
        
        print(f"\n{'='*70}")
        
        # Verdict
        if results['annual_return_pct'] > 20 and results['sharpe_ratio'] > 1.5 and results['max_drawdown_pct'] < 25:
            print("✅ EXCELLENT PERFORMANCE - Ready for live trading!")
        elif results['annual_return_pct'] > 15 and results['sharpe_ratio'] > 1.0:
            print("✅ GOOD PERFORMANCE - Consider optimization")
        else:
            print("⚠️ NEEDS IMPROVEMENT - Review strategy")
        
        print(f"{'='*70}")


# Run complete backtest
if __name__ == "__main__":
    backtester = CompleteBacktester(
        initial_capital=100000,
        commission=1.0,
        slippage_pct=0.001,
        position_size_pct=0.10,
        stop_loss_pct=0.02,
        max_positions=10
    )
    
    # Test on 5 tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    
    results = backtester.run_complete_backtest(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # Save results
    results['trades_history'].to_csv('tests/phase5_backtest_trades.csv', index=False)
    results['portfolio_history'].to_csv('tests/phase5_portfolio_history.csv', index=False)
    
    print(f"\n📊 Results saved to:")
    print(f"   - tests/phase5_backtest_trades.csv")
    print(f"   - tests/phase5_portfolio_history.csv")
