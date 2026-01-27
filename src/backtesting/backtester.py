"""
Backtesting Engine - Phase 5 Step 1
Simulates trading over historical data to validate strategy performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.enhanced_preprocess import build_enhanced_model_input
from models.model_trainer import ModelTrainer
from models.model_selector import ModelSelector

class Backtester:
    """
    Complete backtesting engine for trading strategy validation
    
    Features:
    - Multi-ticker support (test on all 45 stocks)
    - Realistic trade execution
    - Transaction costs
    - Position sizing
    - Risk management
    - Performance tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 1.0,
        slippage_pct: float = 0.001,
        position_size_pct: float = 0.10
    ):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting portfolio value ($100,000 default)
            commission: Commission per trade ($1 default)
            slippage_pct: Slippage as percentage (0.1% default)
            position_size_pct: Position size as % of portfolio (10% default)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        
        # Portfolio tracking
        self.cash = initial_capital
        self.positions = {}  # {ticker: {'shares': int, 'entry_price': float}}
        self.portfolio_value_history = []
        self.trades = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value (cash + positions)
        
        Args:
            current_prices: Dict of {ticker: current_price}
            
        Returns:
            Total portfolio value
        """
        position_value = 0
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                position_value += position['shares'] * current_prices[ticker]
        
        return self.cash + position_value
    
    def execute_buy(
        self,
        ticker: str,
        price: float,
        date: str,
        signal_strength: float
    ) -> bool:
        """
        Execute a buy order
        
        Args:
            ticker: Stock ticker
            price: Current price
            date: Trade date
            signal_strength: Model prediction strength
            
        Returns:
            True if trade executed, False otherwise
        """
        # Calculate position size
        position_value = self.cash * self.position_size_pct
        
        # Apply slippage (buy at slightly higher price)
        execution_price = price * (1 + self.slippage_pct)
        
        # Calculate shares (round down to whole shares)
        shares = int(position_value / execution_price)
        
        if shares == 0:
            return False
        
        # Calculate total cost
        cost = shares * execution_price + self.commission
        
        # Check if we have enough cash
        if cost > self.cash:
            return False
        
        # Execute trade
        self.cash -= cost
        
        if ticker in self.positions:
            # Add to existing position (average up)
            old_shares = self.positions[ticker]['shares']
            old_price = self.positions[ticker]['entry_price']
            new_shares = old_shares + shares
            new_avg_price = ((old_shares * old_price) + (shares * execution_price)) / new_shares
            
            self.positions[ticker] = {
                'shares': new_shares,
                'entry_price': new_avg_price
            }
        else:
            # New position
            self.positions[ticker] = {
                'shares': shares,
                'entry_price': execution_price
            }
        
        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': execution_price,
            'cost': cost,
            'signal_strength': signal_strength
        })
        
        self.total_trades += 1
        
        return True
    
    def execute_sell(
        self,
        ticker: str,
        price: float,
        date: str,
        signal_strength: float,
        reason: str = 'SIGNAL'
    ) -> bool:
        """
        Execute a sell order
        
        Args:
            ticker: Stock ticker
            price: Current price
            date: Trade date
            signal_strength: Model prediction strength
            reason: Reason for sell (SIGNAL, STOP_LOSS, etc.)
            
        Returns:
            True if trade executed, False otherwise
        """
        # Check if we have a position
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        shares = position['shares']
        entry_price = position['entry_price']
        
        # Apply slippage (sell at slightly lower price)
        execution_price = price * (1 - self.slippage_pct)
        
        # Calculate proceeds
        proceeds = shares * execution_price - self.commission
        
        # Calculate profit/loss
        pnl = proceeds - (shares * entry_price)
        pnl_pct = (execution_price - entry_price) / entry_price * 100
        
        # Execute trade
        self.cash += proceeds
        del self.positions[ticker]
        
        # Track win/loss
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
        
        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': execution_price,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'signal_strength': signal_strength,
            'reason': reason
        })
        
        self.total_trades += 1
        
        return True
    
    def run_backtest(
        self,
        tickers: List[str],
        start_date: str = '2004-01-01',
        end_date: str = '2024-12-31',
        verbose: bool = True
    ) -> Dict:
        """
        Run complete backtest on multiple tickers
        
        Args:
            tickers: List of stock tickers to trade
            start_date: Backtest start date
            end_date: Backtest end date
            verbose: Print progress
            
        Returns:
            Dictionary with backtest results
        """
        if verbose:
            print("="*70)
            print("PHASE 5 STEP 1: BACKTESTING ENGINE")
            print("="*70)
            print(f"\n🎯 Backtesting {len(tickers)} tickers from {start_date} to {end_date}")
            print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
            print(f"📊 Position Size: {self.position_size_pct*100:.1f}% per trade")
        
        # Train models for each ticker
        models = {}
        predictions = {}
        
        for ticker in tickers:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Training model for {ticker}")
                print(f"{'='*70}")
            
            try:
                # Load data
                df = build_enhanced_model_input(
                    ticker=ticker,
                    timeframes=['1d'],
                    start=start_date,
                    end=end_date,
                    validate_data=True,
                    create_features=True
                )
                
                if verbose:
                    print(f"✅ Loaded {len(df)} days for {ticker}")
                
                # Train model
                trainer = ModelTrainer(use_log_returns=True, n_features=25)
                X, y = trainer.prepare_data(df, target_col='close')
                
                # Split data
                n_samples = len(X)
                train_size = int(0.70 * n_samples)
                
                X_train = X.iloc[:train_size]
                y_train = y.iloc[:train_size]
                X_test = X.iloc[train_size:]
                y_test = y.iloc[train_size:]
                
                # Feature selection
                selected_features = trainer.select_features(X_train, y_train, method='correlation')
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                
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
                
                # Store model and data
                models[ticker] = {
                    'model': model,
                    'features': selected_features,
                    'data': df,
                    'X': X,
                    'y': y
                }
                
                if verbose:
                    print(f"✅ Model trained for {ticker}")
                
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error training {ticker}: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULATING TRADES")
            print(f"{'='*70}")
        
        # Simulate trading day-by-day
        # (This is a simplified version - will be enhanced in later steps)
        
        # Get all unique dates across all tickers
        all_dates = set()
        for ticker_data in models.values():
            # Convert index to date strings
            for idx in ticker_data['X'].index:
                date_str = str(idx)[:10] if 'Period' not in str(idx) else str(idx).split("'")[1]
                all_dates.add(date_str)
        
        all_dates = sorted(list(all_dates))
        
        if verbose:
            print(f"\n📅 Simulating {len(all_dates)} trading days...")
        
        # For now, just track portfolio value
        # Full trading logic will be added in next steps
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.cash,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'trades': self.trades,
            'portfolio_history': self.portfolio_value_history
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("BACKTEST COMPLETE")
            print(f"{'='*70}")
            print(f"\n💰 Final Capital: ${self.cash:,.2f}")
            print(f"📊 Total Trades: {self.total_trades}")
            print(f"✅ Winning Trades: {self.winning_trades}")
            print(f"❌ Losing Trades: {self.losing_trades}")
        
        return results


def run_simple_backtest():
    """
    Simple backtest example to test the framework
    """
    # Initialize backtester
    backtester = Backtester(
        initial_capital=100000,
        commission=1.0,
        slippage_pct=0.001,
        position_size_pct=0.10
    )
    
    # Test on 3 tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Run backtest
    results = backtester.run_backtest(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-12-31',
        verbose=True
    )
    
    return results


if __name__ == "__main__":
    results = run_simple_backtest()
