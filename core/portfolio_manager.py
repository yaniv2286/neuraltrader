#!/usr/bin/env python3
"""
NeuralTrader Portfolio Manager
Manages a single portfolio CSV file with daily AI signal updates
Tracks positions, P&L, and performance metrics for CAGR calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

class PortfolioManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.portfolio_file = project_root / 'data' / 'portfolio.csv'
        self.reports_dir = project_root / 'reports'
        self.logger = logging.getLogger(__name__)
        
        # Portfolio parameters
        self.max_positions = 20
        self.position_size_usd = 5000  # $5,000 per position
        
        # Initialize portfolio file if doesn't exist
        if not self.portfolio_file.exists():
            self._create_empty_portfolio()
    
    def _create_empty_portfolio(self):
        """Create empty portfolio CSV with headers"""
        empty_df = pd.DataFrame(columns=[
            'Date', 'Ticker', 'Action', 'EntryPrice', 'CurrentPrice', 
            'Quantity', 'PnL_Pct', 'PnL_USD', 'Status', 'DaysHeld',
            'EntryDate', 'ExitDate', 'ExitPrice', 'ExitReason'
        ])
        empty_df.to_csv(self.portfolio_file, index=False)
        self.logger.info(f"[PORTFOLIO] Created new portfolio file: {self.portfolio_file}")
    
    def load_portfolio(self) -> pd.DataFrame:
        """Load existing portfolio from CSV"""
        try:
            if self.portfolio_file.stat().st_size > 0:
                return pd.read_csv(self.portfolio_file)
            else:
                return pd.DataFrame(columns=[
                    'Date', 'Ticker', 'Action', 'EntryPrice', 'CurrentPrice', 
                    'Quantity', 'PnL_Pct', 'PnL_USD', 'Status', 'DaysHeld',
                    'EntryDate', 'ExitDate', 'ExitPrice', 'ExitReason'
                ])
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error loading portfolio: {e}")
            return pd.DataFrame()
    
    def save_portfolio(self, df: pd.DataFrame):
        """Save portfolio to CSV"""
        df.to_csv(self.portfolio_file, index=False)
        self.logger.info(f"[PORTFOLIO] Saved portfolio with {len(df)} positions")
    
    def update_portfolio(self, signals_df: pd.DataFrame, current_date: str = None) -> pd.DataFrame:
        """
        Update portfolio with new signals
        - Close positions where signal changed
        - Add new positions for top signals
        - Update prices for existing positions
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Load existing portfolio
        portfolio = self.load_portfolio()
        
        # Get active positions only
        active_positions = portfolio[portfolio['Status'] == 'ACTIVE'].copy()
        
        # Create signal lookup (handle both lowercase and uppercase column names)
        # NORMALIZE all tickers AND actions to UPPERCASE for consistent matching
        if 'ticker' in signals_df.columns and 'action' in signals_df.columns:
            signal_lookup = dict(zip(signals_df['ticker'].str.upper(), signals_df['action'].str.upper()))
        elif 'Ticker' in signals_df.columns and 'Action' in signals_df.columns:
            signal_lookup = dict(zip(signals_df['Ticker'].str.upper(), signals_df['Action'].str.upper()))
        else:
            self.logger.error(f"[PORTFOLIO] Invalid signals DataFrame columns: {signals_df.columns.tolist()}")
            return portfolio
        
        # Debug: Log signal lookup to verify TSLA and XEL are present
        self.logger.info(f"[PORTFOLIO] Signal lookup created with {len(signal_lookup)} signals")
        for ticker in ['TSLA', 'XEL']:
            if ticker in signal_lookup:
                self.logger.info(f"[PORTFOLIO] {ticker} signal: {signal_lookup[ticker]}")
        
        # Step 1: Close positions ONLY when AI explicitly signals to SELL
        # DO NOT close positions just because they're not in today's signals (that means HOLD)
        positions_to_close = []
        for _, position in active_positions.iterrows():
            ticker = position['Ticker']
            current_action = position['Action']
            
            # Only close if:
            # 1. Ticker IS in today's signals AND
            # 2. Signal action is opposite (BUY position gets SELL signal, or vice versa)
            if ticker in signal_lookup:
                new_signal = signal_lookup[ticker]
                # Close BUY positions only if we get explicit SELL signal
                if current_action == 'BUY' and new_signal == 'SELL':
                    positions_to_close.append(position)
                    self.logger.info(f"[PORTFOLIO] Will close {ticker}: AI changed from BUY to SELL")
                # Close SELL positions only if we get explicit BUY signal
                elif current_action == 'SELL' and new_signal == 'BUY':
                    positions_to_close.append(position)
                    self.logger.info(f"[PORTFOLIO] Will close {ticker}: AI changed from SELL to BUY")
                # Close SELL_SHORT positions only if we get explicit BUY signal
                elif current_action == 'SELL_SHORT' and new_signal == 'BUY':
                    positions_to_close.append(position)
                    self.logger.info(f"[PORTFOLIO] Will close {ticker}: AI changed from SELL_SHORT to BUY")
            # If ticker NOT in signals, that means HOLD - do nothing
        
        # Close positions
        for position in positions_to_close:
            close_price = self._get_current_price(position['Ticker'])
            pnl_pct = self._calculate_pnl_pct(position['Action'], position['EntryPrice'], close_price)
            pnl_usd = self._calculate_pnl_usd(position['Action'], position['EntryPrice'], close_price, position['Quantity'])
            
            # Update position in portfolio
            mask = (portfolio['Ticker'] == position['Ticker']) & (portfolio['Status'] == 'ACTIVE')
            portfolio.loc[mask, 'CurrentPrice'] = close_price
            portfolio.loc[mask, 'PnL_Pct'] = pnl_pct
            portfolio.loc[mask, 'PnL_USD'] = pnl_usd
            portfolio.loc[mask, 'Status'] = 'CLOSED'
            portfolio.loc[mask, 'ExitDate'] = current_date
            portfolio.loc[mask, 'ExitPrice'] = close_price
            portfolio.loc[mask, 'ExitReason'] = 'Signal Changed'
            
            self.logger.info(f"[PORTFOLIO] Closed {position['Ticker']} position: PnL {pnl_pct:.2f}%")
        
        # Step 2: Update prices for remaining active positions
        active_tickers = set(active_positions[~active_positions['Ticker'].isin([p['Ticker'] for p in positions_to_close])]['Ticker'])
        for ticker in active_tickers:
            current_price = self._get_current_price(ticker)
            mask = (portfolio['Ticker'] == ticker) & (portfolio['Status'] == 'ACTIVE')
            if not portfolio[mask].empty:
                entry_price = portfolio.loc[mask, 'EntryPrice'].iloc[0]
                action = portfolio.loc[mask, 'Action'].iloc[0]
                quantity = portfolio.loc[mask, 'Quantity'].iloc[0]
                
                pnl_pct = self._calculate_pnl_pct(action, entry_price, current_price)
                pnl_usd = self._calculate_pnl_usd(action, entry_price, current_price, quantity)
                
                portfolio.loc[mask, 'CurrentPrice'] = current_price
                portfolio.loc[mask, 'PnL_Pct'] = pnl_pct
                portfolio.loc[mask, 'PnL_USD'] = pnl_usd
                portfolio.loc[mask, 'DaysHeld'] = portfolio.loc[mask, 'DaysHeld'] + 1
        
        # Step 3: Add new positions (only if we have capacity)
        current_active_count = len(portfolio[portfolio['Status'] == 'ACTIVE'])
        available_slots = self.max_positions - current_active_count
        
        if available_slots > 0:
            # Get top signals that aren't already in portfolio (handle both column cases)
            existing_tickers = set(portfolio[portfolio['Status'] == 'ACTIVE']['Ticker'])
            
            # Handle both lowercase and uppercase column names
            ticker_col = 'ticker' if 'ticker' in signals_df.columns else 'Ticker'
            action_col = 'action' if 'action' in signals_df.columns else 'Action'
            price_col = 'price' if 'price' in signals_df.columns else 'Price'
            
            # CRITICAL FIX: Only add BUY and SELL_SHORT signals as new positions
            # SELL signals should only close existing positions, never create new ones
            new_signals = signals_df[
                (~signals_df[ticker_col].isin(existing_tickers)) & 
                (signals_df[action_col].str.upper().isin(['BUY', 'SELL_SHORT']))
            ]
            
            # Add top N new BUY positions
            new_positions = new_signals.head(available_slots)
            
            for _, signal in new_positions.iterrows():
                entry_price = signal[price_col]
                action = signal[action_col].upper()
                
                # For short positions, quantity is negative (shares sold)
                if action == 'SELL_SHORT':
                    quantity = -int(self.position_size_usd / entry_price)
                else:
                    quantity = int(self.position_size_usd / entry_price)
                
                new_position = pd.DataFrame({
                    'Date': [current_date],
                    'Ticker': [signal[ticker_col].upper()],  # Normalize to uppercase
                    'Action': [action],  # Normalize to uppercase
                    'EntryPrice': [entry_price],
                    'CurrentPrice': [entry_price],
                    'Quantity': [quantity],
                    'PnL_Pct': [0.0],
                    'PnL_USD': [0.0],
                    'Status': ['ACTIVE'],
                    'DaysHeld': [0],
                    'EntryDate': [current_date],
                    'ExitDate': [None],
                    'ExitPrice': [None],
                    'ExitReason': [None]
                })
                
                portfolio = pd.concat([portfolio, new_position], ignore_index=True)
                self.logger.info(f"[PORTFOLIO] Added {signal[ticker_col].upper()} {action} position: {quantity} shares @ ${entry_price:.2f}")
        
        # Step 4: Save updated portfolio
        self.save_portfolio(portfolio)
        
        # Step 5: Generate TradingView CSV
        self._export_tradingview_csv(portfolio, current_date)
        
        # Step 6: Calculate performance metrics
        self._log_performance_metrics(portfolio)
        
        return portfolio
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for ticker from real Tiingo parquet data - NO SILENT FAILURES"""
        # Try to get price from today's signals file first
        try:
            reports_dir = self.project_root / 'reports'
            signal_files = sorted(reports_dir.glob('tradingview_signals_*.csv'))
            if signal_files:
                latest_signals = pd.read_csv(signal_files[-1])
                ticker_row = latest_signals[latest_signals['Ticker'] == ticker]
                if not ticker_row.empty:
                    price = float(ticker_row['Price'].iloc[0])
                    if price > 0 and price < 1000000:  # Validate price is reasonable
                        self.logger.info(f"[PRICE] {ticker}: ${price:.2f} (from signals)")
                        return price
        except Exception as e:
            self.logger.warning(f"[PRICE] Could not get {ticker} price from signals: {e}")
        
        # Fallback to Tiingo parquet data (REAL DATA - NO PLACEHOLDERS)
        try:
            raw_data_dir = self.project_root / 'data' / 'raw'
            parquet_file = raw_data_dir / f"{ticker}.parquet"
            
            if not parquet_file.exists():
                # Try lowercase
                parquet_file = raw_data_dir / f"{ticker.lower()}.parquet"
            
            if not parquet_file.exists():
                self.logger.error(f"[PRICE] [FATAL] Parquet file not found for {ticker}")
                raise FileNotFoundError(f"No parquet file for {ticker}")
            
            # Read parquet and get latest price
            df = pd.read_parquet(parquet_file)
            
            if df.empty:
                self.logger.error(f"[PRICE] [FATAL] Empty data for {ticker}")
                raise ValueError(f"Empty parquet data for {ticker}")
            
            # Get latest close price (adjClose from Tiingo)
            if 'adjClose' in df.columns:
                current_price = float(df['adjClose'].iloc[-1])
            elif 'close' in df.columns:
                current_price = float(df['close'].iloc[-1])
            elif 'Close' in df.columns:
                current_price = float(df['Close'].iloc[-1])
            else:
                self.logger.error(f"[PRICE] [FATAL] No price column for {ticker}. Columns: {df.columns.tolist()}")
                raise ValueError(f"No price column in parquet for {ticker}")
            
            # Validate price
            if current_price <= 0 or current_price > 1000000:
                self.logger.error(f"[PRICE] [FATAL] Invalid price for {ticker}: ${current_price}")
                raise ValueError(f"Invalid price for {ticker}: ${current_price}")
            
            self.logger.info(f"[PRICE] {ticker}: ${current_price:.2f} (real Tiingo data)")
            return current_price
            
        except Exception as e:
            self.logger.error(f"[PRICE] [FATAL] Failed to get real price for {ticker}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # NO SILENT FAILURES - Raise error instead of returning placeholder
            raise RuntimeError(f"[FATAL] Cannot get real price for {ticker} - NO PLACEHOLDER ALLOWED") from e
    
    def _calculate_pnl_pct(self, action: str, entry_price: float, current_price: float) -> float:
        """Calculate P&L percentage"""
        if action == 'BUY':
            # Long position: profit when price goes up
            return ((current_price - entry_price) / entry_price) * 100
        elif action == 'SELL_SHORT':
            # Short position: profit when price goes down
            return ((entry_price - current_price) / entry_price) * 100
        else:  # Legacy SELL (close position)
            return ((entry_price - current_price) / entry_price) * 100
    
    def _calculate_pnl_usd(self, action: str, entry_price: float, current_price: float, quantity: int) -> float:
        """Calculate P&L in USD"""
        return self._calculate_pnl_pct(action, entry_price, current_price) * quantity * entry_price / 100
    
    def _export_tradingview_csv(self, portfolio: pd.DataFrame, current_date: str):
        """Export portfolio to TradingView-compatible CSV"""
        # Get active positions only
        active_positions = portfolio[portfolio['Status'] == 'ACTIVE'].copy()
        
        if active_positions.empty:
            # Create empty CSV
            tradingview_df = pd.DataFrame(columns=['Ticker', 'Action', 'Confidence', 'Price', 'Rank', 'Status'])
        else:
            # Sort by P&L (best performers first)
            active_positions = active_positions.sort_values('PnL_Pct', ascending=False)
            
            # Create TradingView format
            tradingview_df = pd.DataFrame({
                'Ticker': active_positions['Ticker'],
                'Action': active_positions['Action'],
                'Confidence': 0.85,  # Mock confidence for existing positions
                'Price': active_positions['CurrentPrice'],
                'Rank': range(1, len(active_positions) + 1),
                'Status': ['HOLDING'] * len(active_positions)
            })
        
        # Save TradingView CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tradingview_file = self.reports_dir / f'portfolio_tradingview_{timestamp}.csv'
        tradingview_df.to_csv(tradingview_file, index=False)
        
        self.logger.info(f"[PORTFOLIO] Exported {len(active_positions)} positions to TradingView: {tradingview_file}")
        
        # Update global variable for email attachment
        import __main__
        __main__.PORTFOLIO_TRADINGVIEW_FILE = str(tradingview_file)
    
    def _log_performance_metrics(self, portfolio: pd.DataFrame):
        """Calculate and log portfolio performance metrics"""
        active_positions = portfolio[portfolio['Status'] == 'ACTIVE']
        closed_positions = portfolio[portfolio['Status'] == 'CLOSED']
        
        if not portfolio.empty:
            total_pnl_usd = portfolio['PnL_USD'].sum()
            total_invested = len(portfolio) * self.position_size_usd
            
            # Win rate
            winning_trades = len(portfolio[portfolio['PnL_Pct'] > 0])
            total_trades = len(portfolio[portfolio['Status'] == 'CLOSED'])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Average P&L
            avg_pnl_pct = portfolio['PnL_Pct'].mean()
            
            self.logger.info(f"[PORTFOLIO] Performance Summary:")
            self.logger.info(f"  Total P&L: ${total_pnl_usd:,.2f}")
            self.logger.info(f"  Win Rate: {win_rate:.1f}%")
            self.logger.info(f"  Avg P&L: {avg_pnl_pct:.2f}%")
            self.logger.info(f"  Active Positions: {len(active_positions)}")
            self.logger.info(f"  Closed Positions: {len(closed_positions)}")

def update_portfolio_with_signals(signals_file: str, project_root: Path):
    """Main function to update portfolio with new signals"""
    # Load today's signals
    signals_df = pd.read_csv(signals_file)
    
    # Initialize portfolio manager
    manager = PortfolioManager(project_root)
    
    # Update portfolio
    updated_portfolio = manager.update_portfolio(signals_df)
    
    return updated_portfolio

if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).resolve().parent.parent
    signals_file = project_root / 'reports' / 'tradingview_signals_20260302_195852.csv'
    
    portfolio = update_portfolio_with_signals(signals_file, project_root)
    print(f"Portfolio updated with {len(portfolio)} positions")
