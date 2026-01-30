"""
Backtest Engine - Integrated engine following Trading Constitution.
Combines DataStore, VetoGates, CostModel, and BacktestResult.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import uuid

from .data_store import DataStore, get_data_store
from .veto_gates import VetoGates, VetoResult
from .cost_model import CostModel
from .backtest_result import BacktestResult, Trade, BacktestExcelWriter, create_backtest_result


class BacktestEngine:
    """
    Production backtest engine following Trading Constitution v1.
    
    Key principles:
    - Maximize risk-adjusted return
    - Capital preservation priority
    - All trades must pass veto gates
    - All costs must be applied
    - Strict Excel output schema
    """
    
    def __init__(self, config_path: str = "config/trading_constitution.json"):
        self.config = self._load_config(config_path)
        self.data_store = get_data_store()
        self.veto_gates = VetoGates(config_path)
        self.cost_model = CostModel(config_path)
        
        # Portfolio state
        self.initial_capital = 100000
        self.capital = self.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        self.daily_pnl = 0.0
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        
        # Configuration
        self.position_sizing = self.config.get('position_sizing', {})
        self.exit_rules = self.config.get('exit_rules', {})
        
    def _load_config(self, path: str) -> dict:
        """Load trading constitution config."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def run(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        model_predictions: Optional[Dict[str, pd.DataFrame]] = None
    ) -> BacktestResult:
        """
        Run backtest on specified tickers.
        
        Args:
            tickers: List of tickers to trade
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            model_predictions: Optional dict of ticker -> DataFrame with 'prediction' column
        
        Returns:
            BacktestResult with all metrics and trades
        """
        run_id = f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Log price policy as required
        self.data_store.log_price_policy()
        
        # Reset state
        self._reset_state()
        
        # Load data for all tickers
        print(f"\n📊 Loading data for {len(tickers)} tickers...")
        ticker_data = {}
        for ticker in tickers:
            try:
                df = self.data_store.get_ticker_data(ticker, start_date, end_date)
                ticker_data[ticker] = df
            except Exception as e:
                print(f"   ⚠️ {ticker}: {e}")
        
        print(f"   ✅ Loaded {len(ticker_data)} tickers")
        
        if not ticker_data:
            return self._create_empty_result(run_id)
        
        # Get common date range
        all_dates = set()
        for df in ticker_data.values():
            all_dates.update(df.index.tolist())
        trading_days = sorted(all_dates)
        
        print(f"   📅 Trading period: {trading_days[0].date()} to {trading_days[-1].date()}")
        print(f"   📈 Trading days: {len(trading_days)}")
        
        # Run simulation
        print(f"\n🚀 Running backtest simulation...")
        
        for i, date in enumerate(trading_days):
            self._process_day(date, ticker_data, model_predictions)
            
            # Record equity curve
            self.equity_curve.append({
                'date': date,
                'portfolio_equity': self.capital + self._get_unrealized_pnl(date, ticker_data),
                'drawdown_pct': self.current_drawdown * 100
            })
        
        # Close all remaining positions
        self._close_all_positions(trading_days[-1], ticker_data)
        
        # Create result
        result = self._create_result(run_id, ticker_data)
        
        # Validate against success criteria
        criteria = self.config.get('backtest_success_criteria', {})
        result.validate(criteria)
        
        return result
    
    def _reset_state(self):
        """Reset portfolio state for new backtest."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = 0.0
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
    
    def _process_day(
        self,
        date: datetime,
        ticker_data: Dict[str, pd.DataFrame],
        model_predictions: Optional[Dict[str, pd.DataFrame]]
    ):
        """Process a single trading day."""
        # Reset daily P&L
        self.daily_pnl = 0.0
        
        # Update positions and check exits
        self._check_exits(date, ticker_data)
        
        # Generate signals and check entries
        for ticker, df in ticker_data.items():
            if date not in df.index:
                continue
            
            # Skip if already have position
            if ticker in self.positions:
                continue
            
            # Get prediction/signal
            signal = self._get_signal(ticker, date, df, model_predictions)
            if signal == 0:
                continue
            
            # Get current price and volume
            row = df.loc[date]
            price = row['close']
            volume = row['volume']
            
            # Estimate spread (simplified)
            spread_pct = 0.001  # Default estimate
            
            # Calculate confidence and risk/reward
            confidence = self._calculate_confidence(ticker, signal, df, date, model_predictions)
            risk_reward = self._calculate_risk_reward(ticker, signal, price, df)
            
            # Run veto gates
            passed, veto_results = self.veto_gates.check_all(
                ticker=ticker,
                confidence=confidence,
                risk_reward_ratio=risk_reward,
                volume=volume,
                spread_pct=spread_pct,
                current_positions=len(self.positions),
                sector_exposure=0.0,  # Simplified
                daily_pnl_pct=self.daily_pnl / self.capital if self.capital > 0 else 0,
                portfolio_drawdown_pct=self.current_drawdown
            )
            
            if not passed:
                continue
            
            # Calculate position size
            position_size = self._calculate_position_size(confidence, price)
            if position_size <= 0:
                continue
            
            # Execute entry
            self._execute_entry(
                ticker=ticker,
                direction='LONG' if signal > 0 else 'SHORT',
                date=date,
                price=price,
                position_size=position_size,
                confidence=confidence,
                veto_results=veto_results
            )
        
        # Update drawdown
        current_equity = self.capital + self._get_unrealized_pnl(date, ticker_data)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.current_drawdown = (current_equity - self.peak_equity) / self.peak_equity
    
    def _get_signal(
        self,
        ticker: str,
        date: datetime,
        df: pd.DataFrame,
        model_predictions: Optional[Dict[str, pd.DataFrame]]
    ) -> int:
        """Get trading signal for ticker on date. Returns 1 (long), -1 (short), or 0 (no trade)."""
        # Use model predictions if available
        if model_predictions and ticker in model_predictions:
            pred_df = model_predictions[ticker]
            # Try matching by date
            date_str = str(date)[:10]
            matching = pred_df[pred_df.index.astype(str).str[:10] == date_str]
            
            if len(matching) > 0:
                pred = matching.iloc[0]['prediction']
                confidence = abs(pred)
                
                # SELECTIVE TRADING: Only trade on strong signals
                # For log returns, values are typically 0.001-0.01 range
                min_signal_strength = 0.001  # Minimum prediction magnitude (0.1% expected return)
                
                if confidence < min_signal_strength:
                    return 0  # Signal too weak, skip
                
                # Trade both directions based on prediction
                if pred > min_signal_strength:
                    return 1  # Long signal
                elif pred < -min_signal_strength:
                    return -1  # Short signal
        
        return 0
    
    def _calculate_confidence(
        self,
        ticker: str,
        signal: int,
        df: pd.DataFrame,
        date: datetime,
        model_predictions: Optional[Dict[str, pd.DataFrame]] = None
    ) -> float:
        """Calculate confidence score for trade."""
        base_confidence = 0.50
        
        # Use ML prediction confidence if available
        if model_predictions and ticker in model_predictions:
            pred_df = model_predictions[ticker]
            date_str = str(date)[:10]
            matching = pred_df[pred_df.index.astype(str).str[:10] == date_str]
            if len(matching) > 0 and 'confidence' in matching.columns:
                ml_conf = matching.iloc[0]['confidence']
                # Scale ML confidence (0-1 range typically small values)
                base_confidence = 0.45 + min(0.50, ml_conf * 10)
        
        # Boost for high-confidence tickers
        high_conf_tickers = set(self.config.get('high_confidence_tickers', []))
        if ticker in high_conf_tickers:
            base_confidence += 0.05
        
        return min(0.95, max(0.30, base_confidence))
    
    def _calculate_risk_reward(
        self,
        ticker: str,
        signal: int,
        price: float,
        df: pd.DataFrame
    ) -> float:
        """Calculate risk/reward ratio."""
        # Use ATR for stop distance
        if len(df) < 14:
            return 1.5  # Default
        
        high = df['high'].tail(14)
        low = df['low'].tail(14)
        close = df['close'].tail(14)
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.mean()
        
        # Risk = 2 ATR, Reward = 3 ATR
        risk = 2 * atr / price
        reward = 3 * atr / price
        
        return reward / risk if risk > 0 else 1.5
    
    def _calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size based on risk."""
        base_risk = self.position_sizing.get('base_risk_per_trade', 0.01)
        max_risk = self.position_sizing.get('max_risk_per_trade', 0.02)
        
        # Risk-based sizing
        risk_amount = self.capital * base_risk
        
        # Scale by confidence (but never increase risk)
        if confidence > 0.6:
            risk_amount *= 1.5
        
        # Cap at max risk
        risk_amount = min(risk_amount, self.capital * max_risk)
        
        # Convert to position size (assuming 2% stop loss)
        stop_loss_pct = self.exit_rules.get('trailing_stop_pct', 0.02)
        position_size = risk_amount / stop_loss_pct
        
        # Cap at 20% of capital
        position_size = min(position_size, self.capital * 0.20)
        
        return position_size
    
    def _execute_entry(
        self,
        ticker: str,
        direction: str,
        date: datetime,
        price: float,
        position_size: float,
        confidence: float,
        veto_results: List[VetoResult]
    ):
        """Execute trade entry."""
        # Calculate entry costs
        entry_costs = self.cost_model.calculate_entry_costs(position_size)
        
        # Create trade record
        trade = Trade(
            trade_id=f"T_{len(self.trades)+1:04d}",
            ticker=ticker,
            direction=direction,
            entry_date=date,
            entry_price=price,
            position_size_pct=(position_size / self.capital) * 100,
            capital_allocated=position_size,
            risk_at_entry_pct=self.position_sizing.get('base_risk_per_trade', 0.01) * 100,
            confidence_score=confidence,
            entry_reasons='Model signal + veto passed',
            veto_checks_passed=self.veto_gates.get_passed_gates_string(veto_results),
            regime=self._detect_regime(ticker),
            liquidity_status='HIGH',
            status='OPEN'
        )
        
        # Deduct capital and costs
        self.capital -= position_size + entry_costs.total_cost
        
        # Store position
        self.positions[ticker] = {
            'trade': trade,
            'shares': position_size / price,
            'entry_costs': entry_costs.total_cost,
            'stop_loss': price * (1 - self.exit_rules.get('trailing_stop_pct', 0.02)) if direction == 'LONG'
                        else price * (1 + self.exit_rules.get('trailing_stop_pct', 0.02)),
            'highest_price': price if direction == 'LONG' else None,
            'lowest_price': price if direction == 'SHORT' else None
        }
        
        self.trades.append(trade)
    
    def _check_exits(self, date: datetime, ticker_data: Dict[str, pd.DataFrame]):
        """Check and execute exits for open positions."""
        to_close = []
        
        for ticker, pos in self.positions.items():
            if ticker not in ticker_data:
                continue
            
            df = ticker_data[ticker]
            if date not in df.index:
                continue
            
            row = df.loc[date]
            current_price = row['close']
            trade = pos['trade']
            
            # Update trailing stop
            if trade.direction == 'LONG':
                if current_price > pos.get('highest_price', current_price):
                    pos['highest_price'] = current_price
                    pos['stop_loss'] = current_price * (1 - self.exit_rules.get('trailing_stop_pct', 0.02))
                
                # Check stop loss
                if current_price <= pos['stop_loss']:
                    to_close.append((ticker, current_price, 'trailing_stop'))
            else:  # SHORT
                if current_price < pos.get('lowest_price', current_price):
                    pos['lowest_price'] = current_price
                    pos['stop_loss'] = current_price * (1 + self.exit_rules.get('trailing_stop_pct', 0.02))
                
                # Check stop loss
                if current_price >= pos['stop_loss']:
                    to_close.append((ticker, current_price, 'trailing_stop'))
        
        # Execute exits
        for ticker, exit_price, exit_reason in to_close:
            self._execute_exit(ticker, date, exit_price, exit_reason)
    
    def _execute_exit(self, ticker: str, date: datetime, price: float, reason: str):
        """Execute trade exit."""
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        trade = pos['trade']
        shares = pos['shares']
        entry_costs = pos['entry_costs']
        
        # Calculate P&L
        exit_value = shares * price
        entry_value = trade.capital_allocated
        
        if trade.direction == 'LONG':
            gross_pnl = exit_value - entry_value
        else:
            gross_pnl = entry_value - exit_value
        
        # Apply exit costs
        exit_costs = self.cost_model.calculate_exit_costs(exit_value)
        total_costs = entry_costs + exit_costs.total_cost
        net_pnl = gross_pnl - exit_costs.total_cost
        
        # Update trade record
        trade.exit_date = date
        trade.exit_price = price
        trade.gross_pnl = gross_pnl
        trade.costs_paid = total_costs
        trade.net_pnl = net_pnl
        trade.return_on_allocated_pct = (net_pnl / entry_value) * 100 if entry_value > 0 else 0
        trade.return_on_portfolio_pct = (net_pnl / self.initial_capital) * 100
        trade.exit_reason = reason
        trade.status = 'CLOSED'
        
        # Update capital
        self.capital += exit_value - exit_costs.total_cost
        self.daily_pnl += net_pnl
        
        # Remove position
        del self.positions[ticker]
    
    def _close_all_positions(self, date: datetime, ticker_data: Dict[str, pd.DataFrame]):
        """Close all remaining positions at end of backtest."""
        for ticker in list(self.positions.keys()):
            if ticker in ticker_data and date in ticker_data[ticker].index:
                price = ticker_data[ticker].loc[date, 'close']
                self._execute_exit(ticker, date, price, 'backtest_end')
    
    def _get_unrealized_pnl(self, date: datetime, ticker_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate unrealized P&L for open positions."""
        unrealized = 0.0
        
        for ticker, pos in self.positions.items():
            if ticker not in ticker_data:
                continue
            
            df = ticker_data[ticker]
            if date not in df.index:
                continue
            
            current_price = df.loc[date, 'close']
            trade = pos['trade']
            shares = pos['shares']
            
            if trade.direction == 'LONG':
                unrealized += (current_price - trade.entry_price) * shares
            else:
                unrealized += (trade.entry_price - current_price) * shares
        
        return unrealized
    
    def _detect_regime(self, ticker: str) -> str:
        """Detect market regime (simplified)."""
        return 'NORMAL'
    
    def _create_result(self, run_id: str, ticker_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Create BacktestResult from simulation."""
        equity_df = pd.DataFrame(self.equity_curve)
        
        result = create_backtest_result(
            run_id=run_id,
            trades=self.trades,
            equity_curve=equity_df,
            config=self.config,
            initial_capital=self.initial_capital
        )
        
        result.universe_size = len(ticker_data)
        result.cost_model = self.cost_model.get_config_dict()
        
        # Calculate per-ticker summary
        for ticker in ticker_data.keys():
            ticker_trades = [t for t in self.trades if t.ticker == ticker and t.status == 'CLOSED']
            if ticker_trades:
                winning = [t for t in ticker_trades if t.net_pnl > 0]
                total_pnl = sum(t.net_pnl for t in ticker_trades)
                
                result.stock_summary[ticker] = {
                    'trades': len(ticker_trades),
                    'win_rate': len(winning) / len(ticker_trades) * 100,
                    'total_return': total_pnl / self.initial_capital * 100,
                    'max_drawdown': 0,  # Simplified
                    'profit_factor': 1.0,  # Simplified
                    'passed': len(winning) / len(ticker_trades) > 0.4 if ticker_trades else False
                }
        
        return result
    
    def _create_empty_result(self, run_id: str) -> BacktestResult:
        """Create empty result when no data available."""
        return BacktestResult(
            run_id=run_id,
            timestamp=datetime.now(),
            universe_size=0,
            passed=False,
            failure_reasons=['No data available for backtest']
        )
    
    def write_excel(self, result: BacktestResult, output_path: str) -> str:
        """Write backtest result to Excel file."""
        writer = BacktestExcelWriter(result)
        return writer.write(output_path)
