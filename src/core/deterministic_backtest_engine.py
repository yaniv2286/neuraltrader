"""
Deterministic Backtest Engine - Phase 4 Harness
================================================

Implements the backtest invariants (NON-NEGOTIABLE):
- Universe: same 154 tickers
- Data: all available history per ticker
- Timing: Signal at Close (T) -> Trade at Open (T+1). One-bar delay enforced.
- Costs: MUST be applied to every trade. Missing costs -> ABORT.
- Same cost model across all strategies
- Same adjusted-price policy across all strategies
- No lookahead

This engine runs ALL strategies through the SAME pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.core.strategy_registry import StrategyConfig, get_strategy_registry
from src.core.vt_sweetspot_baseline import VTSweetSpotBaseline, create_baseline_strategy, create_variant_strategy
from src.core.data_store import get_data_store


class BacktestEngineError(Exception):
    """Raised when backtest engine encounters a critical error."""
    pass


@dataclass
class CostModel:
    """
    Trading cost model - applied to EVERY trade.
    
    If costs are missing/0 due to bug -> results invalid -> ABORT.
    """
    commission_per_trade: float = 1.0  # $1 per trade
    spread_pct: float = 0.001  # 0.1% spread
    slippage_pct: float = 0.0005  # 0.05% slippage
    
    def calculate_entry_cost(self, price: float, shares: float) -> float:
        """Calculate cost of entering a position."""
        trade_value = price * shares
        spread_cost = trade_value * self.spread_pct
        slippage_cost = trade_value * self.slippage_pct
        return self.commission_per_trade + spread_cost + slippage_cost
    
    def calculate_exit_cost(self, price: float, shares: float) -> float:
        """Calculate cost of exiting a position."""
        trade_value = price * shares
        spread_cost = trade_value * self.spread_pct
        slippage_cost = trade_value * self.slippage_pct
        return self.commission_per_trade + spread_cost + slippage_cost
    
    def calculate_total_cost(self, entry_price: float, exit_price: float, shares: float) -> float:
        """Calculate total round-trip cost."""
        return self.calculate_entry_cost(entry_price, shares) + self.calculate_exit_cost(exit_price, shares)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'commission_per_trade': self.commission_per_trade,
            'spread_pct': self.spread_pct,
            'slippage_pct': self.slippage_pct
        }


@dataclass
class Trade:
    """Complete trade record for audit trail."""
    # Identity
    strategy_id: str
    ticker: str
    trade_id: int
    
    # Timing (ONE-BAR DELAY ENFORCED)
    signal_date: datetime  # Day T (close)
    execution_date: datetime  # Day T+1 (open)
    exit_date: datetime
    
    # Prices
    entry_price: float
    exit_price: float
    
    # Position
    shares: float
    direction: int  # 1 = LONG, -1 = SHORT
    
    # Costs (MANDATORY)
    entry_cost: float
    exit_cost: float
    total_cost: float
    
    # P&L
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    
    # Reasons (for audit)
    open_reason: str
    close_reason: str
    veto_reason: Optional[str] = None
    
    # Indicator snapshot at entry
    rule_snapshot: Dict[str, float] = field(default_factory=dict)
    
    # Confidence
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert datetime to string for serialization
        d['signal_date'] = self.signal_date.isoformat() if self.signal_date else None
        d['execution_date'] = self.execution_date.isoformat() if self.execution_date else None
        d['exit_date'] = self.exit_date.isoformat() if self.exit_date else None
        return d


@dataclass
class Position:
    """Active position tracking."""
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: float
    direction: int
    peak_price: float
    signal_date: datetime
    open_reason: str
    rule_snapshot: Dict[str, float]
    confidence: float


@dataclass
class BacktestResult:
    """Complete backtest result for a strategy."""
    strategy_id: str
    strategy_name: str
    
    # Performance metrics
    initial_capital: float
    final_capital: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    
    # Yearly breakdown
    yearly_returns: Dict[int, float]
    best_year: int
    worst_year: int
    best_year_return: float
    worst_year_return: float
    
    # Costs
    total_costs_paid: float
    avg_cost_per_trade: float
    
    # PASS/FAIL determination
    pass_fail: str
    pass_fail_reasons: List[str]
    
    # Raw data
    trades: List[Trade]
    equity_curve: pd.DataFrame
    
    # Metadata
    config: StrategyConfig
    ticker_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return_pct': self.total_return_pct,
            'cagr_pct': self.cagr_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': self.win_rate_pct,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'best_year': self.best_year,
            'worst_year': self.worst_year,
            'best_year_return': self.best_year_return,
            'worst_year_return': self.worst_year_return,
            'total_costs_paid': self.total_costs_paid,
            'avg_cost_per_trade': self.avg_cost_per_trade,
            'pass_fail': self.pass_fail,
            'pass_fail_reasons': self.pass_fail_reasons
        }


class DeterministicBacktestEngine:
    """
    Deterministic backtest engine that enforces all invariants.
    
    INVARIANTS (NON-NEGOTIABLE):
    1. One-bar delay: Signal at Close(T) -> Execute at Open(T+1)
    2. Costs applied to EVERY trade
    3. Same cost model for all strategies
    4. Adjusted prices only
    5. No lookahead bias
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        cost_model: Optional[CostModel] = None
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.data_store = get_data_store()
        
    def run_backtest(
        self,
        strategy: VTSweetSpotBaseline,
        signals_df: pd.DataFrame,
        ticker_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run backtest for a single strategy.
        
        Args:
            strategy: Strategy instance
            signals_df: DataFrame of signals
            ticker_data: Pre-loaded ticker data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult with complete metrics
        """
        config = strategy.config
        print(f"\n💰 Running backtest for {config.strategy_id}...")
        
        # Initialize state
        capital = self.initial_capital
        positions: Dict[str, Position] = {}
        trades: List[Trade] = []
        equity_curve = []
        trade_counter = 0
        total_costs = 0.0
        
        # Get all trading days from SPY
        spy_data = self.data_store.get_ticker_data('SPY', start_date, end_date)
        all_dates = spy_data.index.tolist()
        
        # Group signals by execution date
        if not signals_df.empty:
            signals_df['execution_date'] = pd.to_datetime(signals_df['execution_date'])
            signals_by_date = signals_df.groupby(signals_df['execution_date'].dt.date)
        else:
            signals_by_date = {}
        
        for current_date in all_dates:
            date_key = current_date.date()
            
            # Calculate current portfolio value
            portfolio_value = capital
            for ticker, pos in positions.items():
                if ticker in ticker_data and current_date in ticker_data[ticker].index:
                    current_price = ticker_data[ticker].loc[current_date, 'close']
                    portfolio_value += pos.shares * current_price
                    
                    # Update peak price for trailing stop
                    if current_price > pos.peak_price:
                        pos.peak_price = current_price
            
            # Record equity
            equity_curve.append({
                'date': current_date,
                'equity': portfolio_value,
                'cash': capital,
                'positions_count': len(positions)
            })
            
            # Check exits for existing positions
            tickers_to_exit = []
            for ticker, pos in positions.items():
                if ticker not in ticker_data or current_date not in ticker_data[ticker].index:
                    continue
                
                df = ticker_data[ticker]
                row = df.loc[current_date]
                current_price = row['close']
                
                exit_reason = None
                
                # Check stochastic exit
                threshold = config.stoch_exit_threshold
                if 'daily_stoch_k' in row and row['daily_stoch_k'] < threshold:
                    exit_reason = f"daily_K < {threshold}"
                elif 'daily_stoch_d' in row and row['daily_stoch_d'] < threshold:
                    exit_reason = f"daily_D < {threshold}"
                
                # Check trailing stop
                if exit_reason is None:
                    stop_price = pos.peak_price * (1 - config.trailing_stop_pct)
                    if current_price <= stop_price:
                        exit_reason = f"trailing_stop ({config.trailing_stop_pct*100:.1f}%)"
                
                if exit_reason:
                    tickers_to_exit.append((ticker, current_price, exit_reason))
            
            # Execute exits
            for ticker, exit_price, exit_reason in tickers_to_exit:
                pos = positions[ticker]
                
                # Calculate costs (MANDATORY)
                exit_cost = self.cost_model.calculate_exit_cost(exit_price, pos.shares)
                entry_cost = self.cost_model.calculate_entry_cost(pos.entry_price, pos.shares)
                total_cost = entry_cost + exit_cost
                
                # Validate costs are non-zero
                if total_cost <= 0:
                    raise BacktestEngineError(
                        f"ABORT: Zero cost detected for trade. Costs MUST be applied. "
                        f"Ticker: {ticker}, Entry: {pos.entry_price}, Exit: {exit_price}"
                    )
                
                # Calculate P&L
                gross_pnl = (exit_price - pos.entry_price) * pos.shares * pos.direction
                net_pnl = gross_pnl - total_cost
                pnl_pct = (net_pnl / (pos.entry_price * pos.shares)) * 100
                
                # Return capital
                exit_value = pos.shares * exit_price - exit_cost
                capital += exit_value
                total_costs += total_cost
                
                # Record trade
                trade_counter += 1
                trade = Trade(
                    strategy_id=config.strategy_id,
                    ticker=ticker,
                    trade_id=trade_counter,
                    signal_date=pos.signal_date,
                    execution_date=pos.entry_date,
                    exit_date=current_date,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.shares,
                    direction=pos.direction,
                    entry_cost=entry_cost,
                    exit_cost=exit_cost,
                    total_cost=total_cost,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    open_reason=pos.open_reason,
                    close_reason=exit_reason,
                    rule_snapshot=pos.rule_snapshot,
                    confidence_score=pos.confidence
                )
                trades.append(trade)
                
                del positions[ticker]
            
            # Check for new entries (signals for this execution date)
            if date_key in signals_by_date.groups:
                day_signals = signals_by_date.get_group(date_key)
                
                for _, signal in day_signals.iterrows():
                    ticker = signal['ticker']
                    
                    # Skip if already in position
                    if ticker in positions:
                        continue
                    
                    # Skip if max positions reached
                    if len(positions) >= config.max_positions:
                        continue
                    
                    # Skip if ticker data not available
                    if ticker not in ticker_data or current_date not in ticker_data[ticker].index:
                        continue
                    
                    # Get execution price (OPEN of T+1 - one bar delay enforced)
                    execution_price = ticker_data[ticker].loc[current_date, 'open']
                    
                    # Calculate position size
                    position_value = portfolio_value * config.max_position_pct
                    shares = position_value / execution_price
                    
                    # Calculate entry cost
                    entry_cost = self.cost_model.calculate_entry_cost(execution_price, shares)
                    
                    # Check if we have enough capital
                    total_entry_cost = (shares * execution_price) + entry_cost
                    if total_entry_cost > capital:
                        continue
                    
                    # Deduct from capital
                    capital -= total_entry_cost
                    
                    # Create position
                    positions[ticker] = Position(
                        ticker=ticker,
                        entry_date=current_date,
                        entry_price=execution_price,
                        shares=shares,
                        direction=1,  # LONG only in v1
                        peak_price=execution_price,
                        signal_date=pd.to_datetime(signal['signal_date']),
                        open_reason="Sweet Spot entry conditions met",
                        rule_snapshot={
                            'daily_stoch_k': signal.get('daily_stoch_k', np.nan),
                            'daily_stoch_d': signal.get('daily_stoch_d', np.nan),
                            'weekly_stoch_k': signal.get('weekly_stoch_k', np.nan),
                            'weekly_stoch_d': signal.get('weekly_stoch_d', np.nan),
                            'sma_25': signal.get('sma_25', np.nan),
                            'sma_50': signal.get('sma_50', np.nan),
                            'sma_100': signal.get('sma_100', np.nan),
                            'sma_200': signal.get('sma_200', np.nan)
                        },
                        confidence=signal.get('confidence', 1.0)
                    )
        
        # Close remaining positions at end
        final_date = all_dates[-1]
        for ticker, pos in list(positions.items()):
            if ticker in ticker_data:
                exit_price = ticker_data[ticker].iloc[-1]['close']
                
                exit_cost = self.cost_model.calculate_exit_cost(exit_price, pos.shares)
                entry_cost = self.cost_model.calculate_entry_cost(pos.entry_price, pos.shares)
                total_cost = entry_cost + exit_cost
                
                gross_pnl = (exit_price - pos.entry_price) * pos.shares * pos.direction
                net_pnl = gross_pnl - total_cost
                pnl_pct = (net_pnl / (pos.entry_price * pos.shares)) * 100
                
                exit_value = pos.shares * exit_price - exit_cost
                capital += exit_value
                total_costs += total_cost
                
                trade_counter += 1
                trade = Trade(
                    strategy_id=config.strategy_id,
                    ticker=ticker,
                    trade_id=trade_counter,
                    signal_date=pos.signal_date,
                    execution_date=pos.entry_date,
                    exit_date=final_date,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.shares,
                    direction=pos.direction,
                    entry_cost=entry_cost,
                    exit_cost=exit_cost,
                    total_cost=total_cost,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    open_reason=pos.open_reason,
                    close_reason="end_of_backtest",
                    rule_snapshot=pos.rule_snapshot,
                    confidence_score=pos.confidence
                )
                trades.append(trade)
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        final_capital = capital
        
        # Validate costs were applied
        if len(trades) > 0 and total_costs <= 0:
            raise BacktestEngineError(
                f"ABORT: Total costs is {total_costs} for {len(trades)} trades. "
                f"Costs MUST be applied to every trade."
            )
        
        result = self._calculate_metrics(
            config=config,
            trades=trades,
            equity_df=equity_df,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_costs=total_costs
        )
        
        return result
    
    def _calculate_metrics(
        self,
        config: StrategyConfig,
        trades: List[Trade],
        equity_df: pd.DataFrame,
        initial_capital: float,
        final_capital: float,
        total_costs: float
    ) -> BacktestResult:
        """Calculate all performance metrics."""
        
        # Basic returns
        total_return = (final_capital / initial_capital - 1) * 100
        
        # CAGR
        if len(equity_df) > 0:
            years = (equity_df['date'].max() - equity_df['date'].min()).days / 365.25
            cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        else:
            cagr = 0
            years = 0
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade statistics
        if trades:
            trades_df = pd.DataFrame([t.to_dict() for t in trades])
            winning = trades_df[trades_df['net_pnl'] > 0]
            losing = trades_df[trades_df['net_pnl'] <= 0]
            
            win_rate = len(winning) / len(trades_df) * 100
            
            gross_profit = winning['net_pnl'].sum() if len(winning) > 0 else 0
            gross_loss = abs(losing['net_pnl'].sum()) if len(losing) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_cost = total_costs / len(trades)
        else:
            win_rate = 0
            profit_factor = 0
            avg_cost = 0
            winning = pd.DataFrame()
            losing = pd.DataFrame()
        
        # Yearly returns
        equity_df['year'] = pd.to_datetime(equity_df['date']).dt.year
        yearly_returns = {}
        
        for year in equity_df['year'].unique():
            year_data = equity_df[equity_df['year'] == year]
            if len(year_data) > 1:
                start_eq = year_data.iloc[0]['equity']
                end_eq = year_data.iloc[-1]['equity']
                yearly_returns[int(year)] = (end_eq / start_eq - 1) * 100
        
        if yearly_returns:
            best_year = max(yearly_returns, key=yearly_returns.get)
            worst_year = min(yearly_returns, key=yearly_returns.get)
            best_year_return = yearly_returns[best_year]
            worst_year_return = yearly_returns[worst_year]
        else:
            best_year = worst_year = 0
            best_year_return = worst_year_return = 0
        
        # Sharpe & Sortino (simplified)
        if len(equity_df) > 1:
            daily_returns = equity_df['equity'].pct_change().dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                downside = daily_returns[daily_returns < 0]
                sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
            else:
                sharpe = sortino = 0
        else:
            sharpe = sortino = 0
        
        # PASS/FAIL determination
        pass_fail_reasons = []
        
        # Acceptance criteria from spec
        # (Will be compared against baseline in final report)
        if cagr < 0:
            pass_fail_reasons.append(f"Negative CAGR: {cagr:.2f}%")
        if max_drawdown < -50:
            pass_fail_reasons.append(f"Excessive drawdown: {max_drawdown:.2f}%")
        if len(trades) == 0:
            pass_fail_reasons.append("No trades executed")
        
        pass_fail = "PASS" if len(pass_fail_reasons) == 0 else "FAIL"
        
        return BacktestResult(
            strategy_id=config.strategy_id,
            strategy_name=config.strategy_name,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return,
            cagr_pct=cagr,
            max_drawdown_pct=max_drawdown,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            yearly_returns=yearly_returns,
            best_year=best_year,
            worst_year=worst_year,
            best_year_return=best_year_return,
            worst_year_return=worst_year_return,
            total_costs_paid=total_costs,
            avg_cost_per_trade=avg_cost,
            pass_fail=pass_fail,
            pass_fail_reasons=pass_fail_reasons,
            trades=trades,
            equity_curve=equity_df,
            config=config,
            ticker_summary={}
        )
