"""
BacktestResult Contract - Authoritative schema for all backtest outputs.
Every backtest MUST output one Excel file with this strict schema.
If a sheet or column is missing → the run is invalid.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import os

@dataclass
class Trade:
    """Single trade record with all required fields."""
    trade_id: str
    ticker: str
    direction: str  # LONG or SHORT
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    position_size_pct: float = 0.0
    capital_allocated: float = 0.0
    risk_at_entry_pct: float = 0.0
    confidence_score: float = 0.0
    gross_pnl: float = 0.0
    costs_paid: float = 0.0
    net_pnl: float = 0.0
    return_on_allocated_pct: float = 0.0
    return_on_portfolio_pct: float = 0.0
    entry_reasons: str = ""
    exit_reason: str = ""
    veto_checks_passed: str = ""
    regime: str = ""
    liquidity_status: str = ""
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED

@dataclass
class BacktestResult:
    """
    Complete backtest result with all required metrics.
    This is the authoritative contract for backtest outputs.
    """
    # Run metadata
    run_id: str
    timestamp: datetime
    universe_size: int
    price_policy: str = "adjusted"
    costs_applied: bool = True
    schema_version: str = "1.0"
    
    # Configuration snapshot
    config: Dict[str, Any] = field(default_factory=dict)
    feature_list: List[str] = field(default_factory=list)
    model_params: Dict[str, Any] = field(default_factory=dict)
    signal_rules: Dict[str, Any] = field(default_factory=dict)
    risk_rules: Dict[str, Any] = field(default_factory=dict)
    cost_model: Dict[str, Any] = field(default_factory=dict)
    
    # Overall performance metrics
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    worst_year_pct: float = 0.0
    best_year_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Pass/Fail status
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    
    # Trade records
    trades: List[Trade] = field(default_factory=list)
    
    # Per-ticker summary
    stock_summary: Dict[str, Dict] = field(default_factory=dict)
    
    # Equity curve
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def validate(self, criteria: Dict) -> bool:
        """Validate backtest against success criteria."""
        self.failure_reasons = []
        
        # Check all required criteria
        if self.cagr_pct < criteria.get('min_annualized_return', 0):
            self.failure_reasons.append(f"CAGR {self.cagr_pct:.1f}% < min {criteria['min_annualized_return']*100:.1f}%")
        
        if abs(self.max_drawdown_pct) > criteria.get('max_drawdown', 0.25):
            self.failure_reasons.append(f"Max DD {self.max_drawdown_pct:.1f}% > max {criteria['max_drawdown']*100:.1f}%")
        
        if self.win_rate_pct < criteria.get('min_win_rate', 0.40) * 100:
            self.failure_reasons.append(f"Win rate {self.win_rate_pct:.1f}% < min {criteria['min_win_rate']*100:.1f}%")
        
        if self.profit_factor < criteria.get('min_profit_factor', 1.0):
            self.failure_reasons.append(f"Profit factor {self.profit_factor:.2f} < min {criteria['min_profit_factor']:.2f}")
        
        self.passed = len(self.failure_reasons) == 0
        return self.passed


class BacktestExcelWriter:
    """
    Writes BacktestResult to Excel with strict schema.
    All sheets and columns are mandatory.
    """
    
    def __init__(self, result: BacktestResult):
        self.result = result
    
    def write(self, output_path: str) -> str:
        """Write backtest result to Excel file."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self._write_how_to_read(writer)
            self._write_overall_performance(writer)
            self._write_all_trades(writer)
            self._write_stock_summary(writer)
            self._write_equity_curve(writer)
            self._write_config_snapshot(writer)
        
        return output_path
    
    def _write_how_to_read(self, writer):
        """Sheet 1: How_To_Read - Run metadata and schema info."""
        data = {
            'Field': [
                'Run ID',
                'Timestamp',
                'Universe Size',
                'Price Policy',
                'Costs Applied',
                'Schema Version',
                'Objective',
                'Status'
            ],
            'Value': [
                self.result.run_id,
                self.result.timestamp.isoformat(),
                self.result.universe_size,
                self.result.price_policy,
                'Yes' if self.result.costs_applied else 'No',
                self.result.schema_version,
                'Maximize risk-adjusted return with 25%+ ARR target',
                'PASS' if self.result.passed else 'FAIL'
            ]
        }
        pd.DataFrame(data).to_excel(writer, sheet_name='How_To_Read', index=False)
    
    def _write_overall_performance(self, writer):
        """Sheet 2: Overall_Performance - Key metrics."""
        data = {
            'Metric': [
                'Total Return %',
                'CAGR %',
                'Max Drawdown %',
                'Profit Factor',
                'Win Rate %',
                'Total Trades',
                'Worst Year %',
                'Best Year %',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Status'
            ],
            'Value': [
                f"{self.result.total_return_pct:.2f}%",
                f"{self.result.cagr_pct:.2f}%",
                f"{self.result.max_drawdown_pct:.2f}%",
                f"{self.result.profit_factor:.2f}",
                f"{self.result.win_rate_pct:.2f}%",
                self.result.total_trades,
                f"{self.result.worst_year_pct:.2f}%",
                f"{self.result.best_year_pct:.2f}%",
                f"{self.result.sharpe_ratio:.2f}",
                f"{self.result.sortino_ratio:.2f}",
                'PASS' if self.result.passed else 'FAIL'
            ]
        }
        
        if not self.result.passed:
            data['Metric'].append('Failure Reasons')
            data['Value'].append('; '.join(self.result.failure_reasons))
        
        pd.DataFrame(data).to_excel(writer, sheet_name='Overall_Performance', index=False)
    
    def _write_all_trades(self, writer):
        """Sheet 3: All_Trades - One row per trade (MOST IMPORTANT)."""
        if not self.result.trades:
            # Write empty template with headers
            columns = [
                'trade_id', 'ticker', 'direction', 'entry_date', 'entry_price',
                'exit_date', 'exit_price', 'position_size_pct', 'capital_allocated',
                'risk_at_entry_pct', 'confidence_score', 'gross_pnl', 'costs_paid',
                'net_pnl', 'return_on_allocated_pct', 'return_on_portfolio_pct',
                'entry_reasons', 'exit_reason', 'veto_checks_passed', 'regime',
                'liquidity_status', 'status'
            ]
            pd.DataFrame(columns=columns).to_excel(writer, sheet_name='All_Trades', index=False)
            return
        
        trades_data = []
        for t in self.result.trades:
            trades_data.append({
                'trade_id': t.trade_id,
                'ticker': t.ticker,
                'direction': t.direction,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'position_size_pct': f"{t.position_size_pct:.2f}%",
                'capital_allocated': f"${t.capital_allocated:,.2f}",
                'risk_at_entry_pct': f"{t.risk_at_entry_pct:.2f}%",
                'confidence_score': f"{t.confidence_score:.2f}",
                'gross_pnl': f"${t.gross_pnl:,.2f}",
                'costs_paid': f"${t.costs_paid:,.2f}",
                'net_pnl': f"${t.net_pnl:,.2f}",
                'return_on_allocated_pct': f"{t.return_on_allocated_pct:.2f}%",
                'return_on_portfolio_pct': f"{t.return_on_portfolio_pct:.2f}%",
                'entry_reasons': t.entry_reasons,
                'exit_reason': t.exit_reason,
                'veto_checks_passed': t.veto_checks_passed,
                'regime': t.regime,
                'liquidity_status': t.liquidity_status,
                'status': t.status
            })
        
        pd.DataFrame(trades_data).to_excel(writer, sheet_name='All_Trades', index=False)
    
    def _write_stock_summary(self, writer):
        """Sheet 4: Stock_Summary - Per ticker metrics."""
        if not self.result.stock_summary:
            columns = ['ticker', 'trades', 'win_rate', 'total_return', 'max_drawdown', 
                      'profit_factor', 'status']
            pd.DataFrame(columns=columns).to_excel(writer, sheet_name='Stock_Summary', index=False)
            return
        
        summary_data = []
        for ticker, metrics in self.result.stock_summary.items():
            summary_data.append({
                'ticker': ticker,
                'trades': metrics.get('trades', 0),
                'win_rate': f"{metrics.get('win_rate', 0):.1f}%",
                'total_return': f"{metrics.get('total_return', 0):.1f}%",
                'max_drawdown': f"{metrics.get('max_drawdown', 0):.1f}%",
                'profit_factor': f"{metrics.get('profit_factor', 0):.2f}",
                'status': 'PASS' if metrics.get('passed', False) else 'FAIL'
            })
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Stock_Summary', index=False)
    
    def _write_equity_curve(self, writer):
        """Sheet 5: Equity_Curve - Date, equity, drawdown."""
        if self.result.equity_curve.empty:
            columns = ['date', 'portfolio_equity', 'drawdown_pct']
            pd.DataFrame(columns=columns).to_excel(writer, sheet_name='Equity_Curve', index=False)
            return
        
        self.result.equity_curve.to_excel(writer, sheet_name='Equity_Curve', index=False)
    
    def _write_config_snapshot(self, writer):
        """Sheet 6: Config_Snapshot - Full config used for the run."""
        config_items = []
        
        # Add all config sections
        sections = [
            ('General Config', self.result.config),
            ('Feature List', {'features': self.result.feature_list}),
            ('Model Params', self.result.model_params),
            ('Signal Rules', self.result.signal_rules),
            ('Risk Rules', self.result.risk_rules),
            ('Cost Model', self.result.cost_model)
        ]
        
        for section_name, section_data in sections:
            config_items.append({'Section': section_name, 'Key': '', 'Value': ''})
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    config_items.append({
                        'Section': '',
                        'Key': key,
                        'Value': str(value) if not isinstance(value, (list, dict)) else json.dumps(value)
                    })
        
        pd.DataFrame(config_items).to_excel(writer, sheet_name='Config_Snapshot', index=False)


def create_backtest_result(
    run_id: str,
    trades: List[Trade],
    equity_curve: pd.DataFrame,
    config: Dict,
    initial_capital: float = 100000
) -> BacktestResult:
    """
    Factory function to create a BacktestResult with computed metrics.
    """
    result = BacktestResult(
        run_id=run_id,
        timestamp=datetime.now(),
        universe_size=len(set(t.ticker for t in trades)),
        config=config
    )
    
    result.trades = trades
    result.equity_curve = equity_curve
    result.total_trades = len(trades)
    
    # Compute metrics from trades
    if trades:
        closed_trades = [t for t in trades if t.status == 'CLOSED']
        if closed_trades:
            winning = [t for t in closed_trades if t.net_pnl > 0]
            result.win_rate_pct = len(winning) / len(closed_trades) * 100
            
            total_profit = sum(t.net_pnl for t in closed_trades if t.net_pnl > 0)
            total_loss = abs(sum(t.net_pnl for t in closed_trades if t.net_pnl < 0))
            result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            result.total_return_pct = sum(t.return_on_portfolio_pct for t in closed_trades)
    
    # Compute from equity curve
    if not equity_curve.empty and 'portfolio_equity' in equity_curve.columns:
        final_equity = equity_curve['portfolio_equity'].iloc[-1]
        result.total_return_pct = (final_equity - initial_capital) / initial_capital * 100
        
        # Max drawdown
        peak = equity_curve['portfolio_equity'].expanding().max()
        drawdown = (equity_curve['portfolio_equity'] - peak) / peak * 100
        result.max_drawdown_pct = drawdown.min()
        
        # CAGR (assuming daily data)
        days = len(equity_curve)
        years = days / 252
        if years > 0:
            result.cagr_pct = ((final_equity / initial_capital) ** (1/years) - 1) * 100
    
    return result
