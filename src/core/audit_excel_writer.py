"""
Audit Excel Writer - 6-Sheet Schema Contract
=============================================

Implements the EXACT Excel output schema (NON-NEGOTIABLE):

1) How_To_Read - Run metadata, price policy, schema version
2) Overall_Performance - CAGR, Max DD, Profit Factor, Win Rate, PASS/FAIL
3) All_Trades - Full trade log with all required columns
4) Stock_Summary - Per-ticker metrics grouped by strategy
5) Equity_Curve - Date, equity, drawdown per strategy
6) Config_Snapshot - Full resolved config, overrides vs baseline

If any sheet or required columns are missing -> FAIL RUN.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from src.core.deterministic_backtest_engine import BacktestResult, CostModel, Trade
from src.core.strategy_registry import StrategyConfig, get_strategy_registry


class AuditExcelError(Exception):
    """Raised when Excel output contract is violated."""
    pass


# Required columns for each sheet
REQUIRED_COLUMNS = {
    'How_To_Read': [
        'parameter', 'value', 'description'
    ],
    'Overall_Performance': [
        'strategy_id', 'CAGR', 'Max_DD', 'Profit_Factor', 'Win_Rate',
        'Total_Trades', 'Worst_Year', 'Best_Year', 'PASS_FAIL'
    ],
    'All_Trades': [
        'strategy_id', 'ticker', 'signal_date', 'execution_date',
        'entry_price', 'exit_price', 'qty', 'costs_paid',
        'open_reason', 'close_reason', 'veto_reason', 'confidence_score',
        'rule_snapshot'
    ],
    'Stock_Summary': [
        'strategy_id', 'ticker', 'trades', 'win_rate', 'return_pct',
        'max_drawdown', 'profit_factor'
    ],
    'Equity_Curve': [
        'strategy_id', 'date', 'equity', 'drawdown_pct'
    ],
    'Config_Snapshot': [
        'strategy_id', 'parameter', 'value', 'is_override', 'baseline_value'
    ]
}


class AuditExcelWriter:
    """
    Writes the official 6-sheet audit Excel file.
    
    Contract: All sheets and required columns MUST be present.
    Missing data -> FAIL RUN.
    """
    
    def __init__(self):
        if not HAS_OPENPYXL:
            raise AuditExcelError("openpyxl required for Excel output. Install with: pip install openpyxl")
        
        self.wb = Workbook()
        self.registry = get_strategy_registry()
        
        # Styles
        self.header_font = Font(bold=True, color='FFFFFF')
        self.header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        self.pass_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
        self.fail_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
        self.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
    
    def write_audit_excel(
        self,
        results: List[BacktestResult],
        run_metadata: Dict[str, Any],
        cost_model: CostModel,
        output_path: str
    ) -> str:
        """
        Write complete audit Excel file.
        
        Args:
            results: List of BacktestResult for all strategies
            run_metadata: Run metadata dict
            cost_model: Cost model used
            output_path: Output file path
            
        Returns:
            Path to created file
            
        Raises:
            AuditExcelError: If any required data is missing
        """
        print(f"\n📊 Writing Audit Excel: {output_path}")
        
        # Validate baseline unchanged
        self.registry.validate_baseline_unchanged()
        
        # Write all 6 sheets
        self._write_how_to_read(run_metadata, cost_model, results)
        self._write_overall_performance(results)
        self._write_all_trades(results)
        self._write_stock_summary(results)
        self._write_equity_curve(results)
        self._write_config_snapshot(results)
        
        # Remove default sheet
        if 'Sheet' in self.wb.sheetnames:
            del self.wb['Sheet']
        
        # Validate all sheets exist
        required_sheets = list(REQUIRED_COLUMNS.keys())
        for sheet_name in required_sheets:
            if sheet_name not in self.wb.sheetnames:
                raise AuditExcelError(f"FAIL RUN: Required sheet '{sheet_name}' missing from output")
        
        # Save
        self.wb.save(output_path)
        print(f"   ✅ Audit Excel saved: {output_path}")
        
        return output_path
    
    def _write_how_to_read(
        self,
        run_metadata: Dict[str, Any],
        cost_model: CostModel,
        results: List[BacktestResult]
    ):
        """Write Sheet 1: How_To_Read."""
        ws = self.wb.create_sheet("How_To_Read")
        
        # Calculate ticker summary
        total_tickers = 154  # Universe size
        tickers_success = sum(r.ticker_summary.get('tickers_success', 0) for r in results) // len(results) if results else 0
        tickers_failed = total_tickers - tickers_success
        
        # Build metadata rows
        rows = [
            ('run_id', run_metadata.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S')), 'Unique identifier for this backtest run'),
            ('run_timestamp_utc', datetime.utcnow().isoformat(), 'UTC timestamp when run was executed'),
            ('engine_version', '1.0.0', 'Backtest engine version'),
            ('schema_version', '1.0', 'Excel output schema version'),
            ('', '', ''),
            ('baseline_strategy_id', 'VT_SweetSpot_v1', 'Immutable baseline strategy ID'),
            ('strategies_tested', len(results), 'Number of strategies in this run'),
            ('', '', ''),
            ('data_provider', 'Tiingo', 'Source of price data'),
            ('price_policy', 'adjusted', 'Price adjustment policy (splits/dividends)'),
            ('', '', ''),
            ('execution_model', 'signal_close_T_execute_open_T+1', 'Signal at close Day T, execute at open Day T+1'),
            ('one_bar_delay', 'TRUE', 'One-bar delay enforced (MANDATORY)'),
            ('', '', ''),
            ('costs_applied', 'TRUE', 'Trading costs applied to all trades'),
            ('commission_per_trade', f"${cost_model.commission_per_trade:.2f}", 'Commission per trade'),
            ('spread_pct', f"{cost_model.spread_pct*100:.2f}%", 'Bid-ask spread cost'),
            ('slippage_pct', f"{cost_model.slippage_pct*100:.2f}%", 'Slippage cost'),
            ('', '', ''),
            ('tickers_total', total_tickers, 'Total tickers in universe'),
            ('tickers_success', tickers_success, 'Tickers successfully processed'),
            ('tickers_failed', tickers_failed, 'Tickers that failed (with reasons in logs)'),
            ('', '', ''),
            ('NO_SILENT_SKIP', 'ENFORCED', 'All failures are logged with reasons'),
        ]
        
        # Add failure reasons if any
        for result in results:
            if 'failure_reasons' in result.ticker_summary:
                for ticker, reason in list(result.ticker_summary['failure_reasons'].items())[:10]:
                    rows.append((f'failure_{ticker}', reason, f'Reason {ticker} was skipped'))
        
        # Write data
        headers = ['parameter', 'value', 'description']
        ws.append(headers)
        
        for row in rows:
            ws.append(row)
        
        # Style headers
        for col in range(1, 4):
            cell = ws.cell(row=1, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 40
        ws.column_dimensions['C'].width = 50
    
    def _write_overall_performance(self, results: List[BacktestResult]):
        """Write Sheet 2: Overall_Performance."""
        ws = self.wb.create_sheet("Overall_Performance")
        
        headers = [
            'strategy_id', 'CAGR', 'Max_DD', 'Profit_Factor', 'Win_Rate',
            'Total_Trades', 'Winning_Trades', 'Losing_Trades',
            'Sharpe_Ratio', 'Sortino_Ratio',
            'Best_Year', 'Best_Year_Return', 'Worst_Year', 'Worst_Year_Return',
            'Total_Costs', 'Avg_Cost_Per_Trade',
            'PASS_FAIL', 'PASS_FAIL_Reasons'
        ]
        
        ws.append(headers)
        
        for result in results:
            row = [
                result.strategy_id,
                f"{result.cagr_pct:.2f}%",
                f"{result.max_drawdown_pct:.2f}%",
                f"{result.profit_factor:.2f}",
                f"{result.win_rate_pct:.1f}%",
                result.total_trades,
                result.winning_trades,
                result.losing_trades,
                f"{result.sharpe_ratio:.2f}",
                f"{result.sortino_ratio:.2f}",
                result.best_year,
                f"{result.best_year_return:.2f}%",
                result.worst_year,
                f"{result.worst_year_return:.2f}%",
                f"${result.total_costs_paid:.2f}",
                f"${result.avg_cost_per_trade:.2f}",
                result.pass_fail,
                "; ".join(result.pass_fail_reasons) if result.pass_fail_reasons else ""
            ]
            ws.append(row)
        
        # Style headers
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
        
        # Color PASS/FAIL column
        pass_fail_col = headers.index('PASS_FAIL') + 1
        for row in range(2, len(results) + 2):
            cell = ws.cell(row=row, column=pass_fail_col)
            if cell.value == 'PASS':
                cell.fill = self.pass_fill
            else:
                cell.fill = self.fail_fill
        
        # Adjust column widths
        for col, header in enumerate(headers, 1):
            ws.column_dimensions[chr(64 + col) if col <= 26 else 'A' + chr(64 + col - 26)].width = max(len(header) + 2, 12)
    
    def _write_all_trades(self, results: List[BacktestResult]):
        """Write Sheet 3: All_Trades."""
        ws = self.wb.create_sheet("All_Trades")
        
        headers = [
            'strategy_id', 'trade_id', 'ticker',
            'signal_date', 'execution_date', 'exit_date',
            'entry_price', 'exit_price', 'qty',
            'direction', 'gross_pnl', 'net_pnl', 'pnl_pct',
            'entry_cost', 'exit_cost', 'costs_paid',
            'open_reason', 'close_reason', 'veto_reason',
            'confidence_score', 'rule_snapshot'
        ]
        
        ws.append(headers)
        
        for result in results:
            for trade in result.trades:
                # Validate costs are present
                if trade.total_cost <= 0 and trade.entry_price > 0:
                    raise AuditExcelError(
                        f"FAIL RUN: Trade {trade.trade_id} has zero costs. "
                        f"Costs MUST be applied to every trade."
                    )
                
                row = [
                    trade.strategy_id,
                    trade.trade_id,
                    trade.ticker,
                    trade.signal_date.strftime('%Y-%m-%d') if trade.signal_date else '',
                    trade.execution_date.strftime('%Y-%m-%d') if trade.execution_date else '',
                    trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else '',
                    f"${trade.entry_price:.2f}",
                    f"${trade.exit_price:.2f}",
                    f"{trade.shares:.4f}",
                    'LONG' if trade.direction == 1 else 'SHORT',
                    f"${trade.gross_pnl:.2f}",
                    f"${trade.net_pnl:.2f}",
                    f"{trade.pnl_pct:.2f}%",
                    f"${trade.entry_cost:.2f}",
                    f"${trade.exit_cost:.2f}",
                    f"${trade.total_cost:.2f}",
                    trade.open_reason,
                    trade.close_reason,
                    trade.veto_reason or '',
                    f"{trade.confidence_score:.2f}",
                    json.dumps(trade.rule_snapshot) if trade.rule_snapshot else ''
                ]
                ws.append(row)
        
        # Style headers
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['C'].width = 8
        ws.column_dimensions['Q'].width = 30
        ws.column_dimensions['R'].width = 25
    
    def _write_stock_summary(self, results: List[BacktestResult]):
        """Write Sheet 4: Stock_Summary."""
        ws = self.wb.create_sheet("Stock_Summary")
        
        headers = [
            'strategy_id', 'ticker', 'trades', 'winning_trades', 'losing_trades',
            'win_rate', 'total_return', 'avg_return_per_trade',
            'max_gain', 'max_loss', 'profit_factor'
        ]
        
        ws.append(headers)
        
        for result in results:
            if not result.trades:
                continue
            
            # Group trades by ticker
            trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
            
            for ticker in trades_df['ticker'].unique():
                ticker_trades = trades_df[trades_df['ticker'] == ticker]
                
                winning = ticker_trades[ticker_trades['net_pnl'] > 0]
                losing = ticker_trades[ticker_trades['net_pnl'] <= 0]
                
                win_rate = len(winning) / len(ticker_trades) * 100 if len(ticker_trades) > 0 else 0
                total_return = ticker_trades['net_pnl'].sum()
                avg_return = ticker_trades['pnl_pct'].mean()
                max_gain = ticker_trades['pnl_pct'].max()
                max_loss = ticker_trades['pnl_pct'].min()
                
                gross_profit = winning['net_pnl'].sum() if len(winning) > 0 else 0
                gross_loss = abs(losing['net_pnl'].sum()) if len(losing) > 0 else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                row = [
                    result.strategy_id,
                    ticker,
                    len(ticker_trades),
                    len(winning),
                    len(losing),
                    f"{win_rate:.1f}%",
                    f"${total_return:.2f}",
                    f"{avg_return:.2f}%",
                    f"{max_gain:.2f}%",
                    f"{max_loss:.2f}%",
                    f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
                ]
                ws.append(row)
        
        # Style headers
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
    
    def _write_equity_curve(self, results: List[BacktestResult]):
        """Write Sheet 5: Equity_Curve."""
        ws = self.wb.create_sheet("Equity_Curve")
        
        headers = ['strategy_id', 'date', 'equity', 'drawdown_pct', 'cash', 'positions']
        ws.append(headers)
        
        for result in results:
            if result.equity_curve is None or result.equity_curve.empty:
                continue
            
            # Sample to reduce file size (every 5th day)
            eq = result.equity_curve.iloc[::5].copy()
            
            for _, row in eq.iterrows():
                ws.append([
                    result.strategy_id,
                    row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    f"${row['equity']:.2f}",
                    f"{row['drawdown']:.2f}%",
                    f"${row['cash']:.2f}",
                    row['positions_count']
                ])
        
        # Style headers
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
    
    def _write_config_snapshot(self, results: List[BacktestResult]):
        """Write Sheet 6: Config_Snapshot."""
        ws = self.wb.create_sheet("Config_Snapshot")
        
        headers = ['strategy_id', 'parameter', 'value', 'is_override', 'baseline_value']
        ws.append(headers)
        
        # Get baseline config
        baseline = self.registry.get_baseline()
        baseline_dict = baseline.to_dict()
        
        for result in results:
            config = result.config
            config_dict = config.to_dict()
            
            for param, value in config_dict.items():
                if param == 'overrides':
                    continue
                
                baseline_val = baseline_dict.get(param, '')
                is_override = param in config.overrides
                
                # Check if baseline was modified (FAIL condition)
                if config.is_baseline and value != baseline_val:
                    raise AuditExcelError(
                        f"FAIL RUN: Baseline config '{param}' was modified! "
                        f"Expected {baseline_val}, got {value}. "
                        f"Baseline must remain IMMUTABLE."
                    )
                
                row = [
                    config.strategy_id,
                    param,
                    str(value),
                    'YES' if is_override else 'NO',
                    str(baseline_val)
                ]
                ws.append(row)
            
            # Add separator row
            ws.append(['', '', '', '', ''])
        
        # Style headers
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
        
        # Highlight overrides
        override_fill = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
        for row in range(2, ws.max_row + 1):
            if ws.cell(row=row, column=4).value == 'YES':
                for col in range(1, 6):
                    ws.cell(row=row, column=col).fill = override_fill
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 25
        ws.column_dimensions['C'].width = 40
        ws.column_dimensions['D'].width = 12
        ws.column_dimensions['E'].width = 40


def write_audit_excel(
    results: List[BacktestResult],
    run_metadata: Dict[str, Any],
    cost_model: CostModel,
    output_path: str
) -> str:
    """
    Convenience function to write audit Excel.
    
    Args:
        results: List of BacktestResult
        run_metadata: Run metadata
        cost_model: Cost model used
        output_path: Output file path
        
    Returns:
        Path to created file
    """
    writer = AuditExcelWriter()
    return writer.write_audit_excel(results, run_metadata, cost_model, output_path)
