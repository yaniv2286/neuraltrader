"""
Simple CSV Enhancement Script
Creates a formatted Excel file with colors and analysis
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import os

def enhance_trades_csv():
    """
    Create enhanced Excel file with formatting and analysis
    """
    print("🎨 Creating Enhanced Trade Analysis Excel...")
    
    # Load the original CSV
    input_file = os.path.join(os.path.dirname(__file__), '..', 'tests', 'phase5_final_trades.csv')
    output_file = os.path.join(os.path.dirname(__file__), '..', 'tests', 'phase5_trades_analysis.xlsx')
    
    df = pd.read_csv(input_file)
    
    # Create workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Trade Analysis"
    
    # Define styles
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=12)
    header_alignment = Alignment(horizontal="center", vertical="center")
    
    buy_fill = PatternFill(start_color="C6E0B4", end_color="C6E0B4", fill_type="solid")  # Light green
    sell_fill = PatternFill(start_color="BDD7EE", end_color="BDD7EE", fill_type="solid")  # Light blue
    profit_fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")  # Dark green
    loss_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")    # Red
    stop_loss_fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")  # Orange
    
    strong_buy_fill = PatternFill(start_color="006100", end_color="006100", fill_type="solid")  # Dark green
    strong_sell_fill = PatternFill(start_color="C00000", end_color="C00000", fill_type="solid")  # Dark red
    
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Write headers
    headers = df.columns.tolist()
    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = thin_border
    
    # Write data
    for row_num, row_data in enumerate(dataframe_to_rows(df, index=False, header=False), 2):
        for col_num, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_num, column=col_num, value=value)
            cell.border = thin_border
            
            # Apply conditional formatting based on content
            col_name = headers[col_num-1]
            
            # Color code BUY/SELL actions
            if col_name == 'action':
                if value == 'BUY':
                    cell.fill = buy_fill
                elif value == 'SELL':
                    cell.fill = sell_fill
            
            # Color code profit/loss
            elif col_name == 'pnl_pct':
                if pd.notna(value):
                    if value > 0:
                        cell.fill = profit_fill
                        cell.font = Font(color="FFFFFF", bold=True)
                    elif value < 0:
                        cell.fill = loss_fill
                        cell.font = Font(color="FFFFFF", bold=True)
            
            # Color code stop losses
            elif col_name == 'reason':
                if value == 'STOP_LOSS':
                    cell.fill = stop_loss_fill
                    cell.font = Font(color="000000", bold=True)
            
            # Color code signal strength
            elif col_name == 'signal':
                if pd.notna(value):
                    if value > 0.05:  # Strong buy (>5%)
                        cell.fill = strong_buy_fill
                        cell.font = Font(color="FFFFFF", bold=True)
                    elif value < -0.05:  # Strong sell (<-5%)
                        cell.fill = strong_sell_fill
                        cell.font = Font(color="FFFFFF", bold=True)
    
    # Auto-filter
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(headers)).coordinate}"
    
    # Freeze first row
    ws.freeze_panes = "A2"
    
    # Adjust column widths
    column_widths = {
        'date': 12,
        'ticker': 8,
        'action': 8,
        'shares': 8,
        'price': 10,
        'cost': 12,
        'signal': 10,
        'proceeds': 12,
        'pnl': 10,
        'pnl_pct': 10,
        'reason': 12
    }
    
    for col_num, header in enumerate(headers, 1):
        width = column_widths.get(header, 10)
        ws.column_dimensions[chr(64 + col_num)].width = width
    
    # Create summary sheet
    summary_ws = wb.create_sheet("Summary")
    
    # Calculate summary statistics
    total_trades = len(df)
    buy_trades = len(df[df['action'] == 'BUY'])
    sell_trades = len(df[df['action'] == 'SELL'])
    
    profitable_trades = len(df[(df['pnl_pct'] > 0) & (df['pnl_pct'].notna())])
    losing_trades = len(df[(df['pnl_pct'] < 0) & (df['pnl_pct'].notna())])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df['pnl'].sum()
    avg_win = df.loc[(df['pnl_pct'] > 0) & (df['pnl_pct'].notna()), 'pnl_pct'].mean() if profitable_trades > 0 else 0
    avg_loss = df.loc[(df['pnl_pct'] < 0) & (df['pnl_pct'].notna()), 'pnl_pct'].mean() if losing_trades > 0 else 0
    profit_factor = abs(df.loc[(df['pnl_pct'] > 0) & (df['pnl_pct'].notna()), 'pnl'].sum() / df.loc[(df['pnl_pct'] < 0) & (df['pnl_pct'].notna()), 'pnl'].sum()) if losing_trades > 0 else float('inf')
    
    stop_loss_trades = len(df[df['reason'] == 'STOP_LOSS'])
    signal_trades = len(df[df['reason'] == 'SIGNAL'])
    
    # Write summary
    summary_data = [
        ["NeuralTrader - Trade Analysis Summary", ""],
        ["", ""],
        ["Trade Statistics", ""],
        ["Total Trades", total_trades],
        ["Buy Trades", buy_trades],
        ["Sell Trades", sell_trades],
        ["", ""],
        ["Performance Metrics", ""],
        ["Win Rate", f"{win_rate:.2f}%"],
        ["Total P&L", f"${total_pnl:,.2f}"],
        ["Average Win", f"{avg_win:.2f}%"],
        ["Average Loss", f"{avg_loss:.2f}%"],
        ["Profit Factor", f"{profit_factor:.2f}"],
        ["", ""],
        ["Risk Management", ""],
        ["Stop Loss Trades", stop_loss_trades],
        ["Signal-Based Exits", signal_trades],
        ["Stop Loss Rate", f"{(stop_loss_trades/sell_trades*100):.2f}%"],
        ["", ""],
        ["Signal Analysis", ""],
        ["Strong Buy Signals (>5%)", len(df[(df['signal'] > 0.05) & (df['action'] == 'BUY')])],
        ["Strong Sell Signals (<-5%)", len(df[(df['signal'] < -0.05) & (df['action'] == 'SELL')])],
        ["", ""],
        ["Color Legend", ""],
        ["BUY", "Light Green"],
        ["SELL", "Light Blue"],
        ["Profit", "Dark Green"],
        ["Loss", "Red"],
        ["Stop Loss", "Orange"],
        ["Strong Buy (>5%)", "Dark Green"],
        ["Strong Sell (<-5%)", "Dark Red"],
    ]
    
    for row_num, (label, value) in enumerate(summary_data, 1):
        label_cell = summary_ws.cell(row=row_num, column=1, value=label)
        value_cell = summary_ws.cell(row=row_num, column=2, value=value)
        
        # Format headers
        if row_num in [1, 4, 10, 16, 20, 26]:
            label_cell.fill = header_fill
            label_cell.font = header_font
            value_cell.fill = header_fill
            value_cell.font = header_font
        
        # Add borders
        label_cell.border = thin_border
        value_cell.border = thin_border
    
    # Adjust summary column widths
    summary_ws.column_dimensions['A'].width = 25
    summary_ws.column_dimensions['B'].width = 15
    
    # Save the workbook
    wb.save(output_file)
    
    print(f"✅ Enhanced Excel file created: {output_file}")
    print(f"\n📊 Features:")
    print(f"   • Frozen header row")
    print(f"   • Auto-filter on all columns")
    print(f"   • Color-coded trades:")
    print(f"     - BUY: Light Green")
    print(f"     - SELL: Light Blue")
    print(f"     - Profit: Dark Green")
    print(f"     - Loss: Red")
    print(f"     - Stop Loss: Orange")
    print(f"     - Strong Buy (>5%): Dark Green")
    print(f"     - Strong Sell (<-5%): Dark Red")
    print(f"   • Summary sheet with key metrics")
    print(f"   • All columns have borders and proper width")
    
    return output_file

if __name__ == "__main__":
    enhance_trades_csv()
