#!/usr/bin/env python3
"""
NeuralTrader HTML Dashboard Report Generator
Creates high-fidelity HTML dashboard with professional styling
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTMLDashboardGenerator:
    """High-fidelity HTML dashboard generator for NeuralTrader"""
    
    def __init__(self, portfolio_file: str = "data/portfolio.json"):
        self.portfolio_file = portfolio_file
        self.project_root = Path(__file__).parent.parent
        
    def generate_dashboard(self, 
                         buy_signals: List[str] = None,
                         sell_signals: List[str] = None, 
                         hold_signals: List[str] = None,
                         trades_executed: List[Dict] = None) -> str:
        """Generate comprehensive HTML dashboard"""
        
        # Load portfolio data
        portfolio_data = self._load_portfolio_data()
        
        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(portfolio_data)
        
        # Generate HTML
        html_content = self._create_html_dashboard(
            portfolio_data, 
            metrics, 
            buy_signals or [],
            sell_signals or [],
            hold_signals or [],
            trades_executed or []
        )
        
        # Save HTML file
        html_file = self.project_root / "reports" / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_file.parent.mkdir(exist_ok=True)
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML Dashboard generated: {html_file}")
        return str(html_file)
    
    def _load_portfolio_data(self) -> Dict:
        """Load portfolio data from JSON file"""
        try:
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            return self._get_default_portfolio()
    
    def _get_default_portfolio(self) -> Dict:
        """Return default portfolio structure"""
        return {
            'cash': 100000.0,
            'positions': {},
            'performance': {
                'total_value': 100000.0,
                'total_return': 0.0,
                'total_return_pct': 0.0
            }
        }
    
    def _calculate_portfolio_metrics(self, portfolio_data: Dict) -> Dict:
        """Calculate portfolio metrics"""
        cash = portfolio_data.get('cash', 100000.0)
        positions = portfolio_data.get('positions', {})
        performance = portfolio_data.get('performance', {})
        
        total_value = performance.get('total_value', cash)
        total_return = performance.get('total_return', 0.0)
        total_return_pct = performance.get('total_return_pct', 0.0)
        
        position_count = len(positions)
        max_positions = 10  # NeuralTrader max positions
        
        return {
            'portfolio_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'available_cash': cash,
            'position_count': f"{position_count}/{max_positions}",
            'positions': positions
        }
    
    def _create_html_dashboard(self, 
                              portfolio_data: Dict,
                              metrics: Dict,
                              buy_signals: List[str],
                              sell_signals: List[str],
                              hold_signals: List[str],
                              trades_executed: List[Dict]) -> str:
        """Create HTML dashboard content"""
        
        # Generate positions table
        positions_html = self._create_positions_table(metrics['positions'])
        
        # Generate market activity section
        market_activity_html = self._create_market_activity_section(
            buy_signals, sell_signals, hold_signals, trades_executed
        )
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuralTrader Dashboard - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.8;
        }}
        
        .executive-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .positive {{
            color: #27ae60 !important;
        }}
        
        .negative {{
            color: #e74c3c !important;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        
        .positions-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .positions-table th {{
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }}
        
        .positions-table td {{
            padding: 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .positions-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .market-activity {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .activity-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .activity-card h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .signal-list {{
            list-style: none;
        }}
        
        .signal-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .signal-list li:last-child {{
            border-bottom: none;
        }}
        
        .buy {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .sell {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .hold {{
            color: #f39c12;
            font-weight: bold;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .executive-summary {{
                padding: 20px;
                grid-template-columns: 1fr;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .positions-table {{
                font-size: 0.9em;
            }}
            
            .positions-table th,
            .positions-table td {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 NeuralTrader Dashboard</h1>
            <div class="subtitle">Algorithmic Trading Performance Report</div>
            <div class="subtitle">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
        </div>
        
        <div class="executive-summary">
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${metrics['portfolio_value']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if metrics['total_return'] >= 0 else 'negative'}">
                    ${metrics['total_return']:,.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return %</div>
                <div class="metric-value {'positive' if metrics['total_return_pct'] >= 0 else 'negative'}">
                    {metrics['total_return_pct']:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Available Cash</div>
                <div class="metric-value">${metrics['available_cash']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Position Count</div>
                <div class="metric-value">{metrics['position_count']}</div>
            </div>
        </div>
        
        <div class="content">
            <h2 class="section-title">📊 Current Holdings</h2>
            {positions_html}
            
            <h2 class="section-title">📈 Market Activity</h2>
            {market_activity_html}
        </div>
        
        <div class="footer">
            <div>NeuralTrader Automated Trading System v5.0</div>
            <div>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Sector Authority Active</div>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _create_positions_table(self, positions: Dict) -> str:
        """Create HTML table for positions"""
        if not positions:
            return "<p>No active positions</p>"
        
        rows_html = ""
        for ticker, pos_data in positions.items():
            # Use the correct keys from portfolio.json
            shares = pos_data.get('shares', 0)
            cost_basis = pos_data.get('cost_basis', 0)
            current_price = pos_data.get('current_price', 0)
            
            # Calculate derived values
            market_value = shares * current_price
            total_cost_basis = shares * cost_basis
            gain_loss = market_value - total_cost_basis
            gain_loss_pct = (gain_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0
            
            color_class = 'positive' if gain_loss_pct >= 0 else 'negative'
            
            rows_html += f"""
            <tr>
                <td><strong>{ticker}</strong></td>
                <td>{shares}</td>
                <td>${cost_basis:.2f}</td>
                <td>${current_price:.2f}</td>
                <td>${market_value:.2f}</td>
                <td>${gain_loss:.2f}</td>
                <td class="{color_class}">{gain_loss_pct:.2f}%</td>
            </tr>
            """
        
        return f"""
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Shares</th>
                    <th>Cost Basis</th>
                    <th>Current Price</th>
                    <th>Market Value</th>
                    <th>Gain/Loss</th>
                    <th>Gain/Loss %</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
    
    def _create_market_activity_section(self, buy_signals: List[str],
                                      sell_signals: List[str],
                                      hold_signals: List[str],
                                      trades_executed: List[Dict]) -> str:
        """Create market activity section"""
        
        buy_html = ""
        for signal in buy_signals:
            buy_html += f'<li class="buy">📈 BUY: {signal}</li>'
        
        sell_html = ""
        for signal in sell_signals:
            sell_html += f'<li class="sell">📉 SELL: {signal}</li>'
        
        hold_html = ""
        for signal in hold_signals:
            hold_html += f'<li class="hold">⏸️ HOLD: {signal}</li>'
        
        trades_html = ""
        for trade in trades_executed:
            ticker = trade.get('ticker', 'Unknown')
            action = trade.get('action', 'Unknown')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            trades_html += f'<li>{action.upper()} {quantity} shares of {ticker} @ ${price:.2f}</li>'
        
        return f"""
        <div class="market-activity">
            <div class="activity-card">
                <h3>🟢 Buy Signals</h3>
                <ul class="signal-list">
                    {buy_html if buy_html else '<li>No buy signals today</li>'}
                </ul>
            </div>
            
            <div class="activity-card">
                <h3>🔴 Sell Signals</h3>
                <ul class="signal-list">
                    {sell_html if sell_html else '<li>No sell signals today</li>'}
                </ul>
            </div>
            
            <div class="activity-card">
                <h3>🟡 Hold Signals</h3>
                <ul class="signal-list">
                    {hold_html if hold_html else '<li>No hold signals today</li>'}
                </ul>
            </div>
            
            <div class="activity-card">
                <h3>💼 Trades Executed</h3>
                <ul class="signal-list">
                    {trades_html if trades_html else '<li>No trades executed today</li>'}
                </ul>
            </div>
        </div>
        """
    
    def get_today_log_file(self) -> Optional[str]:
        """Get today's log file path"""
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = self.project_root / "logs" / f"NeuralTrader_{today}.log"
        
        if log_file.exists():
            return str(log_file)
        return None

def main():
    """Command line interface for HTML dashboard generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralTrader HTML Dashboard Generator')
    parser.add_argument('--portfolio-file', type=str, default='data/portfolio.json',
                       help='Portfolio JSON file path')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for HTML dashboard')
    
    args = parser.parse_args()
    
    # Generate dashboard
    generator = HTMLDashboardGenerator(args.portfolio_file)
    dashboard_file = generator.generate_dashboard()
    
    print(f"\n🎉 HTML Dashboard generated successfully!")
    print(f"📁 Dashboard: {dashboard_file}")

if __name__ == "__main__":
    main()
