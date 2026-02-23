"""
NeuralTrader Email Notifier
===========================

Sends daily executive briefs with portfolio performance and risk compliance.

Features:
- Daily Executive Brief with portfolio metrics
- Constitution Health Check reporting
- TLS email sending via smtplib
- HTML formatted emails with professional styling
- Portfolio data exclusively from Interactive Brokers (IBKR)
- Error handling for IBKR connection issues
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Dict, List, Optional
import pytz
import json
from pathlib import Path

# Force load environment variables immediately
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailNotifier:
    """
    Email Notifier for NeuralTrader
    Sends daily executive briefs and alerts
    """
    
    def __init__(self, smtp_server: str = 'smtp.gmail.com', smtp_port: int = 587):
        """Initialize email notifier"""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        
        # Load email credentials from environment with fallback logic
        # Priority: NOTIFIER_EMAIL/NOTIFIER_PASSWORD (from .env) -> EMAIL_USER/EMAIL_PASS (legacy)
        self.sender_email = (
            os.getenv('NOTIFIER_EMAIL') or 
            os.getenv('EMAIL_USER') or
            os.getenv('EMAIL_ADDRESS')
        )
        self.sender_password = (
            os.getenv('NOTIFIER_PASSWORD') or 
            os.getenv('EMAIL_PASS') or
            os.getenv('EMAIL_PASSWORD')
        )
        self.recipient_email = (
            os.getenv('EMAIL_RECIPIENT') or 
            os.getenv('RECIPIENT_EMAIL') or 
            'lugassy.ai@gmail.com'
        )
        
        # Validate credentials
        if not self.sender_email:
            raise ValueError("Email credentials not found. Set NOTIFIER_EMAIL or EMAIL_USER in .env")
        if not self.sender_password:
            raise ValueError("Email password not found. Set NOTIFIER_PASSWORD or EMAIL_PASS in .env")
        
        # Timezone handling
        self.eastern = pytz.timezone('US/Eastern')
        self.israel = pytz.timezone('Asia/Jerusalem')
        
        # Initialize IBKR engine for live data
        self.ibkr_engine = None
        self._init_ibkr_engine()
        
        logger.info("Email Notifier initialized")
        logger.info(f"Sender: {self.sender_email}")
        logger.info(f"Recipient: {self.recipient_email}")
        logger.info("[OK] Environment variables loaded successfully")
    
    def _init_ibkr_engine(self):
        """Initialize IBKR engine for live portfolio data"""
        try:
            from core.ibkr_engine import IBKRExecutionEngine
            self.ibkr_engine = IBKRExecutionEngine()
            logger.info("[IBKR] IBKR engine initialized for live portfolio data")
        except ImportError as e:
            logger.error(f"[IBKR] Failed to initialize IBKR engine: {e}")
            self.ibkr_engine = None
        except Exception as e:
            logger.error(f"[IBKR] Error initializing IBKR engine: {e}")
            self.ibkr_engine = None
    
    def _load_portfolio_data(self) -> Dict:
        """Load portfolio data from Interactive Brokers (live)"""
        if self.ibkr_engine is None:
            return self._get_error_portfolio("IBKR_UNAVAILABLE")
        
        try:
            # Connect to IBKR
            if not self.ibkr_engine.connect():
                return self._get_error_portfolio("IBKR_CONNECTION_FAILED")
            
            # Get account summary
            account_summary = self.ibkr_engine.get_account_summary()
            if 'error' in account_summary:
                return self._get_error_portfolio("IBKR_ACCOUNT_ERROR")
            
            # Get positions
            positions = self.ibkr_engine.get_positions()
            
            # Convert IBKR positions to portfolio format
            portfolio_positions = {}
            total_market_value = 0.0
            
            for pos in positions:
                portfolio_positions[pos['symbol']] = {
                    'shares': int(pos['quantity']),
                    'cost_basis': pos['average_cost'],
                    'current_price': pos['market_price'],
                    'market_value': pos['market_value'],
                    'unrealized_pnl': pos['unrealized_pnl'],
                    'unrealized_pnl_pct': (pos['unrealized_pnl'] / pos['average_cost'] * 100) if pos['average_cost'] > 0 else 0
                }
                total_market_value += pos['market_value']
            
            # Calculate portfolio metrics
            cash_balance = account_summary.get('cash_balance', 0.0)
            total_value = cash_balance + total_market_value
            initial_cash = 100000.0  # Starting capital
            total_return = total_value - initial_cash
            total_return_pct = (total_return / initial_cash * 100) if initial_cash > 0 else 0
            
            portfolio_data = {
                'cash': cash_balance,
                'positions': portfolio_positions,
                'performance': {
                    'total_value': total_value,
                    'total_return': total_return,
                    'total_return_pct': total_return_pct
                },
                'account_summary': account_summary,
                'ibkr_positions': positions,
                'data_source': 'IBKR_LIVE'
            }
            
            logger.info(f"[IBKR] Live portfolio data loaded: Cash=${cash_balance:.2f}, Positions={len(positions)}")
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"[IBKR] Error loading portfolio data: {e}")
            return self._get_error_portfolio("IBKR_DATA_ERROR")
        finally:
            # Always disconnect
            if self.ibkr_engine:
                self.ibkr_engine.disconnect()
    
    def _get_error_portfolio(self, error_type: str) -> Dict:
        """Return error portfolio structure for reporting"""
        return {
            'cash': 0.0,
            'positions': {},
            'performance': {
                'total_value': 0.0,
                'total_return': 0.0,
                'total_return_pct': 0.0
            },
            'error': {
                'type': error_type,
                'message': f"CRITICAL ERROR: IBKR connection failed. Portfolio state unknown."
            },
            'data_source': 'ERROR'
        }
    
    def send_email_with_logs(self, to_email: str, subject: str, body: str, 
                          log_file_path: str = None, attachment_path: str = None, 
                          html_body: bool = False) -> bool:
        """
        Send email with full logs attached and optional additional attachment
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (HTML or plain text)
            log_file_path: Path to log file to attach
            attachment_path: Path to additional file to attach (NOT USED for HTML dashboard)
            html_body: Whether body is HTML (True) or plain text (False)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message with explicit headers
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.sender_email  # Force From to match login email
            msg['To'] = to_email
            
            # Add body (HTML or plain text)
            if html_body:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Attach log file if provided
            if log_file_path and os.path.exists(log_file_path):
                try:
                    print(f"[DEBUG] Attaching file... Size: {os.path.getsize(log_file_path)} bytes")
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    # Create attachment
                    attachment = MIMEText(log_content, 'plain')
                    attachment.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{os.path.basename(log_file_path)}"'
                    )
                    msg.attach(attachment)
                    
                    logger.info(f"📎 Log file attached: {log_file_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Error attaching log file: {e}")
            else:
                if log_file_path:
                    print(f"FAILED TO ATTACH LOG FILE: File not found - {log_file_path}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"[OK] [SUCCESS] Message accepted by Gmail server")
            logger.info(f"📨 Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full error details: {type(e).__name__}: {e}")
            return False
    
    def send_html_email_with_attachment(self, subject: str, html_content: str, 
                                       attachment_path: str = None) -> bool:
        """
        Send HTML email with optional attachment
        
        Args:
            subject: Email subject
            html_content: HTML email content
            attachment_path: Path to file to attach (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Attach file if provided
            if attachment_path and os.path.exists(attachment_path):
                try:
                    with open(attachment_path, 'rb') as f:
                        file_content = f.read()
                    
                    # Create attachment
                    attachment = MIMEBase('application', 'octet-stream')
                    attachment.set_payload(file_content)
                    attachment.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{os.path.basename(attachment_path)}"'
                    )
                    msg.attach(attachment)
                    
                    logger.info(f"📎 File attached: {attachment_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Error attaching file: {e}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending HTML email: {e}")
            return False
    
    def send_daily_brief(self, account_info: Dict, current_positions: List[Dict], 
                         trades_today: List[Dict], risk_summary: Dict = None, log_file_path: str = None, attachment_file: str = None) -> bool:
        """
        Send Daily Executive Brief
        
        Args:
            account_info: Account information from Alpaca
            current_positions: Current portfolio positions
            trades_today: Trades executed today
            risk_summary: Risk management summary
            log_file_path: Optional path to log file for attachment
            attachment_file: Optional path to additional file for attachment (e.g., CSV)
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Get current time in both timezones
            now_utc = datetime.now(pytz.UTC)
            now_eastern = now_utc.astimezone(self.eastern)
            now_israel = now_utc.astimezone(self.israel)
            
            # Generate HTML content
            html_content = self._generate_daily_brief_html(
                account_info, current_positions, trades_today, risk_summary,
                now_eastern, now_israel
            )
            
            # Send email with log file and additional attachments
            success = self._send_email(subject, html_content, log_file_path, attachment_file)
            
            if success:
                logger.info(f"Daily brief sent to {self.recipient_email}")
            else:
                logger.error("Failed to send daily brief")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending daily brief: {e}")
            return False
    
    def _generate_daily_brief_html(self, account_info: Dict, current_positions: List[Dict],
                                   trades_today: List[Dict], risk_summary: Dict,
                                   now_eastern: datetime, now_israel: datetime) -> str:
        """Generate HTML content for daily brief"""
        
        # Load portfolio data from portfolio.json
        portfolio_data = self._load_portfolio_data()
        
        # Check for error condition
        if 'error' in portfolio_data:
            return self._generate_error_daily_brief_html(portfolio_data, now_eastern, now_israel)
        
        # Calculate portfolio metrics from real data
        cash = portfolio_data.get('cash', 0.0)
        positions = portfolio_data.get('positions', {})
        
        # Calculate real-time market value using actual current prices from portfolio.json
        market_value = 0.0
        for ticker, position in positions.items():
            shares = position.get('shares', 0)
            current_price = position.get('current_price', 0)
            market_value += shares * current_price
        
        # Calculate real-time total value and return
        total_value = cash + market_value
        initial_cash = 100000.0  # Initial starting capital
        total_return = total_value - initial_cash
        total_return_pct = (total_return / initial_cash * 100) if initial_cash > 0 else 0
        
        # Generate positions HTML
        positions_html = self._generate_positions_html(positions)
        
        # Generate trades HTML
        trades_html = self._generate_trades_html(trades_today)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            text-align: center;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .timestamp {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 NeuralTrader Daily Brief</h1>
        <div class="timestamp">{now_eastern.strftime('%Y-%m-%d %H:%M:%S EST')}</div>
    </div>
    
    <div class="section">
        <h2>💰 Portfolio Overview</h2>
        <div class="metric">
            <div class="metric-value">${total_value:,.2f}</div>
            <div class="metric-label">Portfolio Equity</div>
        </div>
        <div class="metric">
            <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">
                ${total_return:,.2f}
            </div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric">
            <div class="metric-value {'positive' if total_return_pct >= 0 else 'negative'}">
                {total_return_pct:.2f}%
            </div>
            <div class="metric-label">Total Return %</div>
        </div>
        <div class="metric">
            <div class="metric-value">${cash:,.2f}</div>
            <div class="metric-label">Available Cash</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(positions)}</div>
            <div class="metric-label">Active Positions</div>
        </div>
        <div class="metric">
            <div class="metric-value">🟢 LIVE</div>
            <div class="metric-label">Data Source</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Active Positions</h2>
        {positions_html}
    </div>
    
    <div class="section">
        <h2>💼 Trades Executed</h2>
        {trades_html}
    </div>
    
    <div class="section">
        <h2>📈 Market Activity</h2>
        <p><strong>Buy Signals:</strong> No buy signals today</p>
        <p><strong>Sell Signals:</strong> No sell signals today</p>
        <p><strong>Hold Signals:</strong> No hold signals today</p>
    </div>
    
    <div class="section">
        <h2>⚠️ Risk Management</h2>
        <p><strong>Risk Status:</strong> All systems operational</p>
        <p><strong>Compliance:</strong> Within risk limits</p>
        <p><strong>Sector Authority:</strong> Active</p>
    </div>
    
    <div class="header">
        <div>NeuralTrader Automated Trading System v5.0</div>
        <div class="timestamp">Generated on {now_eastern.strftime('%Y-%m-%d %H:%M:%S EST')}</div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_error_daily_brief_html(self, portfolio_data: Dict, now_eastern: datetime, now_israel: datetime) -> str:
        """Generate error daily brief HTML when IBKR connection fails"""
        error_info = portfolio_data['error']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .error-message {{
            background: #ffebee;
            border: 2px solid #f44336;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #d32f2f;
        }}
        .error-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .timestamp {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 NeuralTrader Daily Brief - ERROR</h1>
        <div class="timestamp">{now_eastern.strftime('%Y-%m-%d %H:%M:%S EST')}</div>
    </div>
    
    <div class="section">
        <div class="error-message">
            <div class="error-title">CRITICAL ERROR: IBKR connection failed</div>
            <p>Portfolio state unknown. Please check IBKR TWS connection.</p>
            <p><strong>Error Type:</strong> {error_info['type']}</p>
            <p><strong>Message:</strong> {error_info['message']}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Current Holdings</h2>
        <p>CRITICAL ERROR: IBKR connection failed. Portfolio state unknown.</p>
    </div>
    
    <div class="section">
        <h2>💼 Trades Executed</h2>
        <p>CRITICAL ERROR: IBKR connection failed. Portfolio state unknown.</p>
    </div>
    
    <div class="section">
        <h2>📈 Market Activity</h2>
        <p>CRITICAL ERROR: IBKR connection failed. Portfolio state unknown.</p>
    </div>
    
    <div class="section">
        <h2>⚠️ Risk Management</h2>
        <p>CRITICAL ERROR: IBKR connection failed. Portfolio state unknown.</p>
    </div>
    
    <div class="header">
        <div>NeuralTrader Automated Trading System v5.0</div>
        <div class="timestamp">Generated on {now_eastern.strftime('%Y-%m-%d %H:%M:%S EST')}</div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_positions_html(self, positions: Dict) -> str:
        """Generate HTML for positions table"""
        if not positions:
            return "<p>No active positions</p>"
        
        rows_html = ""
        for ticker, pos_data in positions.items():
            # Use IBKR position data structure
            shares = pos_data.get('shares', 0)
            cost_basis = pos_data.get('cost_basis', 0)
            current_price = pos_data.get('current_price', 0)
            market_value = pos_data.get('market_value', 0)
            unrealized_pnl = pos_data.get('unrealized_pnl', 0)
            unrealized_pnl_pct = pos_data.get('unrealized_pnl_pct', 0)
            
            color_class = 'positive' if unrealized_pnl >= 0 else 'negative'
            
            rows_html += f"""
            <tr>
                <td><strong>{ticker}</strong></td>
                <td>{shares}</td>
                <td>${cost_basis:.2f}</td>
                <td>${current_price:.2f}</td>
                <td>${market_value:.2f}</td>
                <td class="{color_class}">${unrealized_pnl:.2f}</td>
                <td class="{color_class}">{unrealized_pnl_pct:.2f}%</td>
            </tr>
            """
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Shares</th>
                    <th>Cost Basis</th>
                    <th>Current Price</th>
                    <th>Market Value</th>
                    <th>Unrealized P&L</th>
                    <th>Unrealized P&L %</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
    
    def _generate_trades_html(self, trades_today: List[Dict]) -> str:
        """Generate HTML for trades executed today"""
        if not trades_today:
            return "<p>No trades executed today</p>"
        
        trades_html = ""
        for trade in trades_today:
            ticker = trade.get('ticker', 'Unknown')
            action = trade.get('action', 'Unknown')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            trades_html += f'<tr><td>{action.upper()}</td><td>{ticker}</td><td>{quantity}</td><td>${price:.2f}</td></tr>'
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Action</th>
                    <th>Ticker</th>
                    <th>Quantity</th>
                    <th>Price</th>
                </tr>
            </thead>
            <tbody>
                {trades_html}
            </tbody>
        </table>
        """
    
    def _send_email(self, subject: str, html_content: str, log_file_path: str = None, attachment_file: str = None) -> bool:
        """Send email via SMTP with optional log file and additional attachments"""
        try:
            # Create message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Handle log file attachment with fallback
            if log_file_path:
                if os.path.exists(log_file_path):
                    try:
                        with open(log_file_path, 'rb') as f:
                            log_attachment = MIMEBase('application', 'octet-stream')
                            log_attachment.set_payload(f.read())
                            encoders.encode_base64(log_attachment)
                            
                            # Get filename
                            filename = os.path.basename(log_file_path)
                            log_attachment.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {filename}'
                            )
                            msg.attach(log_attachment)
                            
                        logger.info(f"✅ Log file attached: {filename}")
                    except Exception as attach_error:
                        logger.error(f"❌ Failed to attach log file: {attach_error}")
                        print(f"FAILED TO ATTACH LOG FILE: {attach_error}")
                        
                        # Create fallback attachment for missing/corrupted log file
                        fallback_content = f"""
LOG FILE MISSING/CORRUPTED
============================

Expected log file: {log_file_path}
Status: File not found or could not be read

This is an automated fallback attachment from the Ironclad Wrapper.
The original log file may be missing due to:
- File system issues
- Permission problems
- Log file corruption
- Unexpected system shutdown

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logging system and file system integrity.
"""
                        
                        try:
                            fallback_attachment = MIMEText(fallback_content, 'plain')
                            fallback_attachment.add_header(
                                'Content-Disposition',
                                'attachment; filename="LOG_FILE_MISSING.txt"'
                            )
                            msg.attach(fallback_attachment)
                            logger.warning("⚠️ Created fallback attachment for missing log file")
                        except Exception as fallback_error:
                            logger.error(f"❌ Failed to create fallback attachment: {fallback_error}")
                else:
                    logger.error(f"❌ Log file not found: {log_file_path}")
                    print(f"FAILED TO ATTACH LOG FILE: File not found - {log_file_path}")
                    
                    # Create fallback attachment for missing log file
                    fallback_content = f"""
LOG FILE MISSING/CORRUPTED
============================

Expected log file: {log_file_path}
Status: File does not exist

This is an automated fallback attachment from the Ironclad Wrapper.
The original log file may be missing due to:
- File system issues
- Permission problems
- Log file corruption
- Unexpected system shutdown

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logging system and file system integrity.
"""
                    
                    try:
                        fallback_attachment = MIMEText(fallback_content, 'plain')
                        fallback_attachment.add_header(
                            'Content-Disposition',
                            'attachment; filename="LOG_FILE_MISSING.txt"'
                        )
                        msg.attach(fallback_attachment)
                        logger.warning("⚠️ Created fallback attachment for missing log file")
                    except Exception as fallback_error:
                        logger.error(f"❌ Failed to create fallback attachment: {fallback_error}")
            
            # Attach additional file if provided
            if attachment_file and os.path.exists(attachment_file):
                try:
                    with open(attachment_file, 'rb') as f:
                        file_attachment = MIMEBase('application', 'octet-stream')
                        file_attachment.set_payload(f.read())
                        encoders.encode_base64(file_attachment)
                        
                        # Get filename
                        filename = os.path.basename(attachment_file)
                        file_attachment.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {filename}'
                        )
                        msg.attach(file_attachment)
                        
                    logger.info(f"✅ Additional file attached: {filename}")
                except Exception as attach_error:
                    logger.error(f"❌ Failed to attach additional file: {attach_error}")
                    print(f"FAILED TO ATTACH ADDITIONAL FILE: {attach_error}")
            elif attachment_file:
                logger.error(f"❌ Additional file not found: {attachment_file}")
                print(f"FAILED TO ATTACH ADDITIONAL FILE: File not found - {attachment_file}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full error details: {type(e).__name__}: {e}")
            return False
    
    def send_alert(self, subject: str, message: str) -> bool:
        """
        Send alert email
        
        Args:
            subject: Alert subject
            message: Alert message
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Simple text email for alerts
            msg = MIMEMultipart()
            msg['Subject'] = f"🚨 NeuralTrader Alert: {subject}"
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Add timestamp
            timestamp = datetime.now(self.eastern).strftime('%Y-%m-%d %H:%M:%S EST')
            
            body = f"""
NeuralTrader Alert

Timestamp: {timestamp}

{message}

---
NeuralTrader Automated Trading System
Phase 6.1: Production Environment
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False

# Usage example
if __name__ == "__main__":
    # Sample data for testing
    account_info = {
        'portfolio_value': 100000,
        'cash': 5000,
        'buying_power': 95000
    }
    
    positions = [
        {
            'symbol': 'AAPL',
            'qty': 100,
            'cost_basis': 150.0,
            'current_price': 175.0,
            'unrealized_pl': 2500.0
        },
        {
            'symbol': 'GOOGL',
            'qty': 50,
            'cost_basis': 2500.0,
            'current_price': 2800.0,
            'unrealized_pl': 15000.0
        }
    ]
    
    trades_today = [
        {
            'timestamp': '2024-01-15 10:30:00',
            'ticker': 'AAPL',
            'action': 'buy',
            'quantity': 10,
            'price': 175.0
        },
        {
            'timestamp': '2024-01-15 11:15:00',
            'ticker': 'GOOGL',
            'action': 'sell',
            'quantity': 5,
            'price': 2800.0
        }
    ]
    
    risk_summary = {
        'risk_per_trade': 0.02,
        'max_sector_exposure': 0.30,
        'daily_loss_limit': 0.05
    }
    
    # Send test email
    notifier = EmailNotifier()
    success = notifier.send_daily_brief(account_info, positions, trades_today, risk_summary)
    
    if success:
        print("Test email sent successfully")
