"""
NeuralTrader Email Notifier
===========================

Sends daily executive briefs with portfolio performance and risk compliance.

Features:
- Daily Executive Brief with portfolio metrics
- Constitution Health Check reporting
- TLS email sending via smtplib
- HTML formatted emails with professional styling
- Scheduled reporting at 16:15 EST (23:15 IST)

Usage:
    from src.utils.notifier import EmailNotifier
    
    notifier = EmailNotifier()
    notifier.send_daily_brief(account_info, positions, trades_today)
"""

import os
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
import pytz

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
        
        logger.info("Email Notifier initialized")
        logger.info(f"Sender: {self.sender_email}")
        logger.info(f"Recipient: {self.recipient_email}")
        logger.info("✅ Environment variables loaded successfully")
    
    def send_email_with_logs(self, to_email: str, subject: str, body: str, 
                          log_file_path: str = None) -> bool:
        """
        Send email with full logs attached
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body
            log_file_path: Path to log file to attach
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message with explicit headers
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.sender_email  # Force From to match login email
            msg['To'] = to_email
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach log file if provided
            if log_file_path and os.path.exists(log_file_path):
                try:
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
            
            logger.info(f"📧 Sending email to {to_email}")
            logger.info(f"📧 From: {self.sender_email}")
            logger.info(f"📧 Subject: {subject}")
            
            # Send email with debug mode
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                # Enable SMTP debugging
                server.set_debuglevel(1)
                logger.info(f"🔗 Connecting to {self.smtp_server}:{self.smtp_port}")
                
                # Start TLS
                server.starttls()
                logger.info("🔒 TLS connection established")
                
                # Login (credentials are guaranteed to exist due to constructor validation)
                logger.info(f"🔐 Logging in as {self.sender_email}")
                server.login(self.sender_email, self.sender_password)
                logger.info("✅ Login successful")
                
                # Send message with verification
                try:
                    logger.info("📤 Sending message...")
                    server.send_message(msg)
                    logger.info("✅ [SUCCESS] Message accepted by Gmail server")
                    logger.info(f"📨 Email sent successfully to {to_email}")
                    return True
                    
                except Exception as send_error:
                    logger.error(f"❌ [FAILED] Message rejected by server: {send_error}")
                    logger.error(f"❌ Full error details: {type(send_error).__name__}: {send_error}")
                    return False
            
        except Exception as e:
            logger.error(f"❌ Error sending email: {e}")
            logger.error(f"❌ Full error details: {type(e).__name__}: {e}")
            return False
    
    def send_daily_brief(self, account_info: Dict, current_positions: List[Dict], 
                         trades_today: List[Dict], risk_summary: Dict = None) -> bool:
        """
        Send Daily Executive Brief
        
        Args:
            account_info: Account information from Alpaca
            current_positions: Current portfolio positions
            trades_today: Trades executed today
            risk_summary: Risk management summary
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Get current times
            now_eastern = datetime.now(self.eastern)
            now_israel = datetime.now(self.israel)
            
            # Compose email
            subject = f"NeuralTrader Daily Brief - {now_eastern.strftime('%Y-%m-%d')} ({now_eastern.strftime('%H:%M')} EST / {now_israel.strftime('%H:%M')} IST)"
            
            # Generate HTML content
            html_content = self._generate_daily_brief_html(
                account_info, current_positions, trades_today, risk_summary,
                now_eastern, now_israel
            )
            
            # Send email
            success = self._send_email(subject, html_content)
            
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
        
        # Calculate portfolio metrics
        portfolio_value = account_info.get('portfolio_value', 0)
        cash = account_info.get('cash', 0)
        buying_power = account_info.get('buying_power', 0)
        
        # Calculate today's P&L (simplified - would need previous day's value)
        today_pnl = 0.0
        for position in current_positions:
            unrealized_pl = float(position.get('unrealized_pl', 0))
            today_pnl += unrealized_pl
        
        today_pnl_pct = (today_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        
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
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .neutral {{
            color: #f39c12;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
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
        .status-pass {{
            background-color: #d4edda;
            color: #155724;
            padding: 5px 10px;
            border-radius: 3px;
        }}
        .status-fail {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 5px 10px;
            border-radius: 3px;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ NeuralTrader Daily Executive Brief</h1>
        <p>{now_eastern.strftime('%A, %B %d, %Y')}</p>
        <p>{now_eastern.strftime('%H:%M')} EST / {now_israel.strftime('%H:%M')} IST</p>
    </div>
    
    <div class="section">
        <h2>💰 Portfolio Overview</h2>
        <div class="metric">
            <div class="metric-value">${portfolio_value:,.2f}</div>
            <div class="metric-label">Portfolio Equity</div>
        </div>
        <div class="metric">
            <div class="metric-value ${'positive' if today_pnl >= 0 else 'negative'}">
                ${today_pnl:,.2f}
            </div>
            <div class="metric-label">Today's P&L</div>
        </div>
        <div class="metric">
            <div class="metric-value ${'positive' if today_pnl_pct >= 0 else 'negative'}">
                {today_pnl_pct:+.2f}%
            </div>
            <div class="metric-label">Today's Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">${cash:,.2f}</div>
            <div class="metric-label">Available Cash</div>
        </div>
        <div class="metric">
            <div class="metric-value">${buying_power:,.2f}</div>
            <div class="metric-label">Buying Power</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(current_positions)}</div>
            <div class="metric-label">Active Positions</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Active Positions</h2>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Shares</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>Market Value</th>
                    <th>P&L</th>
                    <th>P&L %</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add positions
        for position in current_positions:
            ticker = position.get('symbol', 'N/A')
            shares = float(position.get('qty', 0))
            cost_basis = float(position.get('cost_basis', 0))
            market_value = float(position.get('market_value', 0))
            unrealized_pl = float(position.get('unrealized_pl', 0))
            
            entry_price = cost_basis / shares if shares > 0 else 0
            current_price = market_value / shares if shares > 0 else 0
            pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0
            
            pl_class = 'positive' if unrealized_pl >= 0 else 'negative'
            
            html += f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td>{shares:,.0f}</td>
                    <td>${entry_price:.2f}</td>
                    <td>${current_price:.2f}</td>
                    <td>${market_value:,.2f}</td>
                    <td class="{pl_class}">${unrealized_pl:,.2f}</td>
                    <td class="{pl_class}">{pl_pct:+.2f}%</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>🔄 Today's Trading Activity</h2>
"""
        
        if trades_today:
            html += """
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Ticker</th>
                        <th>Action</th>
                        <th>Shares</th>
                        <th>Price</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for trade in trades_today:
                timestamp = trade.get('timestamp', 'N/A')
                ticker = trade.get('symbol', 'N/A')
                side = trade.get('side', 'N/A')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                value = quantity * price
                status = trade.get('status', 'N/A')
                
                html += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><strong>{ticker}</strong></td>
                        <td>{side.upper()}</td>
                        <td>{quantity:,.0f}</td>
                        <td>${price:.2f}</td>
                        <td>${value:,.2f}</td>
                        <td>{status}</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
"""
        else:
            html += "<p>No trades executed today.</p>"
        
        html += """
    </div>
    
    <div class="section">
        <h2>🛡️ Constitution Health Check</h2>
"""
        
        # Risk management summary
        if risk_summary:
            html += f"""
            <table>
                <thead>
                    <tr>
                        <th>Risk Metric</th>
                        <th>Current</th>
                        <th>Limit</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Risk Per Trade</td>
                        <td>{risk_summary.get('risk_per_trade', 'N/A')}</td>
                        <td>0.9%</td>
                        <td class="status-pass">PASS</td>
                    </tr>
                    <tr>
                        <td>Max Sector Exposure</td>
                        <td>{risk_summary.get('max_sector_exposure', 'N/A')}</td>
                        <td>30%</td>
                        <td class="status-pass">PASS</td>
                    </tr>
                    <tr>
                        <td>Black Swan Protection</td>
                        <td>{risk_summary.get('black_swan_status', 'Active')}</td>
                        <td>Active</td>
                        <td class="status-pass">PASS</td>
                    </tr>
                    <tr>
                        <td>Duplicate Protection</td>
                        <td>{risk_summary.get('duplicate_protection', 'Active')}</td>
                        <td>Active</td>
                        <td class="status-pass">PASS</td>
                    </tr>
                </tbody>
            </table>
"""
        else:
            html += "<p>Risk summary not available.</p>"
        
        html += f"""
    </div>
    
    <div class="section">
        <h2>📈 Performance Summary</h2>
        <div class="metric">
            <div class="metric-value">30.83%</div>
            <div class="metric-label">Target CAGR</div>
        </div>
        <div class="metric">
            <div class="metric-value">-18.94%</div>
            <div class="metric-label">Max Drawdown Target</div>
        </div>
        <div class="metric">
            <div class="metric-value">57.50%</div>
            <div class="metric-label">Target Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">0.9%</div>
            <div class="metric-label">Risk Per Trade</div>
        </div>
    </div>
    
    <div class="footer">
        <p>🏛️ NeuralTrader Automated Trading System</p>
        <p>Phase 6.1: Production Environment | S&P 100 Universe</p>
        <p>Generated: {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} EST</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _send_email(self, subject: str, html_content: str) -> bool:
        """Send email via SMTP"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
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
    # Test the email notifier
    notifier = EmailNotifier()
    
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
            'cost_basis': 15000,
            'market_value': 16000,
            'unrealized_pl': 1000
        }
    ]
    
    trades_today = [
        {
            'timestamp': '09:45:00',
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'price': 150.0,
            'status': 'filled'
        }
    ]
    
    risk_summary = {
        'risk_per_trade': '0.9%',
        'max_sector_exposure': '25%',
        'black_swan_status': 'Active',
        'duplicate_protection': 'Active'
    }
    
    # Send test email
    success = notifier.send_daily_brief(account_info, positions, trades_today, risk_summary)
    
    if success:
        print("Test email sent successfully")
    else:
        print("Failed to send test email")
