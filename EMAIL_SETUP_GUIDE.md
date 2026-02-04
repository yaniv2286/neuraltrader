# NeuralTrader Email Setup Guide
================================

## 📧 Enhanced Email System with Full Log Attachments

The NeuralTrader email system has been enhanced to provide comprehensive notifications with **full execution logs attached** to every email. This ensures complete verification and debugging capabilities for all Task Scheduler runs.

## 🎯 What You'll Receive

### **Enhanced Email Features**
- **Full Log Attachments**: Complete execution logs (automation_YYYYMMDD_HHMMSS.log) attached to every email
- **Detailed Metrics**: Performance data, duration, success rates
- **Error Reporting**: Complete error information with stack traces
- **Unicode Compatibility**: ASCII-only content for Task Scheduler
- **Multiple Recipients**: Support for multiple email addresses

### **Task Scheduler Email Types**
| Task | Schedule | Email Content | Log Attachment |
|------|----------|---------------|-----------------|
| **Data Fetch** | Daily 16:45 IST | Data fetch results, tickers processed | ✅ Full automation log |
| **Daily Report** | Daily 23:15 IST | Portfolio performance, risk status | ✅ Full automation log |
| **Saturday Retrain** | Weekly Saturday | Model retraining results, performance | ✅ Full automation log |

## 🔧 Quick Setup (5 Minutes)

### Option 1: Gmail (Recommended)

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" for the app
   - Generate a 16-character password
   - Copy this password

3. **Create .env file**:
```cmd
copy .env.example .env
```

4. **Edit .env file** with your credentials:
```env
# Gmail Configuration
NOTIFIER_EMAIL=your-gmail@gmail.com
NOTIFIER_PASSWORD=your-16-character-app-password
EMAIL_RECIPIENT=lugassy.ai@gmail.com

# Optional: Multiple recipients
EMAIL_RECIPIENT=recipient1@gmail.com,recipient2@gmail.com
```

### Option 2: Outlook/Hotmail

1. **Enable 2-Factor Authentication** on your Microsoft account
2. **Generate App Password**:
   - Go to: https://account.live.com/proofs/manage/additional
   - Create an app password
   - Copy the password

3. **Edit .env file**:
```env
# Outlook Configuration
NOTIFIER_EMAIL=your-email@outlook.com
NOTIFIER_PASSWORD=your-app-password
EMAIL_RECIPIENT=lugassy.ai@gmail.com
```

## 🧪 Test Email Setup

After configuring .env, test the email system:

```cmd
python tests\test_email.py
```

Expected output:
```
📧 NOTIFIER_EMAIL: ✅ Set
🔑 NOTIFIER_PASSWORD: ✅ Set
✅ Test email sent successfully!
📨 Check your inbox: lugassy.ai@gmail.com
```

## 📊 Enhanced Email System Status

### ✅ Working Components:
- **Email Notifier** with log attachment support
- **SMTP connection** to Gmail/Outlook with enhanced error handling
- **Email formatting** with ASCII-only content for Task Scheduler
- **Error handling** and comprehensive logging
- **Daily brief generation** with portfolio metrics
- **Full log attachments** for complete execution verification

### 🔧 Enhanced Features:
- **send_email_with_logs()**: New method for attaching execution logs
- **Unicode compatibility**: ASCII-only content for Task Scheduler
- **Multiple recipients**: Support for multiple email addresses
- **Enhanced error reporting**: Complete stack traces and debugging info
- **Timestamped logs**: Unique log files for each execution

## 🔄 What Happens When Email Is Configured

Once you set up email credentials, you'll receive:

### **1. Data Fetch Notifications** (16:45 IST)
```
[DATA] NeuralTrader Data Fetch Complete - 2026-02-04 16:45 IST
[DATA] DATA FETCH RESULTS: 97 tickers fetched
[SUCCESS] Market data updated
[EMAIL] FULL LOGS ATTACHED: Complete automation log attached
```

### **2. Daily Executive Brief** (23:15 IST)
```
📧 NeuralTrader Daily Executive Brief - 2026-02-04 23:15 IST
📊 PORTFOLIO OVERVIEW: $100,000.00 total value
📈 ACTIVE POSITIONS: 0 positions
📋 TRADING ACTIVITY: 0 trades today
[EMAIL] FULL LOGS ATTACHED: Complete automation log attached
```

### **3. Saturday Retrain Notifications** (Weekly Saturday)
```
🔄 NeuralTrader Saturday Retrain - 2026-02-04 10:00 IST
📊 RETRAIN RESULTS: 5 models updated
📈 PERFORMANCE: 2.3% improvement
[EMAIL] FULL LOGS ATTACHED: Complete automation log attached
```

## 📧 Email Content Examples

### **Data Fetch Email**
```
[DATA] NeuralTrader Data Fetch Complete - 2026-02-04 16:45 IST

[DATA] DATA FETCH RESULTS:
---------------------
Tickers Fetched: 97
Data Source: Yahoo Finance
Status: [SUCCESS]

[SUCCESS] Market data updated
[SUCCESS] Risk systems operational
[SUCCESS] Virtual portfolio ready
[SUCCESS] Email notifications enabled

[EMAIL] FULL LOGS ATTACHED:
-------------------
Complete automation log attached for detailed analysis.

---
NeuralTrader Automated Trading System
Phase 6.2: Task Scheduler Integration
```

### **Daily Report Email**
```
📧 NeuralTrader Daily Executive Brief - 2026-02-04 23:15 IST

📊 PORTFOLIO OVERVIEW:
-------------------------
Total Portfolio Value: $100,000.00
Cash Balance: $100,000.00
Position Value: $0.00

Performance Metrics:
• Total Return: $0.00
• Total Return %: 0.00%
• Unrealized P&L: $0.00
• Realized P&L: $0.00

📈 ACTIVE POSITIONS (0):

📋 TRADING ACTIVITY:
-------------------
Total Trades: 0
Recent Trades:

🌍 MARKET STATUS:
------------------
Current Time: 2026-02-04 23:15:00 IST
EST Time: 2026-02-04 15:45:00 EST
Market Hours: CLOSED

🏛️ CONSTITUTION HEALTH CHECK:
---------------------------------
✅ 0.9% Risk Per Trade: Active
✅ 30% Technology Sector Cap: Active
✅ Black Swan Exit: VXX Monitoring Active
✅ Duplicate Position Protection: Active
✅ Capital Preservation: Priority #1

🎯 READINESS FOR TOMORROW:
-------------------------
✅ All risk systems operational
✅ Virtual portfolio ready for trading
✅ Market data feed active (Yahoo Finance)
✅ Email notifications enabled
✅ Shadow trading simulation running

[EMAIL] FULL LOGS ATTACHED:
-------------------
Complete automation log attached for detailed analysis.

---
NeuralTrader Automated Trading System
Phase 6.2: Task Scheduler Integration
```

## 🛡️ Security Notes

- **Never commit .env file to Git** (it's already in .gitignore)
- **Use App Passwords**, not your main password
- **Email credentials are stored locally only**
- **Gmail App Passwords are more secure than regular passwords**
- **Log files contain system information** - handle attachments securely

## 🚀 Quick Start Commands

```cmd
# 1. Copy the example file
copy .env.example .env

# 2. Edit .env with your credentials
notepad .env

# 3. Test email sending
python tests\test_email.py

# 4. Test Task Scheduler with logs
.\run_neural_venv.bat fetch

# 5. Check email with log attachment
# Check your inbox for the email with automation_*.log attachment
```

## 📞 Troubleshooting

### **"Authentication Required" Error**
- Generate a new App Password
- Ensure 2-factor authentication is enabled
- Check email address and password spelling

### **"Connection Timeout" Error**
- Check internet connection
- Verify SMTP server settings
- Try again in a few minutes

### **"Email Not Received" Error**
- Check spam/junk folder
- Verify recipient email address
- Check email provider's sending limits

### **"Log File Not Attached" Error**
- Check if automation_*.log file exists
- Verify file permissions
- Check log file size (large files may be blocked)

### **"Unicode Encoding Error"**
- This has been fixed with ASCII-only content
- All emails now use Task Scheduler-compatible encoding
- Unicode characters are converted to ASCII equivalents

## 🔍 Log File Management

### **Log File Naming**
- **Format**: `automation_YYYYMMDD_HHMMSS.log`
- **Location**: `logs/` directory
- **Purpose**: Unique log files avoid file locking issues

### **Log File Content**
- **Complete execution logs** with timestamps
- **Error information** with stack traces
- **Performance metrics** and duration
- **System status** and module initialization

### **Log File Access**
```cmd
# View latest log file
Get-Content -Tail 50 logs\automation_*.log

# Search for errors in logs
Select-String -Pattern "ERROR" -Path logs\automation_*.log

# Check today's logs
Get-ChildItem logs\automation_*.log | Where-Object {$_.Name -like "*$(Get-Date -Format 'yyyyMMdd')*"}
```

## 📧 Multiple Recipients

### **Setup Multiple Recipients**
```env
# Single recipient
EMAIL_RECIPIENT=recipient@gmail.com

# Multiple recipients (comma-separated)
EMAIL_RECIPIENT=recipient1@gmail.com,recipient2@gmail.com,recipient3@gmail.com
```

### **Recipient Verification**
```cmd
# Test multiple recipients
python tests\test_email.py --recipients="recipient1@gmail.com,recipient2@gmail.com"
```

## 🎯 Email Verification

### **Test All Email Features**
```cmd
# Test basic email
python tests\test_email.py

# Test email with logs
.\run_neural_venv.bat fetch

# Test daily report email
.\run_neural_venv.bat report

# Test Saturday retrain email
.\run_neural_venv.bat saturday_retrain
```

### **Email Content Verification**
- **Check for log attachments**: Each email should have a `.log` file attached
- **Verify ASCII content**: No Unicode characters in email body
- **Confirm timestamps**: Email subject should include current date/time
- **Check metrics**: Performance data should be included

---

**Status**: Enhanced email system with full log attachments is ready! 🎯

## 📋 Task Scheduler Email Schedule

| Time (IST) | Task | Email Type | Log Attachment |
|------------|------|------------|----------------|
| **16:45** | Data Fetch | Data fetch notification | ✅ automation_*.log |
| **23:15** | Daily Report | Executive brief | ✅ automation_*.log |
| **Saturday 10:00** | Retrain | Model retrain results | ✅ automation_*.log |

---

*Last Updated: February 4, 2026*
