# 🏛️ NeuralTrader: AI-Driven Quantitative Fund
**Target:** 25%+ ARR | **Risk Limit:** <20% Drawdown
**Current Status:** [Active Production]
**Architecture:** Core Kernel (Protected) + Execution Layer (Flexible)

> **⚠️ ARCHITECTURAL WARNING:**
> This system operates under the **"No Silent Failures"** protocol.
> 1. The `core/` directory is **IMMUTABLE** without explicit authorization.
> 2. The Orchestrator will refuse to run if `core/integrity.py` validation fails.
> 3. **Priority #1** is always Capital Preservation.

---

## 🏆 CURRENT MILESTONE: PHASE 6.2 (COMPLETE)
**Status:** ✅ **PRODUCTION READY** | **Task Scheduler:** Fully Automated

## 🎉 **MAJOR MILESTONE: PHASE 6.2 COMPLETE - PRODUCTION READY!**

**Date**: February 4, 2026  
**Status**: ✅ **TASK SCHEDULER FULLY OPERATIONAL**  
**Features**: **Automated daily execution with enhanced email notifications and supervision**

### **🚀 Latest Achievements**
- ✅ **Task Scheduler Integration**: All three tasks working perfectly
- ✅ **Enhanced Email Notifications**: Full logs attached to every email
- ✅ **Daily Supervision System**: Complete run tracking and verification
- ✅ **Unicode Compatibility**: Fixed encoding issues for Task Scheduler
- ✅ **Error Handling**: Robust error detection and reporting
- ✅ **Production Ready**: Fully tested and operational

### **🔥 Breakthrough Features**
- **Automated Data Fetch**: Daily S&P 100 data collection (16:45 IST)
- **Daily Executive Brief**: Portfolio performance reports (23:15 IST)
- **Saturday Retraining**: Weekly model updates and optimization
- **Full Log Attachments**: Complete execution logs in every email
- **Supervision Dashboard**: Real-time monitoring and verification
- **Trade Verification**: System to identify missed opportunities

---

## 🚀 Quick Start

### **Task Scheduler Setup**
```bash
# Setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Configure email credentials
# Edit .env file with your Gmail App Password
NOTIFIER_EMAIL=your@gmail.com
NOTIFIER_PASSWORD=your-app-password
EMAIL_RECIPIENT=recipient@gmail.com

# Test Task Scheduler scripts
.\run_neural_venv.bat fetch
.\run_neural_venv.bat report
.\run_neural_venv.bat saturday_retrain
```

### **Manual Testing**
```bash
# Test individual modes
python main_orchestrator_ist.py --mode=fetch
python main_orchestrator_ist.py --mode=report
python main_orchestrator_ist.py --mode=saturday_retrain

# Verify trades (check for missed opportunities)
python scripts\verify_trades.py --yesterday --send-email

# View supervision dashboard
python scripts\supervision_dashboard.py

# Generate daily supervision report
python scripts\daily_supervision_report.py --send-email
```

## 🎯 Task Scheduler Configuration

### **Automated Tasks**
| Task | Schedule | Command | Purpose |
|------|----------|---------|---------|
| **NeuralTrader_DataFetch** | Daily 16:45 IST | `run_neural_venv.bat fetch` | Fetch S&P 100 data |
| **NeuralTrader_DailyReport** | Daily 23:15 IST | `run_neural_venv.bat report` | Send executive brief |
| **NeuralTrader_SaturdayRetrain** | Weekly Saturday | `run_neural_venv.bat saturday_retrain` | Model retraining |

### **Email Notifications**
Each Task Scheduler run sends:
- **Status Report**: Success/failure with detailed metrics
- **Full Log Attachment**: Complete execution logs (automation_YYYYMMDD_HHMMSS.log)
- **Performance Metrics**: Duration, tickers processed, portfolio status
- **Error Details**: Complete error information if failures occur

### **Supervision Features**
- **Daily Tracking**: Every run logged with unique ID and timestamp
- **Performance Monitoring**: Execution duration and success rates
- **Error Detection**: Automatic error logging and notification
- **Historical Analysis**: Complete audit trail of all executions
- **Interactive Dashboard**: Real-time monitoring interface

## 📁 Project Structure

```
NeuralTrader/
├── 📁 Task Scheduler Files
│   ├── run_neural_venv.bat        # Main Task Scheduler script
│   ├── main_orchestrator_ist.py   # Core orchestrator with supervision
│   └── .env                       # Email credentials
├── 📁 Core System
│   ├── src/
│   │   ├── data/yfinance_manager.py    # Data fetching
│   │   ├── trading/risk_manager.py      # Risk management
│   │   ├── trading/virtual_engine.py    # Virtual trading engine
│   │   ├── utils/notifier.py            # Email notifications
│   │   ├── utils/daily_logger.py        # Daily supervision
│   │   └── utils/trade_verifier.py     # Trade verification
│   └── src/reporting/ist_scheduler.py    # Daily reporting
├── 📁 Scripts & Tools
│   ├── scripts/
│   │   ├── verify_trades.py             # Trade verification
│   │   ├── daily_supervision_report.py   # Daily reports
│   │   └── supervision_dashboard.py      # Interactive dashboard
│   └── tests/test_email.py               # Email testing
├── 📁 Logs & Data
│   ├── logs/automation_*.log              # Task Scheduler logs
│   ├── logs/supervision/                  # Daily supervision logs
│   └── logs/verification/                 # Trade verification logs
└── 📁 Configuration
    ├── .env.example                      # Email setup template
    └── EMAIL_SETUP_GUIDE.md              # Email setup instructions
```

## 🤖 Trading Strategy

### **Automation Workflow**
1. **Daily Data Fetch** (16:45 IST)
   - Fetch S&P 100 market data
   - Update price databases
   - Send completion notification with logs

2. **Daily Executive Brief** (23:15 IST)
   - Generate portfolio performance report
   - Risk compliance check
   - Send comprehensive brief with logs

3. **Saturday Retraining** (Weekly)
   - Retrain ML models on latest data
   - Validate model performance
   - Update trading parameters

### **Risk Management**
- **Capital Preservation**: Priority #1
- **Position Sizing**: 0.9% risk per trade
- **Sector Limits**: 30% max exposure per sector
- **Black Swan Protection**: VXX surge detection
- **Daily Loss Limit**: 5% maximum loss
- **Maximum Drawdown**: 20% hard stop

## 📧 Email Notification System

### **Enhanced Email Features**
- **Full Log Attachments**: Complete execution logs attached to every email
- **Detailed Metrics**: Performance data, duration, success rates
- **Error Reporting**: Complete error information with stack traces
- **Unicode Compatibility**: ASCII-only content for Task Scheduler
- **Multiple Recipients**: Support for multiple email addresses

### **Email Content Examples**
```
[DATA] NeuralTrader Data Fetch Complete - 2026-02-04 16:45 IST
[DATA] DATA FETCH RESULTS: 97 tickers fetched
[SUCCESS] Market data updated
[EMAIL] FULL LOGS ATTACHED: Complete automation log attached
```

## 🔧 Configuration

### **Environment Setup (.env)**
```env
# Email Configuration
NOTIFIER_EMAIL=your@gmail.com
NOTIFIER_PASSWORD=your-gmail-app-password
EMAIL_RECIPIENT=recipient@gmail.com

# Optional: Multiple recipients
EMAIL_RECIPIENT=recipient1@gmail.com,recipient2@gmail.com
```

### **Task Scheduler Setup**
1. **Create Tasks**: Use Windows Task Scheduler
2. **Set Triggers**: Daily at specified times
3. **Actions**: Run `run_neural_venv.bat` with mode parameter
4. **Settings**: Run whether user is logged on or not

## 📊 Supervision & Monitoring

### **Daily Supervision**
```bash
# View today's supervision
python scripts\supervision_dashboard.py

# Generate daily report
python scripts\daily_supervision_report.py --send-email

# Verify trades (check for missed opportunities)
python scripts\verify_trades.py --yesterday --detailed
```

### **Monitoring Dashboard**
- **Real-time Status**: Current system status
- **Daily Summary**: Execution statistics
- **Weekly Overview**: 7-day performance trends
- **Error Tracking**: Failed executions and issues
- **Performance Metrics**: Duration and success rates

## �️ Troubleshooting

### **Common Issues & Solutions**

#### **Unicode Encoding Errors**
**Problem**: Task Scheduler can't handle Unicode characters
**Solution**: Fixed with ASCII-only content and UTF-8 encoding

#### **File Permission Errors**
**Problem**: Log file locked by another process
**Solution**: Timestamped log files to avoid conflicts

#### **Email Authentication Issues**
**Problem**: Gmail authentication failures
**Solution**: Use Gmail App Password, not regular password

#### **Task Scheduler Failures**
**Problem**: Scripts failing in Task Scheduler
**Solution**: Enhanced error checking and debugging logs

### **Debug Mode**
```bash
# Enable detailed logging
.\run_neural_venv.bat fetch

# Check automation logs
Get-Content -Tail 50 logs\automation_*.log

# Test email configuration
python tests\test_email.py
```

## 📦 Dependencies

### **Core Requirements**
- `pandas>=2.0.0` - Data processing
- `yfinance>=0.2.0` - Market data
- `pytz>=2023.0` - Timezone handling
- `python-dotenv>=1.0.0` - Environment variables
- `schedule>=1.2.0` - Task scheduling

### **Email Requirements**
- `smtplib` (built-in) - Email sending
- `email.mime` (built-in) - Email formatting

### **Optional Dependencies**
- `schedule>=1.2.0` - Task scheduling
- `pyarrow>=15.0.0` - Parquet support

## 🎯 Usage Examples

### **Task Scheduler Testing**
```bash
# Test all Task Scheduler modes
.\run_neural_venv.bat fetch
.\run_neural_venv.bat report
.\run_neural_venv.bat saturday_retrain
```

### **Manual Supervision**
```bash
# Check for missed trades
python scripts\verify_trades.py --yesterday --send-email

# View supervision dashboard
python scripts\supervision_dashboard.py

# Generate daily report
python scripts\daily_supervision_report.py --print
```

### **Email Testing**
```bash
# Test email configuration
python tests\test_email.py

# Send test supervision report
python scripts\daily_supervision_report.py --send-email
```

## 📊 Performance & Monitoring

### **Daily Reports**
- **Portfolio Performance**: Total value, P&L, returns
- **Risk Compliance**: All risk metrics and limits
- **System Status**: Module health and connectivity
- **Execution Summary**: Success rates and duration

### **Weekly Analysis**
- **Performance Trends**: 7-day performance metrics
- **Risk Metrics**: Drawdown monitoring and alerts
- **System Health**: Module performance and issues
- **Execution Quality**: Success rates and error analysis

## 🚀 Deployment

### **Production Setup**
1. **Environment Setup**: Configure virtual environment and dependencies
2. **Email Configuration**: Set up Gmail App Password and recipients
3. **Task Scheduler**: Create and configure all three tasks
4. **Testing**: Verify all modes work correctly
5. **Monitoring**: Set up supervision and verification

### **Task Scheduler Configuration**
```xml
<!-- Example Task Scheduler XML -->
<Task>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-02-04T16:45:00</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>D:\GitHub\NeuralTrader\run_neural_venv.bat</Command>
      <Arguments>fetch</Arguments>
      <WorkingDirectory>D:\GitHub\NeuralTrader</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

## 🎉 Status

✅ **Phase 1 Complete**: Infrastructure & Data Pipeline  
✅ **Phase 2 Complete**: ML Features & Models  
✅ **Phase 3 Complete**: Risk Management & Execution  
✅ **Phase 4 Complete**: Paper Trading & Validation  
✅ **Phase 5 Complete**: Live Trading Performance  
🚀 **Phase 6.2 Complete**: Task Scheduler Integration & Supervision

## 📞 Support

### **Documentation**
- **EMAIL_SETUP_GUIDE.md**: Complete email setup instructions
- **TASK_SCHEDULER_GUIDE.md**: Task Scheduler configuration guide
- **SUPERVISION_GUIDE.md**: Daily supervision system guide

### **Troubleshooting**
- **Common Issues**: Unicode, permissions, authentication
- **Debug Mode**: Enhanced logging and error reporting
- **Email Support**: Gmail App Password setup

---

⚠️ **Disclaimer**: This is a sophisticated trading system. Past performance does not guarantee future results. Trade at your own risk.

🏆 **Production Ready**: Fully tested and operational Task Scheduler system with comprehensive supervision and monitoring.
