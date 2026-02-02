# NeuralTrader Task Scheduler Setup Guide
=====================================

This guide explains how to set up Windows Task Scheduler for NeuralTrader Phase 6.2 automation.

## 🎯 Overview

NeuralTrader Master Runner supports three operation modes:
- **FETCH MODE** (16:45 IST): Data fetch for previous day's data and opening trends
- **TRADE MODE** (Market hours): Shadow trading execution during market hours
- **REPORT MODE** (23:15 IST): Daily executive brief with portfolio updates

## 📋 Prerequisites

1. **Python Virtual Environment**: Already created at `.venv/`
2. **Required Libraries**: yfinance, pytz, pandas, schedule
3. **Project Directory**: `D:\GitHub\NeuralTrader`
4. **Batch Script**: `run_neural.bat` for Task Scheduler execution

## 🕐 IST Schedule Times

| Mode | IST Time | EST Time | Purpose |
|------|-----------|-----------|---------|
| FETCH | 16:45 IST | 09:45 EST | Data fetch and opening trends |
| TRADE | Market Hours | Market Hours | Shadow trading execution |
| REPORT | 23:15 IST | 16:15 EST | Daily executive brief |

## 🔧 Task Scheduler Setup

### Step 1: Open Task Scheduler
1. Press `Win + R` and type `taskschd.msc`
2. Click "Create Task" in the Actions pane

### Step 2: General Settings
- **Name**: `NeuralTrader Data Fetch`
- **Description**: `Fetch market data at 16:45 IST (09:45 EST)`
- **Security options**: 
  - Select "Run whether user is logged on or not"
  - Check "Run with highest privileges"
  - Configure for: Windows 10

### Step 3: Triggers Setup
1. Click "Triggers" tab → "New..."
2. **Begin the task**: Daily
3. **Start time**: 09:45:00 (EST time)
4. **Repeat every**: 1 day
5. Click "OK"

### Step 4: Actions Setup
1. Click "Actions" tab → "New..."
2. **Action**: Start a program
3. **Program/script**: `D:\GitHub\NeuralTrader\run_neural.bat`
4. **Add arguments**: `fetch`
5. **Start in**: `D:\GitHub\NeuralTrader`
6. Click "OK"

### Step 5: Conditions Setup
1. Click "Conditions" tab
2. **Power**: Uncheck "Start the task only if the computer is on AC power"
3. **Network**: Check "Start only if the following network connection is available"
4. Select "Any connection"

### Step 6: Settings Setup
1. Click "Settings" tab
2. **Allow task to be run on demand**: Checked
3. **Stop task if it runs longer than**: 30 minutes
4. **If the task is already running**: Do not start a new instance
5. Click "OK"

## 📋 Additional Tasks

### Task 2: Daily Report (23:15 IST)
- **Name**: `NeuralTrader Daily Report`
- **Start time**: 16:15:00 (EST time)
- **Arguments**: `report`

### Task 3: Trading Session (Optional)
- **Name**: `NeuralTrader Trading`
- **Start time**: 10:00:00 (EST time)
- **Arguments**: `trade`
- **Repeat**: Every 30 minutes during market hours

## 🧪 Testing Tasks

### Test Data Fetch
```cmd
cd D:\GitHub\NeuralTrader
run_neural.bat fetch
```

### Test Report
```cmd
cd D:\GitHub\NeuralTrader
run_neural.bat report
```

### Test Trading
```cmd
cd D:\GitHub\NeuralTrader
run_neural.bat trade
```

## 📊 Monitoring

### Check Automation Log
```cmd
type logs\automation.log
```

### Check Task History
1. In Task Scheduler, select the task
2. Click "History" tab
3. Review execution results

### Check Portfolio Status
```cmd
python main_orchestrator_ist.py --mode=report
```

## 🛡️ Safety Features

### Kill Switch
Create `STOP.txt` file in project root to stop all automation:
```cmd
echo STOP > STOP.txt
```

### Environment Validation
The system automatically checks for:
- Required libraries (yfinance, pytz, pandas, schedule)
- Virtual environment existence
- Project directory structure

### Error Handling
- Comprehensive logging to `logs/automation.log`
- Email notifications for failures
- Automatic retry on network errors

## 🔧 Troubleshooting

### Common Issues

1. **"Library not found" error**
   ```cmd
   .venv\Scripts\activate
   pip install -r requirements_trading.txt
   ```

2. **"Working directory" error**
   - Ensure batch script runs from project root
   - Check "Start in" field in Task Scheduler

3. **"Rate limit" error**
   - Normal for Yahoo Finance API
   - System will retry automatically
   - Check logs for successful fetches

4. **"Virtual environment" error**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements_trading.txt
   ```

### Log Analysis
Check `logs/automation.log` for:
- Session start/end times
- Error messages
- Success/failure status
- Performance metrics

## 📧 Email Notifications

The system sends emails to `lugassy.ai@gmail.com`:
- Data fetch completion notifications
- Daily executive brief reports
- Error notifications (if configured)

## 🎯 Success Indicators

### Successful Data Fetch
- ✅ 96+ tickers fetched successfully
- ✅ Opening trends analyzed
- ✅ Portfolio values updated
- ✅ Notification sent

### Successful Report
- ✅ Portfolio values updated
- ✅ Email report sent
- ✅ Constitution health check completed

### Successful Trading
- ✅ Market data fetched
- ✅ Signals generated
- ✅ Trades executed (if signals present)
- ✅ Portfolio updated

## 🔄 Maintenance

### Weekly Tasks
1. Review automation logs
2. Check portfolio performance
3. Verify email notifications
4. Update risk parameters if needed

### Monthly Tasks
1. Clean old log files
2. Update ticker universe
3. Review and optimize parameters
4. Backup portfolio data

## 📞 Support

For issues:
1. Check `logs/automation.log`
2. Verify Task Scheduler history
3. Test with manual execution
4. Check virtual environment status

---

**Status**: ✅ Task Scheduler integration complete and tested
**Last Updated**: 2026-02-02
**Version**: Phase 6.2 - Master Runner
