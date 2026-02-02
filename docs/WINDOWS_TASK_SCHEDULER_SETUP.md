# Windows Task Scheduler Setup for NeuralTrader

## 🚀 Automated Paper Trading Deployment

This guide walks you through setting up NeuralTrader to run automatically at 9:45 AM EST daily using Windows Task Scheduler.

---

## 📋 Prerequisites

1. **NeuralTrader Installed**: Complete Phase 5 deployment
2. **Alpaca Paper Trading Account**: Active with API keys
3. **Environment Variables Set**: ALPACA_API_KEY and ALPACA_SECRET_KEY
4. **Python Environment**: All dependencies installed

---

## 🔧 Setup Instructions

### Step 1: Verify Environment

1. **Open Command Prompt** and verify Python path:
   ```cmd
   python --version
   cd /d D:\GitHub\NeuralTrader
   python main_orchestrator.py --help
   ```

2. **Test Manual Run**:
   ```cmd
   python main_orchestrator.py --force
   ```

3. **Verify API Keys**:
   ```cmd
   echo %ALPACA_API_KEY%
   echo %ALPACA_SECRET_KEY%
   ```

### Step 2: Create Batch File

Create a batch file to run the orchestrator:

1. **Create file**: `D:\GitHub\NeuralTrader\run_trading.bat`
2. **Add this content**:
   ```batch
   @echo off
   cd /d D:\GitHub\NeuralTrader
   python main_orchestrator.py --auto >> logs\automated_runs.log 2>&1
   ```

3. **Test the batch file**:
   ```cmd
   D:\GitHub\NeuralTrader\run_trading.bat
   ```

### Step 3: Open Task Scheduler

1. **Press Win + R**, type `taskschd.msc`, press Enter
2. **Click "Create Task"** in the right panel
3. **General Tab**:
   - Name: `NeuralTrader Automated Trading`
   - Description: `Runs NeuralTrader paper trading at 9:45 AM EST`
   - Select: `Run whether user is logged on or not`
   - Check: `Run with highest privileges`

### Step 4: Set Trigger

1. **Click "Triggers" tab**
2. **Click "New..."**
3. **Settings**:
   - Begin the task: `On a schedule`
   - Settings: `Daily`
   - Start time: `9:45:00 AM`
   - Recur every: `1 Days`
   - Start: `Today's date`
4. **Check**: `Enabled`
5. **Click OK**

### Step 5: Set Action

1. **Click "Actions" tab**
2. **Click "New..."**
3. **Action**: `Start a program`
4. **Program/script**: `D:\GitHub\NeuralTrader\run_trading.bat`
5. **Start in (optional)**: `D:\GitHub\NeuralTrader`
6. **Click OK**

### Step 6: Configure Conditions

1. **Click "Conditions" tab**
2. **Power settings**:
   - Uncheck: `Start the task only if the computer is on AC power`
   - Uncheck: `Stop if the computer switches to battery power`
   - Check: `Wake the computer to run this task`

### Step 7: Configure Settings

1. **Click "Settings" tab**
2. **Settings**:
   - Check: `Allow task to be run on demand`
   - Check: `Run task as soon as possible after a scheduled start is missed`
   - Stop task if it runs longer than: `1 hour`
   - If the task is already running: `Do not start a new instance`

### Step 8: Final Setup

1. **Click OK** to save the task
2. **Enter your Windows password** when prompted
3. **Verify the task appears** in the Task Scheduler Library

---

## 🧪 Testing the Setup

### Test Manual Trigger

1. **In Task Scheduler**, find your task
2. **Right-click** → `Run`
3. **Check the log**: `D:\GitHub\NeuralTrader\logs\automated_runs.log`

### Test Scheduled Trigger

1. **Change the trigger time** to 2 minutes from now
2. **Wait for the task to run**
3. **Verify the log file** for execution
4. **Change back to 9:45 AM** when confirmed working

---

## 🛡️ Safety Features

### Kill Switch

Create a file named `STOP.txt` in the NeuralTrader root directory to immediately halt all trading:

```cmd
# To stop trading
echo STOP > D:\GitHub\NeuralTrader\STOP.txt

# To resume trading
del D:\GitHub\NeuralTrader\STOP.txt
```

### Log Monitoring

Monitor these log files:

- **Main Log**: `D:\GitHub\NeuralTrader\logs\orchestrator.log`
- **Automated Runs**: `D:\GitHub\NeuralTrader\logs\automated_runs.log`
- **Trade Ledger**: `D:\GitHub\NeuralTrader\reports\live_trade_log.csv`

---

## 🔍 Troubleshooting

### Common Issues

1. **Task doesn't run**:
   - Check Windows Event Viewer for errors
   - Verify batch file path is correct
   - Ensure Python is in system PATH

2. **API errors**:
   - Verify environment variables are set
   - Check Alpaca API status
   - Review log files for specific errors

3. **Permission issues**:
   - Run Task Scheduler as Administrator
   - Check file permissions on NeuralTrader directory
   - Ensure batch file has execute permissions

### Debug Mode

To run in debug mode with detailed logging:

```cmd
python main_orchestrator.py --force --debug
```

---

## 📊 Monitoring

### Daily Checklist

1. **Check automated_runs.log** for successful execution
2. **Verify trades** in live_trade_log.csv
3. **Review portfolio** in Alpaca paper trading account
4. **Monitor for errors** in orchestrator.log

### Weekly Review

1. **Analyze trade performance**
2. **Check risk management compliance**
3. **Verify system health**
4. **Update any configurations**

---

## 🚨 Emergency Procedures

### Immediate Stop

1. **Create STOP.txt file** in root directory
2. **Disable the task** in Task Scheduler
3. **Review any open positions** in Alpaca

### System Recovery

1. **Identify the error** from log files
2. **Fix the issue** (API keys, permissions, etc.)
3. **Remove STOP.txt file**
4. **Re-enable the task**
5. **Test with manual run**

---

## 📈 Success Indicators

Your automated trading is working correctly when:

- ✅ Task runs daily at 9:45 AM without errors
- ✅ Log files show successful execution
- ✅ Trade ledger records all activities
- ✅ Risk management rules are enforced
- ✅ No API errors or permission issues

---

## 🎯 Next Steps

Once automated trading is running successfully:

1. **Monitor performance** for 1-2 weeks
2. **Analyze trade execution quality**
3. **Verify risk management compliance**
4. **Prepare for Phase 7: Live Trading**

---

*Last Updated: February 2, 2026*  
*Status: Ready for Automated Deployment*
