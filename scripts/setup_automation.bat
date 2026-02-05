@echo off
title NeuralTrader - AUTOMATION UPGRADE
echo [SETUP] Upgrading Windows Task Scheduler for Phase 7...

:: --- STEP 1: RETIRE OLD TASKS ---
echo [INFO] Deleting legacy tasks...
schtasks /delete /tn "NeuralTrader_DataFetch" /f >nul 2>&1
schtasks /delete /tn "NeuralTrader_DailyReport" /f >nul 2>&1
schtasks /delete /tn "NeuralTrader_SaturdayRetrain" /f >nul 2>&1
echo [OK] Old tasks retired.

:: --- STEP 2: COMMISSION NEW ENGINE ---
:: Task 1: Phase 7 Master Run (Daily @ 16:45 IST - US Market Open)
:: This runs the fully integrated 'start_paper_trading.bat'
schtasks /create /tn "NeuralTrader_Phase7_Master" /tr "%~dp0start_paper_trading.bat" /sc daily /st 16:45 /f
if %ERRORLEVEL% EQU 0 echo [OK] Master Engine scheduled for 16:45 IST.

:: Task 2: Saturday Retrain (Saturday @ 17:00 IST)
schtasks /create /tn "NeuralTrader_Saturday_Train" /tr "%~dp0..\run_neural_venv.bat saturday_retrain" /sc weekly /d SAT /st 17:00 /f
if %ERRORLEVEL% EQU 0 echo [OK] Saturday Retraining scheduled for 17:00 IST.

echo.
echo [SUCCESS] Automation Upgrade Complete.
echo The system will now auto-launch 'start_paper_trading.bat' every day at 16:45 IST.
pause
