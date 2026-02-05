@echo off
title NeuralTrader - AUTOMATION CONFIG
echo [SETUP] Configuring NeuralTrader Autopilot...

:: --- STEP 1: CLEAN SLATE ---
echo [INFO] Removing old/conflicting tasks...
:: Remove the old name if it exists
schtasks /delete /tn "NeuralTrader_Phase7_Master" /f >nul 2>&1
:: Remove legacy tasks
schtasks /delete /tn "NeuralTrader_DataFetch" /f >nul 2>&1
schtasks /delete /tn "NeuralTrader_DailyReport" /f >nul 2>&1
schtasks /delete /tn "NeuralTrader_SaturdayRetrain" /f >nul 2>&1
echo [OK] Cleanup complete.

:: --- STEP 2: THE TRADER (DAILY) ---
:: Runs every day at 16:45 IST (US Market Open)
:: Name: NeuralTrader_PaperTrading_Live
schtasks /create /tn "NeuralTrader_PaperTrading_Live" /tr "%~dp0start_paper_trading.bat" /sc daily /st 16:45 /f
if %ERRORLEVEL% EQU 0 echo [OK] DAILY TRADING scheduled for 16:45 IST.

:: --- STEP 3: THE STUDENT (WEEKLY) ---
:: Runs every Saturday at 17:00 IST
schtasks /create /tn "NeuralTrader_Saturday_Train" /tr "%~dp0..\run_neural_venv.bat saturday_retrain" /sc weekly /d SAT /st 17:00 /f
if %ERRORLEVEL% EQU 0 echo [OK] WEEKLY RETRAINING scheduled for Saturday 17:00 IST.

echo.
echo [SUCCESS] System is now fully automated.
echo [1] Paper Trading: 16:45 IST (Daily)
echo [2] Model Training: 17:00 IST (Saturday)
pause
