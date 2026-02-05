@echo off
title NeuralTrader - AI PAPER TRADING

:: --- PATH CORRECTION ---
:: Move from /scripts/ to Project Root
cd /d "%~dp0.."
:: -----------------------

echo [LAUNCH] Initiating NeuralTrader Paper Trading Mode...
echo [INFO] Working Directory: %CD%

echo [INFO] Activating Neural Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [CRITICAL] Virtual Environment not found at %CD%\.venv
    pause
    exit /b 1
)

echo [INFO] Starting Orchestrator in PAPER mode...
python main_orchestrator_ist.py --mode=paper

if %ERRORLEVEL% NEQ 0 (
    echo [CRITICAL] System crashed! Check logs above.
    pause
)
pause
