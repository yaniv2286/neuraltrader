@echo off
title NeuralTrader - AI PAPER TRADING
echo [LAUNCH] Initiating NeuralTrader Paper Trading Mode...
echo [INFO] Activating Neural Environment...
call .venv\Scripts\activate

echo [INFO] Starting Orchestrator in PAPER mode...
python main_orchestrator_ist.py --mode=paper

if %ERRORLEVEL% NEQ 0 (
    echo [CRITICAL] System crashed! Check logs.
    pause
)
pause
