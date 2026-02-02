@echo off
REM ===============================================================
REM NeuralTrader Task Scheduler Runner
REM ===============================================================
REM This script activates the virtual environment and runs NeuralTrader
REM Designed for Windows Task Scheduler integration
REM
REM Usage:
REM   run_neural.bat fetch    - Data fetch mode (16:45 IST)
REM   run_neural.bat trade    - Trading mode (market hours)
REM   run_neural.bat report   - Report mode (23:15 IST)
REM ===============================================================

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%

REM Log file for debugging
set LOG_FILE=%PROJECT_ROOT%logs\automation.log

REM Create logs directory if it doesn't exist
if not exist "%PROJECT_ROOT%logs" mkdir "%PROJECT_ROOT%logs"

REM Check if mode parameter is provided
if "%1"=="" (
    echo [ERROR] No mode specified
    echo Usage: %0 [fetch^|trade^|report^|auto]
    echo.
    echo Modes:
    echo   fetch   - Data fetch mode ^(16:45 IST^)
    echo   trade   - Trading mode ^(market hours^)
    echo   report  - Report mode ^(23:15 IST^)
    echo   auto    - Auto mode with scheduling
    goto :error_exit
)

REM Log script start
echo [%date% %time%] === NeuralTrader Task Scheduler Started === >> "%LOG_FILE%"
echo [%date% %time%] Script: %0 >> "%LOG_FILE%"
echo [%date% %time%] Mode: %1 >> "%LOG_FILE%"
echo [%date% %time%] Project Root: %PROJECT_ROOT% >> "%LOG_FILE%"
echo [%date% %time%] Working Directory: %CD% >> "%LOG_FILE%"

echo [%date% %time%] === NeuralTrader Task Scheduler Started ===
echo [%date% %time%] Script: %0
echo [%date% %time%] Mode: %1
echo [%date% %time%] Project Root: %PROJECT_ROOT%
echo [%date% %time%] Working Directory: %CD%

REM Change to project directory
cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
    echo [%date% %time%] [ERROR] Failed to change to project directory: %PROJECT_ROOT% >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] Failed to change to project directory: %PROJECT_ROOT%
    goto :error_exit
)

echo [%date% %time%] Changed to project directory: %CD% >> "%LOG_FILE%"
echo [%date% %time%] Changed to project directory: %CD%

REM Check if virtual environment exists
if not exist "%PROJECT_ROOT%.venv\Scripts\activate.bat" (
    echo [%date% %time%] [ERROR] Virtual environment not found: %PROJECT_ROOT%.venv\Scripts\activate.bat >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] Virtual environment not found: %PROJECT_ROOT%.venv\Scripts\activate.bat
    echo [%date% %time%] Please create virtual environment first:
    echo [%date% %time%] python -m venv .venv
    echo [%date% %time%] .venv\Scripts\activate
    echo [%date% %time%] pip install -r requirements_trading.txt
    goto :error_exit
)

REM Activate virtual environment
echo [%date% %time%] Activating virtual environment... >> "%LOG_FILE%"
echo [%date% %time%] Activating virtual environment...
call "%PROJECT_ROOT%.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [%date% %time%] [ERROR] Failed to activate virtual environment >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] Failed to activate virtual environment
    goto :error_exit
)

echo [%date% %time%] Virtual environment activated >> "%LOG_FILE%"
echo [%date% %time%] Virtual environment activated

REM Check if Python script exists
if not exist "%PROJECT_ROOT%main_orchestrator_ist.py" (
    echo [%date% %time%] [ERROR] Python script not found: %PROJECT_ROOT%main_orchestrator_ist.py >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] Python script not found: %PROJECT_ROOT%main_orchestrator_ist.py
    goto :error_exit
)

REM Run NeuralTrader with specified mode
echo [%date% %time%] Starting NeuralTrader with mode: %1 >> "%LOG_FILE%"
echo [%date% %time%] Starting NeuralTrader with mode: %1

python main_orchestrator_ist.py --mode=%1

REM Check Python exit code
if errorlevel 1 (
    echo [%date% %time%] [ERROR] NeuralTrader failed with exit code: %errorlevel% >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] NeuralTrader failed with exit code: %errorlevel%
    goto :error_exit
) else (
    echo [%date% %time%] [SUCCESS] NeuralTrader completed successfully >> "%LOG_FILE%"
    echo [%date% %time%] [SUCCESS] NeuralTrader completed successfully
)

REM Log script end
echo [%date% %time%] === NeuralTrader Task Scheduler Completed === >> "%LOG_FILE%"
echo [%date% %time%] === NeuralTrader Task Scheduler Completed ===
goto :success_exit

:error_exit
echo [%date% %time%] [FATAL] Task Scheduler script failed >> "%LOG_FILE%"
echo [%date% %time%] [FATAL] Task Scheduler script failed
exit /b 1

:success_exit
echo [%date% %time%] [INFO] Task Scheduler script completed successfully >> "%LOG_FILE%"
echo [%date% %time%] [INFO] Task Scheduler script completed successfully
exit /b 0
