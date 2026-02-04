@echo off
REM ===============================================================
REM NeuralTrader Task Scheduler Runner - Virtual Environment
REM ===============================================================
REM This script activates the virtual environment and runs NeuralTrader
REM Designed for Windows Task Scheduler integration
REM
REM Usage:
REM   run_neural_venv.bat fetch    - Data fetch mode (16:45 IST)
REM   run_neural_venv.bat trade    - Trading mode (market hours)
REM   run_neural_venv.bat report   - Report mode (23:15 IST)
REM ===============================================================

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%

REM Log file for debugging (with date to avoid file locking)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOG_FILE=%PROJECT_ROOT%logs\automation_%datetime:~0,8%.log

REM Create logs directory if it doesn't exist
if not exist "%PROJECT_ROOT%logs" mkdir "%PROJECT_ROOT%logs"

REM Check if mode parameter is provided
if "%1"=="" (
    echo [ERROR] No mode specified
    echo Usage: %0 [fetch^|trade^|report^|auto^|saturday_retrain]
    echo.
    echo Modes:
    echo   fetch           - Data fetch mode ^(16:45 IST^)
    echo   trade           - Trading mode ^(market hours^)
    echo   report          - Report mode ^(23:15 IST^)
    echo   auto            - Auto mode with scheduling
    echo   saturday_retrain - Saturday model retrain
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

REM Log environment variables for debugging
echo [%date% %time%] Environment Variables: >> "%LOG_FILE%"
echo [%date% %time%] PATH: %PATH% >> "%LOG_FILE%"
echo [%date% %time%] PYTHONPATH: %PYTHONPATH% >> "%LOG_FILE%"
echo [%date% %time%] VIRTUAL_ENV: %VIRTUAL_ENV% >> "%LOG_FILE%"

REM Check if virtual environment exists
if not exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    echo [%date% %time%] [ERROR] Virtual environment not found at %PROJECT_ROOT%\.venv >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] Please create virtual environment: python -m venv .venv >> "%LOG_FILE%"
    goto :error_exit
)

REM Check if Python exists in virtual environment
if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [%date% %time%] [ERROR] Python not found in virtual environment >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] Please check virtual environment installation >> "%LOG_FILE%"
    goto :error_exit
)

REM Check if main script exists
if not exist "%PROJECT_ROOT%\main_orchestrator_ist.py" (
    echo [%date% %time%] [ERROR] Main script not found: %PROJECT_ROOT%\main_orchestrator_ist.py >> "%LOG_FILE%"
    goto :error_exit
)

REM Log Python version
echo [%date% %time%] Checking Python version... >> "%LOG_FILE%"
"%PROJECT_ROOT%\.venv\Scripts\python.exe" --version >> "%LOG_FILE%" 2>&1

REM Check required libraries
echo [%date% %time%] Checking required libraries... >> "%LOG_FILE%"
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -c "import sys; print('Python version:', sys.version)" >> "%LOG_FILE%" 2>&1
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -c "import pandas; print('pandas: OK')" >> "%LOG_FILE%" 2>&1
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -c "import yfinance; print('yfinance: OK')" >> "%LOG_FILE%" 2>&1
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -c "import pytz; print('pytz: OK')" >> "%LOG_FILE%" 2>&1
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -c "import dotenv; print('dotenv: OK')" >> "%LOG_FILE%" 2>&1

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

REM Run NeuralTrader with error handling
echo [%date% %time%] Starting NeuralTrader with mode: %1 >> "%LOG_FILE%"
echo [%date% %time%] Starting NeuralTrader with mode: %1

REM Run with full error capture and UTF-8 encoding
chcp 65001 >nul 2>&1
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -X utf8 "%PROJECT_ROOT%\main_orchestrator_ist.py" --mode=%1 >> "%LOG_FILE%" 2>&1
set PYTHON_EXIT_CODE=%errorlevel%

if %PYTHON_EXIT_CODE% EQU 0 (
    echo [%date% %time%] [SUCCESS] NeuralTrader completed successfully >> "%LOG_FILE%"
    echo [%date% %time%] [SUCCESS] NeuralTrader completed successfully
) else (
    echo [%date% %time%] [ERROR] NeuralTrader failed with exit code: %PYTHON_EXIT_CODE% >> "%LOG_FILE%"
    echo [%date% %time%] [ERROR] NeuralTrader failed with exit code: %PYTHON_EXIT_CODE%
    echo [%date% %time%] [FATAL] Task Scheduler script failed >> "%LOG_FILE%"
    echo [%date% %time%] [FATAL] Task Scheduler script failed
    goto :error_exit
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
