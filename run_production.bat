@echo off
REM NeuralTrader Production Runner (Windows Batch Version)
REM =====================================================
REM Executes fetch and trade pipeline in a loop with configurable delay
REM All output is captured to timestamped logs

setlocal enabledelayedexpansion

REM Configuration
set /A DELAY_MINUTES=%1
if "%DELAY_MINUTES%"=="" set DELAY_MINUTES=5
set /A MAX_LOOPS=%2
if "%MAX_LOOPS%"=="" set MAX_LOOPS=0

REM Get project root
set PROJECT_ROOT=%~dp0
set LOG_DIR=%PROJECT_ROOT%logs

REM Ensure logs directory exists
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Function to log with timestamp (simulated)
:log_message
echo [%time%] %~1
goto :eof

REM Main execution loop
set current_loop=0
set /A delay_seconds=DELAY_MINUTES * 60

echo === NeuralTrader Production Runner Started ===
echo Delay: %DELAY_MINUTES% minutes
echo Max Loops: %MAX_LOOPS% (0 = infinite)
echo Log Directory: %LOG_DIR%
echo ==========================================

REM Create status file
echo [%date% %time%] STARTED > "%LOG_DIR%\production_status.txt"

:loop
set /A current_loop+=1

echo --- Loop %current_loop% ---

REM Run fetch mode (always)
echo [%time%] Starting fetch mode...
cd /d "%PROJECT_ROOT%"
call run_neural_venv.bat fetch
if !errorlevel! equ 0 (
    echo [%time%] [SUCCESS] Fetch completed successfully
) else (
    echo [%time%] [ERROR] Fetch failed with exit code: !errorlevel!
)

REM Simple market hours check (weekdays, 10:00-16:00 EST)
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set day=%%a
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set hour=%%a
set /A hour=hour%%100

REM Check if weekend (simplified - assumes format doesn't contain "Sat" or "Sun")
echo %date% | findstr /i "Sat Sun" >nul
if !errorlevel! equ 0 (
    echo [%time%] Market closed (weekend)
) else (
    REM Check if market hours (10:00-16:00)
    if !hour! geq 10 if !hour! lss 16 (
        echo [%time%] Market is open - running trade mode
        
        echo [%time%] Starting trade mode...
        call run_neural_venv.bat trade
        if !errorlevel! equ 0 (
            echo [%time%] [SUCCESS] Trade completed successfully
        ) else (
            echo [%time%] [ERROR] Trade failed with exit code: !errorlevel!
        )
    ) else (
        echo [%time%] Market closed (off hours)
    )
)

REM Check if we should continue
if %MAX_LOOPS% neq 0 if %current_loop% geq %MAX_LOOPS% (
    echo === Maximum loops reached ===
    goto end
)

REM Wait before next iteration
echo [%time%] Waiting %DELAY_MINUTES% minutes before next iteration...
echo [%date% %time%] WAITING > "%LOG_DIR%\production_status.txt"

REM Simple delay using timeout
timeout /t %delay_seconds% /nobreak >nul

if %MAX_LOPS% equ 0 goto loop
if %current_loop% lss %MAX_LOOPS% goto loop

:end
echo === NeuralTrader Production Runner Completed ===
echo [%date% %time%] COMPLETED > "%LOG_DIR%\production_status.txt"

pause
