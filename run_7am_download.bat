@echo off
echo ========================================
echo 7:00 AM Auto Download Launcher
echo ========================================
echo.
echo This will run the download script at 7:00 AM
echo when your API limits renew.
echo.
echo Starting scheduler...
echo.

cd /d "D:\GitHub\NeuralTrader"

uv run python scripts/auto_download_scheduler.py

pause
