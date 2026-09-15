@echo off
setlocal EnableDelayedExpansion

:: NeuralTrader standalone Windows deploy script
:: ---------------------------------------------
:: Run this batch file from the NeuralTrader repo root to package, upload,
:: and restart the NeuralTrader container on the VPS.
::
:: Required environment:
::   SERVER        - VPS SSH target, e.g. root@207.154.218.23
::   SSH_KEY_PATH  - Absolute path to your SSH private key for the VPS
::
:: Accepted ways to pass the key:
::   1. As the first argument: .\deploy_neuraltrader.bat C:\Users\Yaniv\.ssh\id_ed25519
::   2. As the SSH_KEY_PATH environment variable
::   3. As the legacy SSH_KEY environment variable
::   4. Defaults to C:\Users\<username>\.ssh\id_ed25519 if nothing is provided
::
:: Example usage:
::   .\deploy_neuraltrader.bat C:\Users\Yaniv\.ssh\id_ed25519

set "SERVER=root@207.154.218.23"
set "PROJECT=%~dp0"
set "ARCHIVE=%TEMP%\neuraltrader-deploy-%RANDOM%.tar.gz"

:: 1. Resolve SSH key path: argument > SSH_KEY_PATH > SSH_KEY > default
if not "%~1"=="" (
    set "SSH_KEY_PATH=%~1"
) else if not defined SSH_KEY_PATH (
    if defined SSH_KEY (
        set "SSH_KEY_PATH=%SSH_KEY%"
    ) else (
        set "SSH_KEY_PATH=C:\Users\%USERNAME%\.ssh\id_ed25519"
    )
)

echo [INFO] Using SSH key: %SSH_KEY_PATH%
if not exist "%SSH_KEY_PATH%" (
    echo [ERROR] SSH key not found: %SSH_KEY_PATH%
    echo [INFO] Pass the key as the first argument, or set SSH_KEY_PATH / SSH_KEY.
    pause
    exit /b 1
)
set "SSH_OPTS=-i "%SSH_KEY_PATH%""

:: Ensure we are in the NeuralTrader repo
if not exist "%PROJECT%\main_orchestrator_ist.py" (
    echo [ERROR] main_orchestrator_ist.py not found in %PROJECT%
    pause
    exit /b 1
)

:: 2. Quick syntax validation before packaging
echo [INFO] Running py_compile sanity check...
python -m py_compile "%PROJECT%\main_orchestrator_ist.py" "%PROJECT%\scripts\run_full_backtest.py" "%PROJECT%\scripts\run_simulation.py" "%PROJECT%\core\utils\notifier.py" "%PROJECT%\core\utils\telegram_notifier.py"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python syntax check failed. Fix before deploying.
    pause
    exit /b %ERRORLEVEL%
)

:: 3. Confirm SSH connectivity
echo [INFO] Testing SSH connectivity to %SERVER%...
ssh %SSH_OPTS% %SERVER% "echo [OK] SSH connectivity verified" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Cannot SSH to %SERVER% with key %SSH_KEY_PATH%
    echo [INFO] Check that the key is authorized on the VPS and the SSH service is reachable.
    pause
    exit /b 1
)

:: 4. Upload .env (always overwrite with local source of truth)
echo [INFO] Uploading NeuralTrader .env...
scp %SSH_OPTS% "%PROJECT%\.env" %SERVER%:/root/neuraltrader/.env
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload NeuralTrader .env
    pause
    exit /b %ERRORLEVEL%
)

:: 5. Seed market data if not present on VPS
echo [INFO] Checking market data seed on the VPS...
ssh %SSH_OPTS% %SERVER% "test -f /root/neuraltrader/data/raw/AAPL.parquet" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] No market data found on VPS. Building data/raw seed archive locally...
    set "DATA_ARCHIVE=%TEMP%\neuraltrader-data-%RANDOM%.tar.gz"
    tar -czf "!DATA_ARCHIVE!" -C "%PROJECT:~0,-1%" data\raw
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to build data/raw archive
        pause
        exit /b !ERRORLEVEL!
    )
    echo [INFO] Uploading market data seed. This may take several minutes...
    scp %SSH_OPTS% "!DATA_ARCHIVE!" %SERVER%:/root/neuraltrader-data.tar.gz
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to upload data/raw seed
        pause
        exit /b !ERRORLEVEL!
    )
    echo [INFO] Extracting market data seed on VPS...
    ssh %SSH_OPTS% %SERVER% "mkdir -p /root/neuraltrader && cd /root/neuraltrader && tar -xzf /root/neuraltrader-data.tar.gz && rm -f /root/neuraltrader-data.tar.gz"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to extract data/raw seed on VPS
        pause
        exit /b !ERRORLEVEL!
    )
)

:: 6. Build archive excluding local state and secrets
tar -czf "%ARCHIVE%" -C "%PROJECT:~0,-1%"^
 --exclude=".git"^
 --exclude="data"^
 --exclude="logs"^
 --exclude="reports"^
 --exclude="__pycache__"^
 --exclude=".pytest_cache"^
 --exclude="venv"^
 --exclude=".venv"^
 --exclude="*.db"^
 --exclude="*.sqlite3"^
 --exclude=".env"^
 --exclude=".env.*"^
 .

:: 7. Upload
echo [INFO] Uploading NeuralTrader archive to %SERVER%...
scp %SSH_OPTS% "%ARCHIVE%" %SERVER%:/root/neuraltrader-deploy.tar.gz
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] SCP failed
    pause
    exit /b %ERRORLEVEL%
)

:: 8. Deploy on the VPS
echo [INFO] Deploying NeuralTrader on %SERVER%...
ssh %SSH_OPTS% %SERVER% "mkdir -p /root/neuraltrader && cd /root/neuraltrader && rm -rf core scripts models docs temp *.py *.toml *.txt *.sh *.yml *.md && tar -xzf /root/neuraltrader-deploy.tar.gz && rm -f /root/neuraltrader-deploy.tar.gz && docker network inspect family-agents-net >/dev/null 2>&1 || docker network create family-agents-net && docker compose -f docker-compose.yml -p neuraltrader down && docker compose -f docker-compose.yml -p neuraltrader up -d --build && docker compose -p neuraltrader ps"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Remote deployment failed
    pause
    exit /b %ERRORLEVEL%
)

echo [OK] NeuralTrader deployed successfully.
pause
