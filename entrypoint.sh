#!/bin/bash
# NeuralTrader container entrypoint
# ---------------------------------
# Default behaviour: start the 24/7 cron-driven trading loop.
# Override behaviour: pass any command (e.g. `python scripts/run_full_backtest.py`)
# and it will execute that instead, making backtests / ad-hoc runs easy.

set -e

# Ensure the container uses the configured timezone for cron scheduling.
if [ -n "$TZ" ]; then
    ln -sf "/usr/share/zoneinfo/$TZ" /etc/localtime
    echo "$TZ" > /etc/timezone
fi

# Make runtime state directories available even if volumes are not mounted.
mkdir -p /app/data/raw /app/data/cache/backtest /app/data/cache/tiingo /app/data/fred_cache
mkdir -p /app/logs /app/reports

# If /app/models is an empty volume mount, seed it from the image's bootstrap copy
# so the production ensemble is always available on first start.
if [ ! -f /app/models/xgboost_model.pkl ] && [ -d /app/models_bootstrap ]; then
    echo "[ENTRYPOINT] Seeding /app/models from bootstrap copy..."
    cp -a /app/models_bootstrap/* /app/models/
fi

# Export the current container environment into a source-able shell profile so
# cron jobs (which start with a minimal environment) inherit secrets and config.
# Values are quoted to survive spaces/special characters.
mkdir -p /etc/profile.d
# Start with a fresh file, then append each variable quoted for shell safety.
> /etc/profile.d/neuraltrader-env.sh
printenv | while IFS= read -r line; do
    [ -z "$line" ] && continue
    printf 'export %q\n' "$line" >> /etc/profile.d/neuraltrader-env.sh
done
chmod 0644 /etc/profile.d/neuraltrader-env.sh

# ---------------------------------------------------------------------------
# Override mode: the user passed a command -> run it directly.
# Examples:
#   docker compose run --rm neuraltrader python scripts/run_full_backtest.py --start 2020-01-01
#   docker compose run --rm neuraltrader python scripts/run_simulation.py --cash 250000
#   docker compose run --rm neuraltrader python main_orchestrator_ist.py --mode=fetch
# ---------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
    echo "[ENTRYPOINT] Override mode: executing supplied command: $*"
    exec "$@"
fi

# ---------------------------------------------------------------------------
# Default mode: start the scheduled live trading loop.
# The crontab is copied by the Dockerfile to /etc/cron.d/neuraltrader.
# Each scheduled job uses flock to prevent overlapping runs.
# ---------------------------------------------------------------------------
echo "[ENTRYPOINT] Running startup validator..."
python /app/scripts/startup_validator.py

echo "[ENTRYPOINT] Probing IBKR/TWS connectivity (non-blocking)..."
python /app/scripts/ibkr_startup_check.py || true

echo "[ENTRYPOINT] Starting NeuralTrader 24/7 scheduled trading loop..."
echo "[ENTRYPOINT] Cron schedule:"
cat /etc/cron.d/neuraltrader
cron -f
