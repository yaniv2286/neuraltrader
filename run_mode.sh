#!/bin/bash
# Helper invoked by cron to run one NeuralTrader mode under a file lock.
# Keeps long-running AI scans from overlapping while still allowing the
# scheduler to queue the next invocation safely.

set -e

MODE="$1"
if [ -z "$MODE" ]; then
    echo "[RUN_MODE] Error: no mode supplied. Usage: run_mode.sh <fetch|trade|report|paper|auto|backtest|simulation>" >&2
    exit 1
fi

mkdir -p /app/logs

# Inherit environment variables injected by the entrypoint.
if [ -f /etc/profile.d/neuraltrader-env.sh ]; then
    source /etc/profile.d/neuraltrader-env.sh
fi

cd /app

case "$MODE" in
    fetch|trade|report|paper|auto|saturday_retrain|backtest|simulation)
        exec python main_orchestrator_ist.py --mode="$MODE" \
            >> "/app/logs/cron_${MODE}.log" 2>&1
        ;;
    *)
        echo "[RUN_MODE] Unknown mode: $MODE" >&2
        exit 1
        ;;
esac
