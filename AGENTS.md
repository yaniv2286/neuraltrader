# NeuralTrader Deployment Notes

## Containerised Execution

This project can run 24/7 on a VPS via Docker.

- Image build: `docker compose -p neuraltrader build`
- Start live trading loop: `docker compose -p neuraltrader up -d`
- View logs: `docker compose -p neuraltrader logs -f`
- One-off backtest:
  ```bash
  docker compose -p neuraltrader run --rm neuraltrader \
    python scripts/run_full_backtest.py --start 2020-01-01 --end 2025-12-31
  ```

## Deployment

The preferred entry point is the unified `deploy.bat` in the family-agent-platform
repository, which deploys both the family-agent stack and NeuralTrader:

```powershell
cd D:\GitHub\family-agent-platform
set SSH_KEY=C:\Users\Yaniv\.ssh\id_rsa
.\deploy.bat
```

A standalone `deploy_neuraltrader.bat` also exists in this repo for deploying
only NeuralTrader if needed.

## Telegram Alerting

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `/root/neuraltrader/.env`.
The email notifier will automatically mirror execution confirmations, error
alerts, and daily brief summaries to Telegram.

## State Persistence

All mutable state is kept in host-mounted directories under the compose working
folder (e.g. `/root/neuraltrader`):

- `./data`   -> Tiingo parquet files, portfolio JSON, training caches
- `./logs`   -> automation, cron, backtest and simulation logs
- `./reports`-> backtest_metrics.json, trade_log.csv, equity_curve.csv, etc.
- `./models` -> optional retrained model persistence (seeded from image on first run)

## Trading Mode

This deployment is configured for **paper trading only** (`--mode=paper`). The
scheduled execution job at 09:35 ET uses paper mode, so no live IBKR orders are
submitted. If you later want live trading, change `paper` to `trade` in `crontab`
and confirm IB Gateway is reachable.

## Required Environment Variables

Copy `.env.example` to `.env` and fill in real values before starting the
container. Critical variables: `TIINGO_API_KEY`, `IBKR_HOST`/`IBKR_PORT`,
`NOTIFIER_EMAIL`/`NOTIFIER_PASSWORD`, `EMAIL_RECIPIENT`.

## IB Gateway / TWS Connectivity

The default `IBKR_HOST=host.docker.internal` resolves to the VPS host inside the
container. Choose the setup that matches your infrastructure:

1. **IB Gateway on the VPS host** (simplest):
   - Set `IBKR_HOST=host.docker.internal` and `IBKR_PORT=7497` (paper) or `7496` (live).
   - Start IB Gateway on the VPS host and ensure it listens on `0.0.0.0:7497`.
   - The compose file already adds `extra_hosts` for Docker on Linux.

2. **TWS on your local home machine**:
   - A container on a remote VPS cannot reach your LAN directly.
   - Use an encrypted tunnel such as `ssh -R 7497:localhost:7497 root@207.154.218.23`
     or a WireGuard VPN, then set `IBKR_HOST=127.0.0.1` on the VPS/container.

3. **No IBKR socket (paper/data-only mode)**:
   - The orchestrator logs a warning and continues using the last saved portfolio
     state. This is fine for `--mode=paper`/`fetch`/`report` and backtests, but
     `trade` mode will not submit live orders without a reachable socket.

