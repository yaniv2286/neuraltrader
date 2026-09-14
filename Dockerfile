# NeuralTrader Institutional Container
# -------------------------------------
# Optimised Python 3.11 slim image for the live trading loop, backtests, and
# periodic retraining. The image ships with source code + frozen models; runtime
# state (data, logs, reports, optional model retraining output) is mounted from
# the host/named volumes so nothing is lost across container restarts.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout/stderr to be unbuffered
# so logs appear immediately in docker logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Default TZ for market-relative scheduling; override in docker-compose.yml / .env
ENV TZ=America/New_York

WORKDIR /app

# Install system dependencies required by the ML stack and cron scheduler.
# - gcc/g++: compilation fallback for some wheels (xgboost / lightgbm historical builds)
# - libgomp1: OpenMP runtime used by xgboost/lightgbm
# - libsqlite3-0: SQLite support if any local DBs are used later
# - cron + util-linux: in-container scheduled execution and flock locking
# - tzdata: TZ database for cron
# - git: useful for pip installs from VCS if ever needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libgomp1 \
    libsqlite3-0 \
    cron \
    procps \
    util-linux \
    tzdata \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies. We use the Docker-optimised requirements file
# that removes the heavy torch/pytorch-forecasting stack.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy the full application tree (code, docs, scripts, models).
# .dockerignore excludes data/, logs/, reports/, .env, .git, venvs, etc.
COPY . .

# Preserve a pristine copy of the production models. If /app/models is mounted
# as an empty volume (e.g. to persist retrained models), the entrypoint will
# seed it from this bootstrap copy on first run.
RUN cp -a /app/models /app/models_bootstrap

# Copy the production crontab; cron requires strict ownership/permissions.
COPY crontab /etc/cron.d/neuraltrader

# Ensure runtime directories exist and are writable, and scripts are executable.
RUN mkdir -p /app/data/raw /app/data/cache /app/data/cache/backtest /app/data/cache/tiingo \
    /app/logs /app/reports /app/data/fred_cache && \
    chmod 0644 /etc/cron.d/neuraltrader && \
    chmod +x /app/entrypoint.sh /app/run_mode.sh && \
    sed -i 's/\r$//' /app/entrypoint.sh /app/run_mode.sh /etc/cron.d/neuraltrader

# Default entrypoint handles scheduled live trading OR one-off backtest commands.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
