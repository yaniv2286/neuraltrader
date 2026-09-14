"""
NeuralTrader Family Agent Platform Monitoring / Control Adapter
=================================================================
A small, signed FastAPI adapter that exposes read-only monitoring endpoints
and a restricted set of paper/data-only control commands for integration with
the family-agent-platform stack.

Security:
- All requests (including health) must carry an HMAC-SHA256 signature in the
  X-Agent-Signature header together with a Unix timestamp in X-Agent-Timestamp.
- The shared secret is read from the AGENT_SHARED_SECRET environment variable.
- Requests outside the configured replay window are rejected.
- Control commands are an explicit allow-list (fetch / paper / report); the
  live ``trade`` mode is never accepted.  Commands are executed as a curated
  argument list (no shell), and the caller must pass ``confirm: true``.

Output:
- Every response follows the generic envelope:
  {"agent_id", "version", "timestamp", "data", "error"}

Location: core/control_api.py (adapter, not trading/ML/risk logic).
"""

from __future__ import annotations

import csv
import glob
import hmac
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
SYSTEM_MANIFEST = PROJECT_ROOT / "system_manifest.json"

AGENT_ID = os.environ.get("AGENT_ID", "neuraltrader")
API_VERSION = "v1"
SHARED_SECRET = os.environ.get(
    "AGENT_SHARED_SECRET", os.environ.get("AGENT_NEURALTRADER_SECRET", "")
).strip()
REPLAY_WINDOW_SECONDS = int(os.environ.get("AGENT_REPLAY_WINDOW", "300"))
CONTROL_ALLOWED_MODES = {"fetch", "paper", "report"}
CONTROL_DEFAULT_LOG = LOGS_DIR / "control_api.log"

# Process start time used for a lightweight uptime metric.
_PROCESS_START = time.time()

# ASCII-only logging as required by repository conventions.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("NeuralTrader_ControlAPI")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuralTrader Family Agent Adapter",
    version=API_VERSION,
    description="Signed monitoring and control adapter for NeuralTrader.",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ControlCommandRequest(BaseModel):
    mode: str = Field(..., description="One of the allowed control modes.")
    confirm: bool = Field(False, description="Must be true to actually execute the command.")
    dry_run: bool = Field(False, description="If true, only return a preview without running.")

    @field_validator("mode")
    @classmethod
    def mode_must_be_allowed(cls, v: str) -> str:
        if v not in CONTROL_ALLOWED_MODES:
            raise ValueError(f"mode must be one of {sorted(CONTROL_ALLOWED_MODES)}")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _envelope(
    data: Any = None,
    error: Optional[str] = None,
    status_code: int = 200,
) -> JSONResponse:
    body = {
        "agent_id": AGENT_ID,
        "version": API_VERSION,
        "timestamp": _now_iso(),
        "data": data,
        "error": error,
    }
    return JSONResponse(content=body, status_code=status_code)


def _sign_request(method: str, path: str, timestamp: str, body: bytes) -> str:
    """Return the hex HMAC-SHA256 for a request."""
    body_hash = hashlib.sha256(body or b"").hexdigest()
    base = f"{method.upper()}\n{path}\n{timestamp}\n{body_hash}"
    return hmac.new(SHARED_SECRET.encode("utf-8"), base.encode("utf-8"), hashlib.sha256).hexdigest()


async def _read_request_body(request: Request) -> bytes:
    try:
        return await request.body()
    except Exception as exc:
        logger.warning(f"[CONTROL] Unable to read request body: {exc}")
        return b""


async def require_hmac_auth(request: Request) -> None:
    """Dependency that enforces HMAC-signed requests within the replay window."""
    if not SHARED_SECRET:
        raise HTTPException(status_code=403, detail="HMAC authentication is not configured")

    timestamp_header = request.headers.get("X-Agent-Timestamp")
    signature_header = request.headers.get("X-Agent-Signature")

    if not timestamp_header or not signature_header:
        raise HTTPException(status_code=403, detail="Missing authentication headers")

    try:
        timestamp = int(timestamp_header)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid timestamp format")

    now = int(time.time())
    if abs(now - timestamp) > REPLAY_WINDOW_SECONDS:
        raise HTTPException(status_code=403, detail="Request outside replay window")

    body = await _read_request_body(request)
    expected = _sign_request(request.method, request.url.path, timestamp_header, body)
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=403, detail="Invalid signature")

    # Cache the body so routes can reuse it without re-reading the stream.
    request.state.body = body


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"[CONTROL] Failed to read {path}: {exc}")
        return default


def _latest_file(pattern: str) -> Optional[Path]:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None


def _read_csv_head(path: Path, rows: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= rows:
                    break
                out.append({k: v for k, v in row.items() if v is not None})
    except Exception as exc:
        logger.warning(f"[CONTROL] Failed to read CSV {path}: {exc}")
    return out


# Patterns that indicate a line contains sensitive material.
_SECRET_KEY_RE = re.compile(
    r"([A-Za-z0-9_]*(SECRET|TOKEN|PASSWORD|API_KEY|CREDENTIAL|PASSWD)|"
    r"AGENT_SHARED_SECRET|TIINGO_API_KEY|NOTIFIER_EMAIL|NOTIFIER_PASSWORD|"
    r"EMAIL_RECIPIENT|IBKR_ACCOUNT|TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID)\s*[=:]\s*",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _sanitize_log_line(line: str) -> str:
    """Sanitize a single log line: redact emails and secrets."""
    line = _EMAIL_RE.sub("[EMAIL]", line)
    if _SECRET_KEY_RE.search(line):
        line = _SECRET_KEY_RE.sub(r"\1[REDACTED] ", line)
    return line


def _recent_log_lines(n: int = 100) -> List[str]:
    log_file = _latest_file(str(LOGS_DIR / "automation_*.log"))
    if not log_file or not log_file.exists():
        return []
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        logger.warning(f"[CONTROL] Failed to read recent log {log_file}: {exc}")
        return []
    return [_sanitize_log_line(line.rstrip("\n")) for line in lines[-n:]]


def _portfolio_path() -> Path:
    return DATA_DIR / "portfolio_paper.json"


def _load_portfolio() -> Dict[str, Any]:
    data = _load_json(_portfolio_path(), default={})
    if not isinstance(data, dict):
        return {}
    return data


def _portfolio_summary() -> Dict[str, Any]:
    data = _load_portfolio()
    cash = float(data.get("cash", 0.0))
    positions = data.get("positions", {}) or {}
    peak = float(data.get("peak_portfolio_value", cash))
    total = cash + sum(
        float(pos.get("shares", 0)) * float(pos.get("current_price", 0.0))
        for pos in positions.values()
    )
    drawdown = 0.0
    if peak > 0:
        drawdown = (peak - total) / peak
    return {
        "mode": data.get("mode", "paper"),
        "cash": round(cash, 2),
        "total_value": round(total, 2),
        "peak_value": round(peak, 2),
        "drawdown_pct": round(drawdown * 100, 2),
        "positions_count": len(positions),
        "last_saved": data.get("last_saved"),
        "circuit_breaker_cooldown_until": data.get("circuit_breaker_cooldown_until"),
    }


def _position_list() -> List[Dict[str, Any]]:
    data = _load_portfolio()
    positions = data.get("positions", {}) or {}
    total = _portfolio_summary()["total_value"]
    out = []
    for ticker, pos in positions.items():
        shares = float(pos.get("shares", 0))
        price = float(pos.get("current_price", 0.0))
        cost = float(pos.get("cost_basis", 0.0))
        market_value = shares * price
        cost_basis_total = shares * cost
        pnl_usd = market_value - cost_basis_total
        pnl_pct = (pnl_usd / cost_basis_total * 100) if cost_basis_total else 0.0
        out.append(
            {
                "ticker": ticker,
                "shares": int(shares),
                "cost_basis": round(cost, 4),
                "current_price": round(price, 4),
                "market_value": round(market_value, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "weight_pct": round((market_value / total * 100), 2) if total else 0.0,
                "entry_date": pos.get("entry_date"),
                "peak_price": pos.get("peak_price"),
            }
        )
    return sorted(out, key=lambda x: x["market_value"], reverse=True)


def _orders_and_trades(limit: int = 50) -> List[Dict[str, Any]]:
    """Return active positions as open orders plus closed history entries."""
    data = _load_portfolio()
    positions = data.get("positions", {}) or {}
    history = data.get("history", []) or []
    out = []

    # Treat current positions as open BUY orders.
    for ticker, pos in positions.items():
        out.append(
            {
                "date": pos.get("entry_date"),
                "ticker": ticker,
                "action": "BUY",
                "quantity": int(pos.get("shares", 0)),
                "price": round(float(pos.get("cost_basis", 0.0)), 4),
                "status": "OPEN",
                "exit_reason": None,
            }
        )

    # Append closed trades from portfolio history.
    for trade in history[-limit:]:
        if not isinstance(trade, dict):
            continue
        out.append(
            {
                "date": trade.get("date") or trade.get("exit_date"),
                "ticker": trade.get("ticker"),
                "action": trade.get("action", "SELL"),
                "quantity": trade.get("quantity"),
                "price": trade.get("price") or trade.get("exit_price"),
                "status": "CLOSED",
                "exit_reason": trade.get("reason") or trade.get("exit_reason"),
            }
        )

    return out[-limit:]


def _performance_history(limit: int = 60) -> List[Dict[str, Any]]:
    equity_file = REPORTS_DIR / "equity_curve.csv"
    if equity_file.exists():
        try:
            with equity_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            return [
                {"date": row.get("date"), "value": float(row.get("value", 0))}
                for row in rows[-limit:]
            ]
        except Exception as exc:
            logger.warning(f"[CONTROL] Failed to read equity_curve.csv: {exc}")

    # Fallback: derive from portfolio history if present.
    data = _load_portfolio()
    history = data.get("history", []) or []
    return [
        {
            "date": trade.get("date") or trade.get("exit_date"),
            "value": trade.get("portfolio_value"),
        }
        for trade in history[-limit:]
        if isinstance(trade, dict) and trade.get("portfolio_value")
    ]


def _signals(limit: int = 20) -> List[Dict[str, Any]]:
    signal_file = _latest_file(str(REPORTS_DIR / "tradingview_signals_*.csv"))
    if not signal_file:
        return []
    return _read_csv_head(signal_file, rows=limit)


def _model_summary() -> Dict[str, Any]:
    manifest = _load_json(SYSTEM_MANIFEST, default={})
    ensemble = manifest.get("ensemble", {})
    models = manifest.get("models", {})
    files = models.get("files", {}) or {}
    present = {}
    for key, fname in files.items():
        fpath = MODELS_DIR / fname
        present[key] = {
            "filename": fname,
            "present": fpath.exists(),
            "size_bytes": fpath.stat().st_size if fpath.exists() else None,
            "mtime": datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc).isoformat()
            if fpath.exists()
            else None,
        }
    return {
        "weights": ensemble.get("weights", {}),
        "feature_count": ensemble.get("feature_count"),
        "feature_breakdown": ensemble.get("feature_breakdown", {}),
        "training_rows": models.get("training_rows"),
        "training_tickers": models.get("training_tickers"),
        "training_date": models.get("training_date"),
        "files": present,
    }


def _regime_summary() -> Dict[str, Any]:
    manifest = _load_json(SYSTEM_MANIFEST, default={})
    gate = manifest.get("regime_gate", {})
    thresholds = gate.get("thresholds", {})

    # Try to extract the most recent regime label from the logs.
    last_regime = None
    for line in _recent_log_lines(n=200):
        match = re.search(r"\[REGIME\]\s*(\w+)", line)
        if match:
            last_regime = match.group(1).upper()
            break

    return {
        "features": gate.get("features"),
        "sources": gate.get("sources", []),
        "thresholds": thresholds,
        "contrarian_override": gate.get("contrarian_override", {}),
        "last_observed_regime": last_regime,
    }


def _alert_lines(limit: int = 20) -> List[Dict[str, Any]]:
    alert_tags = ("[FAIL]", "[FATAL]", "[SHIELD]", "[EMERGENCY]")
    alerts = []
    for line in _recent_log_lines(n=1000):
        if any(tag in line for tag in alert_tags):
            alerts.append({"timestamp": _now_iso(), "message": line})
        if len(alerts) >= limit:
            break
    return alerts


def _workflow_summary() -> Dict[str, Any]:
    jobs = {}
    failure_markers = ("[ERROR]", "[CRITICAL]", "[FATAL]", "Traceback")
    success_markers = ("[OK]", "completed", "finished", "sent")
    for mode in ("fetch", "paper", "report"):
        path = LOGS_DIR / f"cron_{mode}.log"
        lines = []
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
            except Exception as exc:
                logger.error(f"[CONTROL] Failed to inspect workflow log {path}: {exc}", exc_info=True)
        failed = any(marker in line for line in lines for marker in failure_markers)
        succeeded = any(marker.lower() in line.lower() for line in lines for marker in success_markers)
        jobs[mode] = {
            "status": "failed" if failed else "success" if succeeded else "never_run",
            "last_run": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None,
            "last_error": next((line for line in reversed(lines) if any(marker in line for marker in failure_markers)), None),
        }
    data_files = list((DATA_DIR / "raw").glob("*.parquet"))
    newest = max(data_files, key=lambda path: path.stat().st_mtime) if data_files else None
    newest_at = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc) if newest else None
    age_hours = (datetime.now(timezone.utc) - newest_at).total_seconds() / 3600 if newest_at else None
    return {
        "jobs": jobs,
        "market_data": {
            "files": len(data_files),
            "newest_file": newest.name if newest else None,
            "newest_at": newest_at.isoformat() if newest_at else None,
            "age_hours": round(age_hours, 1) if age_hours is not None else None,
            "fresh": age_hours is not None and age_hours <= 48,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return _envelope(error=exc.detail, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return _envelope(error=f"Validation error: {exc}", status_code=422)


@app.exception_handler(Exception)
async def _generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"[CONTROL] Unhandled adapter exception: {exc}")
    return _envelope(error=f"Internal adapter error: {exc}", status_code=500)


@app.get("/v1/health")
async def health(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(
        data={
            "status": "ok",
            "agent_id": AGENT_ID,
            "version": API_VERSION,
            "uptime_seconds": round(time.time() - _PROCESS_START, 2),
            "auth": "hmac",
        }
    )


@app.get("/v1/capabilities")
async def capabilities(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(
        data={
            "endpoints": [
                "GET /v1/health",
                "GET /v1/capabilities",
                "GET /v1/status",
                "GET /v1/portfolio",
                "GET /v1/positions",
                "GET /v1/orders/trades",
                "GET /v1/performance/history",
                "GET /v1/signals",
                "GET /v1/model/summary",
                "GET /v1/regime/summary",
                "GET /v1/alerts",
                "GET /v1/workflows",
                "GET /v1/logs/recent",
                "POST /v1/control/command",
            ],
            "allowed_control_modes": sorted(CONTROL_ALLOWED_MODES),
            "signature_scheme": "HMAC-SHA256(method\\npath\\ntimestamp\\nsha256(body))",
            "required_headers": ["X-Agent-Timestamp", "X-Agent-Signature"],
        }
    )


@app.get("/v1/status")
async def status(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    summary = _portfolio_summary()
    return _envelope(
        data={
            "agent_id": AGENT_ID,
            "mode": summary.get("mode", "paper"),
            "portfolio_value": summary.get("total_value"),
            "cash": summary.get("cash"),
            "positions_count": summary.get("positions_count"),
            "drawdown_pct": summary.get("drawdown_pct"),
            "last_saved": summary.get("last_saved"),
            "circuit_breaker_cooldown_until": summary.get("circuit_breaker_cooldown_until"),
            "uptime_seconds": round(time.time() - _PROCESS_START, 2),
            "active_alerts": len(_alert_lines(limit=100)),
        }
    )


@app.get("/v1/portfolio")
async def portfolio(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_portfolio_summary())


@app.get("/v1/positions")
async def positions(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_position_list())


@app.get("/v1/orders/trades")
async def orders_trades(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_orders_and_trades(limit=50))


@app.get("/v1/performance/history")
async def performance_history(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_performance_history(limit=60))


@app.get("/v1/signals")
async def signals(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_signals(limit=20))


@app.get("/v1/model/summary")
async def model_summary(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_model_summary())


@app.get("/v1/regime/summary")
async def regime_summary(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_regime_summary())


@app.get("/v1/alerts")
async def alerts(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_alert_lines(limit=20))


@app.get("/v1/workflows")
async def workflows(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data=_workflow_summary())


@app.get("/v1/logs/recent")
async def logs_recent(_: None = Depends(require_hmac_auth)) -> JSONResponse:
    return _envelope(data={"source": str(_latest_file(str(LOGS_DIR / "automation_*.log"))), "lines": _recent_log_lines(n=100)})


@app.post("/v1/control/command")
async def control_command(
    request: Request,
    payload: ControlCommandRequest,
    _: None = Depends(require_hmac_auth),
) -> JSONResponse:
    mode = payload.mode
    preview = {
        "mode": mode,
        "confirm_required": True,
        "command_preview": [
            sys.executable,
            str(PROJECT_ROOT / "main_orchestrator_ist.py"),
            "--mode",
            mode,
        ],
        "note": "Pass confirm=true to execute. fetch/paper/report only; trade is blocked.",
    }

    if payload.dry_run or not payload.confirm:
        return _envelope(data=preview)

    # Explicit execution path: no shell, curated argument list, allow-list validated.
    job_id = str(uuid.uuid4())
    log_path = CONTROL_DEFAULT_LOG
    try:
        with log_path.open("a", encoding="utf-8") as log_fh:
            cmd = [sys.executable, str(PROJECT_ROOT / "main_orchestrator_ist.py"), "--mode", mode]
            log_fh.write(
                f"{_now_iso()} [CONTROL] job={job_id} mode={mode} cmd={' '.join(cmd)}\n"
            )
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
    except Exception as exc:
        logger.error(f"[CONTROL] Failed to start command for mode={mode}: {exc}")
        return _envelope(
            error=f"Failed to start command: {exc}",
            status_code=500,
        )

    return _envelope(
        data={
            "job_id": job_id,
            "mode": mode,
            "status": "started",
            "pid": proc.pid,
            "started_at": _now_iso(),
            "log": str(log_path),
            "note": "Command is running asynchronously. Check logs/control_api.log for output.",
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return _envelope(
        data={
            "message": "NeuralTrader Family Agent Adapter",
            "docs": "/v1/capabilities (requires HMAC auth)",
        }
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CONTROL_API_PORT", "8000"))
    host = os.environ.get("CONTROL_API_HOST", "0.0.0.0")
    logger.info(f"[CONTROL] Starting adapter on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
