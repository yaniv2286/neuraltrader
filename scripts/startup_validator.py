"""NeuralTrader startup validator.

Exits non-zero if critical configuration, secrets, data, or models are missing.
Warnings are logged but do not block startup.

Run manually:
    python scripts/startup_validator.py

Or call validate() from the entrypoint before starting cron.
"""

import logging
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("startup_validator")


@dataclass
class ValidationResult:
    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_MODEL_FILES = [
    "models/xgboost_model.pkl",
    "models/lightgbm_model.pkl",
    "models/rf_model.pkl",
    "models/regime_classifier.pkl",
    "models/feature_scaler.pkl",
    "models/feature_names.pkl",
]

CRITICAL_ENV = {
    "TIINGO_API_KEY": "Required to fetch market data.",
}

OPTIONAL_ENV = {
    "NOTIFIER_EMAIL": "Gmail sender for daily briefs.",
    "NOTIFIER_PASSWORD": "Gmail app password for daily briefs.",
    "EMAIL_RECIPIENT": "Destination email for daily briefs.",
    "TELEGRAM_BOT_TOKEN": "Telegram bot token for alerts.",
    "TELEGRAM_CHAT_ID": "Telegram chat ID for alerts.",
}


def _validate_python_version(result: ValidationResult) -> None:
    if sys.version_info < (3, 10):
        result.add_error(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old; Python 3.10+ is required.")


def _validate_env(result: ValidationResult) -> None:
    for key, description in CRITICAL_ENV.items():
        value = os.getenv(key)
        if not value or value.strip() == "" or "your" in value.lower():
            result.add_error(f"{key} is missing or looks like a placeholder. {description}")

    for key, description in OPTIONAL_ENV.items():
        value = os.getenv(key)
        if not value or value.strip() == "" or "your" in value.lower():
            result.add_warning(f"{key} is not configured. {description}")


def _validate_models(result: ValidationResult) -> None:
    for relative in REQUIRED_MODEL_FILES:
        path = PROJECT_ROOT / relative
        if not path.exists():
            result.add_error(f"Required model file not found: {path}")
        elif path.stat().st_size == 0:
            result.add_error(f"Required model file is empty: {path}")


def _validate_market_data(result: ValidationResult, freshness_hours: int = 48) -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    if not data_dir.exists():
        result.add_error(f"Market data directory does not exist: {data_dir}")
        return

    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        result.add_error(f"No *.parquet market data files found in {data_dir}")
        return

    expected_tickers: Set[str] = {
        "SPY",
        "VXX",
    }
    found = {p.stem.upper() for p in parquet_files}
    missing = expected_tickers - found
    if missing:
        result.add_warning(f"Expected benchmark tickers missing from data/raw: {sorted(missing)}")

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=freshness_hours)
    stale = [p for p in parquet_files if datetime.utcfromtimestamp(p.stat().st_mtime) < cutoff]
    if stale:
        names = [p.name for p in stale]
        result.add_warning(f"Market data files older than {freshness_hours}h: {names}")


def _validate_directories(result: ValidationResult) -> None:
    for relative in ["data", "logs", "reports"]:
        path = PROJECT_ROOT / relative
        if not path.exists():
            logger.info("Creating missing runtime directory: %s", path)
            path.mkdir(parents=True, exist_ok=True)


def validate(freshness_hours: int = 48) -> ValidationResult:
    """Run all startup checks and return a ValidationResult."""
    result = ValidationResult()
    _validate_python_version(result)
    _validate_env(result)
    _validate_models(result)
    _validate_market_data(result, freshness_hours=freshness_hours)
    _validate_directories(result)
    return result


def _print_report(result: ValidationResult) -> None:
    print("=" * 60)
    print("NeuralTrader Startup Validator")
    print("=" * 60)
    if result.warnings:
        print("\nWARNINGS (startup will continue):")
        for w in result.warnings:
            print(f"  - {w}")
    if result.errors:
        print("\nERRORS (startup blocked):")
        for e in result.errors:
            print(f"  - {e}")
    print("\n" + ("OK" if result.ok else "FAILED"))
    print("=" * 60)


def main() -> int:
    result = validate()
    _print_report(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
