"""Lightweight IBKR/TWS socket reachability probe for container startup.

This script is intentionally standalone (no heavy project imports) so it can
run before the orchestrator starts. It only checks that the configured
IBKR_HOST:IBKR_PORT is reachable; it does NOT perform a full IB login.
"""
import os
import socket
import sys


def main() -> int:
    host = os.environ.get("IBKR_HOST", "host.docker.internal")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    timeout = int(os.environ.get("IBKR_PROBE_TIMEOUT", "5"))

    print(f"[IBKR-PROBE] Probing {host}:{port} (timeout {timeout}s)...")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"[IBKR-PROBE] [OK] TWS/IB Gateway socket is reachable at {host}:{port}")
            return 0
    except (OSError, socket.timeout) as exc:
        print(f"[IBKR-PROBE] [WARN] Cannot reach TWS/IB Gateway at {host}:{port}: {exc}")
        print("[IBKR-PROBE] [INFO] Paper/data-only modes will continue using cached portfolio state.")
        print("[IBKR-PROBE] [INFO] For live trading, ensure TWS/IB Gateway is running and accepts connections.")
        return 0  # Non-blocking: do not prevent cron from starting.


if __name__ == "__main__":
    sys.exit(main())
