"""Signed Docker healthcheck for the NeuralTrader control adapter.

Uses AGENT_SHARED_SECRET (or AGENT_NEURALTRADER_SECRET) to sign a GET request
to /v1/health and exits 0 on HTTP 200.
"""

import hashlib
import hmac
import os
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("CONTROL_HOST", "127.0.0.1")
PORT = os.environ.get("CONTROL_PORT", "8000")
SECRET = os.environ.get("AGENT_SHARED_SECRET", os.environ.get("AGENT_NEURALTRADER_SECRET", "")).strip()
PATH = "/v1/health"
URL = f"http://{HOST}:{PORT}{PATH}"


def _sign(method: str, path: str, timestamp: str, body: bytes) -> str:
    body_hash = hashlib.sha256(body or b"").hexdigest()
    base = f"{method.upper()}\n{path}\n{timestamp}\n{body_hash}"
    return hmac.new(SECRET.encode("utf-8"), base.encode("utf-8"), hashlib.sha256).hexdigest()


def main() -> int:
    if not SECRET:
        print("[HEALTHCHECK] No shared secret configured; falling back to socket probe.")
        import socket
        try:
            socket.create_connection((HOST, int(PORT)), timeout=5)
        except Exception as exc:
            print(f"[HEALTHCHECK] Socket probe failed: {exc}")
            return 1
        return 0

    timestamp = str(int(time.time()))
    signature = _sign("GET", PATH, timestamp, b"")
    req = urllib.request.Request(
        URL,
        headers={
            "X-Agent-Timestamp": timestamp,
            "X-Agent-Signature": signature,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("[HEALTHCHECK] OK")
                return 0
            print(f"[HEALTHCHECK] Unexpected status: {resp.status}")
            return 1
    except urllib.error.HTTPError as exc:
        print(f"[HEALTHCHECK] HTTP error: {exc.code}")
        return 1
    except Exception as exc:
        print(f"[HEALTHCHECK] Request failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
