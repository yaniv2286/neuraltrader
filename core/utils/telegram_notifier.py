"""
NeuralTrader Telegram Notifier
==============================

Lightweight Telegram alerting that mirrors the email notifier.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment and sends
execution confirmations, error alerts, and daily trade summaries via direct
POST request to the Telegram Bot API.

This module is intentionally dependency-light (only stdlib + requests) so it
can be imported from core/utils/notifier.py without bloating the runtime.
"""

import json
import os
import logging
import re
import requests
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_TEXT_LENGTH = 4000  # Leave headroom below Telegram's 4096 limit


class _HTMLStripper(HTMLParser):
    """Convert HTML email bodies to readable plain text for Telegram."""

    # Tags whose contents should never appear in the output.
    _SKIP_TAGS = frozenset({"style", "script", "head", "title", "meta", "link", "noscript"})
    # Tags that act as block separators in plain text.
    _BLOCK_TAGS = frozenset({"p", "div", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"})

    def __init__(self):
        super().__init__()
        self.text = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br":
            self.text.append("\n")
        elif tag in self._BLOCK_TAGS:
            if self.text and not self.text[-1].endswith("\n"):
                self.text.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in self._BLOCK_TAGS:
            if self.text and not self.text[-1].endswith("\n"):
                self.text.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        self.text.append(data)

    def get_text(self):
        text = "".join(self.text)
        # Collapse repeated blank lines while preserving intentional line breaks.
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def _strip_html(html_text: str) -> str:
    # Remove embedded CSS/JS blocks up-front so the parser never sees their text.
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    stripper = _HTMLStripper()
    try:
        stripper.feed(cleaned)
        return stripper.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", "", cleaned)


class TelegramNotifier:
    """
    Telegram alerting wrapper.

    If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are missing the class becomes a
    silent no-op so email alerting continues to work unchanged.
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token) and bool(self.chat_id)

        if self.enabled:
            logger.info("[TELEGRAM] Notifier enabled")
        else:
            logger.info("[TELEGRAM] Notifier disabled (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set)")

    def _post(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """Send a plain-text message to the configured chat."""
        if not self.enabled:
            return False

        url = TELEGRAM_API_URL.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": text[:MAX_TEXT_LENGTH],
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info("[TELEGRAM] Message sent successfully")
            return True
        except Exception as e:
            logger.error(f"[TELEGRAM] Failed to send message: {type(e).__name__}: {e}")
            return False

    def send_message(self, text: str) -> bool:
        """Send a generic Telegram message."""
        return self._post(text)

    def send_execution_confirmation(self, mode: str, status: str, details: str = "") -> bool:
        """Notify that a trading execution cycle completed."""
        header = f"[OK] NeuralTrader {mode.upper()} Execution Confirmation"
        body = f"Status: {status}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}"
        if details:
            body += f"\n\n{details}"
        return self._post(f"{header}\n\n{body}")

    def send_error_alert(self, mode: str, error_summary: str, traceback_text: str = "") -> bool:
        """Notify that NeuralTrader encountered a crash/error."""
        header = f"[CRASH] NeuralTrader {mode.upper()} FAILED"
        body = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}\nError: {error_summary}"
        if traceback_text:
            snippet = traceback_text[-1800:] if len(traceback_text) > 1800 else traceback_text
            body += f"\n\nTraceback snippet:\n{snippet}"
        return self._post(f"{header}\n\n{body}")

    def send_daily_summary(
        self,
        portfolio_value: float,
        cash: float,
        active_positions: int,
        total_trades: int,
        daily_pnl: float = 0.0,
        status: str = "OK",
    ) -> bool:
        """Send a compact daily P&L / position summary."""
        header = f"[DAILY] NeuralTrader Summary - {datetime.now().strftime('%Y-%m-%d')}"
        body = (
            f"Status: {status}\n"
            f"Portfolio Value: ${portfolio_value:,.2f}\n"
            f"Cash: ${cash:,.2f}\n"
            f"Active Positions: {active_positions}\n"
            f"Total Trades: {total_trades}\n"
            f"Daily P&L: ${daily_pnl:,.2f}"
        )
        return self._post(f"{header}\n\n{body}")

    def send_run_summary(self, summary: dict) -> bool:
        """Send one compact, informative summary at the end of a run.

        This replaces the old pattern of mirroring every email to Telegram,
        which produced duplicate, stripped-HTML noise.
        """
        if not self.enabled:
            return False

        # Deduplicate accidental rapid re-runs of the same mode+status.
        mode = str(summary.get("mode", "UNKNOWN")).upper()
        status = summary.get("status", "UNKNOWN")
        portfolio_value = float(summary.get("portfolio_value", 0) or 0)
        total_pnl = float(summary.get("total_pnl", 0) or 0)
        timestamp = summary.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Round to the nearest $100 for deduplication so tiny market wiggles
        # do not cause repeated messages.
        sig = f"{mode}:{status}:{round(portfolio_value / 100):.0f}:{round(total_pnl / 100):.0f}"
        last_path = Path("logs/last_telegram.json")
        now = datetime.now()
        try:
            if last_path.exists():
                last = json.loads(last_path.read_text(encoding="utf-8"))
                if (last.get("signature") == sig and
                        (now - datetime.fromisoformat(last["sent_at"])).total_seconds() < 300):
                    logger.info("[TELEGRAM] Skipping duplicate summary within 5-minute window")
                    return False
        except Exception:
            logger.exception("[TELEGRAM] Could not read last_telegram dedup file")

        if status == "OK":
            header = f"[OK] NeuralTrader {mode} completed - {timestamp}"
        elif status == "WARN":
            header = f"[WARN] NeuralTrader {mode} completed with issues - {timestamp}"
        else:
            header = f"[CRASH] NeuralTrader {mode} FAILED - {timestamp}"

        lines = [
            header,
            f"",
            f"Portfolio: ${portfolio_value:,.2f}",
            f"Cash: ${float(summary.get('cash', 0) or 0):,.2f}",
            f"P&L: ${total_pnl:,.2f}",
            f"Active Positions: {int(summary.get('active_positions', 0) or 0)}",
            f"Buy Signals: {int(summary.get('buy_signals', 0) or 0)}",
            f"Sell Signals: {int(summary.get('sell_signals', 0) or 0)}",
        ]

        email_sent = bool(summary.get("email_sent"))
        if email_sent:
            lines.append("Email: sent")
        else:
            lines.append("Email: failed (see log file)")

        exception_text = summary.get("exception_text") or ""
        if exception_text:
            # Show the first exception line, not a full traceback dump.
            first_line = exception_text.strip().splitlines()[0]
            if len(first_line) > 200:
                first_line = first_line[:200]
            lines.append(f"Error: {first_line}")

        text = "\n".join(lines)

        sent = self._post(text)
        if sent and last_path.parent.exists():
            try:
                last_path.write_text(json.dumps({"signature": sig, "sent_at": now.isoformat()}), encoding="utf-8")
            except Exception:
                logger.exception("[TELEGRAM] Could not write last_telegram dedup file")
        return sent

    def send_html_email_fallback(self, subject: str, html_body: str) -> bool:
        """
        Strip tags from an HTML email and send a short Telegram summary.
        Kept for backwards compatibility; new code should use send_run_summary.
        """
        text = _strip_html(html_body)
        return self._post(f"{subject}\n\n{text[:MAX_TEXT_LENGTH]}")
