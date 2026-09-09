#!/usr/bin/env python3
"""
Verification script for core/utils/telegram_notifier.py.

Validates that HTML email bodies mirrored to Telegram have CSS/style/script
blocks removed and that other Telegram messages stay plain-text safe.

Exit code 0 = all checks pass.
Exit code 1 = at least one check failed.
"""
import os
import sys
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.utils.telegram_notifier import _strip_html, TelegramNotifier


def assert_check(condition: bool, message: str) -> bool:
    if condition:
        print(f"[OK] {message}")
        return True
    print(f"[FAIL] {message}")
    return False


def main() -> int:
    failures = 0

    sample_html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background-color: white; }
        </style>
        <script>alert('xss');</script>
    </head>
    <body>
        <div class="header">
            <h2>NeuralTrader Daily Report</h2>
            <p><b>Mode:</b> PAPER</p>
        </div>
        <div class="section">
            <h3>Summary</h3>
            <p>Portfolio Value: $100,000.00</p>
            <p>Cash &lt; 5%</p>
        </div>
    </body>
    </html>
    """

    plain = _strip_html(sample_html)

    # CSS and JS content must not leak into plain text.
    failures += not assert_check("font-family" not in plain, "CSS style content removed")
    failures += not assert_check("background-color" not in plain, "CSS property content removed")
    failures += not assert_check("alert('xss')" not in plain, "Script content removed")
    # Visible text must be preserved.
    failures += not assert_check("NeuralTrader Daily Report" in plain, "Report title preserved")
    failures += not assert_check("Mode: PAPER" in plain, "Mode value preserved")
    failures += not assert_check("Portfolio Value: $100,000.00" in plain, "Portfolio value preserved")
    # HTML tag markup should be gone.
    failures += not assert_check(__import__("re").search(r"<[^>]+>", plain) is None, "No stray HTML tag markup in output")

    # Verify the email mirror path sends clean plain text without parse_mode.
    captured = []

    class CaptureNotifier(TelegramNotifier):
        def __init__(self):
            super().__init__()
            self.token = "dummy"
            self.chat_id = "12345"
            self.enabled = True

        def _post(self, text: str, parse_mode: Optional[str] = None) -> bool:
            captured.append((text, parse_mode))
            return True

    notifier = CaptureNotifier()
    notifier.send_html_email_fallback(
        subject="[OK] NeuralTrader PAPER completed - 2026-09-07 17:17:17",
        html_body=sample_html,
    )

    failures += not assert_check(len(captured) == 1, "One Telegram message captured")
    text, parse_mode = captured[0]
    failures += not assert_check(parse_mode is None, "Email mirror uses plain text mode")
    failures += not assert_check("[OK] NeuralTrader PAPER completed" in text, "Subject preserved")
    failures += not assert_check("font-family" not in text, "No CSS in sent message")
    failures += not assert_check("NeuralTrader Daily Report" in text, "Visible content in sent message")

    # Verify disabled notifier is a silent no-op.
    silent = TelegramNotifier()
    silent.token = ""
    silent.chat_id = ""
    silent.enabled = False
    failures += not assert_check(silent.send_message("test") is False, "Disabled notifier returns False")

    if failures:
        print(f"\n[FAIL] {failures} check(s) failed.")
        return 1

    print("\n[OK] All NeuralTrader Telegram notifier checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
