"""
Alert system — Discord webhook notifications for trades, exits, drawdown, daily summaries.

Setup:
  1. In Discord: Server Settings -> Integrations -> Webhooks -> New Webhook
  2. Copy the webhook URL
  3. Add to .env: DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy
"""

import os
import time
import threading
import requests
from datetime import datetime
from utils import setup_logger

log = setup_logger("alerts")


class AlertManager:
    def __init__(self, config: dict):
        self.enabled = config.get("alerts", {}).get("enabled", False)
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.notify_trades = config.get("alerts", {}).get("notify_trades", True)
        self.notify_drawdown = config.get("alerts", {}).get("notify_drawdown", True)
        self.notify_daily = config.get("alerts", {}).get("notify_daily_summary", True)

        # Rate limiter: max 20 messages per minute
        self._msg_timestamps: list[float] = []
        self._lock = threading.Lock()

        if self.enabled and self.webhook_url:
            log.info("Discord alerts enabled")
        elif self.enabled:
            log.warning("Alerts enabled but DISCORD_WEBHOOK_URL not set in .env")
            self.enabled = False

    def send_trade_alert(self, side: str, qty: int, symbol: str,
                         entry_price: float, stop_loss: float,
                         take_profit: float, regime: str = "",
                         confluence: int = 0):
        """Notify on new trade entry."""
        if not self.enabled or not self.notify_trades:
            return

        rr = abs(take_profit - entry_price) / abs(entry_price - stop_loss) \
            if abs(entry_price - stop_loss) > 0 else 0
        risk = abs(entry_price - stop_loss) * qty

        emoji = "🟢" if side == "buy" else "🔴"
        msg = (
            f"{emoji} *NEW TRADE*\n"
            f"`{side.upper()} {qty} {symbol}`\n"
            f"Entry: `${entry_price:.2f}`\n"
            f"SL: `${stop_loss:.2f}` | TP: `${take_profit:.2f}`\n"
            f"R:R: `{rr:.1f}` | Risk: `${risk:.2f}`\n"
            f"Regime: `{regime}` | Confluence: `{confluence}`"
        )
        self._send(msg)

    def send_exit_alert(self, symbol: str, side: str, entry_price: float,
                        exit_price: float, pnl: float, pnl_pct: float,
                        reason: str):
        """Notify on position exit."""
        if not self.enabled or not self.notify_trades:
            return

        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} *CLOSED: {symbol}*\n"
            f"Side: `{side}` | Reason: `{reason}`\n"
            f"Entry: `${entry_price:.2f}` -> Exit: `${exit_price:.2f}`\n"
            f"P&L: `${pnl:+,.2f}` (`{pnl_pct:+.2%}`)"
        )
        self._send(msg)

    def send_drawdown_warning(self, drawdown_pct: float, peak: float,
                              current: float):
        """Warning when drawdown approaches limit."""
        if not self.enabled or not self.notify_drawdown:
            return

        msg = (
            f"⚠️ *DRAWDOWN WARNING*\n"
            f"Current: `{drawdown_pct:.1%}`\n"
            f"Peak: `${peak:,.2f}` -> Now: `${current:,.2f}`\n"
            f"Loss: `${peak - current:,.2f}`"
        )
        self._send(msg)

    def send_drawdown_halt(self, drawdown_pct: float, peak: float,
                           current: float):
        """Critical: trading halted due to max drawdown."""
        if not self.enabled:
            return

        msg = (
            f"🛑 *TRADING HALTED — MAX DRAWDOWN*\n"
            f"Drawdown: `{drawdown_pct:.1%}`\n"
            f"Peak: `${peak:,.2f}` -> Now: `${current:,.2f}`\n"
            f"All trading stopped. Manual review required."
        )
        self._send(msg)

    def send_daily_summary(self, stats: dict, equity: float,
                           positions: int):
        """End-of-day performance summary."""
        if not self.enabled or not self.notify_daily:
            return

        if not stats:
            msg = f"📊 *DAILY SUMMARY*\nNo trades recorded yet.\nEquity: `${equity:,.2f}`"
        else:
            msg = (
                f"📊 *DAILY SUMMARY — {datetime.now():%Y-%m-%d}*\n"
                f"Equity: `${equity:,.2f}` | Positions: `{positions}`\n"
                f"───────────────\n"
                f"Trades: `{stats['total_trades']}` | "
                f"Win%: `{stats['win_pct']}%`\n"
                f"Total P&L: `${stats['total_pnl']:+,.2f}`\n"
                f"Sharpe: `{stats['sharpe_ratio']}` | "
                f"Profit Factor: `{stats['profit_factor']}`\n"
                f"Expectancy: `${stats['expectancy']:+,.2f}`/trade\n"
                f"Max DD: `${stats['max_drawdown']:,.2f}`"
            )
            if stats.get("r_expectancy") is not None:
                msg += f"\nR-Expectancy: `{stats['r_expectancy']:+.2f}R`"

        self._send(msg)

    def send_error(self, error_msg: str):
        """Critical error notification."""
        if not self.enabled:
            return
        msg = f"🔥 *BOT ERROR*\n`{error_msg[:500]}`"
        self._send(msg)

    def _send(self, message: str):
        """Send message via Discord webhook with rate limiting."""
        if not self.enabled or not self.webhook_url:
            return

        with self._lock:
            now = time.time()
            self._msg_timestamps = [t for t in self._msg_timestamps if now - t < 60]
            if len(self._msg_timestamps) >= 20:
                log.warning("Alert rate limit hit (20/min), skipping message")
                return
            self._msg_timestamps.append(now)

        # Convert Markdown bold (*text*) to Discord bold (**text**)
        discord_msg = message.replace("*", "**")
        # Convert Markdown code (`text`) — Discord uses same format, keep as-is

        try:
            resp = requests.post(self.webhook_url, json={
                "content": discord_msg,
                "username": "Trading Bot",
            }, timeout=5)

            if resp.status_code not in (200, 204):
                log.warning(f"Discord webhook error: {resp.status_code}")
        except Exception as e:
            # Alerts must NEVER crash the bot
            log.warning(f"Alert send failed: {e}")
