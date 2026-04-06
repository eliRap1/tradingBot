"""
Alert system — Discord webhook notifications + bot commands.

Setup:
  1. In Discord: Server Settings -> Integrations -> Webhooks -> New Webhook
  2. Copy the webhook URL
  3. Add to .env: DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy

  For !stat command:
  4. Go to https://discord.com/developers/applications -> New Application -> Bot
  5. Enable MESSAGE CONTENT INTENT under Bot -> Privileged Gateway Intents
  6. Copy the bot token
  7. Add to .env: DISCORD_BOT_TOKEN=your_bot_token_here
  8. Invite bot to server with: OAuth2 -> URL Generator -> bot -> Send Messages + Read Messages
"""

import os
import time
import threading
import requests
from datetime import datetime, timezone, timedelta
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

    def send_market_closed(self, next_open, stats: dict = None,
                           equity: float = 0, positions: int = 0,
                           daily_pnl: float = 0, daily_trades: int = 0):
        """Send alert when market closes: next open (Israel time) + full stats."""
        if not self.enabled:
            return

        try:
            # Israel is UTC+3 (IDT, daylight saving most of the year)
            israel_tz = timezone(timedelta(hours=3))

            if hasattr(next_open, 'astimezone'):
                israel_time = next_open.astimezone(israel_tz)
            elif hasattr(next_open, 'timestamp'):
                from datetime import datetime as dt
                israel_time = dt.fromtimestamp(next_open.timestamp(), tz=israel_tz)
            else:
                israel_time = None

            if israel_time:
                # Calculate hours until open
                now_utc = datetime.now(timezone.utc)
                if hasattr(next_open, 'timestamp'):
                    delta_sec = next_open.timestamp() - now_utc.timestamp()
                else:
                    delta_sec = 0
                h = int(delta_sec // 3600)
                m = int((delta_sec % 3600) // 60)

                time_str = israel_time.strftime("%A %H:%M")
                msg = (
                    f"🌙 *MARKET CLOSED*\n"
                    f"Next open: `{time_str}` Israel time\n"
                    f"Opens in: `{h}h {m}m`\n"
                    f"Crypto watchers still active 24/7"
                )
            else:
                msg = (
                    f"🌙 *MARKET CLOSED*\n"
                    f"Next open: `{next_open}`\n"
                    f"Crypto watchers still active 24/7"
                )
        except Exception:
            msg = (
                f"🌙 *MARKET CLOSED*\n"
                f"Next open: `{next_open}`\n"
                f"Crypto watchers still active 24/7"
            )

        # Add today's stats
        if daily_trades > 0 or daily_pnl != 0:
            msg += (
                f"\n\n📅 *TODAY*\n"
                f"Trades: `{daily_trades}` | P&L: `${daily_pnl:+,.2f}`"
            )
            if equity > 0:
                msg += f"\nEquity: `${equity:,.2f}` | Open positions: `{positions}`"

        # Add overall stats
        if stats and stats.get("total_trades", 0) > 0:
            msg += (
                f"\n\n📊 *ALL-TIME STATS*\n"
                f"Trades: `{stats['total_trades']}` | "
                f"Win%: `{stats['win_pct']}%`\n"
                f"Total P&L: `${stats['total_pnl']:+,.2f}`\n"
                f"Avg P&L: `${stats['avg_pnl']:+,.2f}`/trade\n"
                f"Sharpe: `{stats['sharpe_ratio']}` | "
                f"PF: `{stats['profit_factor']}`\n"
                f"Max DD: `${stats['max_drawdown']:,.2f}` | "
                f"Calmar: `{stats['calmar_ratio']}`"
            )
            if stats.get("r_expectancy") is not None:
                msg += f"\nR-Exp: `{stats['r_expectancy']:+.2f}R`"
            if stats.get("apr") is not None:
                msg += f"\nAPR: `{stats['apr']:+.1f}%`"
            msg += (
                f"\nBest: `${stats['largest_win']:+,.2f}` | "
                f"Worst: `${stats['largest_loss']:+,.2f}`\n"
                f"Streaks: `{stats['max_consecutive_wins']}W` / "
                f"`{stats['max_consecutive_losses']}L`"
            )

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


class DiscordBot:
    """
    Discord bot that listens for !stat command and replies with full stats.

    Runs in a background thread so it doesn't block the trading bot.
    Requires DISCORD_BOT_TOKEN in .env.
    """

    def __init__(self, tracker, broker=None):
        self.tracker = tracker
        self.broker = broker
        self._thread = None
        self.token = os.getenv("DISCORD_BOT_TOKEN", "")

        if not self.token:
            log.info("DISCORD_BOT_TOKEN not set — !stat command disabled")

    def start(self):
        """Start the Discord bot in a background thread."""
        if not self.token:
            return

        self._thread = threading.Thread(
            target=self._run, name="discord-bot", daemon=True
        )
        self._thread.start()
        log.info("Discord bot started — listening for !stat")

    def _run(self):
        """Run the Discord bot (blocking, runs in thread)."""
        try:
            import discord
            import asyncio

            intents = discord.Intents.default()
            intents.message_content = True
            client = discord.Client(intents=intents)

            tracker = self.tracker
            broker = self.broker

            @client.event
            async def on_ready():
                log.info(f"Discord bot connected as {client.user}")

            @client.event
            async def on_message(message):
                if message.author == client.user:
                    return

                if message.content.strip().lower() in ("!stat", "!stats", "!status"):
                    stats = tracker.get_stats()
                    if not stats:
                        await message.channel.send("📊 No trades recorded yet.")
                        return

                    # Get equity if broker available
                    equity_str = ""
                    positions_str = ""
                    if broker:
                        try:
                            eq = broker.get_equity()
                            equity_str = f"\n💰 **Equity:** `${eq:,.2f}`"
                            positions = broker.get_positions()
                            positions_str = f"\n📌 **Open positions:** `{len(positions)}`"
                        except Exception:
                            pass

                    # Today's trades
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    today_trades = [t for t in tracker.trades
                                    if t.get("closed_at", "").startswith(today_str)]
                    daily_pnl = sum(t["pnl"] for t in today_trades)

                    msg = (
                        f"📅 **TODAY**\n"
                        f"Trades: `{len(today_trades)}` | P&L: `${daily_pnl:+,.2f}`"
                        f"{equity_str}{positions_str}"
                        f"\n\n📊 **ALL-TIME STATS**\n"
                        f"Trades: `{stats['total_trades']}` | "
                        f"Win%: `{stats['win_pct']}%`\n"
                        f"Total P&L: `${stats['total_pnl']:+,.2f}`\n"
                        f"Avg P&L: `${stats['avg_pnl']:+,.2f}`/trade\n"
                        f"Sharpe: `{stats['sharpe_ratio']}` | "
                        f"PF: `{stats['profit_factor']}`\n"
                        f"Max DD: `${stats['max_drawdown']:,.2f}` | "
                        f"Calmar: `{stats['calmar_ratio']}`"
                    )
                    if stats.get("r_expectancy") is not None:
                        msg += f"\nR-Exp: `{stats['r_expectancy']:+.2f}R`"
                    if stats.get("apr") is not None:
                        msg += f"\nAPR: `{stats['apr']:+.1f}%`"
                    msg += (
                        f"\nBest: `${stats['largest_win']:+,.2f}` | "
                        f"Worst: `${stats['largest_loss']:+,.2f}`\n"
                        f"Streaks: `{stats['max_consecutive_wins']}W` / "
                        f"`{stats['max_consecutive_losses']}L`"
                    )

                    # Top 5 recent trades
                    recent = tracker.trades[-5:]
                    if recent:
                        msg += "\n\n📝 **Last 5 trades:**\n"
                        for t in reversed(recent):
                            emoji = "✅" if t["pnl"] >= 0 else "❌"
                            msg += (
                                f"{emoji} `{t['symbol']}` "
                                f"${t['pnl']:+,.2f} "
                                f"({t.get('reason', '')})\n"
                            )

                    await message.channel.send(msg)

            # Run the bot
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(client.start(self.token))
        except Exception as e:
            log.error(f"Discord bot error: {e}")
