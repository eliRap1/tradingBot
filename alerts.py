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

    def send_regime_change(self, old_regime: str, new_regime: str, description: str):
        """Notify when market regime changes (bull/bear/chop transition)."""
        if not self.enabled:
            return
        emoji = {"bull": "🟢", "bear": "🔴", "chop": "🟡"}.get(new_regime, "📊")
        msg = (
            f"{emoji} *REGIME CHANGE*\n"
            f"`{old_regime.upper()} → {new_regime.upper()}`\n"
            f"{description}"
        )
        self._send(msg)

    _ETF_NAMES = {
        "XLK": "Tech/Semi/Cloud", "XLE": "Energy",       "XLF": "Financials",
        "XLV": "Healthcare",      "XLY": "Consumer Disc", "XLP": "Consumer Staples",
        "XLI": "Industrials",     "XLRE": "REITs",        "XLU": "Utilities",
        "crypto": "Crypto",
    }

    def send_sector_regime_change(self, etf_key: str, old_regime: str,
                                   new_regime: str, description: str = ""):
        """Discord alert when a sector ETF regime transitions."""
        if not self.enabled:
            return
        emoji = {"bull": "🟢", "bear": "🔴", "chop": "🟡"}.get(new_regime, "📊")
        name = self._ETF_NAMES.get(etf_key, etf_key)
        msg = (
            f"{emoji} *SECTOR CHANGE: {etf_key} ({name})*\n"
            f"`{old_regime.upper()} -> {new_regime.upper()}`\n"
            f"{description}"
        )
        self._send(msg)

    def send_hmm_caution(self, confidence: float, hmm_state: str,
                         ema_state: str, size_mult: float):
        """Notify when HMM disagrees with EMA and position sizes are capped."""
        if not self.enabled:
            return
        msg = (
            f"⚠️ *HMM CAUTION*\n"
            f"EMA: `{ema_state.upper()}` | HMM: `{hmm_state.upper()} @ {confidence:.0%}`\n"
            f"Position sizes capped at `{size_mult:.0%}` — elevated volatility detected"
        )
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

    def __init__(self, tracker, broker=None, coordinator=None):
        self.tracker = tracker
        self.broker = broker
        self.coordinator = coordinator
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

            tracker = self.tracker
            broker = self.broker
            coordinator = self.coordinator

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            backoff = 5

            while True:
                intents = discord.Intents.default()
                intents.message_content = True
                client = discord.Client(intents=intents)

                @client.event
                async def on_ready():
                    log.info(f"Discord bot connected as {client.user}")
                    log.info(f"Discord bot intents: message_content={client.intents.message_content}")

                @client.event
                async def on_message(message):
                    if message.author == client.user:
                        return

                    safe_content = message.content.encode("ascii", errors="replace").decode("ascii")
                    log.debug(f"Discord message from {message.author}: '{safe_content}'")
                    if message.content.strip().lower() in ("!stat", "!stats", "!status"):
                        # Get equity first so APR uses real starting equity
                        eq = 100_000.0
                        equity_str = ""
                        positions_str = ""
                        if broker:
                            try:
                                eq = broker.get_equity()
                                equity_str = f"\n💰 **Equity:** `${eq:,.2f}`"
                            except Exception as e:
                                log.warning(f"!stat: get_equity failed: {e}")
                                equity_str = "\n💰 **Equity:** `(unavailable)`"
                            try:
                                positions = broker.get_positions()
                                positions_str = f"\n📌 **Open positions:** `{len(positions)}`"
                            except Exception as e:
                                log.warning(f"!stat: get_positions failed: {e}")
                                positions_str = ""

                        stats = tracker.get_stats(starting_equity=eq)
                        if not stats:
                            await message.channel.send("📊 No trades recorded yet.")
                            return

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

                    elif message.content.strip().lower() == "!help":
                        msg = (
                            "📖 **Bot Commands**\n"
                            "`!stat` / `!stats` — full performance stats\n"
                            "`!positions` — open positions with live P&L\n"
                            "`!regime` — current market regime + HMM state\n"
                            "`!top` — top 5 pending/confirmed signals\n"
                            "`!risk` — drawdown, daily P&L, exposure\n"
                            "`!pause` — pause new trade entries\n"
                            "`!resume` — resume trading"
                        )
                        await message.channel.send(msg)

                    elif message.content.strip().lower() == "!positions":
                        if not broker:
                            await message.channel.send("❌ Broker not connected.")
                            return
                        try:
                            positions = broker.get_positions()
                            if not positions:
                                await message.channel.send("📭 No open positions.")
                                return
                            eq = broker.get_equity()
                            msg = f"📌 **Open Positions** ({len(positions)})\n"
                            total_unreal = 0.0
                            for p in positions:
                                side = getattr(p, "side", "long")
                                qty = float(getattr(p, "qty", 0))
                                entry = float(getattr(p, "avg_price", getattr(p, "avg_entry_price", 0)))
                                unreal = float(getattr(p, "unrealized_pl", 0))
                                qty_abs = abs(qty) or 1
                                unreal_pct = (unreal / (entry * qty_abs) * 100) if entry else 0.0
                                total_unreal += unreal
                                e = "🟢" if unreal >= 0 else "🔴"
                                msg += (
                                    f"{e} `{p.symbol}` {side.upper()} {qty:.4g} "
                                    f"@ ${entry:.2f} | "
                                    f"P&L: `${unreal:+,.2f}` (`{unreal_pct:+.1f}%`)\n"
                                )
                            sign = "+" if total_unreal >= 0 else ""
                            msg += f"\n**Total unrealized:** `${sign}{total_unreal:,.2f}`"
                            msg += f"\n**Equity:** `${eq:,.2f}`"
                        except Exception as e:
                            msg = f"❌ Error fetching positions: `{e}`"
                        await message.channel.send(msg)

                    elif message.content.strip().lower() == "!regime":
                        if not coordinator or not hasattr(coordinator, "regime"):
                            await message.channel.send("❌ Regime data unavailable.")
                            return
                        try:
                            r = coordinator.regime.get_regime()
                            paused = getattr(coordinator, "_trading_paused", False)
                            hmm_line = ""
                            if r.get("hmm_regime"):
                                hmm_line = (
                                    f"\nHMM: `{r['hmm_regime'].upper()} "
                                    f"@ {r['hmm_confidence']:.0%}`"
                                )
                            msg = (
                                f"📊 **Market Regime**\n"
                                f"Regime: `{r['regime'].upper()}`\n"
                                f"SPY trend: `{r['spy_trend'].upper()}` | "
                                f"RSI: `{r['spy_rsi']:.0f}`\n"
                                f"Breadth: `{r['breadth_pct']:.0f}%` above 50 EMA"
                                f"{hmm_line}\n"
                                f"Allow longs: `{'YES' if r['allow_longs'] else 'NO'}`\n"
                                f"Size mult: `{r['size_multiplier']:.0%}`\n"
                                f"ATR: `{r.get('atr_regime', 'normal').upper()}`\n"
                                f"Trading: `{'⏸ PAUSED' if paused else '▶ ACTIVE'}`\n"
                                f"Detail: _{r['description']}_"
                            )
                        except Exception as e:
                            msg = f"❌ Error fetching regime: `{e}`"
                        await message.channel.send(msg)

                    elif message.content.strip().lower() == "!top":
                        if not coordinator or not hasattr(coordinator, "watchers"):
                            await message.channel.send("❌ Watcher data unavailable.")
                            return
                        try:
                            watchers = list(coordinator.watchers.values())
                            active = [
                                w for w in watchers
                                if w.state.status in ("signal", "pending", "analyzing")
                                and w.state.score != 0
                            ]
                            active.sort(key=lambda w: abs(w.state.score), reverse=True)
                            top = active[:5]
                            if not top:
                                await message.channel.send("🔍 No active signals right now.")
                                return
                            msg = f"🔍 **Top Signals** ({len(active)} active watchers)\n"
                            for w in top:
                                direction = "LONG" if w.state.score > 0 else "SHORT"
                                status_emoji = "✅" if w.state.status == "signal" else "⏳"
                                msg += (
                                    f"{status_emoji} `{w.symbol}` {direction} "
                                    f"score=`{w.state.score:+.3f}` "
                                    f"conf=`{w.state.num_agreeing}` "
                                    f"regime=`{w.state.regime}` "
                                    f"[{w.state.status}]\n"
                                )
                        except Exception as e:
                            msg = f"❌ Error fetching signals: `{e}`"
                        await message.channel.send(msg)

                    elif message.content.strip().lower() == "!risk":
                        try:
                            eq = 100_000.0
                            peak = eq
                            if broker:
                                eq = broker.get_equity()
                            if coordinator and hasattr(coordinator, "risk"):
                                peak = coordinator.risk.peak_equity
                            dd_pct = (peak - eq) / peak if peak > 0 else 0
                            today_str = datetime.now().strftime("%Y-%m-%d")
                            today_trades = [t for t in tracker.trades
                                            if t.get("closed_at", "").startswith(today_str)]
                            daily_pnl = sum(t["pnl"] for t in today_trades)
                            max_dd = coordinator.config["risk"]["max_drawdown_pct"] if coordinator else 0.10
                            daily_limit = coordinator.config["risk"].get("daily_loss_limit_pct", 0.025) * eq if coordinator else 0
                            positions = broker.get_positions() if broker else []
                            max_pos = coordinator.config["signals"].get("max_positions", 10) if coordinator else 10
                            msg = (
                                f"⚖️ **Risk Dashboard**\n"
                                f"Equity: `${eq:,.2f}` | Peak: `${peak:,.2f}`\n"
                                f"Drawdown: `{dd_pct:.1%}` / max `{max_dd:.0%}`\n"
                                f"Daily P&L: `${daily_pnl:+,.2f}` | "
                                f"Daily limit: `${daily_limit:,.0f}`\n"
                                f"Positions: `{len(positions)}/{max_pos}`\n"
                                f"Daily trades: `{len(today_trades)}`"
                            )
                        except Exception as e:
                            msg = f"❌ Error fetching risk data: `{e}`"
                        await message.channel.send(msg)

                    elif message.content.strip().lower() == "!pause":
                        if coordinator:
                            coordinator._trading_paused = True
                            await message.channel.send("⏸ **Trading PAUSED** — no new entries until `!resume`")
                            log.info("Trading paused via Discord !pause command")
                        else:
                            await message.channel.send("❌ Coordinator not connected.")

                    elif message.content.strip().lower() == "!resume":
                        if coordinator:
                            coordinator._trading_paused = False
                            await message.channel.send("▶ **Trading RESUMED** — new entries enabled")
                            log.info("Trading resumed via Discord !resume command")
                        else:
                            await message.channel.send("❌ Coordinator not connected.")

                try:
                    loop.run_until_complete(client.start(self.token))
                    break  # clean shutdown (e.g. client.close() called)
                except Exception as e:
                    log.warning(f"Discord bot disconnected: {e} — retrying in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    # Fresh client + handlers created at top of while loop

        except Exception as e:
            log.error(f"Discord bot thread fatal error: {e}", exc_info=True)
