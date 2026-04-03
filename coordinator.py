"""
Coordinator — manages all StockWatcher threads.

Responsibilities:
  - Spawn one watcher thread per stock in the universe
  - Collect signals from watchers
  - Enforce global risk limits (max positions, regime, drawdown)
  - Execute trades when watchers report confirmed signals
  - Manage existing positions (trailing stops, exits)
"""

import time
import sys
import threading
from datetime import datetime

from utils import load_config, setup_logger
from broker import Broker
from data import DataFetcher
from screener import Screener
from risk import RiskManager
from portfolio import PortfolioManager
from regime import RegimeFilter
from filters import SmartFilters
from watcher import StockWatcher, Action

log = setup_logger("coordinator")


class Coordinator:
    def __init__(self):
        log.info("=" * 60)
        log.info("TRADING BOT STARTING (Threaded Architecture)")
        log.info("=" * 60)

        self.config = load_config()
        self.broker = Broker(self.config)
        self.data = DataFetcher(self.broker)
        self.screener = Screener(self.config, self.data)
        self.risk = RiskManager(self.config)
        self.portfolio = PortfolioManager(self.config, self.broker)
        self.regime = RegimeFilter(self.data)
        self.filters = SmartFilters(tracker=self.portfolio.tracker)

        # Watcher threads: {symbol: StockWatcher}
        self.watchers: dict[str, StockWatcher] = {}
        self._lock = threading.Lock()

        # Verify connection
        account = self.broker.get_account()
        equity = float(account.equity)
        log.info(f"Account: equity=${equity:,.2f} "
                 f"cash=${float(account.cash):,.2f} "
                 f"buying_power=${float(account.buying_power):,.2f}")
        self.risk.peak_equity = max(self.risk.peak_equity, equity)
        self.risk.set_starting_equity(equity)

    def start_watchers(self):
        """Spawn a watcher thread for each stock in the universe."""
        universe = self.screener.get_universe()
        if not universe:
            # Fall back to config universe if screener can't fetch
            universe = self.config["screener"]["universe"]

        cycle_interval = self.config["schedule"]["cycle_interval_sec"]
        # Each watcher checks every cycle_interval seconds
        watcher_interval = max(60, cycle_interval)

        log.info(f"Starting {len(universe)} watcher threads "
                 f"(interval={watcher_interval}s each)")

        for sym in universe:
            if sym not in self.watchers:
                watcher = StockWatcher(
                    symbol=sym,
                    config=self.config,
                    data_fetcher=self.data,
                    interval=watcher_interval
                )
                self.watchers[sym] = watcher
                watcher.start()
                # Stagger thread starts to avoid API rate limits
                time.sleep(0.5)

        log.info(f"All {len(self.watchers)} watchers running")

    def stop_watchers(self):
        """Stop all watcher threads."""
        log.info("Stopping all watchers...")
        for sym, watcher in self.watchers.items():
            watcher.stop()
        self.watchers.clear()
        log.info("All watchers stopped")

    def run(self):
        """Main coordinator loop."""
        log.info("Bot is running. Press Ctrl+C to stop.")

        while True:
            try:
                clock = self.broker.get_clock()

                if not clock.is_open:
                    # Stop watchers while market is closed
                    if self.watchers:
                        self.stop_watchers()
                    next_open = clock.next_open
                    log.info(f"Market closed. Next open: {next_open}")
                    self._wait_until(next_open)

                    # Reset daily tracking
                    equity = self.broker.get_equity()
                    self.risk.set_starting_equity(equity)

                    delay = self.config["schedule"]["market_open_delay_min"]
                    log.info(f"Market opened. Waiting {delay}min...")
                    time.sleep(delay * 60)
                    continue

                # Check close buffer
                next_close = clock.next_close
                buffer = self.config["schedule"]["market_close_buffer_min"]
                if hasattr(next_close, 'timestamp'):
                    close_ts = next_close.timestamp()
                else:
                    close_ts = next_close.replace(tzinfo=None).timestamp()

                if time.time() > close_ts - (buffer * 60):
                    log.info("Too close to market close. Waiting for next day.")
                    if self.watchers:
                        self.stop_watchers()
                    self._wait_until(clock.next_open)
                    continue

                # Start watchers if not running
                if not self.watchers:
                    self.start_watchers()

                # Coordinator cycle: collect signals and execute
                self._coordinator_cycle()

                # Sleep between coordinator checks
                interval = self.config["schedule"]["cycle_interval_sec"]
                log.info(f"Coordinator cycle done. Next check in {interval}s")
                time.sleep(interval)

            except KeyboardInterrupt:
                log.info("Shutting down gracefully...")
                self.stop_watchers()
                break
            except Exception as e:
                log.error(f"Coordinator error: {e}", exc_info=True)
                time.sleep(60)

    def _coordinator_cycle(self):
        """
        Collect signals from all watchers, apply global filters, execute trades.
        """
        log.info("-" * 40)
        log.info(f"COORDINATOR CYCLE: {datetime.now():%H:%M:%S}")

        # 1. Check drawdown
        equity = self.broker.get_equity()
        if self.risk.check_drawdown(equity):
            log.critical("DRAWDOWN LIMIT HIT — halting all trading")
            self.stop_watchers()
            sys.exit(1)

        # 2. Market regime
        regime = self.regime.get_regime()
        log.info(f"Market: {regime['description']}")

        # 3. Manage existing positions
        positions = self.portfolio.get_current_positions()
        self.portfolio.log_portfolio_status(positions)
        held_symbols = list(positions.keys())

        if positions:
            prices = self.data.get_latest_prices(list(positions.keys()))
            to_close = self.portfolio.check_trailing_stops(positions, prices)
            if to_close:
                self.portfolio.execute_exits(to_close, positions)
                positions = self.portfolio.get_current_positions()
                held_symbols = list(positions.keys())

        # 4. Block new trades in bear regime
        if not regime["allow_longs"]:
            log.warning("BEAR REGIME — no new longs allowed")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # 5. Check position capacity
        max_pos = self.config["signals"]["max_positions"]
        if len(held_symbols) >= max_pos:
            log.info(f"At max positions ({max_pos}). Skipping new entries.")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # 6. Collect confirmed BUY signals from watchers
        buy_signals = []
        for sym, watcher in self.watchers.items():
            if sym in held_symbols:
                continue  # Already holding this stock
            if watcher.state.action == Action.BUY and watcher.state.confirmed:
                buy_signals.append(watcher)

        if not buy_signals:
            log.info("No confirmed signals from watchers")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # Sort by score (best signals first)
        buy_signals.sort(key=lambda w: w.state.score, reverse=True)

        log.info(f"Confirmed signals from: "
                 f"{[w.symbol for w in buy_signals]}")

        # 7. Apply global filters
        # Sector cap
        from filters import SECTOR_MAP, MAX_PER_SECTOR
        sector_count = {}
        for sym in held_symbols:
            sector = SECTOR_MAP.get(sym, "other")
            sector_count[sector] = sector_count.get(sector, 0) + 1

        # Gap filter + sector filter
        filtered = []
        for watcher in buy_signals:
            sym = watcher.symbol
            bars = watcher.get_bars()

            # Gap filter
            if bars is not None and len(bars) >= 2 and "open" in bars.columns:
                prev_close = float(bars["close"].iloc[-2])
                today_open = float(bars["open"].iloc[-1])
                gap_pct = abs(today_open - prev_close) / prev_close if prev_close > 0 else 0
                if gap_pct > 0.02:
                    log.info(f"GAP FILTER: Skipping {sym} (gapped {gap_pct:.1%})")
                    continue

            # Sector cap
            sector = SECTOR_MAP.get(sym, "other")
            if sector_count.get(sector, 0) >= MAX_PER_SECTOR:
                log.info(f"SECTOR CAP: Skipping {sym} (sector '{sector}' full)")
                continue

            filtered.append(watcher)
            sector_count[sector] = sector_count.get(sector, 0) + 1

        if not filtered:
            log.info("No signals passed global filters")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # 8. Size and execute orders
        slots_available = max_pos - len(held_symbols)
        cooldown_mult = self.filters.get_loss_cooldown_mult()
        size_mult = regime["size_multiplier"] * cooldown_mult

        for watcher in filtered[:slots_available]:
            sym = watcher.symbol
            bars = watcher.get_bars()
            if bars is None:
                continue

            price = self.data.get_latest_price(sym)
            if not price:
                continue

            orders = self.risk.size_orders(
                opportunities=[_make_opportunity(watcher)],
                bars={sym: bars},
                prices={sym: price},
                equity=equity,
                num_existing=len(held_symbols),
                regime_size_mult=size_mult
            )

            for order in orders:
                try:
                    self.broker.submit_bracket_order(
                        symbol=order.symbol,
                        qty=order.qty,
                        side=order.side,
                        take_profit=order.take_profit,
                        stop_loss=order.stop_loss
                    )
                    log.info(
                        f"ORDER: {order.side} {order.qty} {order.symbol} "
                        f"@ ~${order.entry_price:.2f} | "
                        f"SL=${order.stop_loss:.2f} TP=${order.take_profit:.2f} | "
                        f"regime={watcher.state.regime} "
                        f"confluence={watcher.state.num_agreeing}"
                    )
                    held_symbols.append(order.symbol)
                except Exception as e:
                    log.error(f"Order failed for {sym}: {e}")

        # 9. Log stats
        self.portfolio.tracker.log_stats()
        self._log_watcher_status()

    def _log_watcher_status(self):
        """Log a summary of all watcher threads."""
        statuses = {}
        for sym, w in self.watchers.items():
            s = w.state.status
            statuses[s] = statuses.get(s, 0) + 1

        signals = [w.symbol for w in self.watchers.values()
                   if w.state.score > 0.2]
        pending = [w.symbol for w in self.watchers.values()
                   if w.state.status == "pending"]

        log.info(f"Watchers: {len(self.watchers)} threads | "
                 f"Status: {statuses} | "
                 f"Signals: {signals} | Pending: {pending}")

    def get_all_watcher_states(self) -> list[dict]:
        """Return all watcher states for the dashboard."""
        states = []
        for sym, watcher in sorted(self.watchers.items()):
            s = watcher.state
            states.append({
                "symbol": s.symbol,
                "status": s.status,
                "score": s.score,
                "num_agreeing": s.num_agreeing,
                "strategy_scores": s.strategy_scores,
                "strategy_weights": s.strategy_weights,
                "regime": s.regime,
                "regime_reason": s.regime_reason,
                "candle_patterns": s.candle_patterns,
                "adx": s.adx,
                "trend": s.trend_direction,
                "above_200ema": s.above_200ema,
                "above_vwap": s.above_vwap,
                "weekly_trend_up": s.weekly_trend_up,
                "last_price": s.last_price,
                "last_update": s.last_update,
                "confirmed": s.confirmed,
                "error": s.error,
            })
        return states

    def _wait_until(self, target_time):
        while True:
            now = datetime.now(target_time.tzinfo)
            remaining = (target_time - now).total_seconds()
            if remaining <= 0:
                break
            if remaining > 300:
                log.info(f"Waiting... {remaining / 60:.0f} minutes until market open")
                time.sleep(300)
            elif remaining > 60:
                time.sleep(60)
            else:
                time.sleep(remaining)


def _make_opportunity(watcher: StockWatcher):
    """Convert a watcher's signal into an Opportunity for the risk manager."""
    from signals import Opportunity
    return Opportunity(
        symbol=watcher.symbol,
        score=watcher.state.score,
        direction="buy",
        strategy_scores=watcher.state.strategy_scores,
        num_agreeing=watcher.state.num_agreeing
    )
