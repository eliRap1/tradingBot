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
from broker import Broker, CRYPTO_SYMBOLS
from data import DataFetcher
from screener import Screener
from risk import RiskManager
from portfolio import PortfolioManager
from regime import RegimeFilter
from filters import SmartFilters
from watcher import StockWatcher, Action
from alerts import AlertManager

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
        self.filters = SmartFilters(tracker=self.portfolio.tracker, config=self.config)
        self.alerts = AlertManager(self.config)

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

    def start_watchers(self, crypto_only: bool = False):
        """Spawn a watcher thread for each stock/crypto in the universe."""
        if crypto_only:
            universe = self.config["screener"].get("crypto", [])
        else:
            universe = self.screener.get_universe()
            if not universe:
                universe = self.config["screener"]["universe"]
            # Add crypto symbols
            universe = list(universe) + self.config["screener"].get("crypto", [])

        cycle_interval = self.config["schedule"]["cycle_interval_sec"]
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
                time.sleep(0.5)

        log.info(f"All {len(self.watchers)} watchers running")

    def stop_watchers(self, stocks_only: bool = False):
        """Stop watcher threads. If stocks_only, keep crypto running."""
        if stocks_only:
            log.info("Stopping stock watchers (crypto continues)...")
            to_stop = [s for s in self.watchers if s not in CRYPTO_SYMBOLS]
            for sym in to_stop:
                self.watchers[sym].stop()
                del self.watchers[sym]
        else:
            log.info("Stopping all watchers...")
            for sym, watcher in self.watchers.items():
                watcher.stop()
            self.watchers.clear()
        log.info(f"Watchers remaining: {len(self.watchers)}")

    def run(self):
        """Main coordinator loop."""
        log.info("Bot is running. Press Ctrl+C to stop.")

        while True:
            try:
                clock = self.broker.get_clock()

                if not clock.is_open:
                    # Stop stock watchers but keep crypto running
                    has_stock_watchers = any(
                        s not in CRYPTO_SYMBOLS for s in self.watchers
                    )
                    if has_stock_watchers:
                        self.stop_watchers(stocks_only=True)

                    # Start crypto watchers if not already running
                    crypto_running = any(
                        s in CRYPTO_SYMBOLS for s in self.watchers
                    )
                    if not crypto_running:
                        self.start_watchers(crypto_only=True)

                    next_open = clock.next_open
                    log.info(f"Market closed. Crypto watchers active. "
                             f"Next stock open: {next_open}")

                    # Still run coordinator cycle for crypto signals
                    self._coordinator_cycle()

                    # Wait a cycle then check again
                    interval = self.config["schedule"]["cycle_interval_sec"]
                    time.sleep(interval)

                    # Check if market opened
                    clock = self.broker.get_clock()
                    if clock.is_open:
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
                    log.info("Too close to market close — stocks paused, crypto continues.")
                    self.stop_watchers(stocks_only=True)
                    # Don't wait — keep looping for crypto
                    self._coordinator_cycle()
                    time.sleep(self.config["schedule"]["cycle_interval_sec"])
                    continue

                # Start all watchers if not running
                has_stock_watchers = any(
                    s not in CRYPTO_SYMBOLS for s in self.watchers
                )
                if not has_stock_watchers:
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
            dd_pct = (self.risk.peak_equity - equity) / self.risk.peak_equity
            log.critical("DRAWDOWN LIMIT HIT — halting all trading")
            self.alerts.send_drawdown_halt(dd_pct, self.risk.peak_equity, equity)
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

        # 4. Check position capacity (crypto has separate slots)
        max_pos = self.config["signals"]["max_positions"]
        stock_positions = [s for s in held_symbols if s not in CRYPTO_SYMBOLS]
        crypto_positions = [s for s in held_symbols if s in CRYPTO_SYMBOLS]
        max_crypto = self.config["signals"].get("max_crypto_positions", 2)
        stocks_full = len(stock_positions) >= max_pos
        crypto_full = len(crypto_positions) >= max_crypto
        if stocks_full and crypto_full:
            log.info(f"At max positions (stocks={len(stock_positions)}/{max_pos}, "
                     f"crypto={len(crypto_positions)}/{max_crypto}). Skipping new entries.")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # 5. Collect confirmed signals from watchers (LONG + SHORT)
        trade_signals = []
        for sym, watcher in self.watchers.items():
            if sym in held_symbols:
                continue
            if not watcher.state.confirmed:
                continue

            # Crypto is NOT gated by SPY regime — it trades independently
            is_crypto = sym in CRYPTO_SYMBOLS

            if watcher.state.action == Action.BUY:
                if not is_crypto and not regime["allow_longs"]:
                    continue  # Bear regime blocks stock longs only
                trade_signals.append(watcher)

            elif watcher.state.action == Action.SHORT:
                # Shorts allowed in bear/chop regime, restricted in bull
                if not is_crypto and regime["regime"] == "bull" and not self.config["signals"].get("allow_shorts_in_bull", False):
                    continue
                trade_signals.append(watcher)

        if not trade_signals:
            if not regime["allow_longs"]:
                log.info("BEAR REGIME — scanning for shorts only")
            log.info("No confirmed signals from watchers")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # Sort by absolute score (strongest signals first)
        trade_signals.sort(key=lambda w: abs(w.state.score), reverse=True)

        longs = [w for w in trade_signals if w.state.action == Action.BUY]
        shorts = [w for w in trade_signals if w.state.action == Action.SHORT]
        log.info(f"Confirmed signals — Longs: {[w.symbol for w in longs]} "
                 f"Shorts: {[w.symbol for w in shorts]}")

        # 7. Apply global filters
        # Sector cap
        from filters import SECTOR_MAP, MAX_PER_SECTOR
        sector_count = {}
        for sym in held_symbols:
            sector = SECTOR_MAP.get(sym, "other")
            sector_count[sector] = sector_count.get(sector, 0) + 1

        # Gap filter + sector filter
        filtered = []
        for watcher in trade_signals:
            sym = watcher.symbol
            bars = watcher.get_bars()

            # Gap filter (skip for crypto — gaps are normal)
            if sym not in CRYPTO_SYMBOLS and bars is not None and len(bars) >= 2 and "open" in bars.columns:
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

        # 7b. Dynamic correlation filter
        if held_symbols:
            all_bars = {}
            for w in filtered:
                daily = self.data.get_bars([w.symbol], timeframe="1Day", days=60)
                if w.symbol in daily:
                    all_bars[w.symbol] = daily[w.symbol]
            for sym in held_symbols:
                if sym not in all_bars:
                    daily = self.data.get_bars([sym], timeframe="1Day", days=60)
                    if sym in daily:
                        all_bars[sym] = daily[sym]

            candidate_syms = [w.symbol for w in filtered]
            passed_syms = self.filters.filter_correlated(
                candidate_syms, held_symbols, all_bars
            )
            filtered = [w for w in filtered if w.symbol in passed_syms]

            if not filtered:
                log.info("No signals passed correlation filter")
                self.portfolio.tracker.log_stats()
                self._log_watcher_status()
                return

        # 8. Size and execute orders (separate slots for stocks vs crypto)
        stock_slots = max_pos - len(stock_positions) if not stocks_full else 0
        crypto_slots = max_crypto - len(crypto_positions) if not crypto_full else 0
        cooldown_mult = self.filters.get_loss_cooldown_mult()
        size_mult = regime["size_multiplier"] * cooldown_mult

        # Filter by available slots per type
        executable = []
        for w in filtered:
            is_crypto = w.symbol in CRYPTO_SYMBOLS
            if is_crypto and crypto_slots > 0:
                executable.append(w)
                crypto_slots -= 1
            elif not is_crypto and stock_slots > 0:
                executable.append(w)
                stock_slots -= 1

        for watcher in executable:
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
                regime_size_mult=size_mult,
                tracker_stats=self.portfolio.tracker.get_stats(),
            )

            for order in orders:
                try:
                    use_smart = self.config.get("execution", {}).get("smart_orders", False)

                    if order.symbol in CRYPTO_SYMBOLS:
                        self.broker.submit_crypto_order(
                            symbol=order.symbol,
                            qty=order.qty,
                            side=order.side,
                            take_profit=order.take_profit,
                            stop_loss=order.stop_loss
                        )
                    elif use_smart:
                        self.broker.submit_smart_order(
                            symbol=order.symbol,
                            qty=order.qty,
                            side=order.side,
                            take_profit=order.take_profit,
                            stop_loss=order.stop_loss,
                            limit_offset_pct=self.config.get("execution", {}).get(
                                "limit_offset_pct", 0.0002),
                            timeout_sec=self.config.get("execution", {}).get(
                                "limit_timeout_sec", 30),
                        )
                    elif order.side == "sell":
                        self.broker.submit_short_bracket(
                            symbol=order.symbol,
                            qty=order.qty,
                            take_profit=order.take_profit,
                            stop_loss=order.stop_loss
                        )
                    else:
                        self.broker.submit_bracket_order(
                            symbol=order.symbol,
                            qty=order.qty,
                            side=order.side,
                            take_profit=order.take_profit,
                            stop_loss=order.stop_loss
                        )
                    # Track position risk for R-multiple calculation
                    self.portfolio.set_position_risk(
                        order.symbol, order.entry_price,
                        order.stop_loss, order.qty
                    )
                    log.info(
                        f"ORDER: {order.side} {order.qty} {order.symbol} "
                        f"@ ~${order.entry_price:.2f} | "
                        f"SL=${order.stop_loss:.2f} TP=${order.take_profit:.2f} | "
                        f"regime={watcher.state.regime} "
                        f"confluence={watcher.state.num_agreeing}"
                    )
                    self.alerts.send_trade_alert(
                        side=order.side, qty=order.qty,
                        symbol=order.symbol,
                        entry_price=order.entry_price,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        regime=watcher.state.regime,
                        confluence=watcher.state.num_agreeing,
                    )
                    held_symbols.append(order.symbol)
                except Exception as e:
                    log.error(f"Order failed for {sym}: {e}")
                    self.alerts.send_error(f"Order failed for {sym}: {e}")

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
    direction = "sell" if watcher.state.action == Action.SHORT else "buy"
    return Opportunity(
        symbol=watcher.symbol,
        score=watcher.state.score,
        direction=direction,
        strategy_scores=watcher.state.strategy_scores,
        num_agreeing=watcher.state.num_agreeing
    )
