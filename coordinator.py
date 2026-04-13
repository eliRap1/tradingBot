"""
Coordinator — manages all StockWatcher threads.

Responsibilities:
  - Spawn one watcher thread per stock in the universe
  - Collect signals from watchers
  - Enforce global risk limits (max positions, regime, drawdown)
  - Execute trades when watchers report confirmed signals
  - Manage existing positions (trailing stops, exits)
  - ENSURE LIVE TRADING with real-time data validation
"""

import time
import sys
import threading
from datetime import datetime

from utils import load_config, setup_logger
from ib_broker import IBBroker
from ib_data import IBDataFetcher
from instrument_classifier import InstrumentClassifier
from strategy_router import StrategyRouter
from portfolio import _normalize_symbol
from screener import Screener
from risk import RiskManager
from portfolio import PortfolioManager
from regime import RegimeFilter
from filters import SmartFilters
from watcher import StockWatcher, Action
from alerts import AlertManager, DiscordBot
from live_trading import LiveTradingManager, DataFreshnessValidator, ensure_live_trading_mode
from walk_forward_optimizer import apply_optimized_params
from ml_model import MLMetaModel

log = setup_logger("coordinator")


class Coordinator:
    # Hard limit: max orders per hour to prevent runaway order loops
    MAX_ORDERS_PER_HOUR = 20

    def __init__(self):
        log.info("")
        log.info("=" * 55)
        log.info("     TRADING BOT v2.0 - LIVE MODE")
        log.info("=" * 55)
        
        # Verify we're in live/paper mode, not backtest
        if not ensure_live_trading_mode():
            log.error("Invalid trading mode - exiting")
            sys.exit(1)

        self.config = load_config()
        
        # Apply optimized parameters if available
        self.config = apply_optimized_params(self.config)
        
        # Build broker + data layers — IB only, no Alpaca
        self._clf = InstrumentClassifier(self.config)
        self.broker = IBBroker(self.config)
        self.data = IBDataFetcher(
            self.broker._ib,
            self.broker._contracts,
            self.config
        )

        # Strategy router — assigns optimal strategy set per instrument type
        self._strategy_router = StrategyRouter(self.config)
        self.screener = Screener(self.config, self.data)
        self.risk = RiskManager(self.config)
        self.portfolio = PortfolioManager(self.config, self.broker)
        self.regime = RegimeFilter(self.data, universe=self.config["screener"]["universe"])
        self._sector_regime_enabled = self.config.get("sector_regime", {}).get("enabled", True)
        if self._sector_regime_enabled:
            from sector_regime import SectorRegimeFilter
            self.sector_regime = SectorRegimeFilter(self.data, config=self.config)
        else:
            self.sector_regime = None
        self._last_sector_regimes: dict = {}
        self.filters = SmartFilters(tracker=self.portfolio.tracker, config=self.config)
        self.alerts = AlertManager(self.config)
        self._trading_paused = False  # can be set via !pause Discord command
        self._last_regime_str = ""    # track regime changes for alerts
        self.discord_bot = DiscordBot(
            tracker=self.portfolio.tracker, broker=self.broker, coordinator=self
        )
        self.discord_bot.start()
        self.ml_model = MLMetaModel()
        self._ml_train_counter = 0  # retrain every N cycles

        # LIVE TRADING: Real-time data validation
        self.live_manager = LiveTradingManager(self.broker)
        self.freshness_validator = DataFreshnessValidator()

        # Watcher threads: {symbol: StockWatcher}
        self.watchers: dict[str, StockWatcher] = {}
        self._lock = threading.Lock()

        # Order rate limiter: track timestamps of recent orders
        self._order_timestamps: list[float] = []

        # Verify connection and show market status
        equity = self.broker.get_equity()
        cash = self.broker.get_cash()
        buying_power = self.broker.get_buying_power()
        log.info(f"Account: equity=${equity:,.2f} "
                 f"cash=${cash:,.2f} "
                 f"buying_power=${buying_power:,.2f}")
        self.risk.peak_equity = max(self.risk.peak_equity, equity)
        self.risk.set_starting_equity(equity)
        
        # Show market status
        market_status = self.live_manager.get_market_status()
        log.info(f"Market Status: {'OPEN' if market_status.is_open else 'CLOSED'} "
                 f"(session: {market_status.session})")
        if market_status.next_open:
            log.info(f"Next market open: {market_status.next_open}")

        # Pre-validate stock universe against IB — log skipped symbols once at boot
        self._validate_ib_universe()

    def _validate_ib_universe(self):
        """Pre-qualify all stock symbols against IB at startup.

        Symbols that can't be resolved are added to ib_data/_bad_contracts
        immediately so they never spam the log during normal operation.
        Prints one clean summary line instead of per-symbol errors.
        """
        stocks = [s for s in self.config["screener"].get("universe", [])
                  if self._clf.classify(s) == "stock"]
        skipped = []
        for sym in stocks:
            contract = self.data._resolve_contract(sym)
            if contract is None:
                skipped.append(sym)
        if skipped:
            log.warning(f"IB: {len(skipped)} symbol(s) not available — will be skipped: "
                        f"{', '.join(skipped)}")
        ok = len(stocks) - len(skipped)
        log.info(f"IB universe validation: {ok}/{len(stocks)} stocks OK"
                 + (f", {len(skipped)} skipped" if skipped else ""))

    def start_watchers(self, crypto_only: bool = False, non_stock_only: bool = False):
        """Spawn a watcher thread for each stock/crypto/futures in the universe.

        Args:
            crypto_only:    (legacy) start only crypto symbols
            non_stock_only: start crypto + futures but NOT stocks (used when NYSE is closed)
        """
        if crypto_only or non_stock_only:
            # Crypto symbols
            universe = list(self.config["screener"].get("crypto", []))
            if non_stock_only:
                # Also include futures so they trade during their CME session
                futures_roots = [
                    c["root"] for c in self.config.get("futures", {}).get("contracts", [])
                ]
                universe = list(dict.fromkeys(universe + futures_roots))
        else:
            universe = self.screener.get_universe()
            if len(universe) < 10:
                log.warning(f"Screener returned only {len(universe)} stocks — using full config universe")
                universe = self.config["screener"]["universe"]
            # Add crypto symbols
            universe = list(universe) + self.config["screener"].get("crypto", [])
            # Add futures roots (NQ, ES, CL, GC)
            futures_roots = [c["root"] for c in self.config.get("futures", {}).get("contracts", [])]
            universe = list(universe) + futures_roots
            # Deduplicate (preserves order) — fixes duplicate AMZN etc.
            universe = list(dict.fromkeys(universe))

        cycle_interval = self.config["schedule"]["cycle_interval_sec"]
        watcher_interval = max(60, cycle_interval)

        log.info(f"Starting {len(universe)} watcher threads "
                 f"(interval={watcher_interval}s each)")

        for sym in universe:
            if sym not in self.watchers:
                instrument_type = self._clf.classify(sym)
                strat_weights = self._strategy_router.get_strategies(instrument_type)
                watcher = StockWatcher(
                    symbol=sym,
                    config=self.config,
                    data_fetcher=self.data,
                    interval=watcher_interval,
                    sector_regime_getter=(
                        self.sector_regime.get_regime_for_sector
                        if self._sector_regime_enabled and self.sector_regime else None
                    ),
                    strategies=strat_weights,
                )
                self.watchers[sym] = watcher
                watcher.start()
                time.sleep(2)  # stagger to avoid API rate limits

        log.info(f"All {len(self.watchers)} watchers running")

    def stop_watchers(self, stocks_only: bool = False):
        """Stop watcher threads. If stocks_only, keep crypto and futures running."""
        if stocks_only:
            log.info("Stopping stock watchers (crypto + futures continue)...")
            to_stop = [s for s in self.watchers if self._clf.classify(s) == "stock"]
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
                    # Stop stock watchers ONCE, keep crypto + futures running
                    has_stock_watchers = any(
                        self._clf.classify(s) == "stock" for s in self.watchers
                    )
                    if has_stock_watchers:
                        self.stop_watchers(stocks_only=True)

                    # Start crypto watchers if not already running
                    crypto_running = any(
                        self._clf.classify(s) in ("crypto", "futures") for s in self.watchers
                    )
                    if not crypto_running:
                        self.start_watchers(non_stock_only=True)

                    next_open = clock.next_open
                    log.info(f"Market closed. Crypto watchers active. "
                             f"Next stock open: {next_open}")
                    # Send ONE Discord alert with Israel time + full stats
                    equity = self.broker.get_equity()
                    positions = self.portfolio.get_current_positions()
                    stats = self.portfolio.tracker.get_stats(starting_equity=equity)
                    # Count today's trades
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    today_trades = [t for t in self.portfolio.tracker.trades
                                    if t.get("closed_at", "").startswith(today_str)]
                    daily_pnl = sum(t["pnl"] for t in today_trades)
                    self.alerts.send_market_closed(
                        next_open, stats=stats, equity=equity,
                        positions=len(positions), daily_pnl=daily_pnl,
                        daily_trades=len(today_trades),
                    )

                    # Run one immediate cycle to catch any signals confirmed
                    # right before/at market close (e.g. crypto signals that fired
                    # during the close-buffer skip window).
                    self._coordinator_cycle()

                    # Market-closed loop: keep running crypto cycles
                    interval = self.config["schedule"]["cycle_interval_sec"]
                    cycle_count = 0
                    while True:
                        time.sleep(interval)
                        cycle_count += 1

                        # Check if market opened
                        clock = self.broker.get_clock()
                        if clock.is_open:
                            equity = self.broker.get_equity()
                            self.risk.set_starting_equity(equity)
                            delay = self.config["schedule"]["market_open_delay_min"]
                            log.info(f"Market opened. Waiting {delay}min...")
                            time.sleep(delay * 60)
                            break

                        # Run crypto cycle
                        self._coordinator_cycle()

                        # Hourly status update
                        if cycle_count % max(1, 3600 // interval) == 0:
                            hours_until = ""
                            try:
                                import datetime as _dt
                                now = _dt.datetime.now(_dt.timezone.utc)
                                if hasattr(next_open, 'timestamp'):
                                    delta = next_open.timestamp() - now.timestamp()
                                else:
                                    delta = 0
                                if delta > 0:
                                    h = int(delta // 3600)
                                    m = int((delta % 3600) // 60)
                                    hours_until = f" ({h}h {m}m until open)"
                                    log.info(f"Market still closed{hours_until}. "
                                             f"Crypto watchers running. "
                                             f"Next open: {next_open}")
                            except Exception:
                                log.info(f"Market still closed. Next open: {next_open}")

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
                    self._clf.classify(s) == "stock" for s in self.watchers
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
        
        LIVE TRADING: Validates market hours and data freshness before trading.
        """
        log.info("")
        log.info("=" * 50)
        log.info(f"  CYCLE @ {datetime.now():%H:%M:%S}")
        log.info("=" * 50)

        # 0a. Check if we should skip this cycle entirely.
        # We only skip during the last 15 min of NYSE close (widening spreads).
        # Futures and crypto continue to run 24/7 regardless of NYSE hours.
        should_skip, skip_reason = self.live_manager.should_skip_cycle()
        if should_skip:
            log.info(f"Skipping cycle: {skip_reason}")
            return

        # 0b. Log market status
        market_status = self.live_manager.get_market_status()
        from live_trading import _is_cme_futures_open, _is_crypto_open
        open_sessions = []
        if market_status.is_open:
            open_sessions.append("NYSE")
        if _is_cme_futures_open():
            open_sessions.append("CME")
        if _is_crypto_open():
            open_sessions.append("PAXOS")
        log.info(f"Sessions open: {', '.join(open_sessions) if open_sessions else 'NONE'}")

        # 0c. Prime bar cache for all watchers (one bulk fetch per timeframe instead of N individual calls)
        all_symbols = list(self.watchers.keys())
        if all_symbols:
            self.data.prime_intraday_cache(all_symbols, timeframe="5Min", days=5)
            self.data.prime_intraday_cache(all_symbols, timeframe="1Hour", days=10)
            spy_symbols = list(set(all_symbols) | {"SPY"})
            self.data.prime_intraday_cache(spy_symbols, timeframe="1Day", days=250)
            # Prime sector ETF daily bars (Layer 2 regime)
            if self._sector_regime_enabled:
                from sector_regime import SECTOR_ETFS
                self.data.prime_intraday_cache(
                    list(set(SECTOR_ETFS) | {"BTC/USD"}), timeframe="1Day", days=120
                )

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

        # 2a. Send regime-change and HMM-caution alerts
        new_regime_str = regime["regime"]
        if self._last_regime_str and self._last_regime_str != new_regime_str:
            self.alerts.send_regime_change(
                self._last_regime_str, new_regime_str, regime["description"]
            )
        elif (regime.get("hmm_confidence") and regime["hmm_confidence"] > 0.92
              and regime.get("hmm_regime") == "bear" and new_regime_str == "bull"
              and "HMM CAUTION" in regime["description"]
              and self._last_regime_str == new_regime_str):
            # Only send once per regime session (when str hasn't changed but HMM fires)
            self.alerts.send_hmm_caution(
                regime["hmm_confidence"], regime["hmm_regime"],
                new_regime_str, regime["size_multiplier"]
            )
        self._last_regime_str = new_regime_str

        # 2d. Sector regimes (Layer 2)
        sector_regimes = {}
        if self._sector_regime_enabled and self.sector_regime:
            sector_regimes = self.sector_regime.get_sector_regimes()
            for etf_key, sdata in sector_regimes.items():
                prev = self._last_sector_regimes.get(etf_key)
                curr = sdata["regime"]
                if prev is not None and prev != curr:
                    self.alerts.send_sector_regime_change(
                        etf_key=etf_key, old_regime=prev,
                        new_regime=curr, description=sdata["description"],
                    )
                self._last_sector_regimes[etf_key] = curr
            # Log non-bull sectors to keep logs concise
            non_bull = {k: v["regime"] for k, v in sector_regimes.items() if v["regime"] != "bull"}
            if non_bull:
                log.info(f"Sector regimes (non-bull): {non_bull}")

        # 2b. Check pause flag (set via !pause Discord command)
        if self._trading_paused:
            log.info("Trading PAUSED via Discord command — skipping signal processing")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # 2c. Retrain ML meta-model periodically (every 50 cycles)
        self._ml_train_counter += 1
        if self._ml_train_counter % 50 == 1:
            self.ml_model.train(self.portfolio.tracker.trades)

        # 2c. Update alpha decay factors and distribute to watchers
        decay_factors = self.portfolio.tracker.get_strategy_alpha_decay()
        if decay_factors:
            decaying = {s: f for s, f in decay_factors.items() if f < 0.8}
            if decaying:
                log.info(f"Alpha decay: {', '.join(f'{s}={f:.2f}x' for s, f in decaying.items())}")
            for w in self.watchers.values():
                w._alpha_decay = decay_factors

        # 3. Manage existing positions + reconcile with broker
        positions = self.portfolio.get_current_positions()
        held_symbols = list(positions.keys())
        self._reconcile_positions(held_symbols)
        self.portfolio.log_portfolio_status(positions)

        if positions:
            prices = self.data.get_latest_prices(list(positions.keys()))
            # Fetch bars for ATR-based Chandelier trailing stops
            pos_bars = {}
            for sym in positions:
                w = self.watchers.get(sym)
                if w and w.get_bars() is not None:
                    pos_bars[sym] = w.get_bars()
            to_close, partial_exits = self.portfolio.check_trailing_stops(positions, prices, pos_bars)
            if partial_exits:
                for p in partial_exits:
                    pos = positions.get(p["symbol"])
                    if pos:
                        self.alerts.send_exit_alert(
                            symbol=p["symbol"], side=pos.get("side", "long"),
                            entry_price=pos["entry_price"],
                            exit_price=pos["current_price"],
                            pnl=pos.get("unrealized_pl", 0.0) * (p["qty"] / pos["qty"]) if pos["qty"] > 0 else 0,
                            pnl_pct=(pos["current_price"] - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0,
                            reason=p.get("reason", "partial_exit"),
                        )
                self.portfolio.execute_partial_exits(partial_exits, positions)
            if to_close:
                # Send Discord exit alerts before closing
                for sym in to_close:
                    pos = positions.get(sym)
                    if pos:
                        meta = self.portfolio.position_meta.get(sym, {})
                        is_long = pos.get("side", "long") == "long"
                        pnl = pos.get("unrealized_pl", 0.0)
                        if is_long:
                            pnl_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
                        else:
                            pnl_pct = (pos["entry_price"] - pos["current_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
                        reason = "trailing_stop" if pnl > 0 else "stop_loss"
                        self.alerts.send_exit_alert(
                            symbol=sym, side=pos.get("side", "long"),
                            entry_price=pos["entry_price"],
                            exit_price=pos["current_price"],
                            pnl=pnl, pnl_pct=pnl_pct, reason=reason,
                        )
                self.portfolio.execute_exits(to_close, positions)
                positions = self.portfolio.get_current_positions()
                held_symbols = list(positions.keys())

        # 4. Check position capacity (crypto has separate slots; futures share stock pool)
        max_pos = self.config["signals"]["max_positions"]
        stock_positions = [s for s in held_symbols if self._clf.classify(s) != "crypto"]
        crypto_positions = [s for s in held_symbols if self._clf.classify(s) == "crypto"]
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
        # Build normalized set of held symbols so BTC/USD matches BTCUSD
        held_normalized = set(_normalize_symbol(s) for s in held_symbols)
        trade_signals = []
        for sym, watcher in list(self.watchers.items()):
            if _normalize_symbol(sym) in held_normalized:
                continue
            if not watcher.state.confirmed:
                continue

            asset_type = self._clf.classify(sym)
            is_crypto  = asset_type == "crypto"
            is_futures = asset_type == "futures"

            # Long signals ─────────────────────────────────────────────────
            # Crypto:  always allowed (24/7, independent of equity regime)
            # Futures: always allowed — futures can hedge, not purely directional
            # Stocks:  blocked in bear regime (SPY breadth failing)
            if watcher.state.action == Action.BUY:
                if not is_crypto and not is_futures and not regime["allow_longs"]:
                    continue
                trade_signals.append(watcher)

            # Short signals ─────────────────────────────────────────────────
            # Crypto:  NOT supported on IB PAXOS — hard block
            # Futures: always allowed (ES/NQ shorts are standard hedging tools)
            # Stocks:  blocked in pure-bull regime by default (configurable)
            elif watcher.state.action == Action.SHORT:
                if is_crypto:
                    continue   # IB PAXOS does not support crypto shorts
                if not is_futures:
                    if regime["regime"] == "bull" and not self.config["signals"].get("allow_shorts_in_bull", False):
                        continue
                trade_signals.append(watcher)

        if not trade_signals:
            if not regime["allow_longs"]:
                log.info("BEAR REGIME — scanning for shorts only")
            log.info("No confirmed signals from watchers")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        # 5b. ML meta-model filter: skip signals with low predicted win probability
        if self.ml_model.is_ready:
            ml_filtered = []
            for w in trade_signals:
                features = {
                    "strategy_scores": w.state.strategy_scores,
                    "num_agreeing": w.state.num_agreeing,
                    "composite_score": w.state.score,
                }
                prob = self.ml_model.predict(features)
                if prob is not None and prob < 0.4:
                    log.info(f"ML FILTER: {w.symbol} blocked (win_prob={prob:.1%})")
                    continue
                ml_filtered.append(w)
            trade_signals = ml_filtered

            if not trade_signals:
                log.info("All signals filtered by ML model")
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

        # Sector filter (gap filter removed — GapStrategy handles gap analysis)
        filtered = []
        for watcher in trade_signals:
            sym = watcher.symbol

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
                df = self.data.get_intraday_bars(w.symbol, timeframe="1Day", days=60)
                if df is not None:
                    all_bars[w.symbol] = df
            for sym in held_symbols:
                if sym not in all_bars:
                    df = self.data.get_intraday_bars(sym, timeframe="1Day", days=60)
                    if df is not None:
                        all_bars[sym] = df

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
        base_size_mult = regime["size_multiplier"] * cooldown_mult

        # Filter by available slots per type
        executable = []
        for w in filtered:
            is_crypto = self._clf.classify(w.symbol) == "crypto"
            if is_crypto and crypto_slots > 0:
                executable.append(w)
                crypto_slots -= 1
            elif not is_crypto and stock_slots > 0:
                executable.append(w)
                stock_slots -= 1

        # Rate limit check
        if not self._check_order_rate_limit():
            log.critical("ORDER RATE LIMIT HIT — halting new orders this cycle")
            self.portfolio.tracker.log_stats()
            self._log_watcher_status()
            return

        for watcher in executable:
            sym = watcher.symbol
            bars = watcher.get_bars()
            if bars is None:
                continue
            
            # LIVE TRADING: Validate data freshness
            is_fresh, freshness_reason = self.freshness_validator.validate_bars(bars, "5Min", symbol=sym)
            if not is_fresh:
                log.warning(f"Skipping {sym}: {freshness_reason}")
                continue
            
            # LIVE TRADING: Check if we can trade this symbol now
            can_trade, trade_reason = self.live_manager.can_trade_symbol(sym)
            if not can_trade:
                log.info(f"Cannot trade {sym} now: {trade_reason}")
                continue

            # LIVE TRADING: Get verified live price
            price, price_status = self.live_manager.get_live_price(sym, self.data)
            if not price:
                log.warning(f"No live price for {sym}: {price_status}")
                continue
            
            if "stale" in price_status or "closed" in price_status:
                log.warning(f"Price issue for {sym}: {price_status}")
                # For crypto, continue anyway (24/7 trading)
                if self._clf.classify(sym) not in ("crypto", "futures"):
                    continue

            # Apply correlation-adjusted sizing
            corr_mult = self.filters.get_corr_size_mult(sym)

            # Layer 2: sector regime multiplier
            sector_mult = 1.0
            if self._sector_regime_enabled and self.sector_regime:
                from filters import SECTOR_MAP
                sector_label = SECTOR_MAP.get(sym, "other")
                if sector_label != "other":
                    sector_mult = self.sector_regime.compute_size_mult(
                        sector_label, regime["regime"]
                    )
                    # Veto: macro BEAR + sector BEAR blocks new longs
                    if sector_mult == 0.0 and watcher.state.action == Action.BUY:
                        log.info(
                            f"SECTOR VETO: Skipping {sym} long -- "
                            f"macro={regime['regime']} + sector({sector_label})=bear"
                        )
                        continue

            size_mult = base_size_mult * sector_mult * corr_mult

            orders = self.risk.size_orders(
                opportunities=[_make_opportunity(watcher)],
                bars={sym: bars},
                prices={sym: price},
                equity=equity,
                num_existing=len(held_symbols),
                regime_size_mult=size_mult,
                tracker_stats=self.portfolio.tracker.get_stats(),
                tracker=self.portfolio.tracker,
            )

            for order in orders:
                try:
                    use_smart = self.config.get("execution", {}).get("smart_orders", False)

                    if use_smart:
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
                    # Store which strategies fired + full trade context for crypto exit recording
                    # Use normalized symbol to match broker's position key format
                    contributing = [s for s, v in watcher.state.strategy_scores.items()
                                    if abs(v) > 0.1]
                    norm_sym = _normalize_symbol(order.symbol)
                    self.portfolio.position_meta.setdefault(norm_sym, {}).update({
                        "strategies": contributing,
                        "entry_price": order.entry_price,
                        "take_profit": order.take_profit,
                        "stop_loss": order.stop_loss,
                        "original_qty": order.qty,
                        "side": order.side,
                    })
                    self.portfolio._save_meta()
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
                    self._order_timestamps.append(time.time())
                    # Clear watcher signal to prevent re-entry if position stops out
                    watcher.state.confirmed = False
                    watcher.state.prev_signal = False
                    watcher.state.action = Action.NONE
                except Exception as e:
                    log.error(f"Order failed for {sym}: {e}")
                    self.alerts.send_error(f"Order failed for {sym}: {e}")

        # 9. Log stats
        self.portfolio.tracker.log_stats()
        self._log_watcher_status()

    def _check_order_rate_limit(self) -> bool:
        """Returns False if order rate limit exceeded. Safety valve."""
        now = time.time()
        hour_ago = now - 3600
        self._order_timestamps = [t for t in self._order_timestamps if t > hour_ago]
        if len(self._order_timestamps) >= self.MAX_ORDERS_PER_HOUR:
            log.critical(
                f"ORDER RATE LIMIT: {len(self._order_timestamps)} orders in last hour "
                f"(max={self.MAX_ORDERS_PER_HOUR}). Halting new entries."
            )
            self.alerts.send_error(
                f"Order rate limit hit: {len(self._order_timestamps)} orders/hour"
            )
            return False
        return True

    def _reconcile_positions(self, held_symbols: list[str]):
        """
        Reconcile local state with broker's actual positions.
        Cleans up watermarks/meta for positions that no longer exist
        (e.g., bracket SL/TP filled between cycles).
        """
        broker_positions = set(held_symbols)
        stale_hw = [s for s in self.portfolio.high_watermarks if s not in broker_positions]
        stale_lw = [s for s in self.portfolio.low_watermarks if s not in broker_positions]
        stale_meta = [s for s in self.portfolio.position_meta if s not in broker_positions]

        cleaned = False
        for sym in stale_hw:
            self.portfolio.high_watermarks.pop(sym, None)
            cleaned = True
        for sym in stale_lw:
            self.portfolio.low_watermarks.pop(sym, None)
            cleaned = True
        for sym in stale_meta:
            log.info(f"RECONCILE: {sym} no longer held — cleaning up local state")
            self.portfolio.position_meta.pop(sym, None)
            cleaned = True

        if cleaned:
            self.portfolio._save_watermarks()
            self.portfolio._save_meta()

    def _log_watcher_status(self):
        """Log a visual summary of all watcher threads."""
        watchers_snapshot = list(self.watchers.values())
        statuses = {}
        for w in watchers_snapshot:
            s = w.state.status
            statuses[s] = statuses.get(s, 0) + 1

        signals = [f"{w.symbol}({w.state.score:+.2f})" for w in watchers_snapshot if w.state.score > 0.2]
        pending = [w.symbol for w in watchers_snapshot if w.state.status == "pending"]
        errors = [w.symbol for w in watchers_snapshot if w.state.status == "error"]

        status_str = " | ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
        log.info(f"  WATCHERS: {len(watchers_snapshot)} active | {status_str}")
        if signals:
            log.info(f"  SIGNALS:  {', '.join(signals)}")
        if pending:
            log.info(f"  PENDING:  {', '.join(pending)}")
        if errors:
            log.warning(f"  ERRORS:   {', '.join(errors)}")
        log.info("-" * 50)

    def get_all_watcher_states(self) -> list[dict]:
        """Return all watcher states for the dashboard."""
        states = []
        for sym, watcher in sorted(list(self.watchers.items())):
            s = watcher.state
            states.append({
                "symbol": s.symbol,
                "status": s.status,
                "score": s.score,
                "num_agreeing": s.num_agreeing,
                "strategy_scores": s.strategy_scores,
                "strategy_weights": s.strategy_weights,
                "regime": s.regime,
                "regime": s.regime,
                "regime_reason": s.regime_reason,
                "action": s.action.name if hasattr(s.action, 'name') else str(s.action),
                "confirmed": s.confirmed,
                "error": s.error,
            })
        return states


def _make_opportunity(watcher):
    """Convert a watcher signal into an Opportunity for the risk manager."""
    from signals import Opportunity
    from watcher import Action
    direction = "sell" if watcher.state.action == Action.SHORT else "buy"
    contributing = [s for s, v in watcher.state.strategy_scores.items() if abs(v) > 0.1]
    return Opportunity(
        symbol=watcher.symbol,
        score=watcher.state.score,
        direction=direction,
        strategy_scores=watcher.state.strategy_scores,
        num_agreeing=watcher.state.num_agreeing,
        contributing_strategies=contributing,
    )
