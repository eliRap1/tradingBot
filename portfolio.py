"""
Portfolio management with pro-grade exit logic.

Improvements:
- Direction-aware trailing stops (long: high watermark, short: low watermark)
- Time-based stops (close zombie trades after 5 days with <0.5R profit)
- Breakeven stop (move stop to entry after 0.75R profit, persisted)
- Partial exit scaling (1st: 40% at 1.2R, 2nd: 30% at 2.5R)
- Position reconciliation with broker state
- Trade recording for performance tracking
"""

from datetime import datetime
from utils import setup_logger
from state import load_state, save_state
from tracker import TradeTracker
from broker import CRYPTO_SYMBOLS

log = setup_logger("portfolio")


class PortfolioManager:
    def __init__(self, config: dict, broker):
        self.config = config
        self.broker = broker
        self.tracker = TradeTracker()

        # Restore watermarks from persisted state
        state = load_state()
        self.high_watermarks = state.get("high_watermarks", {})  # longs: tracks highs
        self.low_watermarks = state.get("low_watermarks", {})    # shorts: tracks lows

        # Track when positions were opened and their initial risk
        # {symbol: {"opened_at": iso_str, "entry_price": float, "initial_risk": float,
        #           "breakeven_armed": bool, "partial_done": bool}}
        self.position_meta = state.get("position_meta", {})

    def get_current_positions(self) -> dict:
        """Get current positions as {symbol: position_info}."""
        positions = {}
        for pos in self.broker.get_positions():
            positions[pos.symbol] = {
                "qty": float(pos.qty),
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side,
            }

            # Track position open time if new
            if pos.symbol not in self.position_meta:
                self.position_meta[pos.symbol] = {
                    "opened_at": datetime.now().isoformat(),
                    "entry_price": float(pos.avg_entry_price),
                    "initial_risk": 0.0,  # Will be set when order is placed
                }
                self._save_meta()

        return positions

    def check_trailing_stops(self, positions: dict,
                              prices: dict[str, float],
                              bars: dict[str, "pd.DataFrame"] = None
                              ) -> tuple[list[str], list[dict]]:
        """Check trailing stops, partial exits, return symbols to close and partial exits.

        Uses ATR Chandelier trailing stop (highest-high - 3*ATR for longs,
        lowest-low + 3*ATR for shorts) with fallback to percentage-based.
        Time stop: exit if no 1R profit after configurable number of check cycles.
        """
        import ta as ta_lib
        default_trail_pct = self.config["risk"]["trailing_stop_pct"]
        crypto_trail_pct = self.config.get("screener", {}).get(
            "crypto_risk", {}).get("trailing_stop_pct", default_trail_pct)
        chandelier_mult = self.config["risk"].get("chandelier_atr_mult", 3.0)
        to_close = []
        partial_exits = []

        # Partial exit config
        partial_enabled = self.config["risk"].get("partial_exit_enabled", False)
        partial_r = self.config["risk"].get("partial_exit_r", 1.5)
        partial_pct = self.config["risk"].get("partial_exit_pct", 0.50)
        bars = bars or {}

        for sym, pos in positions.items():
            if sym not in prices:
                continue

            current_price = prices[sym]
            entry_price = pos["entry_price"]
            is_long = pos.get("side", "long") == "long"
            meta = self.position_meta.get(sym, {})
            trail_pct = crypto_trail_pct if sym in CRYPTO_SYMBOLS else default_trail_pct

            # === CHANDELIER ATR TRAILING STOP ===
            # Use ATR-based trail if bars available, else fall back to % trail
            atr_val = None
            if sym in bars and len(bars[sym]) >= 15:
                df = bars[sym]
                atr_series = ta_lib.volatility.AverageTrueRange(
                    df["high"], df["low"], df["close"], window=14
                ).average_true_range()
                atr_val = atr_series.iloc[-1] if len(atr_series) > 0 else None

            if is_long:
                if sym not in self.high_watermarks:
                    self.high_watermarks[sym] = max(entry_price, current_price)
                else:
                    self.high_watermarks[sym] = max(self.high_watermarks[sym], current_price)
                watermark = self.high_watermarks[sym]
                if atr_val and atr_val > 0:
                    trail_price = watermark - (chandelier_mult * atr_val)
                else:
                    trail_price = watermark * (1 - trail_pct)
            else:
                if sym not in self.low_watermarks:
                    self.low_watermarks[sym] = min(entry_price, current_price)
                else:
                    self.low_watermarks[sym] = min(self.low_watermarks[sym], current_price)
                watermark = self.low_watermarks[sym]
                if atr_val and atr_val > 0:
                    trail_price = watermark + (chandelier_mult * atr_val)
                else:
                    trail_price = watermark * (1 + trail_pct)

            # === PARTIAL EXIT at 1.2R (first partial) ===
            if partial_enabled and not meta.get("partial_done", False):
                initial_risk = meta.get("initial_risk", 0.0)
                qty = pos["qty"]
                if initial_risk > 0 and qty > 1:
                    original_qty = meta.get("original_qty", qty)
                    risk_per_share = initial_risk / original_qty if original_qty > 0 else 0
                    if is_long:
                        current_r = (current_price - entry_price) / risk_per_share if risk_per_share > 0 else 0
                    else:
                        current_r = (entry_price - current_price) / risk_per_share if risk_per_share > 0 else 0

                    if current_r >= partial_r:
                        is_crypto = sym in CRYPTO_SYMBOLS
                        close_qty = qty * partial_pct if is_crypto else max(1, int(qty * partial_pct))
                        partial_exits.append({
                            "symbol": sym,
                            "qty": close_qty,
                            "reason": f"partial_exit_{partial_r}R",
                            "r_multiple": round(current_r, 2),
                        })
                        log.info(
                            f"PARTIAL EXIT 1: {sym} — {current_r:.1f}R reached, "
                            f"closing {close_qty}/{qty} shares ({partial_pct:.0%})"
                        )
                        meta["partial_done"] = True
                        if "original_qty" not in meta:
                            meta["original_qty"] = qty
                        self.position_meta[sym] = meta
                        self._save_meta()

            # === SECOND PARTIAL EXIT at 2.5R ===
            second_partial_enabled = self.config["risk"].get("second_partial_enabled", False)
            second_partial_r = self.config["risk"].get("second_partial_r", 2.5)
            second_partial_pct = self.config["risk"].get("second_partial_pct", 0.30)
            
            if second_partial_enabled and meta.get("partial_done") and not meta.get("second_partial_done", False):
                initial_risk = meta.get("initial_risk", 0.0)
                qty = pos["qty"]
                if initial_risk > 0 and qty > 1:
                    # Use saved original_qty for accurate R-multiple
                    original_qty = meta.get("original_qty", qty)
                    risk_per_share = initial_risk / original_qty if original_qty > 0 else 0
                    if is_long:
                        current_r = (current_price - entry_price) / risk_per_share if risk_per_share > 0 else 0
                    else:
                        current_r = (entry_price - current_price) / risk_per_share if risk_per_share > 0 else 0

                    if current_r >= second_partial_r:
                        is_crypto = sym in CRYPTO_SYMBOLS
                        close_qty = qty * second_partial_pct if is_crypto else max(1, int(qty * second_partial_pct))
                        if close_qty < qty:  # Don't close everything
                            partial_exits.append({
                                "symbol": sym,
                                "qty": close_qty,
                                "reason": f"partial_exit_{second_partial_r}R",
                                "r_multiple": round(current_r, 2),
                            })
                            log.info(
                                f"PARTIAL EXIT 2: {sym} — {current_r:.1f}R reached, "
                                f"closing {close_qty}/{qty} shares ({second_partial_pct:.0%})"
                            )
                            meta["second_partial_done"] = True
                            self.position_meta[sym] = meta
                            self._save_meta()

            # === BREAKEVEN STOP (persisted) ===
            breakeven_triggered = False
            if is_long:
                profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            else:
                profit_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0

            # Arm breakeven when profit hits 0.75R (persist so it survives restarts)
            if profit_pct >= 0.015 and not meta.get("breakeven_armed", False):
                meta["breakeven_armed"] = True
                self.position_meta[sym] = meta
                self._save_meta()
                log.info(f"BREAKEVEN ARMED: {sym} — profit reached {profit_pct:.1%}")

            if meta.get("breakeven_armed", False):
                if is_long and current_price <= entry_price:
                    log.info(f"BREAKEVEN STOP: {sym} (long) — was in profit, now at/below entry")
                    to_close.append(sym)
                    breakeven_triggered = True
                elif not is_long and current_price >= entry_price:
                    log.info(f"BREAKEVEN STOP: {sym} (short) — was in profit, now at/above entry")
                    to_close.append(sym)
                    breakeven_triggered = True

            # === TRAILING STOP (direction-aware) ===
            if not breakeven_triggered and sym not in to_close:
                if is_long and current_price <= trail_price and pos["unrealized_pl"] > 0:
                    log.info(
                        f"TRAILING STOP: {sym} (long) price={current_price:.2f} "
                        f"trail={trail_price:.2f} hwm={watermark:.2f} "
                        f"P&L=${pos['unrealized_pl']:.2f}"
                    )
                    to_close.append(sym)
                elif not is_long and current_price >= trail_price and pos["unrealized_pl"] > 0:
                    log.info(
                        f"TRAILING STOP: {sym} (short) price={current_price:.2f} "
                        f"trail={trail_price:.2f} lwm={watermark:.2f} "
                        f"P&L=${pos['unrealized_pl']:.2f}"
                    )
                    to_close.append(sym)

            # === TIME-BASED STOP (cycle count + days) ===
            if not breakeven_triggered and sym not in to_close:
                # Increment check count (tracks how many cycles this position has been evaluated)
                check_count = meta.get("check_count", 0) + 1
                meta["check_count"] = check_count
                self.position_meta[sym] = meta

                # Fast time stop: no 1R profit after N checks (~10 cycles = ~50 min on 5-min)
                max_checks_no_profit = self.config["risk"].get("max_checks_no_1r", 60)
                initial_risk = meta.get("initial_risk", 0.0)
                original_qty = meta.get("original_qty", pos["qty"])
                if initial_risk > 0 and original_qty > 0:
                    risk_per_share = initial_risk / original_qty
                    if is_long:
                        current_r = (current_price - entry_price) / risk_per_share if risk_per_share > 0 else 0
                    else:
                        current_r = (entry_price - current_price) / risk_per_share if risk_per_share > 0 else 0

                    if check_count >= max_checks_no_profit and current_r < 1.0:
                        log.info(
                            f"TIME STOP: {sym} — {check_count} checks, only {current_r:.1f}R profit. "
                            f"Closing stale trade."
                        )
                        to_close.append(sym)

                # Legacy day-based stop (catches anything the cycle counter misses)
                opened_str = meta.get("opened_at", "")
                if sym not in to_close and opened_str:
                    try:
                        opened = datetime.fromisoformat(opened_str)
                        days_held = (datetime.now() - opened).days
                        if days_held >= 5 and profit_pct < 0.01:
                            log.info(
                                f"TIME STOP: {sym} — held {days_held} days "
                                f"with only {profit_pct:.1%} profit. Closing zombie trade."
                            )
                            to_close.append(sym)
                    except (ValueError, TypeError):
                        pass

        self._save_watermarks()
        return to_close, partial_exits

    def execute_partial_exits(self, partial_exits: list[dict], positions: dict):
        """Execute partial position exits (close a portion of the position)."""
        for partial in partial_exits:
            sym = partial["symbol"]
            qty = partial["qty"]
            try:
                pos = positions.get(sym)
                if not pos:
                    continue

                side = "sell" if pos.get("side", "long") == "long" else "buy"
                self.broker.submit_market_order(sym, qty, side)
                log.info(f"Partial exit executed: {side} {qty} {sym} ({partial['reason']})")

                # Record the partial as a trade
                self.tracker.record_trade(
                    symbol=sym,
                    side=pos.get("side", "long"),
                    qty=qty,
                    entry_price=pos["entry_price"],
                    exit_price=pos["current_price"],
                    reason=partial["reason"],
                    risk_dollars=self.position_meta.get(sym, {}).get("initial_risk", 0.0) * (qty / self.position_meta.get(sym, {}).get("original_qty", pos["qty"])),
                )
            except Exception as e:
                log.error(f"Failed partial exit for {sym}: {e}")

    def execute_exits(self, symbols_to_close: list[str], positions: dict = None):
        """Close positions for given symbols and record trades."""
        for sym in symbols_to_close:
            try:
                # Grab position info and meta BEFORE closing/popping
                pos = positions.get(sym) if positions else None
                meta = self.position_meta.get(sym, {})

                self.broker.close_position(sym)
                log.info(f"Closed position: {sym}")

                # Clean up both watermark dicts
                self.high_watermarks.pop(sym, None)
                self.low_watermarks.pop(sym, None)

                if pos:
                    # Determine exit reason
                    is_long = pos.get("side", "long") == "long"
                    if is_long:
                        profit_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
                    else:
                        profit_pct = (pos["entry_price"] - pos["current_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0

                    opened_str = meta.get("opened_at", "")
                    days_held = 0
                    if opened_str:
                        try:
                            days_held = (datetime.now() - datetime.fromisoformat(opened_str)).days
                        except (ValueError, TypeError):
                            pass

                    if days_held >= 5 and profit_pct < 0.01:
                        reason = "time_stop"
                    elif meta.get("breakeven_armed") and pos["unrealized_pl"] <= 0:
                        reason = "breakeven_stop"
                    elif pos["unrealized_pl"] > 0:
                        reason = "trailing_stop"
                    else:
                        reason = "stop_loss"

                    risk_dollars = meta.get("initial_risk", 0.0)
                    self.tracker.record_trade(
                        symbol=sym,
                        side=pos["side"],
                        qty=pos["qty"],
                        entry_price=pos["entry_price"],
                        exit_price=pos["current_price"],
                        reason=reason,
                        risk_dollars=risk_dollars,
                        strategies=meta.get("strategies", []),
                    )

                # Pop meta AFTER using it
                self.position_meta.pop(sym, None)
            except Exception as e:
                log.error(f"Failed to close {sym}: {e}")

        self._save_watermarks()
        self._save_meta()

    def log_portfolio_status(self, positions: dict):
        """Log current portfolio state with visual formatting."""
        if not positions:
            log.info("  [no open positions]")
            return

        total_pl = sum(p["unrealized_pl"] for p in positions.values())
        total_value = sum(p["market_value"] for p in positions.values())
        pl_icon = "+" if total_pl >= 0 else "-"

        log.info(f"  PORTFOLIO: {len(positions)} positions | "
                 f"Value ${total_value:,.0f} | P&L=${total_pl:+,.2f}")
        log.info(f"  {'Symbol':<10} {'Side':<6} {'Qty':>8} {'Entry':>10} "
                 f"{'Now':>10} {'P&L':>12} {'%':>8} {'Days':>5}")
        log.info(f"  {'------':<10} {'----':<6} {'---':>8} {'-----':>10} "
                 f"{'---':>10} {'---':>12} {'--':>8} {'----':>5}")

        for sym, pos in sorted(positions.items(), key=lambda x: x[1]["unrealized_pl"], reverse=True):
            days_held = ""
            meta = self.position_meta.get(sym, {})
            if meta.get("opened_at"):
                try:
                    d = (datetime.now() - datetime.fromisoformat(meta["opened_at"])).days
                    days_held = f"{d}"
                except (ValueError, TypeError):
                    pass

            side = pos.get("side", "long").upper()
            qty_str = f"{pos['qty']:.4f}" if isinstance(pos['qty'], float) and pos['qty'] < 10 else f"{pos['qty']:.0f}"
            pl_str = f"${pos['unrealized_pl']:+,.2f}"
            pct_str = f"{pos['unrealized_plpc']:+.1%}"

            log.info(
                f"  {sym:<10} {side:<6} {qty_str:>8} "
                f"${pos['entry_price']:>9,.2f} ${pos['current_price']:>9,.2f} "
                f"{pl_str:>12} {pct_str:>8} {days_held:>5}"
            )

    def set_position_risk(self, symbol: str, entry_price: float,
                          stop_loss: float, qty: int):
        """Record the initial risk for a position (for R-multiple tracking)."""
        risk_per_share = abs(entry_price - stop_loss)
        risk_dollars = risk_per_share * qty
        if symbol not in self.position_meta:
            self.position_meta[symbol] = {
                "opened_at": datetime.now().isoformat(),
                "entry_price": entry_price,
                "initial_risk": risk_dollars,
                "original_qty": qty,
            }
        else:
            self.position_meta[symbol]["initial_risk"] = risk_dollars
            if "original_qty" not in self.position_meta[symbol]:
                self.position_meta[symbol]["original_qty"] = qty
        self._save_meta()

    def _save_watermarks(self):
        """Persist watermarks to state file."""
        state = load_state()
        state["high_watermarks"] = self.high_watermarks
        state["low_watermarks"] = self.low_watermarks
        save_state(state)

    def _save_meta(self):
        """Persist position metadata to state file."""
        state = load_state()
        state["position_meta"] = self.position_meta
        save_state(state)
