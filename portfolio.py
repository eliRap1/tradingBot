"""
Portfolio management with pro-grade exit logic.

Improvements:
- Direction-aware trailing stops (long: high watermark, short: low watermark)
- Time-based stops (close zombie trades after 5 days with <0.5R profit)
- Breakeven stop (move stop to entry after 0.75R profit, persisted)
- Partial exit scaling (close 50% at 1.5R)
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
                "qty": int(pos.qty),
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
                              prices: dict[str, float]) -> tuple[list[str], list[dict]]:
        """Check trailing stops, partial exits, return symbols to close and partial exits.

        Direction-aware:
        - Long: high watermark tracks price peaks, trail triggers on drop from peak
        - Short: low watermark tracks price troughs, trail triggers on rise from trough
        """
        default_trail_pct = self.config["risk"]["trailing_stop_pct"]
        crypto_trail_pct = self.config.get("screener", {}).get(
            "crypto_risk", {}).get("trailing_stop_pct", default_trail_pct)
        to_close = []
        partial_exits = []  # [{"symbol": str, "qty": int, "reason": str}]

        # Partial exit config
        partial_enabled = self.config["risk"].get("partial_exit_enabled", False)
        partial_r = self.config["risk"].get("partial_exit_r", 1.5)
        partial_pct = self.config["risk"].get("partial_exit_pct", 0.50)

        for sym, pos in positions.items():
            if sym not in prices:
                continue

            current_price = prices[sym]
            entry_price = pos["entry_price"]
            is_long = pos.get("side", "long") == "long"
            meta = self.position_meta.get(sym, {})
            trail_pct = crypto_trail_pct if sym in CRYPTO_SYMBOLS else default_trail_pct

            # === DIRECTION-AWARE WATERMARK ===
            if is_long:
                if sym not in self.high_watermarks:
                    self.high_watermarks[sym] = max(entry_price, current_price)
                else:
                    self.high_watermarks[sym] = max(self.high_watermarks[sym], current_price)
                watermark = self.high_watermarks[sym]
                trail_price = watermark * (1 - trail_pct)
            else:
                # Short: track the lowest price (best for short)
                if sym not in self.low_watermarks:
                    self.low_watermarks[sym] = min(entry_price, current_price)
                else:
                    self.low_watermarks[sym] = min(self.low_watermarks[sym], current_price)
                watermark = self.low_watermarks[sym]
                trail_price = watermark * (1 + trail_pct)

            # === PARTIAL EXIT at 1.2R (first partial) ===
            if partial_enabled and not meta.get("partial_done", False):
                initial_risk = meta.get("initial_risk", 0.0)
                qty = pos["qty"]
                if initial_risk > 0 and qty > 1:
                    risk_per_share = initial_risk / qty
                    if is_long:
                        current_r = (current_price - entry_price) / risk_per_share if risk_per_share > 0 else 0
                    else:
                        current_r = (entry_price - current_price) / risk_per_share if risk_per_share > 0 else 0

                    if current_r >= partial_r:
                        close_qty = max(1, int(qty * partial_pct))
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
                    # Recalculate risk per share with remaining quantity
                    original_qty = meta.get("original_qty", qty * 2)  # Estimate original qty
                    risk_per_share = initial_risk / original_qty if original_qty > 0 else 0
                    if is_long:
                        current_r = (current_price - entry_price) / risk_per_share if risk_per_share > 0 else 0
                    else:
                        current_r = (entry_price - current_price) / risk_per_share if risk_per_share > 0 else 0

                    if current_r >= second_partial_r:
                        close_qty = max(1, int(qty * second_partial_pct))
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

            # === TIME-BASED STOP ===
            if not breakeven_triggered and sym not in to_close:
                opened_str = meta.get("opened_at", "")
                if opened_str:
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
                    risk_dollars=self.position_meta.get(sym, {}).get("initial_risk", 0.0) * (qty / pos["qty"]),
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
                    )

                # Pop meta AFTER using it
                self.position_meta.pop(sym, None)
            except Exception as e:
                log.error(f"Failed to close {sym}: {e}")

        self._save_watermarks()
        self._save_meta()

    def log_portfolio_status(self, positions: dict):
        """Log current portfolio state."""
        if not positions:
            log.info("No open positions")
            return

        total_pl = sum(p["unrealized_pl"] for p in positions.values())
        total_value = sum(p["market_value"] for p in positions.values())

        log.info(f"--- Portfolio: {len(positions)} positions | "
                 f"Value=${total_value:,.2f} | P&L=${total_pl:+,.2f} ---")
        for sym, pos in positions.items():
            days_held = ""
            meta = self.position_meta.get(sym, {})
            if meta.get("opened_at"):
                try:
                    d = (datetime.now() - datetime.fromisoformat(meta["opened_at"])).days
                    days_held = f" | {d}d"
                except (ValueError, TypeError):
                    pass

            log.info(
                f"  {sym}: {pos['qty']} shares @ ${pos['entry_price']:.2f} | "
                f"Now ${pos['current_price']:.2f} | "
                f"P&L=${pos['unrealized_pl']:+.2f} ({pos['unrealized_plpc']:+.1%}){days_held}"
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
            }
        else:
            self.position_meta[symbol]["initial_risk"] = risk_dollars
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
