"""
Portfolio management with pro-grade exit logic.

Improvements:
- Trailing stops (persisted across restarts)
- Time-based stops (close zombie trades after 5 days with <0.5R profit)
- Breakeven stop (move stop to entry after 0.75R profit)
- Trade recording for performance tracking
"""

from datetime import datetime
from utils import setup_logger
from state import load_state, save_state
from tracker import TradeTracker

log = setup_logger("portfolio")


class PortfolioManager:
    def __init__(self, config: dict, broker):
        self.config = config
        self.broker = broker
        self.tracker = TradeTracker()

        # Restore watermarks from persisted state
        state = load_state()
        self.high_watermarks = state.get("high_watermarks", {})

        # Track when positions were opened and their initial risk
        # {symbol: {"opened_at": iso_str, "entry_price": float, "initial_risk": float}}
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
        """Check trailing stops, partial exits, return symbols to close and partial exits."""
        trail_pct = self.config["risk"]["trailing_stop_pct"]
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

            # Update high watermark
            if sym not in self.high_watermarks:
                self.high_watermarks[sym] = max(entry_price, current_price)
            else:
                self.high_watermarks[sym] = max(self.high_watermarks[sym], current_price)

            hwm = self.high_watermarks[sym]
            trail_price = hwm * (1 - trail_pct)

            # === PARTIAL EXIT at 1.5R ===
            meta = self.position_meta.get(sym, {})
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
                            f"PARTIAL EXIT: {sym} — {current_r:.1f}R reached, "
                            f"closing {close_qty}/{qty} shares ({partial_pct:.0%})"
                        )
                        # Mark partial done so we don't repeat
                        meta["partial_done"] = True
                        self.position_meta[sym] = meta
                        self._save_meta()

            # === BREAKEVEN STOP ===
            profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            breakeven_triggered = False
            if profit_pct >= 0.015:  # ~0.75R for typical 2% risk
                if current_price <= entry_price and pos["unrealized_pl"] <= 0:
                    log.info(f"BREAKEVEN STOP: {sym} — was in profit, now back at entry")
                    to_close.append(sym)
                    breakeven_triggered = True

            # === TRAILING STOP ===
            if not breakeven_triggered and current_price <= trail_price and pos["unrealized_pl"] > 0:
                log.info(
                    f"TRAILING STOP: {sym} price={current_price:.2f} "
                    f"trail={trail_price:.2f} hwm={hwm:.2f} P&L=${pos['unrealized_pl']:.2f}"
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
                # Grab position info before closing so we can record the trade
                pos = positions.get(sym) if positions else None
                self.broker.close_position(sym)
                log.info(f"Closed position: {sym}")
                self.high_watermarks.pop(sym, None)
                self.position_meta.pop(sym, None)

                if pos:
                    # Determine exit reason
                    reason = "exit"
                    profit_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
                    meta = self.position_meta.get(sym, {})
                    opened_str = meta.get("opened_at", "")
                    days_held = 0
                    if opened_str:
                        try:
                            days_held = (datetime.now() - datetime.fromisoformat(opened_str)).days
                        except (ValueError, TypeError):
                            pass

                    if days_held >= 5 and profit_pct < 0.01:
                        reason = "time_stop"
                    elif pos["unrealized_pl"] <= 0 and profit_pct >= -0.005:
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
        """Persist high watermarks to state file."""
        state = load_state()
        state["high_watermarks"] = self.high_watermarks
        save_state(state)

    def _save_meta(self):
        """Persist position metadata to state file."""
        state = load_state()
        state["position_meta"] = self.position_meta
        save_state(state)
