"""Trade tracker — logs closed trades and calculates performance stats."""

import json
import os
from datetime import datetime
from utils import setup_logger

log = setup_logger("tracker")

TRADES_FILE = os.path.join(os.path.dirname(__file__), "trades.json")


class TradeTracker:
    def __init__(self):
        self.trades = self._load()
        if self.trades:
            log.info(f"Loaded {len(self.trades)} historical trades")

    # ── Recording ────────────────────────────────────────────

    def record_trade(self, symbol: str, side: str, qty: int,
                     entry_price: float, exit_price: float,
                     reason: str = ""):
        """Record a completed (closed) trade."""
        pnl = (exit_price - entry_price) * qty if side == "buy" else \
              (entry_price - exit_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0

        trade = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "reason": reason,
            "closed_at": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._save()

        tag = "WIN" if pnl >= 0 else "LOSS"
        log.info(f"[{tag}] {symbol}: {side} {qty} shares | "
                 f"entry=${entry_price:.2f} exit=${exit_price:.2f} | "
                 f"P&L=${pnl:+.2f} ({pnl_pct:+.2%}) | {reason}")

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Calculate performance statistics from all recorded trades."""
        if not self.trades:
            return {}

        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        stats = {
            "total_trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_pct": round(len(wins) / len(pnls) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(pnls), 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
            "largest_win": round(max(wins), 2) if wins else 0.0,
            "largest_loss": round(min(losses), 2) if losses else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        }
        return stats

    def log_stats(self):
        """Print a performance summary to the log."""
        stats = self.get_stats()
        if not stats:
            log.info("No closed trades yet")
            return

        log.info("=== PERFORMANCE STATS ===")
        log.info(f"  Trades: {stats['total_trades']} | "
                 f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
                 f"Win%: {stats['win_pct']}%")
        log.info(f"  Total P&L: ${stats['total_pnl']:+,.2f} | "
                 f"Avg P&L: ${stats['avg_pnl']:+,.2f}")
        log.info(f"  Avg Win: ${stats['avg_win']:+,.2f} | "
                 f"Avg Loss: ${stats['avg_loss']:+,.2f}")
        log.info(f"  Largest Win: ${stats['largest_win']:+,.2f} | "
                 f"Largest Loss: ${stats['largest_loss']:+,.2f}")
        log.info(f"  Profit Factor: {stats['profit_factor']}")

    # ── Persistence ──────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not os.path.exists(TRADES_FILE):
            return []
        try:
            with open(TRADES_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load trades file: {e}")
            return []

    def _save(self):
        try:
            with open(TRADES_FILE, "w") as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save trades: {e}")
