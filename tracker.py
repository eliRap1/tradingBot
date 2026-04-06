"""
Trade tracker — logs closed trades and calculates institutional-grade performance stats.

Metrics: Win%, Sharpe ratio, expectancy, R-multiples, profit factor, max drawdown,
rolling Sharpe, Calmar ratio.
"""

import json
import os
import math
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
                     reason: str = "", risk_dollars: float = 0.0,
                     strategies: list[str] | None = None):
        """Record a completed (closed) trade."""
        is_long = side in ("buy", "long")
        pnl = (exit_price - entry_price) * qty if is_long else \
              (entry_price - exit_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price if is_long and entry_price else \
                  (entry_price - exit_price) / entry_price if entry_price else 0.0

        # R-multiple: how many R did this trade return
        r_multiple = pnl / risk_dollars if risk_dollars > 0 else None

        trade = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "r_multiple": round(r_multiple, 2) if r_multiple is not None else None,
            "risk_dollars": round(risk_dollars, 2) if risk_dollars > 0 else None,
            "reason": reason,
            "strategies": strategies or [],
            "closed_at": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._save()

        r_str = f" R={r_multiple:+.2f}" if r_multiple is not None else ""
        tag = "WIN" if pnl >= 0 else "LOSS"
        log.info(f"[{tag}] {symbol}: {side} {qty} shares | "
                 f"entry=${entry_price:.2f} exit=${exit_price:.2f} | "
                 f"P&L=${pnl:+.2f} ({pnl_pct:+.2%}){r_str} | {reason}")

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Calculate institutional-grade performance statistics."""
        if not self.trades:
            return {}

        pnls = [t["pnl"] for t in self.trades]
        pnl_pcts = [t["pnl_pct"] for t in self.trades]
        wins = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        n = len(pnls)
        win_rate = len(wins) / n
        loss_rate = 1 - win_rate

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # ── Sharpe Ratio (annualized) ────────────────────────
        mean_ret = sum(pnl_pcts) / n
        if n > 1:
            variance = sum((r - mean_ret) ** 2 for r in pnl_pcts) / (n - 1)
            std_ret = math.sqrt(variance)
            sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # ── Rolling Sharpe (last 20 trades) ──────────────────
        rolling_sharpe = 0.0
        if n >= 20:
            recent = pnl_pcts[-20:]
            r_mean = sum(recent) / 20
            r_var = sum((r - r_mean) ** 2 for r in recent) / 19
            r_std = math.sqrt(r_var)
            rolling_sharpe = (r_mean / r_std) * math.sqrt(252) if r_std > 0 else 0.0

        # ── Expectancy ($ per trade) ─────────────────────────
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # ── R-Expectancy (avg R-multiple) ────────────────────
        r_multiples = [t["r_multiple"] for t in self.trades
                       if t.get("r_multiple") is not None]
        r_expectancy = sum(r_multiples) / len(r_multiples) if r_multiples else None

        # ── Max Drawdown (from trade equity curve) ───────────
        equity_curve = []
        running = 0.0
        for pnl in pnls:
            running += pnl
            equity_curve.append(running)

        peak = 0.0
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        # ── Calmar Ratio (annual return / max drawdown) ──────
        calmar = 0.0
        if max_dd > 0 and n >= 5:
            # Approximate annualized return
            avg_daily_pnl = total_pnl / n
            annual_pnl = avg_daily_pnl * 252
            calmar = annual_pnl / max_dd

        # ── Consecutive wins/losses ──────────────────────────
        max_consec_wins = 0
        max_consec_losses = 0
        cw = cl = 0
        for p in pnls:
            if p >= 0:
                cw += 1
                cl = 0
            else:
                cl += 1
                cw = 0
            max_consec_wins = max(max_consec_wins, cw)
            max_consec_losses = max(max_consec_losses, cl)

        # ── APR (Annualized Percentage Return) ────────────
        apr = 0.0
        try:
            first_date = self.trades[0].get("closed_at", "")
            last_date = self.trades[-1].get("closed_at", "")
            if first_date and last_date:
                d1 = datetime.fromisoformat(first_date)
                d2 = datetime.fromisoformat(last_date)
                days_active = max(1, (d2 - d1).days)
                # APR = (total_pnl / initial_equity_estimate) * (365 / days)
                # Use total_pnl relative to peak equity as proxy
                equity_base = max(abs(total_pnl) * 5, 100000)  # rough estimate
                apr = (total_pnl / equity_base) * (365 / days_active) * 100
        except (ValueError, TypeError, IndexError):
            pass

        stats = {
            "total_trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_pct": round(win_rate * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / n, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(max(wins), 2) if wins else 0.0,
            "largest_loss": round(min(losses), 2) if losses else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            # Institutional metrics
            "sharpe_ratio": round(sharpe, 2),
            "rolling_sharpe_20": round(rolling_sharpe, 2),
            "expectancy": round(expectancy, 2),
            "r_expectancy": round(r_expectancy, 2) if r_expectancy is not None else None,
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 2),
            "apr": round(apr, 1),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
        }
        return stats

    def get_strategy_alpha_decay(self, lookback: int = 20, min_trades: int = 5) -> dict[str, float]:
        """
        Compute per-strategy alpha decay factor.

        Compares each strategy's win rate in the last `lookback` trades vs its
        overall win rate. Returns a multiplier per strategy:
          >1.0 = strategy is hot (recent alpha above average)
          1.0  = no data / no change
          <1.0 = strategy is decaying (reduce weight)

        The multiplier is clamped to [0.3, 1.5] to avoid extreme swings.
        """
        if len(self.trades) < min_trades:
            return {}

        # Build per-strategy trade outcomes
        strat_all: dict[str, list[bool]] = {}
        for t in self.trades:
            for s in t.get("strategies", []):
                strat_all.setdefault(s, []).append(t["pnl"] >= 0)

        decay_factors = {}
        for strat, outcomes in strat_all.items():
            if len(outcomes) < min_trades:
                decay_factors[strat] = 1.0
                continue

            overall_wr = sum(outcomes) / len(outcomes)
            recent = outcomes[-lookback:]
            recent_wr = sum(recent) / len(recent)

            # Decay factor: recent / overall (clamped)
            if overall_wr > 0:
                factor = recent_wr / overall_wr
            else:
                factor = 1.0 if recent_wr == 0 else 1.5

            decay_factors[strat] = max(0.3, min(1.5, factor))

        return decay_factors

    def get_strategy_kelly(self, strategies: list[str], min_trades: int = 10) -> float | None:
        """
        Compute Kelly fraction for a specific set of strategies.

        Looks at trades where ANY of the given strategies contributed,
        computes their win rate and avg win/loss ratio, returns half-Kelly.

        Returns None if insufficient data.
        """
        if not strategies:
            return None

        strat_set = set(strategies)
        relevant = [t for t in self.trades
                    if strat_set.intersection(t.get("strategies", []))]

        if len(relevant) < min_trades:
            return None

        wins = [t for t in relevant if t["pnl"] >= 0]
        losses = [t for t in relevant if t["pnl"] < 0]

        if not wins or not losses:
            return None

        p = len(wins) / len(relevant)
        avg_win = sum(t["pnl"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))

        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        b = min(b, 5.0)  # cap

        kelly_f = p - ((1 - p) / b) if b > 0 else 0
        kelly_f *= 0.5  # half-Kelly for safety

        return kelly_f if kelly_f > 0 else None

    def get_win_rate(self) -> float:
        """Quick access to win rate for Kelly sizing."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["pnl"] >= 0)
        return wins / len(self.trades)

    def get_avg_win_loss_ratio(self) -> float:
        """Quick access to avg win / avg loss ratio for Kelly sizing."""
        wins = [t["pnl"] for t in self.trades if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self.trades if t["pnl"] < 0]
        if not wins or not losses:
            return 0.0
        return (sum(wins) / len(wins)) / (sum(losses) / len(losses))

    def log_stats(self):
        """Print a performance summary to the log."""
        stats = self.get_stats()
        if not stats:
            log.info("No closed trades yet")
            return

        w = stats['wins']
        l = stats['losses']
        t = stats['total_trades']
        bar_len = 20
        w_bar = int((w / t) * bar_len) if t > 0 else 0
        l_bar = bar_len - w_bar
        win_bar = "=" * w_bar + "-" * l_bar

        log.info(f"  STATS: {t} trades [{win_bar}] {stats['win_pct']}% win")
        log.info(f"    P&L: ${stats['total_pnl']:+,.2f}  |  "
                 f"Avg: ${stats['avg_pnl']:+,.2f}  |  "
                 f"PF: {stats['profit_factor']}  |  "
                 f"Sharpe: {stats['sharpe_ratio']}")
        log.info(f"    W/L: ${stats['avg_win']:+,.2f} / ${stats['avg_loss']:+,.2f}  |  "
                 f"Best: ${stats['largest_win']:+,.2f}  |  "
                 f"Worst: ${stats['largest_loss']:+,.2f}")
        if stats.get("r_expectancy") is not None:
            log.info(f"    R-Exp: {stats['r_expectancy']:+.2f}R  |  "
                     f"Sharpe(20): {stats['rolling_sharpe_20']}")
        log.info(f"    DD: ${stats['max_drawdown']:,.2f}  |  "
                 f"Calmar: {stats['calmar_ratio']}  |  "
                 f"Streak: {stats['max_consecutive_wins']}W / {stats['max_consecutive_losses']}L")

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
