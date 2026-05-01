"""Relative strength rank edge.

Ranks the active stock universe by 20-day total return. Top quintile
(strongest names) get long-size boost; bottom quintile get long-block
(let them mean-revert or be shorted).

Classic momentum anomaly — Jegadeesh-Titman 1993, still robust.

Cache: full universe ranking refreshed every `rs_ttl_sec` (default 1h).
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RelativeStrengthSignal:
    rank_pct: float = 0.5        # 0.0 = weakest, 1.0 = strongest
    bucket: str = "mid"          # top / upper / mid / lower / bottom
    long_size_mult: float = 1.0
    block_long: bool = False
    allow_short: bool = False


class RelativeStrength:
    def __init__(self, data_fetcher, config: dict):
        cfg = config.get("edge", {}) or {}
        self.data = data_fetcher
        self.enabled: bool = bool(cfg.get("relative_strength", True))
        self.lookback: int = int(cfg.get("rs_lookback_days", 20))
        self.ttl_sec: int = int(cfg.get("rs_ttl_sec", 3600))
        self.top_boost: float = float(cfg.get("rs_top_boost", 1.20))
        self.bottom_penalty: float = float(cfg.get("rs_bottom_penalty", 0.60))
        self._ranks: dict[str, float] = {}  # symbol → rank in [0,1]
        self._computed_at: float = 0.0

    def evaluate(self, symbol: str, universe: list[str]) -> RelativeStrengthSignal:
        if not self.enabled:
            return RelativeStrengthSignal()
        if time.time() - self._computed_at > self.ttl_sec:
            self._recompute(universe)

        rank = self._ranks.get(symbol.upper(), 0.5)

        if rank >= 0.80:
            return RelativeStrengthSignal(
                rank_pct=rank,
                bucket="top",
                long_size_mult=self.top_boost,
                block_long=False,
                allow_short=False,
            )
        if rank >= 0.60:
            return RelativeStrengthSignal(
                rank_pct=rank,
                bucket="upper",
                long_size_mult=1.10,
                block_long=False,
                allow_short=False,
            )
        if rank >= 0.40:
            return RelativeStrengthSignal(
                rank_pct=rank, bucket="mid",
            )
        if rank >= 0.20:
            return RelativeStrengthSignal(
                rank_pct=rank,
                bucket="lower",
                long_size_mult=0.80,
                block_long=False,
                allow_short=True,
            )
        return RelativeStrengthSignal(
            rank_pct=rank,
            bucket="bottom",
            long_size_mult=self.bottom_penalty,
            block_long=True,
            allow_short=True,
        )

    # ── helpers ────────────────────────────────────────────

    def _recompute(self, universe: list[str]):
        """Fetch daily bars for universe, compute 20d returns, rank."""
        try:
            bars = self.data.get_bars(universe, timeframe="1Day", days=self.lookback + 5)
        except Exception:
            return

        returns: dict[str, float] = {}
        for sym in universe:
            df = bars.get(sym)
            if df is None or len(df) < self.lookback + 1:
                continue
            try:
                start = float(df["close"].iloc[-(self.lookback + 1)])
                end = float(df["close"].iloc[-1])
                if start <= 0:
                    continue
                returns[sym.upper()] = (end / start) - 1.0
            except Exception:
                continue

        if not returns:
            return

        # Rank: ascending → percentile in [0, 1]
        sorted_syms = sorted(returns.items(), key=lambda kv: kv[1])
        n = len(sorted_syms)
        self._ranks = {
            sym: (i / (n - 1)) if n > 1 else 0.5
            for i, (sym, _) in enumerate(sorted_syms)
        }
        self._computed_at = time.time()
