"""Per-symbol gap filter.

Detects overnight gaps from daily bars and classifies:
- flat:    |gap| <= 0.5%          → no modulation
- normal:  0.5% < |gap| <= 3%     → continuation-friendly (1.0x)
- large:   3% < |gap| <= 5%       → reduce size (0.70x)
- extreme: |gap| > 5%             → block entry (exhaustion risk)

Direction-aware: large positive gap hurts long entries (chase risk) but
helps short entries (mean reversion bet). Extreme gaps block either side.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GapSignal:
    gap_pct: float = 0.0
    category: str = "flat"  # flat / normal / large / extreme
    size_mult: float = 1.0
    block: bool = False


class GapFilter:
    def __init__(self, config: dict):
        cfg = config.get("edge", {}) or {}
        self.normal_threshold = float(cfg.get("gap_normal_pct", 0.005))
        self.large_threshold = float(cfg.get("gap_large_pct", 0.03))
        self.extreme_threshold = float(cfg.get("gap_extreme_pct", 0.05))
        self.large_size_mult = float(cfg.get("gap_large_size_mult", 0.70))
        self.enabled: bool = bool(cfg.get("gap_filter", True))

    def evaluate(self, daily_bars: pd.DataFrame | None, side: str = "buy") -> GapSignal:
        """Evaluate gap for one symbol given daily bars (descending or ascending
        by date — last row is today). `side` is 'buy' or 'sell'.
        """
        if not self.enabled:
            return GapSignal()
        if daily_bars is None or len(daily_bars) < 2:
            return GapSignal()

        try:
            prev_close = float(daily_bars["close"].iloc[-2])
            today_open = float(daily_bars["open"].iloc[-1])
        except (KeyError, IndexError, ValueError):
            return GapSignal()

        if prev_close <= 0:
            return GapSignal()

        gap = (today_open - prev_close) / prev_close
        abs_gap = abs(gap)

        if abs_gap <= self.normal_threshold:
            return GapSignal(gap_pct=gap, category="flat", size_mult=1.0, block=False)

        if abs_gap <= self.large_threshold:
            return GapSignal(gap_pct=gap, category="normal", size_mult=1.0, block=False)

        if abs_gap <= self.extreme_threshold:
            # Large gap: penalize only entries that chase the gap direction
            chase = (gap > 0 and side == "buy") or (gap < 0 and side == "sell")
            mult = self.large_size_mult if chase else 1.0
            return GapSignal(gap_pct=gap, category="large", size_mult=mult, block=False)

        # Extreme: exhaustion risk either direction
        return GapSignal(gap_pct=gap, category="extreme", size_mult=0.0, block=True)
