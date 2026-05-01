"""Volume surge gate — filter entries by bar-volume conviction.

Signal bar volume vs 20-bar trailing avg (ex last bar):
  ratio < 0.5:  weak            → block entry (no conviction)
  0.5–1.2:      below average   → size 0.80x
  1.2–1.8:      normal          → no change
  >= 1.8:       surge           → size 1.15x (strong conviction)

Rationale: breakouts/signals without volume expansion have ~35% lower
hit rate historically. This gate filters out low-conviction setups.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class VolumeSignal:
    ratio: float = 1.0
    bucket: str = "normal"   # weak / below / normal / surge
    size_mult: float = 1.0
    block: bool = False


class VolumeGate:
    def __init__(self, config: dict):
        cfg = config.get("edge", {}) or {}
        self.enabled: bool = bool(cfg.get("volume_gate", True))
        self.lookback: int = int(cfg.get("volume_lookback_bars", 20))
        self.weak_threshold: float = float(cfg.get("volume_weak_ratio", 0.5))
        self.below_threshold: float = float(cfg.get("volume_below_ratio", 1.2))
        self.surge_threshold: float = float(cfg.get("volume_surge_ratio", 1.8))
        self.below_size_mult: float = float(cfg.get("volume_below_mult", 0.80))
        self.surge_size_mult: float = float(cfg.get("volume_surge_mult", 1.15))
        self.block_on_weak: bool = bool(cfg.get("volume_block_on_weak", True))

    def evaluate(self, bars: pd.DataFrame | None) -> VolumeSignal:
        if not self.enabled:
            return VolumeSignal()
        if bars is None or "volume" not in bars.columns or len(bars) < self.lookback + 1:
            return VolumeSignal()

        try:
            current_vol = float(bars["volume"].iloc[-1])
            # Trailing avg excludes current bar
            avg_vol = float(bars["volume"].iloc[-(self.lookback + 1):-1].mean())
        except (IndexError, ValueError):
            return VolumeSignal()

        if avg_vol <= 0:
            return VolumeSignal()

        ratio = current_vol / avg_vol

        if ratio < self.weak_threshold:
            return VolumeSignal(
                ratio=ratio,
                bucket="weak",
                size_mult=0.5,
                block=self.block_on_weak,
            )
        if ratio < self.below_threshold:
            return VolumeSignal(
                ratio=ratio,
                bucket="below",
                size_mult=self.below_size_mult,
            )
        if ratio < self.surge_threshold:
            return VolumeSignal(ratio=ratio, bucket="normal", size_mult=1.0)
        return VolumeSignal(
            ratio=ratio,
            bucket="surge",
            size_mult=self.surge_size_mult,
        )
