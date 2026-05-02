"""Short-interest squeeze detector.

Flags squeeze candidates: high SI%float + days-to-cover + ATR
expansion + positive RS. Used to:
  - Boost long size on confirmed squeeze setups
  - Block shorts on names with squeeze risk

Free data: FINRA short-interest reports (biweekly), FINRA threshold
list, and quarterly EDGAR. Bot caches per-symbol with 24h TTL.

Inputs (per evaluate call):
  - daily_bars: pd.DataFrame for ATR + price
  - rs_signal: optional RS bucket from edge.relative_strength
  - si_data: dict {symbol: {short_pct_float, days_to_cover}} (caller-supplied)

Output:
  SqueezeSignal with size_mult and block_short flag.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SqueezeSignal:
    candidate: bool = False
    short_pct_float: float = 0.0
    days_to_cover: float = 0.0
    atr_expansion: float = 0.0
    rs_bucket: str = ""
    block_short: bool = False
    long_size_mult: float = 1.0
    reason: str = ""


class ShortInterestEdge:
    def __init__(self, config: dict):
        self.config = config.get("edge", {}).get("short_interest", {})
        self.enabled = bool(self.config.get("enabled", True))
        self.si_threshold = float(self.config.get("si_threshold_pct", 20.0))
        self.dtc_threshold = float(self.config.get("dtc_threshold_days", 5.0))
        self.atr_expansion_threshold = float(self.config.get("atr_expansion_min", 1.3))
        self.size_mult = float(self.config.get("size_mult", 1.20))
        self.block_short = bool(self.config.get("block_short_on_candidate", True))
        self.cache_path = self.config.get(
            "cache_path",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "research", "short_interest_cache.json"),
        )
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            self._cache = {}
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f) or {}
        except Exception:
            self._cache = {}

    def update_cache(self, symbol: str, short_pct_float: float, days_to_cover: float):
        """Caller pulls from FINRA / EDGAR / data provider, writes to cache."""
        self._cache[symbol.upper()] = {
            "short_pct_float": float(short_pct_float),
            "days_to_cover": float(days_to_cover),
            "updated_at": time.time(),
        }
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _atr_expansion(df: pd.DataFrame, recent_window: int = 5,
                      lookback_window: int = 20) -> float:
        if df is None or len(df) < lookback_window + 1:
            return 0.0
        tr = (df["high"] - df["low"]).astype(float)
        recent = tr.tail(recent_window).mean()
        avg = tr.tail(lookback_window).mean()
        if avg <= 0:
            return 0.0
        return float(recent / avg)

    def evaluate(self, symbol: str, daily_bars: pd.DataFrame | None = None,
                 rs_bucket: str = "", si_data: dict | None = None) -> SqueezeSignal:
        if not self.enabled:
            return SqueezeSignal(reason="short_interest disabled")
        sym = symbol.upper()
        si = (si_data or {}).get(sym) or self._cache.get(sym)
        if not si:
            return SqueezeSignal(reason=f"no SI data for {sym}")
        spct = float(si.get("short_pct_float", 0.0))
        dtc = float(si.get("days_to_cover", 0.0))
        if spct < self.si_threshold or dtc < self.dtc_threshold:
            return SqueezeSignal(
                short_pct_float=spct, days_to_cover=dtc,
                reason=f"SI/DTC below thresholds (SI={spct:.1f}%, DTC={dtc:.1f}d)",
            )
        atr_exp = self._atr_expansion(daily_bars) if daily_bars is not None else 0.0
        rs_ok = rs_bucket in ("strong", "moderate")
        if atr_exp >= self.atr_expansion_threshold and rs_ok:
            return SqueezeSignal(
                candidate=True,
                short_pct_float=spct, days_to_cover=dtc,
                atr_expansion=atr_exp, rs_bucket=rs_bucket,
                block_short=self.block_short,
                long_size_mult=self.size_mult,
                reason=(
                    f"squeeze: SI={spct:.1f}% DTC={dtc:.1f}d "
                    f"ATR_exp={atr_exp:.2f} RS={rs_bucket}"
                ),
            )
        return SqueezeSignal(
            short_pct_float=spct, days_to_cover=dtc,
            atr_expansion=atr_exp, rs_bucket=rs_bucket,
            block_short=self.block_short,  # still block shorts on heavy SI even if no squeeze
            reason=f"high SI ({spct:.1f}%) but no squeeze trigger (ATR_exp={atr_exp:.2f}, RS={rs_bucket})",
        )
