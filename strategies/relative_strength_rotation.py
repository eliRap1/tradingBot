"""Cross-sectional relative strength rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import setup_logger

log = setup_logger("strategy.relative_strength_rotation")


class RelativeStrengthRotationStrategy:
    """Rank symbols by multi-horizon momentum and favor the leaders."""

    def __init__(self, config: dict):
        self.cfg = config.get("strategies", {}).get("relative_strength_rotation", {})
        self.lookbacks = [int(x) for x in self.cfg.get("lookbacks", [20, 60, 126])]
        self.weights = [float(x) for x in self.cfg.get("lookback_weights", [0.50, 0.30, 0.20])]
        self.top_pct = float(self.cfg.get("top_pct", 0.25))
        self.bottom_pct = float(self.cfg.get("bottom_pct", 0.20))
        self.min_abs_momentum = float(self.cfg.get("min_abs_momentum", 0.02))
        self.allow_shorts = bool(self.cfg.get("allow_shorts", True))

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        scores = {}
        raw = {}
        for sym, df in bars.items():
            try:
                value = self._momentum_score(df)
                if value is not None:
                    raw[sym] = value
            except Exception as e:
                log.error(f"Error analyzing {sym}: {e}")
        if not raw:
            return scores

        if len(raw) == 1:
            sym, value = next(iter(raw.items()))
            if value > self.min_abs_momentum:
                scores[sym] = min(1.0, 0.25 + value)
            elif self.allow_shorts and value < -self.min_abs_momentum:
                scores[sym] = max(-1.0, -0.25 + value)
            return scores

        values = sorted(raw.values())
        market_score = float(np.nanmean(values))
        for sym, value in raw.items():
            rank = sum(1 for x in values if x <= value) / len(values)
            if rank >= 1.0 - self.top_pct and value > self.min_abs_momentum and market_score > -0.02:
                scores[sym] = min(1.0, 0.35 + rank * 0.45 + min(0.20, value))
            elif (
                self.allow_shorts
                and rank <= self.bottom_pct
                and value < -self.min_abs_momentum
                and market_score < 0.02
            ):
                scores[sym] = max(-1.0, -0.35 - (1.0 - rank) * 0.45 + max(-0.20, value))
        return scores

    def _momentum_score(self, df: pd.DataFrame) -> float | None:
        if df is None or len(df) < min(self.lookbacks) + 2:
            return None
        close = df["close"].astype(float)
        total = 0.0
        weight_sum = 0.0
        for lookback, weight in zip(self.lookbacks, self.weights):
            if len(close) <= lookback:
                continue
            total += ((close.iloc[-1] / close.iloc[-lookback - 1]) - 1.0) * weight
            weight_sum += weight
        if weight_sum <= 0:
            return None
        return total / weight_sum
