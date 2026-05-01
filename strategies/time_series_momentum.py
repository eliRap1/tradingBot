"""Multi-horizon time-series momentum strategy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils import setup_logger

log = setup_logger("strategy.time_series_momentum")


class TimeSeriesMomentumStrategy:
    """Trend persistence across 20/60/120 bar lookbacks."""

    def __init__(self, config: dict):
        self.cfg = config.get("strategies", {}).get("time_series_momentum", {})
        self.lookbacks = [int(x) for x in self.cfg.get("lookbacks", [20, 60, 120])]
        self.weights = [float(x) for x in self.cfg.get("lookback_weights", [0.50, 0.30, 0.20])]
        self.ema_fast = int(self.cfg.get("ema_fast", 20))
        self.ema_slow = int(self.cfg.get("ema_slow", 100))
        self.min_abs_score = float(self.cfg.get("min_abs_score", 0.15))
        self.max_realized_vol = float(self.cfg.get("max_realized_vol", 1.20))

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals = {}
        for sym, df in bars.items():
            try:
                score = self._analyze(df)
                if abs(score) >= self.min_abs_score:
                    signals[sym] = score
            except Exception as e:
                log.error(f"Error analyzing {sym}: {e}")
        return signals

    def _analyze(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < max(30, min(self.lookbacks) + 2):
            return 0.0
        close = df["close"].astype(float)
        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return 0.0
        realized_vol = float(returns.tail(20).std() * math.sqrt(252))
        if not np.isfinite(realized_vol) or realized_vol <= 0 or realized_vol > self.max_realized_vol:
            return 0.0

        total = 0.0
        weight_sum = 0.0
        for lookback, weight in zip(self.lookbacks, self.weights):
            if len(close) <= lookback:
                continue
            ret = (close.iloc[-1] / close.iloc[-lookback - 1]) - 1.0
            horizon_vol = max(realized_vol * math.sqrt(lookback / 252), 1e-6)
            total += max(-1.0, min(1.0, ret / horizon_vol)) * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        score = total / weight_sum

        ema_fast = close.ewm(span=self.ema_fast, min_periods=max(5, self.ema_fast // 2)).mean()
        ema_slow = close.ewm(span=self.ema_slow, min_periods=max(10, self.ema_slow // 2)).mean()
        if len(ema_slow.dropna()) > 0:
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                score += 0.15
            elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                score -= 0.15

        # Volatility-managed momentum: high realized vol de-risks the signal.
        target_vol = float(self.cfg.get("target_annual_vol", 0.35))
        score *= max(0.35, min(1.25, target_vol / realized_vol))
        return max(-1.0, min(1.0, score))
