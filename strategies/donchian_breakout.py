"""Donchian channel breakout strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from utils import setup_logger

log = setup_logger("strategy.donchian_breakout")


class DonchianBreakoutStrategy:
    """20/55 bar channel breakout with ATR and volume confirmation."""

    def __init__(self, config: dict):
        self.cfg = config.get("strategies", {}).get("donchian_breakout", {})
        self.fast_lookback = int(self.cfg.get("fast_lookback", 20))
        self.slow_lookback = int(self.cfg.get("slow_lookback", 55))
        self.atr_period = int(self.cfg.get("atr_period", 14))
        self.min_volume_ratio = float(self.cfg.get("min_volume_ratio", 1.05))
        self.max_atr_extension = float(self.cfg.get("max_atr_extension", 3.0))

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals = {}
        for sym, df in bars.items():
            try:
                score = self._analyze(df)
                if score != 0:
                    signals[sym] = score
            except Exception as e:
                log.error(f"Error analyzing {sym}: {e}")
        return signals

    def _analyze(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < self.slow_lookback + 2:
            return 0.0
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=self.atr_period
        ).average_true_range().iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            return 0.0

        prev_fast_high = high.shift(1).rolling(self.fast_lookback).max().iloc[-1]
        prev_fast_low = low.shift(1).rolling(self.fast_lookback).min().iloc[-1]
        prev_slow_high = high.shift(1).rolling(self.slow_lookback).max().iloc[-1]
        prev_slow_low = low.shift(1).rolling(self.slow_lookback).min().iloc[-1]
        if not all(np.isfinite(x) for x in [prev_fast_high, prev_fast_low, prev_slow_high, prev_slow_low]):
            return 0.0

        vol_avg = volume.shift(1).rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg and vol_avg > 0 else 1.0
        if vol_ratio < self.min_volume_ratio:
            return 0.0

        px = close.iloc[-1]
        score = 0.0
        if px > prev_fast_high:
            extension = (px - prev_fast_high) / atr
            if extension <= self.max_atr_extension:
                score = 0.35 + min(0.35, extension * 0.15)
                if px > prev_slow_high:
                    score += 0.20
        elif px < prev_fast_low:
            extension = (prev_fast_low - px) / atr
            if extension <= self.max_atr_extension:
                score = -0.35 - min(0.35, extension * 0.15)
                if px < prev_slow_low:
                    score -= 0.20

        if score:
            score *= min(1.35, max(0.80, vol_ratio / self.min_volume_ratio))
        return max(-1.0, min(1.0, score))
