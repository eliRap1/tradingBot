"""
Draw-on-Liquidity (DOL) Strategy — ICT-style structural bias signal.

Scores every active Order Block, Fair Value Gap, Breaker, and SSL/BSL
cluster on proximity / density / freshness / confluence, then aggregates
signed pull (demand below price + supply above price) into a single
directional verdict in [-1, 1].

Complements `liquidity_sweep` (which trades the reversal on the sweep wick);
DOL is a positional read — it's saying where price is "drawn to."

Primitives detected:
  - Order Blocks (OB): last opposing candle before a displacement break
  - Fair Value Gaps (FVG): 3-bar imbalances
  - Breakers / inversion FVG (iFVG): OB/FVG broken by close through opposite side
  - SSL/BSL: swing-high/low clusters (equal-highs/lows)

Scoring (per level):
  score = 0.35·proximity + 0.25·density + 0.20·freshness + 0.20·confluence
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import ta

from indicators import pivot_high, pivot_low
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.dol")


class DOLStrategy:
    """Draw-on-Liquidity positional bias score."""

    def __init__(self, config: dict):
        self.cfg = config.get("strategies", {}).get("dol", {}) or {}
        self.pivot_lookback = int(self.cfg.get("pivot_lookback", 5))
        self.fvg_min_atr_mult = float(self.cfg.get("fvg_min_atr_mult", 0.1))
        self.ob_displacement_mult = float(self.cfg.get("ob_displacement_mult", 1.0))
        self.freshness_decay_bars = int(self.cfg.get("freshness_decay_bars", 30))
        self.top_n_levels = int(self.cfg.get("top_n_levels", 3))
        self.min_verdict = float(self.cfg.get("min_verdict", 0.20))
        self.require_htf_align = bool(self.cfg.get("require_htf_align", True))
        self.scan_lookback_bars = int(self.cfg.get("scan_lookback_bars", 60))

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals: dict[str, float] = {}
        for sym, df in bars.items():
            if df is None or len(df) < 40:
                continue
            try:
                score = self._analyze(df)
                if score != 0.0:
                    signals[sym] = score
            except Exception as e:
                log.error(f"DOL error {sym}: {e}")
        return signals

    def _analyze(self, df: pd.DataFrame) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]

        atr_series = ta.volatility.AverageTrueRange(
            high, low, close, window=14
        ).average_true_range()
        atr = float(atr_series.iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return 0.0

        ctx = get_trend_context(df)
        px = float(close.iloc[-1])
        n = len(df)

        start_scan = max(self.pivot_lookback + 2, n - self.scan_lookback_bars)

        levels: list[dict] = []
        levels += self._detect_order_blocks(df, atr_series, start_scan)
        levels += self._detect_fvgs(df, atr_series, start_scan)
        self._apply_breakers(levels, df)

        # Prune stale zones (older than 3× decay horizon)
        max_age = self.freshness_decay_bars * 3
        levels = [lv for lv in levels if (n - 1 - lv["bar_formed"]) <= max_age]

        # Add SSL/BSL pivot-cluster levels
        levels += self._detect_pivot_clusters(high, low, atr, n)

        if not levels:
            return 0.0

        # Prior day high/low band for confluence
        pdh, pdl = self._prior_day_hl(df)

        scored: list[tuple[float, str]] = []
        for lv in levels:
            zone_mid = (lv["zone_low"] + lv["zone_high"]) / 2.0
            side = lv["side"]
            # Skip zones on the wrong side of price for their role
            # demand zone must be below price; supply above
            if side == "demand" and zone_mid > px:
                continue
            if side == "supply" and zone_mid < px:
                continue

            proximity = math.exp(-abs(px - zone_mid) / (atr * 3.0))

            density = self._density_score(
                zone_mid, high, low, atr, self.pivot_lookback
            )

            bars_since = max(0, (n - 1) - lv["bar_formed"])
            freshness = math.exp(-bars_since / max(1.0, self.freshness_decay_bars))

            confluence = self._confluence_score(
                zone_mid, atr, ctx, side, pdh, pdl
            )

            score = (
                0.35 * proximity
                + 0.25 * density
                + 0.20 * freshness
                + 0.20 * confluence
            )

            scored.append((score, "bullish" if side == "demand" else "bearish"))

        if not scored:
            return 0.0

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_n_levels]

        signed = 0.0
        for s, direction in top:
            signed += s if direction == "bullish" else -s

        verdict = math.tanh(signed)
        verdict = max(-1.0, min(1.0, verdict))

        if abs(verdict) < self.min_verdict:
            return 0.0

        if self.require_htf_align:
            htf = ctx.get("direction", "neutral")
            if verdict > 0 and htf == "down":
                return 0.0
            if verdict < 0 and htf == "up":
                return 0.0

        return verdict

    # ── primitive detectors ────────────────────────────────────

    def _detect_order_blocks(
        self, df: pd.DataFrame, atr_series: pd.Series, start: int
    ) -> list[dict]:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        open_ = df["open"].values
        n = len(df)
        lb = self.pivot_lookback
        thresh_mult = self.ob_displacement_mult

        levels: list[dict] = []
        for i in range(start, n - 1):
            body_next = abs(close[i + 1] - open_[i + 1])
            atr_here = float(atr_series.iloc[i + 1]) if i + 1 < len(atr_series) else 0.0
            if not np.isfinite(atr_here) or atr_here <= 0:
                continue
            if body_next < thresh_mult * atr_here:
                continue

            window_start = max(0, i - lb)
            prior_high = high[window_start:i].max() if i > window_start else high[i]
            prior_low = low[window_start:i].min() if i > window_start else low[i]

            is_down_candle = close[i] < open_[i]
            is_up_candle = close[i] > open_[i]

            # Bullish OB: down candle + next bar breaks above prior high
            if is_down_candle and close[i + 1] > prior_high:
                zlow = float(min(open_[i], close[i]))
                zhigh = float(max(open_[i], close[i]))
                if zhigh > zlow:
                    levels.append({
                        "kind": "ob",
                        "side": "demand",
                        "zone_low": zlow,
                        "zone_high": zhigh,
                        "bar_formed": i,
                    })

            # Bearish OB: up candle + next bar breaks below prior low
            if is_up_candle and close[i + 1] < prior_low:
                zlow = float(min(open_[i], close[i]))
                zhigh = float(max(open_[i], close[i]))
                if zhigh > zlow:
                    levels.append({
                        "kind": "ob",
                        "side": "supply",
                        "zone_low": zlow,
                        "zone_high": zhigh,
                        "bar_formed": i,
                    })

        return levels

    def _detect_fvgs(
        self, df: pd.DataFrame, atr_series: pd.Series, start: int
    ) -> list[dict]:
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        levels: list[dict] = []
        for i in range(max(2, start), n):
            atr_here = float(atr_series.iloc[i]) if i < len(atr_series) else 0.0
            if not np.isfinite(atr_here) or atr_here <= 0:
                continue
            min_gap = self.fvg_min_atr_mult * atr_here

            # Bull FVG: high[i-2] < low[i]
            if high[i - 2] < low[i]:
                gap = low[i] - high[i - 2]
                if gap >= min_gap:
                    levels.append({
                        "kind": "fvg",
                        "side": "demand",
                        "zone_low": float(high[i - 2]),
                        "zone_high": float(low[i]),
                        "bar_formed": i,
                    })

            # Bear FVG: low[i-2] > high[i]
            if low[i - 2] > high[i]:
                gap = low[i - 2] - high[i]
                if gap >= min_gap:
                    levels.append({
                        "kind": "fvg",
                        "side": "supply",
                        "zone_low": float(high[i]),
                        "zone_high": float(low[i - 2]),
                        "bar_formed": i,
                    })

        return levels

    def _apply_breakers(self, levels: list[dict], df: pd.DataFrame) -> None:
        """If price closes through opposite side of a zone, flip its role."""
        close = df["close"].values
        n = len(df)
        for lv in levels:
            formed = lv["bar_formed"]
            if formed >= n - 1:
                continue
            post = close[formed + 1:]
            if len(post) == 0:
                continue
            if lv["side"] == "demand":
                # Broken if any close below zone_low
                if (post < lv["zone_low"]).any():
                    lv["side"] = "supply"
                    lv["broken"] = True
            else:  # supply
                # Broken if any close above zone_high
                if (post > lv["zone_high"]).any():
                    lv["side"] = "demand"
                    lv["broken"] = True

    def _detect_pivot_clusters(
        self, high: pd.Series, low: pd.Series, atr: float, n: int
    ) -> list[dict]:
        lb = self.pivot_lookback
        ph = pivot_high(high, left_bars=lb, right_bars=lb)
        pl = pivot_low(low, left_bars=lb, right_bars=lb)

        ph_vals = [(idx, float(v)) for idx, v in zip(range(n), ph.values)
                   if not np.isnan(v)]
        pl_vals = [(idx, float(v)) for idx, v in zip(range(n), pl.values)
                   if not np.isnan(v)]

        band = atr * 0.25
        levels: list[dict] = []

        # BSL (buy-side liquidity above = resistance = supply)
        for idx, val in ph_vals[-10:]:
            cluster = [v for _, v in ph_vals if abs(v - val) <= band]
            zlow = min(cluster) if len(cluster) > 1 else val
            zhigh = max(cluster) if len(cluster) > 1 else val
            if zhigh == zlow:
                zhigh = zlow + band * 0.5
            levels.append({
                "kind": "bsl",
                "side": "supply",
                "zone_low": float(zlow),
                "zone_high": float(zhigh),
                "bar_formed": idx,
                "cluster_n": len(cluster),
            })

        # SSL (sell-side liquidity below = support = demand)
        for idx, val in pl_vals[-10:]:
            cluster = [v for _, v in pl_vals if abs(v - val) <= band]
            zlow = min(cluster) if len(cluster) > 1 else val
            zhigh = max(cluster) if len(cluster) > 1 else val
            if zhigh == zlow:
                zhigh = zlow - band * 0.5
                zlow, zhigh = min(zlow, zhigh), max(zlow, zhigh)
            levels.append({
                "kind": "ssl",
                "side": "demand",
                "zone_low": float(zlow),
                "zone_high": float(zhigh),
                "bar_formed": idx,
                "cluster_n": len(cluster),
            })

        return levels

    # ── pillar helpers ─────────────────────────────────────────

    def _density_score(
        self, zone_mid: float, high: pd.Series, low: pd.Series,
        atr: float, lb: int,
    ) -> float:
        band = atr * 0.25
        ph = pivot_high(high, left_bars=lb, right_bars=lb).dropna().values
        pl = pivot_low(low, left_bars=lb, right_bars=lb).dropna().values
        count = 0
        for v in np.concatenate([ph, pl]) if len(ph) + len(pl) > 0 else []:
            if abs(float(v) - zone_mid) <= band:
                count += 1
        return min(1.0, count / 5.0)

    def _confluence_score(
        self, zone_mid: float, atr: float, ctx: dict, side: str,
        pdh: float | None, pdl: float | None,
    ) -> float:
        score = 0.0

        # Prior-day high/low proximity (within 0.5× ATR of either)
        if pdh is not None and abs(zone_mid - pdh) <= atr * 0.5:
            score += 0.33
        elif pdl is not None and abs(zone_mid - pdl) <= atr * 0.5:
            score += 0.33

        # HTF direction agreement
        htf = ctx.get("direction", "neutral")
        if side == "demand" and htf == "up":
            score += 0.33
        elif side == "supply" and htf == "down":
            score += 0.33

        # VWAP proximity
        vwap = ctx.get("vwap")
        if vwap is not None and np.isfinite(vwap):
            if abs(zone_mid - float(vwap)) <= atr * 0.5:
                score += 0.34

        return min(1.0, score)

    def _prior_day_hl(
        self, df: pd.DataFrame
    ) -> tuple[float | None, float | None]:
        """Prior-session high/low. Intraday: prior date's H/L. Daily: bar[-2]."""
        idx = df.index
        try:
            if hasattr(idx, "hour") or (len(idx) > 0 and hasattr(idx[0], "hour")
                and not all(t.hour == 0 and t.minute == 0 for t in idx[-10:])):
                dates = pd.Series(idx).dt.date
                unique_dates = dates.unique()
                if len(unique_dates) >= 2:
                    prev_date = unique_dates[-2]
                    mask = dates.values == prev_date
                    prev_day = df.iloc[mask]
                    if len(prev_day) > 0:
                        return (
                            float(prev_day["high"].max()),
                            float(prev_day["low"].min()),
                        )
        except (AttributeError, TypeError):
            pass

        if len(df) >= 2:
            return (float(df["high"].iloc[-2]), float(df["low"].iloc[-2]))
        return (None, None)
