"""
Liquidity Sweep Strategy — detect and trade institutional stop hunts.

How it works:
  1. Identify liquidity pools (swing highs/lows where stops cluster)
  2. Detect sweep: price wicks through the level then closes back
  3. Confirm reversal: volume spike, wick:body ratio, RSI divergence
  4. Enter on the reversal direction with tight stop beyond the sweep wick

HIGH-CONFLUENCE ONLY: requires 3+ confirmation factors to fire.

LONG setup:  Sweep below swing low + bullish reversal (bearish stop hunt)
SHORT setup: Sweep above swing high + bearish reversal (bullish stop hunt)
"""

import pandas as pd
import numpy as np
import ta
from indicators import pivot_high, pivot_low, rvol
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.liquidity_sweep")


class LiquiditySweepStrategy:
    """
    Detects liquidity sweeps (stop hunts) at key levels and trades the reversal.

    Only fires with high confluence:
      - Sweep of a swing high/low
      - Volume spike on the sweep candle
      - Wick:body ratio >= 2:1
      - At least one confirming factor (RSI divergence, trend alignment, VWAP proximity)
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"].get("liquidity_sweep", {
            "pivot_lookback": 5,
            "sweep_tolerance_pct": 0.005,   # max 0.5% beyond level
            "min_wick_body_ratio": 2.0,
            "min_volume_ratio": 1.5,
            "min_confluence": 3,
            "weight": 0.20,
        })

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals = {}

        for sym, df in bars.items():
            if len(df) < 30:
                continue

            try:
                score = self._analyze(df)
                if score != 0:
                    signals[sym] = score
            except Exception as e:
                log.error(f"Error analyzing {sym}: {e}")

        return signals

    def _analyze(self, df: pd.DataFrame) -> float:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_price = df["open"]
        volume = df["volume"]

        pivot_lb = self.cfg.get("pivot_lookback", 5)
        sweep_tol = self.cfg.get("sweep_tolerance_pct", 0.005)
        min_wb_ratio = self.cfg.get("min_wick_body_ratio", 2.0)
        min_vol_ratio = self.cfg.get("min_volume_ratio", 1.5)
        min_confluence = self.cfg.get("min_confluence", 3)

        # Current candle properties
        curr_close = close.iloc[-1]
        curr_high = high.iloc[-1]
        curr_low = low.iloc[-1]
        curr_open = open_price.iloc[-1]
        curr_body = abs(curr_close - curr_open)
        if curr_body == 0:
            curr_body = 0.001  # avoid division by zero

        # ── Find liquidity levels ─────────────────────────────

        # Swing highs/lows (where stops cluster)
        p_highs = pivot_high(high, left_bars=pivot_lb, right_bars=pivot_lb)
        p_lows = pivot_low(low, left_bars=pivot_lb, right_bars=pivot_lb)

        # Collect recent valid levels (not NaN, within reasonable range)
        recent_swing_highs = p_highs.dropna().tail(10).values
        recent_swing_lows = p_lows.dropna().tail(10).values

        # Also use prior day high/low as key levels
        if len(df) >= 2:
            prev_day_high = high.iloc[-2]
            prev_day_low = low.iloc[-2]
        else:
            prev_day_high = prev_day_low = None

        # Volume analysis
        vol_ratio = rvol(df)

        # RSI for divergence detection
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        curr_rsi = rsi.iloc[-1]

        # Trend context
        ctx = get_trend_context(df)

        # ATR for distance calculations
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        curr_atr = atr.iloc[-1]
        if curr_atr <= 0:
            return 0.0

        # Candle patterns
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ══════════════════════════════════════════════════════
        # BEARISH SWEEP (sweep above swing high → SHORT signal)
        # Price wicks above a key high, then closes back below
        # ══════════════════════════════════════════════════════

        short_score = self._check_bearish_sweep(
            curr_high, curr_low, curr_close, curr_open, curr_body,
            recent_swing_highs, prev_day_high,
            sweep_tol, min_wb_ratio, min_vol_ratio, min_confluence,
            vol_ratio, curr_rsi, rsi, close, high, ctx, candle_bear,
            curr_atr
        )

        if short_score < -0.15:
            return max(-1.0, min(0.0, short_score))

        # ══════════════════════════════════════════════════════
        # BULLISH SWEEP (sweep below swing low → LONG signal)
        # Price wicks below a key low, then closes back above
        # ══════════════════════════════════════════════════════

        long_score = self._check_bullish_sweep(
            curr_high, curr_low, curr_close, curr_open, curr_body,
            recent_swing_lows, prev_day_low,
            sweep_tol, min_wb_ratio, min_vol_ratio, min_confluence,
            vol_ratio, curr_rsi, rsi, close, low, ctx, candle_bull,
            curr_atr
        )

        if long_score > 0.15:
            return max(0.0, min(1.0, long_score))

        return 0.0

    def _check_bearish_sweep(self, curr_high, curr_low, curr_close, curr_open,
                              curr_body, swing_highs, prev_day_high,
                              sweep_tol, min_wb_ratio, min_vol_ratio,
                              min_confluence, vol_ratio, curr_rsi, rsi,
                              close, high, ctx, candle_bear, atr):
        """Check for a bearish liquidity sweep (sweep above high → short)."""

        # Build list of all resistance levels to check
        levels = list(swing_highs)
        if prev_day_high is not None:
            levels.append(prev_day_high)

        if not levels:
            return 0.0

        # Find the best swept level
        best_score = 0.0
        for level in levels:
            if level <= 0:
                continue

            # Check: did current candle wick above this level?
            sweep_distance = (curr_high - level) / level
            if sweep_distance < 0.001:
                continue  # didn't pierce the level
            if sweep_distance > sweep_tol:
                continue  # broke too far — likely real breakout

            # Check: did price close back BELOW the level?
            if curr_close >= level:
                continue  # holding above = breakout, not sweep

            # ── Sweep detected! Now score confluence ──

            score = 0.0
            confluence = 0

            # Factor 1: Sweep itself (base score)
            score -= 0.20
            confluence += 1

            # Factor 2: Wick:body ratio (long upper wick = rejection)
            upper_wick = curr_high - max(curr_close, curr_open)
            wb_ratio = upper_wick / curr_body if curr_body > 0 else 0
            if wb_ratio >= min_wb_ratio:
                score -= 0.15
                confluence += 1
                if wb_ratio >= 3.0:
                    score -= 0.05  # extra strong rejection

            # Factor 3: Volume spike (stops getting triggered)
            if vol_ratio >= 2.0:
                score -= 0.15
                confluence += 1
            elif vol_ratio >= min_vol_ratio:
                score -= 0.10
                confluence += 1

            # Factor 4: RSI divergence (price makes new high, RSI doesn't)
            if len(close) > 10:
                price_higher = curr_high > high.iloc[-6:-1].max()
                rsi_lower = curr_rsi < rsi.iloc[-6:-1].max()
                if price_higher and rsi_lower:
                    score -= 0.15
                    confluence += 1

            # Factor 5: RSI overbought
            if curr_rsi > 70:
                score -= 0.10
                confluence += 1
            elif curr_rsi > 60:
                score -= 0.05

            # Factor 6: HTF trend alignment (sweeping high in downtrend = A+)
            if ctx["direction"] == "down":
                score -= 0.10
                confluence += 1
            elif ctx["direction"] == "up":
                score *= 0.7  # counter-trend sweep = weaker

            # Factor 7: Below VWAP after sweep (institutions selling)
            if not ctx["above_vwap"]:
                score -= 0.05
                confluence += 1

            # Factor 8: Bearish candle confirmation
            if candle_bear > 0.2:
                score -= candle_bear * 0.15
                confluence += 1

            # Factor 9: Displacement — strong reversal candle
            if curr_body >= 1.5 * atr and curr_close < curr_open:
                score -= 0.10
                confluence += 1

            # ── Confluence gate ──
            if confluence < min_confluence:
                continue  # not enough confirmation

            # Volume damping (no volume = no conviction)
            if vol_ratio < 1.0:
                score *= 0.6

            if score < best_score:
                best_score = score
                log.debug(
                    f"BEARISH SWEEP: level={level:.2f} wick={sweep_distance:.3%} "
                    f"wb_ratio={wb_ratio:.1f} vol={vol_ratio:.1f}x "
                    f"confluence={confluence}"
                )

        return best_score

    def _check_bullish_sweep(self, curr_high, curr_low, curr_close, curr_open,
                              curr_body, swing_lows, prev_day_low,
                              sweep_tol, min_wb_ratio, min_vol_ratio,
                              min_confluence, vol_ratio, curr_rsi, rsi,
                              close, low, ctx, candle_bull, atr):
        """Check for a bullish liquidity sweep (sweep below low → long)."""

        levels = list(swing_lows)
        if prev_day_low is not None:
            levels.append(prev_day_low)

        if not levels:
            return 0.0

        best_score = 0.0
        for level in levels:
            if level <= 0:
                continue

            # Check: did current candle wick below this level?
            sweep_distance = (level - curr_low) / level
            if sweep_distance < 0.001:
                continue  # didn't pierce
            if sweep_distance > sweep_tol:
                continue  # broke too far — likely real breakdown

            # Check: did price close back ABOVE the level?
            if curr_close <= level:
                continue  # holding below = breakdown, not sweep

            # ── Sweep detected! Score confluence ──

            score = 0.0
            confluence = 0

            # Factor 1: Sweep itself
            score += 0.20
            confluence += 1

            # Factor 2: Wick:body ratio (long lower wick = buying pressure)
            lower_wick = min(curr_close, curr_open) - curr_low
            wb_ratio = lower_wick / curr_body if curr_body > 0 else 0
            if wb_ratio >= min_wb_ratio:
                score += 0.15
                confluence += 1
                if wb_ratio >= 3.0:
                    score += 0.05

            # Factor 3: Volume spike
            if vol_ratio >= 2.0:
                score += 0.15
                confluence += 1
            elif vol_ratio >= min_vol_ratio:
                score += 0.10
                confluence += 1

            # Factor 4: RSI divergence (price makes new low, RSI doesn't)
            if len(close) > 10:
                price_lower = curr_low < low.iloc[-6:-1].min()
                rsi_higher = curr_rsi > rsi.iloc[-6:-1].min()
                if price_lower and rsi_higher:
                    score += 0.15
                    confluence += 1

            # Factor 5: RSI oversold
            if curr_rsi < 30:
                score += 0.10
                confluence += 1
            elif curr_rsi < 40:
                score += 0.05

            # Factor 6: HTF trend alignment (sweeping low in uptrend = A+)
            if ctx["direction"] == "up":
                score += 0.10
                confluence += 1
            elif ctx["direction"] == "down":
                score *= 0.7  # counter-trend = weaker

            # Factor 7: Above VWAP after sweep (institutions buying)
            if ctx["above_vwap"]:
                score += 0.05
                confluence += 1

            # Factor 8: Bullish candle confirmation
            if candle_bull > 0.2:
                score += candle_bull * 0.15
                confluence += 1

            # Factor 9: Displacement — strong reversal candle
            if curr_body >= 1.5 * atr and curr_close > curr_open:
                score += 0.10
                confluence += 1

            # ── Confluence gate ──
            if confluence < min_confluence:
                continue

            # Volume damping
            if vol_ratio < 1.0:
                score *= 0.6

            if score > best_score:
                best_score = score
                log.debug(
                    f"BULLISH SWEEP: level={level:.2f} wick={sweep_distance:.3%} "
                    f"wb_ratio={wb_ratio:.1f} vol={vol_ratio:.1f}x "
                    f"confluence={confluence}"
                )

        return best_score
