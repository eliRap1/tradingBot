import pandas as pd
import numpy as np
import ta
from indicators import pivot_high, pivot_low, last_pivot_value, bars_since_pivot, supertrend
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.breakout")


class BreakoutStrategy:
    """
    Pivot point breakout — like Pine Script ta.pivothigh/ta.pivotlow setups.

    Uses actual swing highs/lows for support/resistance (how real traders
    draw levels), not just rolling high/low.

    Entry logic:
    1. Identify resistance from confirmed pivot highs (ta.pivothigh)
    2. Price closes above last pivot high (not just wick through)
    3. Volume surge (1.5x+) confirms institutional participation
    4. Strong breakout candle (big body, close near high)
    5. Consolidation before breakout (Bollinger squeeze or tight range)
    6. ADX rising (new trend starting from range)

    Best setup: breakout → pullback to resistance-turned-support → bounce
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"]["breakout"]

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals = {}

        for sym, df in bars.items():
            if len(df) < self.cfg["lookback_days"] + 10:
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
        volume = df["volume"]
        lookback = self.cfg["lookback_days"]

        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]

        # ── Pivot-based support/resistance (Pine Script style) ──
        ph = pivot_high(high, left_bars=5, right_bars=5)
        pl = pivot_low(low, left_bars=5, right_bars=5)

        resistance = last_pivot_value(ph)
        support = last_pivot_value(pl)
        bars_since_res = bars_since_pivot(ph)

        # Fallback to simple high/low if not enough pivots
        if resistance is None:
            resistance = high.iloc[-lookback - 1:-1].max()
        if support is None:
            support = low.iloc[-lookback - 1:-1].min()

        # Second-to-last pivot (for retest detection)
        ph_valid = ph.dropna()
        prev_resistance = ph_valid.iloc[-2] if len(ph_valid) >= 2 else resistance

        # ── Trend context ────────────────────────────────────
        ctx = get_trend_context(df)

        # ── Volume ───────────────────────────────────────────
        avg_volume = volume.iloc[-lookback - 1:-1].mean()
        vol_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # ── ATR ──────────────────────────────────────────────
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close,
            window=self.cfg["atr_period"]
        ).average_true_range()
        current_atr = atr.iloc[-1]

        # ── Consolidation detection ──────────────────────────
        recent_range = (high.iloc[-5:].max() - low.iloc[-5:].min()) / current_price
        lookback_range = (high.iloc[-lookback:].max() - low.iloc[-lookback:].min()) / current_price
        range_compressed = recent_range < lookback_range * 0.5

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2.0)
        bw = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        bb_squeeze = bw.iloc[-1] < bw.tail(20).mean() * 0.7

        consolidation = range_compressed or bb_squeeze

        # ── Candle analysis ──────────────────────────────────
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # Breakout candle quality (big body closing near high)
        if "open" in df.columns:
            candle_body = abs(close.iloc[-1] - df["open"].iloc[-1])
            candle_range = high.iloc[-1] - low.iloc[-1]
            body_ratio = candle_body / candle_range if candle_range > 0 else 0
            close_near_high = (high.iloc[-1] - current_price) / candle_range < 0.2 if candle_range > 0 else False
            strong_candle = body_ratio > 0.6 and close_near_high
        else:
            strong_candle = False
            body_ratio = 0

        # ── Scoring ──────────────────────────────────────────
        score = 0.0

        # === BREAKOUT ABOVE PIVOT RESISTANCE ===
        if current_price > resistance:
            # Close above (not just wick) — critical distinction
            if close.iloc[-1] > resistance:
                score += 0.25
            else:
                score += 0.05  # Only wick — likely fakeout

            # Freshness: pivot should be recent (not stale level from months ago)
            if bars_since_res < 50:
                score += 0.05

            # Volume confirmation (most important fakeout filter)
            if vol_ratio >= self.cfg["volume_multiplier"]:
                score += 0.25
            elif vol_ratio >= 1.2:
                score += 0.08
            else:
                score *= 0.3  # No volume = fakeout

            # Consolidation before breakout (stored energy)
            if consolidation:
                score += 0.15

            # Strong breakout candle
            if strong_candle:
                score += 0.15
            elif candle_bull > 0.2:
                score += 0.08

            # ADX: new trend starting
            if ctx["adx"] > 20 and ctx["di_plus"] > ctx["di_minus"]:
                score += 0.1
            elif ctx["adx"] < 20:
                score += 0.05  # Breaking out of range

            # Breakout magnitude (healthy = 0.5-2x ATR)
            if current_atr > 0:
                atr_mult = (current_price - resistance) / current_atr
                if 0.5 < atr_mult < 2.0:
                    score += 0.05
                elif atr_mult > 3.0:
                    score *= 0.7  # Overextended

            if ctx["above_vwap"]:
                score += 0.05

        # === RETEST SETUP (strongest entry) ===
        # Price broke above resistance earlier, pulled back, now bouncing
        elif (current_price > prev_resistance and
              low.iloc[-1] <= resistance * 1.01 and
              close.iloc[-1] > resistance):
            score += 0.35  # Retest and hold
            if candle_bull > 0.2:
                score += 0.15
            if vol_ratio > 1.0:
                score += 0.1

        # === BREAKDOWN BELOW SUPPORT ===
        elif current_price < support:
            score -= 0.25
            if vol_ratio >= self.cfg["volume_multiplier"]:
                score -= 0.2
            if candle_bear > 0.2:
                score -= 0.1

        # Bearish candle on bullish setup = contradiction
        if score > 0 and candle_bear > 0.3:
            score *= 0.7

        return max(-1.0, min(1.0, score))
