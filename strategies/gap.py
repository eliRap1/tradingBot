"""
Gap Trading Strategy - High probability intraday setups.

Gap strategies exploit overnight sentiment shifts:
1. Gap and Go: Gap up > 3%, holds above VWAP, continuation
2. Gap Fill: Gap fades back toward previous close
3. Opening Range Breakout: First 15-30min range breakout

Professional traders LOVE gaps - they show institutional intent.
"""

import pandas as pd
import numpy as np
import ta
from indicators import rvol
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.gap")


class GapStrategy:
    """
    Gap Trading Strategy for high-probability setups.
    
    Gap types:
    - Full Gap Up: Open > previous high
    - Full Gap Down: Open < previous low
    - Partial Gap: Open above/below previous close but within range
    
    Strategies:
    1. Gap and Go (continuation): Strong gap with volume holds = ride it
    2. Gap Fade (reversal): Weak gap fades back = trade the fill
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"].get("gap", {
            "min_gap_pct": 1.5,
            "strong_gap_pct": 3.0,
            "weight": 0.15
        })

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals = {}

        for sym, df in bars.items():
            if len(df) < 10:
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

        # Need at least 2 days of data
        if len(df) < 2:
            return 0.0

        # Today's open vs yesterday's close
        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        today_open = open_price.iloc[-1]
        current_price = close.iloc[-1]

        # Calculate gap
        gap_pct = (today_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)

        min_gap = self.cfg.get("min_gap_pct", 1.5)
        strong_gap = self.cfg.get("strong_gap_pct", 3.0)

        # No significant gap
        if abs_gap < min_gap:
            return 0.0

        # Gap classification
        gap_up = gap_pct > 0
        gap_down = gap_pct < 0
        full_gap_up = today_open > prev_high
        full_gap_down = today_open < prev_low

        # Current position relative to gap
        if gap_up:
            # Gap up: is price holding above open?
            holding_gap = current_price >= today_open * 0.995  # Within 0.5% of open
            fading = current_price < today_open * 0.99  # More than 1% below open
            filled = current_price <= prev_close * 1.002  # Back to previous close
        else:
            holding_gap = current_price <= today_open * 1.005
            fading = current_price > today_open * 1.01
            filled = current_price >= prev_close * 0.998

        # Volume analysis
        vol_ratio = rvol(df)
        high_volume = vol_ratio > 1.5
        very_high_volume = vol_ratio > 2.5

        # Trend context
        ctx = get_trend_context(df)

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]

        # Candle patterns
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ══════════════════════════════════════════════════════
        # GAP AND GO (Continuation) - Long on gap up, Short on gap down
        # ══════════════════════════════════════════════════════

        if gap_up and holding_gap and abs_gap >= min_gap:
            score = 0.0

            # Strong gap = strong signal
            if abs_gap >= strong_gap:
                score += 0.35
            else:
                score += 0.20

            # Full gap (above previous high) = institutional buying
            if full_gap_up:
                score += 0.15

            # Volume confirmation is critical
            if very_high_volume:
                score += 0.25
            elif high_volume:
                score += 0.15
            else:
                score *= 0.6  # Weak gap without volume

            # Price action confirms
            if current_price > today_open:  # Making new highs after gap
                score += 0.10
            if candle_bull > 0.2:
                score += candle_bull * 0.15

            # RSI not extreme
            if current_rsi > 80:
                score *= 0.7  # Overbought
            elif current_rsi < 60:
                score += 0.05  # Room to run

            # Trend alignment
            if ctx["direction"] == "up" and ctx["above_ema_200"]:
                score += 0.10

            if score > 0.15:
                log.debug(f"GAP AND GO LONG: gap={gap_pct:.1f}% vol={vol_ratio:.1f}x")
                return max(0.0, min(1.0, score))

        elif gap_down and holding_gap and abs_gap >= min_gap:
            score = 0.0

            if abs_gap >= strong_gap:
                score -= 0.35
            else:
                score -= 0.20

            if full_gap_down:
                score -= 0.15

            if very_high_volume:
                score -= 0.25
            elif high_volume:
                score -= 0.15
            else:
                score *= 0.6

            if current_price < today_open:
                score -= 0.10
            if candle_bear > 0.2:
                score -= candle_bear * 0.15

            if current_rsi < 20:
                score *= 0.7
            elif current_rsi > 40:
                score -= 0.05

            if ctx["direction"] == "down" and not ctx["above_ema_200"]:
                score -= 0.10

            if score < -0.15:
                log.debug(f"GAP AND GO SHORT: gap={gap_pct:.1f}% vol={vol_ratio:.1f}x")
                return max(-1.0, min(0.0, score))

        # ══════════════════════════════════════════════════════
        # GAP FADE (Reversal) - Short a fading gap up, Long a fading gap down
        # ══════════════════════════════════════════════════════

        if gap_up and fading and not filled:
            score = 0.0

            # Gap is fading = weakness
            score -= 0.25

            # Smaller gaps fade more often
            if abs_gap < strong_gap:
                score -= 0.15
            
            # Low volume on gap = likely fade
            if vol_ratio < 1.0:
                score -= 0.15

            # Bearish candle confirms fade
            if candle_bear > 0.2:
                score -= candle_bear * 0.20

            # RSI showing momentum loss
            if current_rsi < 50 and rsi.iloc[-2] > 50:
                score -= 0.10  # RSI crossed below 50

            # Below VWAP = institutions selling
            if not ctx["above_vwap"]:
                score -= 0.10

            if score < -0.15:
                log.debug(f"GAP FADE SHORT: gap={gap_pct:.1f}% fading")
                return max(-1.0, min(0.0, score))

        elif gap_down and fading and not filled:
            score = 0.0

            score += 0.25

            if abs_gap < strong_gap:
                score += 0.15

            if vol_ratio < 1.0:
                score += 0.15

            if candle_bull > 0.2:
                score += candle_bull * 0.20

            if current_rsi > 50 and rsi.iloc[-2] < 50:
                score += 0.10

            if ctx["above_vwap"]:
                score += 0.10

            if score > 0.15:
                log.debug(f"GAP FADE LONG: gap={gap_pct:.1f}% bouncing")
                return max(0.0, min(1.0, score))

        return 0.0
