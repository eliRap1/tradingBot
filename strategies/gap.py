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

        if len(df) < 2:
            return 0.0

        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        today_open = open_price.iloc[-1]
        current_price = close.iloc[-1]

        gap_pct = (today_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)

        min_gap = self.cfg.get("min_gap_pct", 1.5)
        strong_gap = self.cfg.get("strong_gap_pct", 3.0)

        if abs_gap < min_gap:
            return 0.0

        gap_up = gap_pct > 0
        gap_down = gap_pct < 0
        full_gap_up = today_open > prev_high
        full_gap_down = today_open < prev_low

        if gap_up:
            holding_gap = current_price >= today_open * 0.995
            fading = current_price < today_open * 0.99
            filled = current_price <= prev_close * 1.002
        else:
            holding_gap = current_price <= today_open * 1.005
            fading = current_price > today_open * 1.01
            filled = current_price >= prev_close * 0.998

        vol_ratio = rvol(df)
        high_volume = vol_ratio > 1.5
        very_high_volume = vol_ratio > 2.5

        ctx = get_trend_context(df)

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        orb_score = self._orb_signal(df, gap_up, gap_down, abs_gap, strong_gap, vol_ratio, ctx)
        if orb_score != 0.0:
            return orb_score

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

    def _orb_signal(self, df: pd.DataFrame, gap_up: bool, gap_down: bool,
                    abs_gap: float, strong_gap: float, vol_ratio: float,
                    ctx: dict) -> float:
        if len(df) < 5:
            return 0.0

        orb_bars = df.iloc[-4:-1]
        or_high = orb_bars["high"].max()
        or_low = orb_bars["low"].min()
        or_vol_avg = orb_bars["volume"].mean()

        vol_avg_50 = df["volume"].rolling(50).mean().iloc[-1]
        or_volume_strong = or_vol_avg > vol_avg_50 if not np.isnan(vol_avg_50) else False

        if not or_volume_strong:
            return 0.0

        current_price = df["close"].iloc[-1]

        if gap_up and current_price > or_high:
            score = 0.30
            if abs_gap >= strong_gap:
                score += 0.15
            if vol_ratio > 1.5:
                score += 0.20
            elif vol_ratio > 1.0:
                score += 0.10
            if ctx["direction"] == "up":
                score += 0.10
            if ctx["above_vwap"]:
                score += 0.05
            log.debug(f"ORB LONG: gap_up, price>{or_high:.2f}, vol={vol_ratio:.1f}x")
            return max(0.0, min(1.0, score))

        if gap_up and current_price < or_low:
            score = -0.20
            if abs_gap < strong_gap:
                score -= 0.10
            if vol_ratio > 1.5:
                score -= 0.10
            if not ctx["above_vwap"]:
                score -= 0.05
            log.debug(f"ORB GAP FADE SHORT: gap_up but price<{or_low:.2f}")
            return max(-1.0, min(0.0, score))

        if gap_down and current_price < or_low:
            score = -0.30
            if abs_gap >= strong_gap:
                score -= 0.15
            if vol_ratio > 1.5:
                score -= 0.20
            elif vol_ratio > 1.0:
                score -= 0.10
            if ctx["direction"] == "down":
                score -= 0.10
            if not ctx["above_vwap"]:
                score -= 0.05
            log.debug(f"ORB SHORT: gap_down, price<{or_low:.2f}, vol={vol_ratio:.1f}x")
            return max(-1.0, min(0.0, score))

        if gap_down and current_price > or_high:
            score = 0.20
            if abs_gap < strong_gap:
                score += 0.10
            if vol_ratio > 1.5:
                score += 0.10
            if ctx["above_vwap"]:
                score += 0.05
            log.debug(f"ORB GAP FADE LONG: gap_down but price>{or_high:.2f}")
            return max(0.0, min(1.0, score))

        return 0.0
