import pandas as pd
import ta
from indicators import supertrend, crossover, crossunder
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.supertrend")


class SuperTrendStrategy:
    """
    SuperTrend trend-following — the workhorse of Pine Script algo traders.

    Entry logic (matches proven Pine Script setups):
    1. SuperTrend flips bullish (price crosses above ST line)
    2. 200 EMA filter: only longs above 200 EMA
    3. Volume surge on the flip bar (1.3x+ average)
    4. Bullish candle confirmation on entry bar
    5. ADX > 20 (some directional movement, not dead market)

    Exit:
    - SuperTrend flips bearish (price crosses below ST line)
    - This IS the trailing stop — SuperTrend tightens automatically
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"]["supertrend"]

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
        volume = df["volume"]

        # ── SuperTrend ───────────────────────────────────────
        st_line, direction = supertrend(
            df,
            period=self.cfg["atr_period"],
            multiplier=self.cfg["multiplier"]
        )

        # Detect the FLIP event (like Pine's ta.crossover(close, st_line))
        current_dir = direction.iloc[-1]
        prev_dir = direction.iloc[-2]
        flip_bullish = (current_dir == -1 and prev_dir == 1)
        flip_bearish = (current_dir == 1 and prev_dir == -1)

        # Recent flip (within last 3 bars)
        recent_flip_bull = any(
            direction.iloc[-j] == -1 and direction.iloc[-j - 1] == 1
            for j in range(1, min(4, len(direction) - 1))
        )

        is_bullish = current_dir == -1

        # ── Trend context ────────────────────────────────────
        ctx = get_trend_context(df)

        # ── 200 EMA filter ───────────────────────────────────
        ema_window = min(200, len(df) - 1)
        ema_200 = ta.trend.EMAIndicator(close, window=ema_window).ema_indicator()
        above_ema200 = close.iloc[-1] > ema_200.iloc[-1]

        # ── Volume ───────────────────────────────────────────
        avg_vol = volume.tail(20).mean()
        vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        # ── Candles ──────────────────────────────────────────
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ── Scoring ──────────────────────────────────────────
        score = 0.0

        # === BULLISH SETUP ===

        # Core signal: SuperTrend flip
        if flip_bullish:
            score += 0.4  # Fresh flip — strongest signal
        elif recent_flip_bull and is_bullish:
            score += 0.2  # Recent flip, still valid
        elif is_bullish:
            score += 0.05  # Already in uptrend, weaker entry

        if score <= 0:
            # Check for bearish
            if flip_bearish:
                return -0.3
            return 0.0

        # 200 EMA filter (don't buy below long-term trend)
        if above_ema200:
            score += 0.1
        else:
            score *= 0.5  # Below 200 EMA — halve confidence

        # ADX filter
        if ctx["adx"] > 25 and ctx["direction"] == "up":
            score += 0.1  # Strong uptrend
        elif ctx["adx"] < 15:
            score *= 0.7  # Dead market — SuperTrend whipsaws here

        # Volume confirmation
        if vol_ratio >= 1.5:
            score += 0.15  # Strong volume on flip
        elif vol_ratio >= 1.2:
            score += 0.05
        elif flip_bullish and vol_ratio < 1.0:
            score *= 0.6  # No volume on flip = suspect

        # Candle confirmation
        if candle_bull > 0.2:
            score += candle_bull * 0.15
        if candle_bear > 0.3:
            score *= 0.7  # Bearish candle contradicts signal

        # VWAP bonus
        if ctx["above_vwap"]:
            score += 0.05

        return max(-1.0, min(1.0, score))
