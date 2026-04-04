import pandas as pd
import ta
from indicators import supertrend, crossover, crossunder, rvol
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.supertrend")


class SuperTrendStrategy:
    """
    SuperTrend trend-following — works for both LONG and SHORT.

    LONG setup:
    - SuperTrend flips bullish (price crosses above ST line)
    - Above 200 EMA, ADX confirms trend, volume surge

    SHORT setup:
    - SuperTrend flips bearish (price crosses below ST line)
    - Below 200 EMA, ADX confirms downtrend, volume surge
    - Bearish candle confirmation
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

        # SuperTrend
        st_line, direction = supertrend(
            df,
            period=self.cfg["atr_period"],
            multiplier=self.cfg["multiplier"]
        )

        current_dir = direction.iloc[-1]
        prev_dir = direction.iloc[-2]
        flip_bullish = (current_dir == -1 and prev_dir == 1)
        flip_bearish = (current_dir == 1 and prev_dir == -1)

        recent_flip_bull = any(
            direction.iloc[-j] == -1 and direction.iloc[-j - 1] == 1
            for j in range(1, min(4, len(direction) - 1))
        )
        recent_flip_bear = any(
            direction.iloc[-j] == 1 and direction.iloc[-j - 1] == -1
            for j in range(1, min(4, len(direction) - 1))
        )

        is_bullish = current_dir == -1
        is_bearish = current_dir == 1

        # Trend context
        ctx = get_trend_context(df)

        # 200 EMA
        ema_window = min(200, len(df) - 1)
        ema_200 = ta.trend.EMAIndicator(close, window=ema_window).ema_indicator()
        above_ema200 = close.iloc[-1] > ema_200.iloc[-1]

        # RVOL (time-of-day adjusted volume)
        vol_ratio = rvol(df)

        # Candles
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ── LONG SCORING ─────────────────────────────────────
        if flip_bullish or (recent_flip_bull and is_bullish) or is_bullish:
            score = 0.0

            if flip_bullish:
                score += 0.4
            elif recent_flip_bull and is_bullish:
                score += 0.2
            elif is_bullish:
                score += 0.05

            if above_ema200:
                score += 0.1
            else:
                score *= 0.5

            if ctx["adx"] > 25 and ctx["direction"] == "up":
                score += 0.1
            elif ctx["adx"] < 15:
                score *= 0.7

            if vol_ratio >= 1.5:
                score += 0.15
            elif vol_ratio >= 1.2:
                score += 0.05
            elif flip_bullish and vol_ratio < 1.0:
                score *= 0.6

            if candle_bull > 0.2:
                score += candle_bull * 0.15
            if candle_bear > 0.3:
                score *= 0.7

            if ctx["above_vwap"]:
                score += 0.05

            return max(0.0, min(1.0, score))

        # ── SHORT SCORING ────────────────────────────────────
        if flip_bearish or (recent_flip_bear and is_bearish) or is_bearish:
            score = 0.0

            # Core signal: SuperTrend flip bearish
            if flip_bearish:
                score -= 0.4
            elif recent_flip_bear and is_bearish:
                score -= 0.2
            elif is_bearish:
                score -= 0.05

            # Below 200 EMA = downtrend confirmed
            if not above_ema200:
                score -= 0.1
            else:
                score *= 0.5  # Above 200 EMA = risky short

            # ADX confirms downtrend
            if ctx["adx"] > 25 and ctx["direction"] == "down":
                score -= 0.1
            elif ctx["adx"] < 15:
                score *= 0.7

            # Volume on the breakdown
            if vol_ratio >= 1.5:
                score -= 0.15
            elif vol_ratio >= 1.2:
                score -= 0.05
            elif flip_bearish and vol_ratio < 1.0:
                score *= 0.6

            # Bearish candle confirmation
            if candle_bear > 0.2:
                score -= candle_bear * 0.15
            if candle_bull > 0.3:
                score *= 0.7  # Bullish candle contradicts short

            # Below VWAP = selling pressure
            if not ctx["above_vwap"]:
                score -= 0.05

            return max(-1.0, min(0.0, score))

        return 0.0
