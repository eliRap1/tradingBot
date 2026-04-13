import pandas as pd
import ta
from indicators import supertrend, crossover, crossunder, rvol

from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.supertrend")


class SuperTrendStrategy:
    """
    SuperTrend as TREND FILTER — does not generate entry signals on its own.

    Requires:
    - SuperTrend direction stable for >= 3 bars (anti-whipsaw on 5-min)
    - ADX > 20 confirming a real trend exists
    - Price momentum agreeing with direction
    - Candle/volume confirmation layered on top

    Scores are intentionally muted when ADX < 25 (choppy regime).
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

        st_line, direction = supertrend(
            df,
            period=self.cfg["atr_period"],
            multiplier=self.cfg["multiplier"]
        )

        current_dir = direction.iloc[-1]
        is_bullish = current_dir == -1
        is_bearish = current_dir == 1

        stable_bars = self._direction_stability(direction)
        if stable_bars < 3:
            return 0.0

        ctx = get_trend_context(df)

        if ctx["adx"] < 20:
            return 0.0

        ema_window = min(200, len(df) - 1)
        ema_200 = ta.trend.EMAIndicator(close, window=ema_window).ema_indicator()
        above_ema200 = close.iloc[-1] > ema_200.iloc[-1]

        momentum_up = close.iloc[-1] > close.iloc[-3]
        momentum_down = close.iloc[-1] < close.iloc[-3]

        vol_ratio = rvol(df)

        choppy = ctx["adx"] < 25

        if is_bullish and momentum_up:
            score = self._score_long(
                ctx, above_ema200, vol_ratio, stable_bars
            )
            if choppy:
                score *= 0.6
            return max(0.0, min(1.0, score))

        if is_bearish and momentum_down:
            score = self._score_short(
                ctx, above_ema200, vol_ratio, stable_bars
            )
            if choppy:
                score *= 0.6
            return max(-1.0, min(0.0, score))

        return 0.0

    def _direction_stability(self, direction: pd.Series) -> int:
        current = direction.iloc[-1]
        count = 0
        for i in range(1, len(direction)):
            if direction.iloc[-i] == current:
                count += 1
            else:
                break
        return count

    def _score_long(self, ctx, above_ema200, vol_ratio, stable_bars):
        if ctx["adx"] <= 20:
            return 0.0

        score = 0.0

        if ctx["adx"] > 30 and ctx["direction"] == "up":
            score += 0.25
        elif ctx["adx"] > 20 and ctx["direction"] == "up":
            score += 0.15

        if above_ema200:
            score += 0.10
        else:
            score *= 0.5

        if stable_bars >= 6:
            score += 0.10
        elif stable_bars >= 3:
            score += 0.05

        if vol_ratio >= 1.5:
            score += 0.15
        elif vol_ratio >= 1.2:
            score += 0.05

        if ctx["above_vwap"]:
            score += 0.05

        return score

    def _score_short(self, ctx, above_ema200, vol_ratio, stable_bars):
        if ctx["adx"] <= 20:
            return 0.0

        score = 0.0

        if ctx["adx"] > 30 and ctx["direction"] == "down":
            score -= 0.25
        elif ctx["adx"] > 20 and ctx["direction"] == "down":
            score -= 0.15

        if not above_ema200:
            score -= 0.10
        else:
            score *= 0.5

        if stable_bars >= 6:
            score -= 0.10
        elif stable_bars >= 3:
            score -= 0.05

        if vol_ratio >= 1.5:
            score -= 0.15
        elif vol_ratio >= 1.2:
            score -= 0.05

        if not ctx["above_vwap"]:
            score -= 0.05

        return score
