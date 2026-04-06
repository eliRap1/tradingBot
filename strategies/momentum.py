import pandas as pd
import numpy as np
import ta
from indicators import rvol
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.momentum")


class MomentumStrategy:
    """
    MACD-driven momentum strategy with adaptive RSI and ADX gating.

    Entry signals require:
    - ADX > 20 (trending market)
    - MACD (12/26/9) crossover as primary trigger
    - Volume confirmation combined with MACD signal
    - Adaptive RSI thresholds: 40/80 in strong trends (ADX>25), 30/70 in ranging

    Returns scores from -1 (strong short) to +1 (strong long).
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"]["momentum"]

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

        ctx = get_trend_context(df)
        adx = ctx["adx"]

        if adx < 20:
            return 0.0

        strong_trend = adx > 25

        rsi = ta.momentum.RSIIndicator(close, window=self.cfg["rsi_period"]).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if strong_trend:
            rsi_oversold, rsi_overbought = 40, 80
        else:
            rsi_oversold, rsi_overbought = 30, 70

        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        macd_diff = macd.macd_diff()

        macd_cross_up = (macd_line.iloc[-1] > signal_line.iloc[-1] and
                         macd_line.iloc[-2] <= signal_line.iloc[-2])
        macd_cross_down = (macd_line.iloc[-1] < signal_line.iloc[-1] and
                           macd_line.iloc[-2] >= signal_line.iloc[-2])

        recent_macd_cross_up = any(
            macd_line.iloc[-j] > signal_line.iloc[-j] and
            macd_line.iloc[-j - 1] <= signal_line.iloc[-j - 1]
            for j in range(1, min(4, len(macd_line)))
        )
        recent_macd_cross_down = any(
            macd_line.iloc[-j] < signal_line.iloc[-j] and
            macd_line.iloc[-j - 1] >= signal_line.iloc[-j - 1]
            for j in range(1, min(4, len(macd_line)))
        )

        macd_hist_rising = macd_diff.iloc[-1] > macd_diff.iloc[-2]
        macd_hist_falling = macd_diff.iloc[-1] < macd_diff.iloc[-2]
        macd_positive = macd_diff.iloc[-1] > 0
        macd_negative = macd_diff.iloc[-1] < 0

        roc = ta.momentum.ROCIndicator(close, window=self.cfg["roc_period"]).roc()
        current_roc = roc.iloc[-1]
        roc_accel = roc.iloc[-1] - roc.iloc[-3] if len(roc) > 3 else 0

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        vol_ratio = rvol(df)
        vol_confirmed = vol_ratio > 1.3
        vol_surge = vol_ratio > 2.0

        range_high = high.rolling(20).max().iloc[-1]
        range_low = low.rolling(20).min().iloc[-1]
        price_location = (close.iloc[-1] - range_low) / (range_high - range_low) if range_high > range_low else 0.5

        price_higher = close.iloc[-1] > close.iloc[-5]
        rsi_higher = current_rsi > rsi.iloc[-5]
        bullish_divergence = not price_higher and rsi_higher
        bearish_divergence = price_higher and not rsi_higher

        # ── LONG SCORING ─────────────────────────────────────
        if ctx["direction"] != "down" or not ctx["trending"]:
            long_score = 0.0

            if macd_cross_up and vol_confirmed:
                long_score += 0.45
            elif recent_macd_cross_up and macd_positive and vol_confirmed:
                long_score += 0.30
            elif macd_positive and macd_hist_rising:
                long_score += 0.15

            if rsi_oversold <= current_rsi <= (rsi_overbought - 15):
                long_score += 0.18
            elif current_rsi < rsi_oversold and prev_rsi < current_rsi:
                long_score += 0.25
            elif current_rsi > rsi_overbought:
                long_score -= 0.35

            if current_roc > 3:
                long_score += 0.12
            if roc_accel > 1:
                long_score += 0.10

            if long_score > 0.1:
                long_score += candle_bull * 0.20
            long_score -= candle_bear * 0.15

            if vol_surge:
                long_score *= 1.30
            elif not vol_confirmed and long_score > 0:
                long_score *= 0.60

            if ctx["above_ema_200"] and ctx["direction"] == "up":
                long_score *= 1.12
            if ctx["above_vwap"]:
                long_score += 0.06

            if price_location < 0.3:
                long_score *= 1.10
            elif price_location > 0.85:
                long_score *= 0.85

            if bullish_divergence:
                long_score *= 1.20

            if strong_trend and ctx["direction"] == "up":
                long_score *= 1.10

            if long_score > 0.12:
                return max(0.0, min(1.0, long_score))

        # ── SHORT SCORING ────────────────────────────────────
        if ctx["direction"] != "up" or not ctx["trending"]:
            short_score = 0.0

            if macd_cross_down and vol_confirmed:
                short_score -= 0.45
            elif recent_macd_cross_down and macd_negative and vol_confirmed:
                short_score -= 0.30
            elif macd_negative and macd_hist_falling:
                short_score -= 0.15

            if (rsi_oversold + 15) <= current_rsi <= rsi_overbought:
                short_score -= 0.18
            elif current_rsi > rsi_overbought and prev_rsi > current_rsi:
                short_score -= 0.25
            elif current_rsi < rsi_oversold:
                short_score += 0.35

            if current_roc < -3:
                short_score -= 0.12
            if roc_accel < -1:
                short_score -= 0.10

            if short_score < -0.1:
                short_score -= candle_bear * 0.20
            short_score += candle_bull * 0.15

            if vol_surge:
                short_score *= 1.30
            elif not vol_confirmed and short_score < 0:
                short_score *= 0.60

            if not ctx["above_ema_200"] and ctx["direction"] == "down":
                short_score *= 1.12
            if not ctx["above_vwap"]:
                short_score -= 0.06

            if price_location > 0.7:
                short_score *= 1.10
            elif price_location < 0.15:
                short_score *= 0.85

            if bearish_divergence:
                short_score *= 1.20

            if strong_trend and ctx["direction"] == "down":
                short_score *= 1.10

            if short_score < -0.12:
                return max(-1.0, min(0.0, short_score))

        return 0.0
