import pandas as pd
import ta
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.momentum")


class MomentumStrategy:
    """
    Momentum strategy — EMA crossover + RSI + MACD for LONG and SHORT.

    LONG: EMA cross up, RSI 40-65, MACD positive, above 200 EMA
    SHORT: EMA cross down, RSI 35-60, MACD negative, below 200 EMA
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
        volume = df["volume"]

        ctx = get_trend_context(df)

        # Indicators
        rsi = ta.momentum.RSIIndicator(close, window=self.cfg["rsi_period"]).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        ema_fast = ta.trend.EMAIndicator(close, window=self.cfg["ema_fast"]).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(close, window=self.cfg["ema_slow"]).ema_indicator()

        ema_cross_up = (ema_fast.iloc[-1] > ema_slow.iloc[-1] and
                        ema_fast.iloc[-2] <= ema_slow.iloc[-2])
        ema_cross_down = (ema_fast.iloc[-1] < ema_slow.iloc[-1] and
                          ema_fast.iloc[-2] >= ema_slow.iloc[-2])
        ema_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        ema_bearish = ema_fast.iloc[-1] < ema_slow.iloc[-1]

        recent_cross_up = any(
            ema_fast.iloc[-j] > ema_slow.iloc[-j] and
            ema_fast.iloc[-j - 1] <= ema_slow.iloc[-j - 1]
            for j in range(1, min(4, len(ema_fast)))
        )
        recent_cross_down = any(
            ema_fast.iloc[-j] < ema_slow.iloc[-j] and
            ema_fast.iloc[-j - 1] >= ema_slow.iloc[-j - 1]
            for j in range(1, min(4, len(ema_fast)))
        )

        roc = ta.momentum.ROCIndicator(close, window=self.cfg["roc_period"]).roc()
        current_roc = roc.iloc[-1]

        macd = ta.trend.MACD(close)
        macd_diff = macd.macd_diff()
        macd_hist_rising = macd_diff.iloc[-1] > macd_diff.iloc[-2]
        macd_hist_falling = macd_diff.iloc[-1] < macd_diff.iloc[-2]
        macd_positive = macd_diff.iloc[-1] > 0
        macd_negative = macd_diff.iloc[-1] < 0

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        avg_vol = volume.tail(20).mean()
        vol_strong = volume.iloc[-1] > avg_vol * 1.1

        # ── LONG SCORING ─────────────────────────────────────
        if ctx["direction"] != "down" or not ctx["trending"]:
            long_score = 0.0

            if ema_cross_up:
                long_score += 0.35
            elif recent_cross_up and ema_bullish:
                long_score += 0.2
            elif ema_bullish:
                long_score += 0.1

            if 40 <= current_rsi <= 65:
                long_score += 0.15
            elif current_rsi < self.cfg["rsi_oversold"] and prev_rsi < current_rsi:
                long_score += 0.2
            elif current_rsi > self.cfg["rsi_overbought"]:
                long_score -= 0.3

            if macd_positive and macd_hist_rising:
                long_score += 0.15
            elif macd_hist_rising:
                long_score += 0.08

            if current_roc > 2:
                long_score += 0.1

            if long_score > 0.1:
                long_score += candle_bull * 0.2
            long_score -= candle_bear * 0.1

            if long_score > 0.1 and vol_strong:
                long_score *= 1.15

            if ctx["above_ema_200"] and ctx["direction"] == "up":
                long_score *= 1.1
            if ctx["above_vwap"]:
                long_score += 0.05

            if long_score > 0.15:
                return max(0.0, min(1.0, long_score))

        # ── SHORT SCORING ────────────────────────────────────
        if ctx["direction"] != "up" or not ctx["trending"]:
            short_score = 0.0

            # EMA cross down = bearish momentum
            if ema_cross_down:
                short_score -= 0.35
            elif recent_cross_down and ema_bearish:
                short_score -= 0.2
            elif ema_bearish:
                short_score -= 0.1

            # RSI in bearish zone (35-60 = room to fall)
            if 35 <= current_rsi <= 60:
                short_score -= 0.15
            elif current_rsi > self.cfg["rsi_overbought"] and prev_rsi > current_rsi:
                short_score -= 0.2  # Falling from overbought
            elif current_rsi < self.cfg["rsi_oversold"]:
                short_score += 0.3  # Already oversold, don't short

            # MACD negative and falling
            if macd_negative and macd_hist_falling:
                short_score -= 0.15
            elif macd_hist_falling:
                short_score -= 0.08

            # Negative ROC = price falling
            if current_roc < -2:
                short_score -= 0.1

            # Bearish candle confirmation
            if short_score < -0.1:
                short_score -= candle_bear * 0.2
            short_score += candle_bull * 0.1  # Bullish candle reduces short conviction

            # Volume on breakdown
            if short_score < -0.1 and vol_strong:
                short_score *= 1.15

            # Below 200 EMA = downtrend confirmed
            if not ctx["above_ema_200"] and ctx["direction"] == "down":
                short_score *= 1.1
            if not ctx["above_vwap"]:
                short_score -= 0.05

            if short_score < -0.15:
                return max(-1.0, min(0.0, short_score))

        return 0.0
