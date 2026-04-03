import pandas as pd
import ta
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.momentum")


class MomentumStrategy:
    """
    Momentum strategy — like a Pine Script EMA crossover + RSI + MACD setup.

    Entry logic (what experienced traders actually check):
    1. Trend filter: ADX > 20 and price above 200 EMA (don't buy in downtrends)
    2. EMA crossover EVENT: fast EMA crosses above slow EMA (not just above)
    3. RSI confirmation: RSI between 40-70 (not overbought, has room to run)
    4. MACD histogram turning positive
    5. Candle confirmation: bullish candle pattern at entry
    6. Volume: above average on the signal bar
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

        # ── Trend context ────────────────────────────────────
        ctx = get_trend_context(df)

        # ── Indicators ───────────────────────────────────────
        rsi = ta.momentum.RSIIndicator(close, window=self.cfg["rsi_period"]).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        ema_fast = ta.trend.EMAIndicator(close, window=self.cfg["ema_fast"]).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(close, window=self.cfg["ema_slow"]).ema_indicator()

        # Detect actual CROSSOVER event (Pine Script's ta.crossover)
        ema_cross_up = (ema_fast.iloc[-1] > ema_slow.iloc[-1] and
                        ema_fast.iloc[-2] <= ema_slow.iloc[-2])
        ema_cross_down = (ema_fast.iloc[-1] < ema_slow.iloc[-1] and
                          ema_fast.iloc[-2] >= ema_slow.iloc[-2])
        ema_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]

        # Recent crossover (within last 3 bars) — still valid signal
        recent_cross_up = any(
            ema_fast.iloc[-j] > ema_slow.iloc[-j] and
            ema_fast.iloc[-j - 1] <= ema_slow.iloc[-j - 1]
            for j in range(1, min(4, len(ema_fast)))
        )

        roc = ta.momentum.ROCIndicator(close, window=self.cfg["roc_period"]).roc()
        current_roc = roc.iloc[-1]

        macd = ta.trend.MACD(close)
        macd_diff = macd.macd_diff()
        macd_hist_rising = macd_diff.iloc[-1] > macd_diff.iloc[-2]
        macd_positive = macd_diff.iloc[-1] > 0

        # ── Candlestick patterns ─────────────────────────────
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ── Volume check ─────────────────────────────────────
        avg_vol = volume.tail(20).mean()
        vol_strong = volume.iloc[-1] > avg_vol * 1.1

        # ── Scoring ──────────────────────────────────────────
        score = 0.0

        # === BULLISH SETUP ===

        # 1. Trend filter: must be in uptrend or neutral (don't buy downtrends)
        if ctx["direction"] == "down" and ctx["trending"]:
            return max(-0.3, -candle_bear * 0.3)  # Strong downtrend, skip buys

        # 2. EMA crossover (the core signal)
        if ema_cross_up:
            score += 0.35  # Fresh crossover is strongest
        elif recent_cross_up and ema_bullish:
            score += 0.2   # Recent crossover, still valid
        elif ema_bullish:
            score += 0.1   # Already above, weaker

        # 3. RSI confirmation
        if 40 <= current_rsi <= 65:
            score += 0.15  # Sweet spot — momentum but not exhausted
        elif current_rsi < self.cfg["rsi_oversold"] and prev_rsi < current_rsi:
            score += 0.2   # Bouncing from oversold
        elif current_rsi > self.cfg["rsi_overbought"]:
            score -= 0.3   # Overbought — don't chase

        # 4. MACD confirmation
        if macd_positive and macd_hist_rising:
            score += 0.15  # Both positive and accelerating
        elif macd_hist_rising:
            score += 0.08  # At least accelerating

        # 5. ROC positive
        if current_roc > 2:
            score += 0.1
        elif current_roc < -5:
            score -= 0.15

        # 6. Candle confirmation
        if score > 0.1:
            score += candle_bull * 0.2  # Boost if bullish candles confirm
        score -= candle_bear * 0.15     # Penalize bearish candles

        # 7. Volume confirmation
        if score > 0.1 and vol_strong:
            score *= 1.15  # Boost signal with volume

        # 8. Trend bonus
        if ctx["above_ema_200"] and ctx["direction"] == "up":
            score *= 1.1
        if ctx["above_vwap"]:
            score += 0.05

        # === BEARISH ===
        if ema_cross_down:
            score -= 0.25
        if not ema_bullish and current_rsi > 60:
            score -= 0.15

        return max(-1.0, min(1.0, score))
