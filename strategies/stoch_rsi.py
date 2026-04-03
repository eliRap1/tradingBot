import pandas as pd
import ta
from indicators import stochastic_rsi, crossover
from candles import detect_patterns, bullish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.stoch_rsi")


class StochRSIStrategy:
    """
    Stochastic RSI pullback strategy — catches dips in uptrends.

    This is the bread-and-butter swing trading setup in Pine Script.
    Uses EMA to define trend, enters on oversold StochRSI crossover.

    Entry logic:
    1. Trend filter: price above 50 EMA (uptrend)
    2. StochRSI %K crosses above %D (momentum turning up)
    3. Crossover happens in oversold zone (%K < 20-25)
    4. Bullish candle on entry bar
    5. Volume above average

    Exit:
    - StochRSI enters overbought (> 80) → take profit
    - Price closes below 50 EMA → trend broken, stop out
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"]["stoch_rsi"]

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

        # ── Stochastic RSI ───────────────────────────────────
        k, d = stochastic_rsi(
            close,
            rsi_period=self.cfg["rsi_period"],
            stoch_period=self.cfg["stoch_period"],
            k_smooth=self.cfg["k_smooth"],
            d_smooth=self.cfg["d_smooth"]
        )

        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        prev_k = k.iloc[-2]

        # Detect K crossing above D (Pine's ta.crossover(k, d))
        k_cross_up = current_k > current_d and prev_k <= d.iloc[-2]

        # Recent crossover (within last 3 bars)
        cross_series = crossover(k, d)
        recent_cross_up = cross_series.iloc[-3:].any()

        # Zone detection
        oversold = current_k < self.cfg["oversold"]
        overbought = current_k > self.cfg["overbought"]
        was_oversold = k.iloc[-3:].min() < self.cfg["oversold"]

        # ── Trend filter (50 EMA) ────────────────────────────
        ema_period = self.cfg["ema_period"]
        ema = ta.trend.EMAIndicator(close, window=ema_period).ema_indicator()
        above_ema = close.iloc[-1] > ema.iloc[-1]

        # ── Trend context ────────────────────────────────────
        ctx = get_trend_context(df)

        # ── Regular RSI for confirmation ─────────────────────
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]

        # ── Candles ──────────────────────────────────────────
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)

        # ── Volume ───────────────────────────────────────────
        avg_vol = volume.tail(20).mean()
        vol_ok = volume.iloc[-1] > avg_vol * 0.8

        # ── Scoring ──────────────────────────────────────────
        score = 0.0

        # Must be in uptrend to buy pullbacks
        if not above_ema:
            if ctx["direction"] == "down":
                return 0.0  # Don't buy pullbacks in downtrends
            # Slightly below EMA might be ok if trend is neutral
            score -= 0.1

        # === CORE SIGNAL: StochRSI crossover from oversold ===
        if k_cross_up and was_oversold:
            score += 0.4  # Fresh crossover from oversold — best signal
        elif recent_cross_up and was_oversold:
            score += 0.25  # Recent crossover, still valid
        elif k_cross_up and current_k < 40:
            score += 0.15  # Crossover not from deep oversold, weaker
        else:
            return score if score < 0 else 0.0  # No crossover = no signal

        # Trend alignment bonus
        if above_ema and ctx["direction"] == "up":
            score += 0.15
        elif above_ema:
            score += 0.05

        # RSI confirmation (not overbought)
        if 35 <= current_rsi <= 60:
            score += 0.1  # Sweet spot for pullback entry
        elif current_rsi > 70:
            score -= 0.2  # Already overbought, skip

        # Candle confirmation
        if candle_bull > 0.2:
            score += candle_bull * 0.15
        elif candle_bull == 0:
            score *= 0.8  # No bullish candle = less conviction

        # Volume
        if vol_ok:
            score += 0.05

        # Higher timeframe alignment
        if ctx["above_ema_200"]:
            score += 0.05

        # Penalty for overbought
        if overbought:
            score -= 0.3

        return max(-1.0, min(1.0, score))
