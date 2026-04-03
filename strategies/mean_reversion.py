import pandas as pd
import numpy as np
import ta
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.mean_reversion")


class MeanReversionStrategy:
    """
    Mean reversion — like Pine Script Bollinger Band bounce setups.

    Key rule experienced traders follow: NEVER mean-revert in a strong trend.
    A stock plummeting with ADX > 40 is not "oversold" — it's trending down.

    Entry logic:
    1. Trend filter: ADX < 30 (ranging market — mean reversion works here)
    2. Price at/below lower Bollinger Band
    3. RSI oversold or recovering from oversold
    4. Bullish reversal candle (hammer, engulfing, morning star)
    5. Volume spike (capitulation selling = near bottom)
    6. VWAP: price below VWAP (discount to fair value)
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"]["mean_reversion"]

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

        # CRITICAL: Don't mean-revert in strong trends
        if ctx["strong_trend"] and ctx["direction"] == "down":
            return 0.0  # Falling knife — stay away
        if ctx["strong_trend"] and ctx["direction"] == "up":
            return 0.0  # Strong uptrend — don't try to short

        # ── Bollinger Bands ──────────────────────────────────
        bb = ta.volatility.BollingerBands(
            close,
            window=self.cfg["bb_period"],
            window_dev=self.cfg["bb_std"]
        )
        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        middle = bb.bollinger_mavg().iloc[-1]
        current_price = close.iloc[-1]
        pct_b = bb.bollinger_pband().iloc[-1]

        # Band width (squeeze detection)
        band_width = (upper - lower) / middle if middle > 0 else 0
        bw_series = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        bw_avg = bw_series.tail(20).mean()
        is_squeeze = band_width < bw_avg * 0.7  # Bands tightening

        # ── Z-score ──────────────────────────────────────────
        rolling_mean = close.rolling(self.cfg["bb_period"]).mean()
        rolling_std = close.rolling(self.cfg["bb_period"]).std()
        if rolling_std.iloc[-1] > 0:
            zscore = (current_price - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        else:
            zscore = 0.0

        # ── RSI ──────────────────────────────────────────────
        rsi = ta.momentum.RSIIndicator(close, window=self.cfg["rsi_period"]).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        rsi_recovering = current_rsi > prev_rsi and prev_rsi < 35

        # ── Candlestick patterns ─────────────────────────────
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ── Volume ───────────────────────────────────────────
        avg_vol = volume.tail(20).mean()
        vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        vol_spike = vol_ratio > 1.5  # Capitulation volume

        # ── Scoring ──────────────────────────────────────────
        score = 0.0

        # === BULLISH MEAN REVERSION (buy the dip) ===

        # Price below lower band — core signal
        if current_price < lower:
            score += 0.25
            if zscore < -self.cfg["zscore_threshold"]:
                score += 0.15  # Extreme deviation — stronger signal

        elif pct_b < 0.2:
            score += 0.1  # Near lower band

        # RSI confirmation
        if current_rsi < 30:
            score += 0.15  # Deeply oversold
        elif rsi_recovering:
            score += 0.2   # Recovering from oversold — best signal

        # Candle confirmation — this is what separates pros from amateurs
        # Don't buy just because it's oversold; wait for a REVERSAL CANDLE
        if score > 0.1:
            if candle_bull > 0.2:
                score += candle_bull * 0.25  # Strong candle confirmation
            else:
                score *= 0.6  # No reversal candle = weak signal, discount it

        # Volume spike at bottom = capitulation = good entry
        if score > 0.1 and vol_spike:
            score += 0.1

        # Below VWAP = buying at a discount
        if score > 0 and not ctx["above_vwap"]:
            score += 0.05

        # Penalty: trending market reduces confidence
        if ctx["trending"]:
            score *= 0.6  # Mean reversion less reliable in trends

        # === BEARISH (price at upper band — avoid buying) ===
        if current_price > upper:
            score -= 0.2
            if zscore > self.cfg["zscore_threshold"]:
                score -= 0.1
            if current_rsi > 70:
                score -= 0.15
            if candle_bear > 0.2:
                score -= candle_bear * 0.2

        # Squeeze — bands tightening means a big move is coming, sit out
        if is_squeeze:
            score *= 0.5

        return max(-1.0, min(1.0, score))
