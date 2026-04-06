import pandas as pd
import numpy as np
import ta
from indicators import rvol, vwap_bands
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.mean_reversion")


class MeanReversionStrategy:
    """
    Mean reversion — Bollinger Band bounces for LONG and SHORT.

    LONG: Price at/below lower BB, RSI oversold, bullish reversal candle
    SHORT: Price at/above upper BB, RSI overbought, bearish reversal candle

    Key rule: NEVER mean-revert in a strong trend.
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

        ctx = get_trend_context(df)

        # Don't mean-revert in strong trends
        if ctx["strong_trend"]:
            return 0.0

        # Bollinger Bands
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

        band_width = (upper - lower) / middle if middle > 0 else 0
        bw_series = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        bw_avg = bw_series.tail(20).mean()
        is_squeeze = band_width < bw_avg * 0.7

        # Detect squeeze-to-expansion: was squeezing, now expanding = breakout
        prev_bw = bw_series.iloc[-2] if len(bw_series) > 1 else band_width
        was_squeezing = prev_bw < bw_avg * 0.7
        squeeze_breakout = was_squeezing and not is_squeeze

        # Pure squeeze (still contracting) = big move coming, skip mean reversion
        if is_squeeze and not squeeze_breakout:
            return 0.0

        # VWAP distance: how far price is from VWAP (for mean reversion strength)
        vwap_data = vwap_bands(df)
        vwap_val = vwap_data["vwap"].iloc[-1] if not vwap_data["vwap"].empty else middle
        vwap_dist = (current_price - vwap_val) / vwap_val if vwap_val > 0 else 0

        # Z-score
        rolling_mean = close.rolling(self.cfg["bb_period"]).mean()
        rolling_std = close.rolling(self.cfg["bb_period"]).std()
        if rolling_std.iloc[-1] > 0:
            zscore = (current_price - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        else:
            zscore = 0.0

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=self.cfg["rsi_period"]).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        rsi_recovering = current_rsi > prev_rsi and prev_rsi < 35
        rsi_declining = current_rsi < prev_rsi and prev_rsi > 65

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        vol_ratio = rvol(df)
        vol_spike = vol_ratio > 1.5

        # ── LONG: Buy at lower band (oversold bounce) ────────
        long_score = 0.0
        if current_price < lower:
            long_score += 0.25
            if zscore < -self.cfg["zscore_threshold"]:
                long_score += 0.15
        elif pct_b < 0.2:
            long_score += 0.1

        if current_rsi < 30:
            long_score += 0.15
        elif rsi_recovering:
            long_score += 0.2

        if long_score > 0.1:
            if candle_bull > 0.2:
                long_score += candle_bull * 0.25
            else:
                long_score *= 0.6  # No reversal candle = weak

        if long_score > 0.1 and vol_spike:
            long_score += 0.1

        if long_score > 0 and not ctx["above_vwap"]:
            long_score += 0.05

        # VWAP proximity: price well below VWAP strengthens long mean reversion
        if long_score > 0 and vwap_dist < -0.01:
            long_score += min(0.15, abs(vwap_dist) * 5)

        # Squeeze breakout bonus: expansion after squeeze + going long
        if squeeze_breakout and current_price > middle:
            long_score += 0.15

        if ctx["trending"]:
            long_score *= 0.6

        # ── SHORT: Sell at upper band (overbought reversal) ──
        short_score = 0.0
        if current_price > upper:
            short_score -= 0.25
            if zscore > self.cfg["zscore_threshold"]:
                short_score -= 0.15
        elif pct_b > 0.8:
            short_score -= 0.1

        if current_rsi > 70:
            short_score -= 0.15
        elif rsi_declining:
            short_score -= 0.2  # Falling from overbought

        # Bearish reversal candle confirmation
        if short_score < -0.1:
            if candle_bear > 0.2:
                short_score -= candle_bear * 0.25
            else:
                short_score *= 0.6  # No reversal candle = weak

        if short_score < -0.1 and vol_spike:
            short_score -= 0.1

        # Above VWAP = overpriced
        if short_score < 0 and ctx["above_vwap"]:
            short_score -= 0.05

        # VWAP proximity: price well above VWAP strengthens short mean reversion
        if short_score < 0 and vwap_dist > 0.01:
            short_score -= min(0.15, vwap_dist * 5)

        # Squeeze breakout bonus: expansion after squeeze + going short
        if squeeze_breakout and current_price < middle:
            short_score -= 0.15

        if ctx["trending"]:
            short_score *= 0.6

        # Return the stronger signal
        if long_score > 0.15 and long_score > abs(short_score):
            return max(0.0, min(1.0, long_score))
        elif short_score < -0.15 and abs(short_score) > long_score:
            return max(-1.0, min(0.0, short_score))

        return 0.0
