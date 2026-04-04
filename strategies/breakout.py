import pandas as pd
import numpy as np
import ta
from indicators import pivot_high, pivot_low, last_pivot_value, bars_since_pivot, supertrend, rvol
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.breakout")


class BreakoutStrategy:
    """
    Pivot point breakout — works for LONG breakouts AND SHORT breakdowns.

    LONG: Price breaks above pivot resistance with volume
    SHORT: Price breaks below pivot support with volume (breakdown)
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"]["breakout"]

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        signals = {}

        for sym, df in bars.items():
            if len(df) < self.cfg["lookback_days"] + 10:
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
        volume = df["volume"]
        lookback = self.cfg["lookback_days"]

        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]

        # Pivot-based support/resistance
        ph = pivot_high(high, left_bars=5, right_bars=5)
        pl = pivot_low(low, left_bars=5, right_bars=5)

        resistance = last_pivot_value(ph)
        support = last_pivot_value(pl)
        bars_since_res = bars_since_pivot(ph)
        bars_since_sup = bars_since_pivot(pl)

        if resistance is None:
            resistance = high.iloc[-lookback - 1:-1].max()
        if support is None:
            support = low.iloc[-lookback - 1:-1].min()

        # Second pivots for retest detection
        ph_valid = ph.dropna()
        pl_valid = pl.dropna()
        prev_resistance = ph_valid.iloc[-2] if len(ph_valid) >= 2 else resistance
        prev_support = pl_valid.iloc[-2] if len(pl_valid) >= 2 else support

        ctx = get_trend_context(df)

        # RVOL (time-of-day adjusted volume)
        vol_ratio = rvol(df)

        # ATR
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close,
            window=self.cfg["atr_period"]
        ).average_true_range()
        current_atr = atr.iloc[-1]

        # Consolidation
        recent_range = (high.iloc[-5:].max() - low.iloc[-5:].min()) / current_price
        lookback_range = (high.iloc[-lookback:].max() - low.iloc[-lookback:].min()) / current_price
        range_compressed = recent_range < lookback_range * 0.5

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2.0)
        bw = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        bb_squeeze = bw.iloc[-1] < bw.tail(20).mean() * 0.7

        consolidation = range_compressed or bb_squeeze

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # Candle quality
        if "open" in df.columns:
            candle_body = abs(close.iloc[-1] - df["open"].iloc[-1])
            candle_range = high.iloc[-1] - low.iloc[-1]
            body_ratio = candle_body / candle_range if candle_range > 0 else 0
            close_near_high = (high.iloc[-1] - current_price) / candle_range < 0.2 if candle_range > 0 else False
            close_near_low = (current_price - low.iloc[-1]) / candle_range < 0.2 if candle_range > 0 else False
            strong_bull_candle = body_ratio > 0.6 and close_near_high
            strong_bear_candle = body_ratio > 0.6 and close_near_low
        else:
            strong_bull_candle = False
            strong_bear_candle = False

        # ── LONG: BREAKOUT ABOVE RESISTANCE ──────────────────
        if current_price > resistance:
            score = 0.0

            if close.iloc[-1] > resistance:
                score += 0.25
            else:
                score += 0.05

            if bars_since_res < 50:
                score += 0.05

            if vol_ratio >= self.cfg["volume_multiplier"]:
                score += 0.25
            elif vol_ratio >= 1.2:
                score += 0.08
            else:
                score *= 0.3

            if consolidation:
                score += 0.15

            if strong_bull_candle:
                score += 0.15
            elif candle_bull > 0.2:
                score += 0.08

            if ctx["adx"] > 20 and ctx["di_plus"] > ctx["di_minus"]:
                score += 0.1

            if current_atr > 0:
                atr_mult = (current_price - resistance) / current_atr
                if 0.5 < atr_mult < 2.0:
                    score += 0.05
                elif atr_mult > 3.0:
                    score *= 0.7

            if ctx["above_vwap"]:
                score += 0.05

            if candle_bear > 0.3:
                score *= 0.7

            return max(0.0, min(1.0, score))

        # LONG: Retest setup
        elif (current_price > prev_resistance and
              low.iloc[-1] <= resistance * 1.01 and
              close.iloc[-1] > resistance):
            score = 0.35
            if candle_bull > 0.2:
                score += 0.15
            if vol_ratio > 1.0:
                score += 0.1
            return max(0.0, min(1.0, score))

        # ── SHORT: BREAKDOWN BELOW SUPPORT ───────────────────
        elif current_price < support:
            score = 0.0

            # Close below support (not just wick)
            if close.iloc[-1] < support:
                score -= 0.25
            else:
                score -= 0.05

            if bars_since_sup < 50:
                score -= 0.05

            # Volume on breakdown
            if vol_ratio >= self.cfg["volume_multiplier"]:
                score -= 0.25
            elif vol_ratio >= 1.2:
                score -= 0.08
            else:
                score *= 0.3  # No volume = fakeout

            # Consolidation before breakdown (stored energy)
            if consolidation:
                score -= 0.15

            # Strong bearish candle
            if strong_bear_candle:
                score -= 0.15
            elif candle_bear > 0.2:
                score -= 0.08

            # ADX: trend accelerating down
            if ctx["adx"] > 20 and ctx["di_minus"] > ctx["di_plus"]:
                score -= 0.1

            # Breakdown magnitude
            if current_atr > 0:
                atr_mult = (support - current_price) / current_atr
                if 0.5 < atr_mult < 2.0:
                    score -= 0.05
                elif atr_mult > 3.0:
                    score *= 0.7  # Overextended

            # Below VWAP = selling pressure
            if not ctx["above_vwap"]:
                score -= 0.05

            # Bullish candle contradicts breakdown
            if candle_bull > 0.3:
                score *= 0.7

            return max(-1.0, min(0.0, score))

        # SHORT: Retest of broken support from below
        elif (current_price < prev_support and
              high.iloc[-1] >= support * 0.99 and
              close.iloc[-1] < support):
            score = -0.35
            if candle_bear > 0.2:
                score -= 0.15
            if vol_ratio > 1.0:
                score -= 0.1
            return max(-1.0, min(0.0, score))

        return 0.0
