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
    Enhanced Momentum strategy — EMA crossover + RSI + MACD + Volume for LONG and SHORT.

    IMPROVEMENTS for profit maximization:
    - Added volume profile scoring (institutional activity detection)
    - Added momentum acceleration detection
    - Added price location vs recent range
    - Added RSI divergence detection
    - More aggressive scoring when multiple factors align
    
    LONG: EMA cross up, RSI 35-65 (room to run), MACD positive, volume surge
    SHORT: EMA cross down, RSI 35-65, MACD negative, below 200 EMA
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
        
        # Rate of change acceleration (momentum of momentum)
        roc_accel = roc.iloc[-1] - roc.iloc[-3] if len(roc) > 3 else 0

        macd = ta.trend.MACD(close)
        macd_diff = macd.macd_diff()
        macd_hist_rising = macd_diff.iloc[-1] > macd_diff.iloc[-2]
        macd_hist_falling = macd_diff.iloc[-1] < macd_diff.iloc[-2]
        macd_positive = macd_diff.iloc[-1] > 0
        macd_negative = macd_diff.iloc[-1] < 0
        
        # MACD cross (signal line cross)
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        macd_cross_up = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
        macd_cross_down = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        vol_ratio = rvol(df)
        vol_strong = vol_ratio > 1.3  # Raised threshold
        vol_surge = vol_ratio > 2.0  # Institutional-level volume
        
        # Price location in 20-day range
        range_high = high.rolling(20).max().iloc[-1]
        range_low = low.rolling(20).min().iloc[-1]
        price_location = (close.iloc[-1] - range_low) / (range_high - range_low) if range_high > range_low else 0.5
        
        # RSI divergence detection
        price_higher = close.iloc[-1] > close.iloc[-5]
        rsi_higher = current_rsi > rsi.iloc[-5]
        bullish_divergence = not price_higher and rsi_higher  # Price lower but RSI higher
        bearish_divergence = price_higher and not rsi_higher  # Price higher but RSI lower

        # ── LONG SCORING ─────────────────────────────────────
        if ctx["direction"] != "down" or not ctx["trending"]:
            long_score = 0.0

            # EMA signals (core)
            if ema_cross_up:
                long_score += 0.40  # Increased from 0.35
            elif recent_cross_up and ema_bullish:
                long_score += 0.25  # Increased from 0.2
            elif ema_bullish:
                long_score += 0.12

            # RSI sweet spot (room to run)
            if 35 <= current_rsi <= 60:  # Tightened range
                long_score += 0.18  # Increased
            elif current_rsi < self.cfg["rsi_oversold"] and prev_rsi < current_rsi:
                long_score += 0.25  # Bouncing from oversold = high conviction
            elif current_rsi > 75:  # More strict overbought
                long_score -= 0.35

            # MACD confirmation
            if macd_cross_up:
                long_score += 0.20  # New: MACD cross is strong signal
            elif macd_positive and macd_hist_rising:
                long_score += 0.15
            elif macd_hist_rising:
                long_score += 0.08

            # ROC momentum
            if current_roc > 3:
                long_score += 0.12
            if roc_accel > 1:  # Accelerating momentum
                long_score += 0.10

            # Candle patterns
            if long_score > 0.1:
                long_score += candle_bull * 0.25  # Increased weight
            long_score -= candle_bear * 0.15

            # Volume confirmation (critical for momentum)
            if vol_surge:
                long_score *= 1.30  # Major boost for institutional volume
            elif vol_strong:
                long_score *= 1.15
            elif vol_ratio < 0.7:  # Weak volume = weak signal
                long_score *= 0.75

            # Trend alignment
            if ctx["above_ema_200"] and ctx["direction"] == "up":
                long_score *= 1.12
            if ctx["above_vwap"]:
                long_score += 0.06
                
            # Price location bonus
            if price_location < 0.3:  # Near range low = better risk/reward
                long_score *= 1.10
            elif price_location > 0.85:  # Near range high = chase risk
                long_score *= 0.85
                
            # RSI divergence bonus
            if bullish_divergence:
                long_score *= 1.20

            if long_score > 0.12:  # Lowered threshold for more signals
                return max(0.0, min(1.0, long_score))

        # ── SHORT SCORING ────────────────────────────────────
        if ctx["direction"] != "up" or not ctx["trending"]:
            short_score = 0.0

            # EMA cross down = bearish momentum
            if ema_cross_down:
                short_score -= 0.40
            elif recent_cross_down and ema_bearish:
                short_score -= 0.25
            elif ema_bearish:
                short_score -= 0.12

            # RSI in bearish zone
            if 40 <= current_rsi <= 65:
                short_score -= 0.18
            elif current_rsi > 75 and prev_rsi > current_rsi:
                short_score -= 0.25  # Falling from overbought
            elif current_rsi < 25:
                short_score += 0.35  # Already oversold, don't short

            # MACD confirmation
            if macd_cross_down:
                short_score -= 0.20
            elif macd_negative and macd_hist_falling:
                short_score -= 0.15
            elif macd_hist_falling:
                short_score -= 0.08

            # Negative ROC = price falling
            if current_roc < -3:
                short_score -= 0.12
            if roc_accel < -1:
                short_score -= 0.10

            # Bearish candle confirmation
            if short_score < -0.1:
                short_score -= candle_bear * 0.25
            short_score += candle_bull * 0.15

            # Volume on breakdown
            if vol_surge:
                short_score *= 1.30
            elif vol_strong:
                short_score *= 1.15
            elif vol_ratio < 0.7:
                short_score *= 0.75

            # Below 200 EMA = downtrend confirmed
            if not ctx["above_ema_200"] and ctx["direction"] == "down":
                short_score *= 1.12
            if not ctx["above_vwap"]:
                short_score -= 0.06
                
            # Price location
            if price_location > 0.7:  # Near range high = good short entry
                short_score *= 1.10
            elif price_location < 0.15:
                short_score *= 0.85
                
            # Bearish divergence
            if bearish_divergence:
                short_score *= 1.20

            if short_score < -0.12:
                return max(-1.0, min(0.0, short_score))

        return 0.0
