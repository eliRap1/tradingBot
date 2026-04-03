"""
Market regime filter — checks SPY/broad market before allowing individual trades.

Like Pine Script's request.security("SPY", ...) to gate entries.
Professional traders NEVER trade stocks in isolation — they check the market first.

Regimes:
  BULL:   SPY above 200 EMA, 50 EMA > 200 EMA → full size, longs allowed
  BEAR:   SPY below 200 EMA, 50 EMA < 200 EMA → reduce size, avoid longs
  CHOP:   Mixed signals → reduce size, be selective
"""

import pandas as pd
import ta
from utils import setup_logger

log = setup_logger("regime")


class RegimeFilter:
    def __init__(self, data_fetcher):
        self.data = data_fetcher
        self._last_regime = None

    def get_regime(self) -> dict:
        """
        Analyze SPY to determine market regime.

        Returns:
            {
                "regime": "bull" | "bear" | "chop",
                "allow_longs": bool,
                "size_multiplier": float,  # 0.0 to 1.0
                "spy_trend": "up" | "down" | "neutral",
                "spy_rsi": float,
                "description": str,
            }
        """
        # Fetch SPY daily data
        bars = self.data.get_bars(["SPY"], timeframe="1Day", days=250)

        if "SPY" not in bars or len(bars["SPY"]) < 50:
            log.warning("Cannot fetch SPY data — defaulting to cautious mode")
            return self._default_regime()

        df = bars["SPY"]
        close = df["close"]
        current_price = close.iloc[-1]

        # ── EMAs ─────────────────────────────────────────────
        ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        ema_200_window = min(200, len(close) - 1)
        ema_200 = ta.trend.EMAIndicator(close, window=ema_200_window).ema_indicator()

        above_200 = current_price > ema_200.iloc[-1]
        ema_50_above_200 = ema_50.iloc[-1] > ema_200.iloc[-1]

        # ── RSI ──────────────────────────────────────────────
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        spy_rsi = rsi.iloc[-1]

        # ── ADX ──────────────────────────────────────────────
        adx = ta.trend.ADXIndicator(
            df["high"], df["low"], close, window=14
        ).adx().iloc[-1]

        # ── Classify regime ──────────────────────────────────
        if above_200 and ema_50_above_200:
            regime = "bull"
            allow_longs = True
            size_mult = 1.0
            trend = "up"
            desc = f"BULL — SPY above 200 EMA, 50>200, RSI={spy_rsi:.0f}"

        elif not above_200 and not ema_50_above_200:
            regime = "bear"
            allow_longs = False  # Block new longs in bear market
            size_mult = 0.3
            trend = "down"
            desc = f"BEAR — SPY below 200 EMA, 50<200, RSI={spy_rsi:.0f}"

        else:
            regime = "chop"
            allow_longs = True
            size_mult = 0.6  # Reduce size in choppy markets
            trend = "neutral"
            desc = f"CHOP — SPY mixed signals, RSI={spy_rsi:.0f}"

        # ── Extreme RSI adjustments ──────────────────────────
        if spy_rsi > 75 and regime == "bull":
            size_mult *= 0.7
            desc += " (SPY overbought — reduce size)"

        if spy_rsi < 30 and regime == "bear":
            # Extremely oversold in bear = might bounce, don't short aggressively
            size_mult = 0.5
            desc += " (SPY deeply oversold — bounce likely)"

        # ── High ADX in bear = strong downtrend, be very cautious ──
        if regime == "bear" and adx > 30:
            size_mult = 0.2
            desc += " (Strong downtrend — minimal exposure)"

        result = {
            "regime": regime,
            "allow_longs": allow_longs,
            "size_multiplier": round(size_mult, 2),
            "spy_trend": trend,
            "spy_rsi": round(spy_rsi, 1),
            "description": desc,
        }

        # Log regime change
        if self._last_regime != regime:
            log.info(f"REGIME: {desc}")
            self._last_regime = regime

        return result

    def _default_regime(self):
        """Fallback when SPY data unavailable — be cautious."""
        return {
            "regime": "chop",
            "allow_longs": True,
            "size_multiplier": 0.5,
            "spy_trend": "neutral",
            "spy_rsi": 50.0,
            "description": "UNKNOWN — SPY data unavailable, cautious mode",
        }
