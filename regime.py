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
    def __init__(self, data_fetcher, universe: list[str] = None):
        self.data = data_fetcher
        self._last_regime = None
        self._universe = universe or []

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

        # ── Market breadth: % of universe above 50 EMA ────────
        breadth = self._get_market_breadth()
        breadth_pct = breadth["pct_above_50ema"]

        if breadth_pct < 30 and regime != "bear":
            # Weak breadth even in "bull" SPY = hidden weakness
            size_mult *= 0.7
            desc += f" (weak breadth: {breadth_pct:.0f}% above 50EMA)"
        elif breadth_pct > 70 and regime == "bull":
            # Strong breadth confirms bull
            size_mult = min(size_mult * 1.1, 1.0)
            desc += f" (strong breadth: {breadth_pct:.0f}%)"
        elif breadth_pct < 50:
            desc += f" (breadth: {breadth_pct:.0f}%)"

        result = {
            "regime": regime,
            "allow_longs": allow_longs,
            "size_multiplier": round(size_mult, 2),
            "spy_trend": trend,
            "spy_rsi": round(spy_rsi, 1),
            "breadth_pct": round(breadth_pct, 1),
            "description": desc,
        }

        # Log regime change
        if self._last_regime != regime:
            log.info(f"REGIME: {desc}")
            self._last_regime = regime

        return result

    def _get_market_breadth(self) -> dict:
        """Calculate % of universe stocks above their 50 EMA."""
        if not self._universe:
            return {"pct_above_50ema": 50.0, "total": 0, "above": 0}

        try:
            # Sample up to 20 stocks for speed
            sample = self._universe[:20]
            bars = self.data.get_bars(sample, timeframe="1Day", days=80)

            above = 0
            total = 0
            for sym, df in bars.items():
                if len(df) < 50:
                    continue
                total += 1
                ema50 = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
                if df["close"].iloc[-1] > ema50.iloc[-1]:
                    above += 1

            pct = (above / total * 100) if total > 0 else 50.0
            return {"pct_above_50ema": pct, "total": total, "above": above}
        except Exception as e:
            log.error(f"Breadth calculation failed: {e}")
            return {"pct_above_50ema": 50.0, "total": 0, "above": 0}

    def _default_regime(self):
        """Fallback when SPY data unavailable — be cautious."""
        return {
            "regime": "chop",
            "allow_longs": True,
            "size_multiplier": 0.5,
            "spy_trend": "neutral",
            "spy_rsi": 50.0,
            "breadth_pct": 50.0,
            "description": "UNKNOWN — SPY data unavailable, cautious mode",
        }
