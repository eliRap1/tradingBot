import pandas as pd
import ta
from indicators import vwap_bands, rvol
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.vwap_reclaim")


class VWAPReclaimStrategy:
    """
    VWAP Reclaim / Rejection — institutional intraday strategy.

    LONG: Price dips below VWAP, then reclaims it with volume
          (institutions buying the dip at fair value)

    SHORT: Price rallies above VWAP, then rejects back below
           (institutions selling into strength)

    Key: VWAP is the institutional "fair value" benchmark.
    Reclaims/rejections with volume are high-probability setups.
    """

    def __init__(self, config: dict):
        self.cfg = config["strategies"].get("vwap_reclaim", {})

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

        if "vwap" not in df.columns:
            # Calculate VWAP from OHLCV
            vwap_data = vwap_bands(df)
            vwap_line = vwap_data["vwap"]
            upper_1 = vwap_data["upper_1"]
            lower_1 = vwap_data["lower_1"]
        else:
            vwap_line = df["vwap"]
            std = (df["high"] - df["low"]).rolling(20).std()
            upper_1 = vwap_line + std
            lower_1 = vwap_line - std

        current_price = close.iloc[-1]
        current_vwap = vwap_line.iloc[-1]

        if pd.isna(current_vwap) or current_vwap <= 0:
            return 0.0

        # Price position relative to VWAP
        above_vwap = current_price > current_vwap
        pct_from_vwap = (current_price - current_vwap) / current_vwap

        # Check recent VWAP crossings (last 3-5 bars)
        was_below = any(close.iloc[-j] < vwap_line.iloc[-j] for j in range(2, min(6, len(close))))
        was_above = any(close.iloc[-j] > vwap_line.iloc[-j] for j in range(2, min(6, len(close))))

        # Reclaim: was below, now above (crossed up through VWAP)
        reclaimed = above_vwap and was_below
        # Rejection: was above, now below (crossed down through VWAP)
        rejected = not above_vwap and was_above

        # RVOL for volume confirmation
        vol_ratio = rvol(df)

        # Trend context
        ctx = get_trend_context(df)

        # RSI for momentum
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]

        # Candle patterns
        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        # ── LONG: VWAP Reclaim ───────────────────────────────
        if reclaimed:
            score = 0.0

            # Core signal: price reclaimed VWAP
            score += 0.30

            # Volume confirmation (institutions buying)
            if vol_ratio >= 1.5:
                score += 0.20
            elif vol_ratio >= 1.2:
                score += 0.10
            else:
                score *= 0.5  # No volume = weak reclaim

            # Close firmly above VWAP (not just barely)
            if pct_from_vwap > 0.002:  # >0.2% above
                score += 0.10
            elif pct_from_vwap > 0.001:
                score += 0.05

            # RSI not overbought
            if 40 <= current_rsi <= 65:
                score += 0.10
            elif current_rsi > 75:
                score -= 0.15

            # Bullish candle on reclaim
            if candle_bull > 0.2:
                score += candle_bull * 0.15

            # Trend alignment
            if ctx["direction"] == "up":
                score += 0.10
            elif ctx["direction"] == "down" and ctx["strong_trend"]:
                score *= 0.5  # Fighting strong downtrend

            if score > 0.15:
                return max(0.0, min(1.0, score))

        # ── SHORT: VWAP Rejection ────────────────────────────
        if rejected:
            score = 0.0

            # Core signal: price rejected from VWAP
            score -= 0.30

            # Volume on rejection
            if vol_ratio >= 1.5:
                score -= 0.20
            elif vol_ratio >= 1.2:
                score -= 0.10
            else:
                score *= 0.5

            # Firmly below VWAP
            if pct_from_vwap < -0.002:
                score -= 0.10
            elif pct_from_vwap < -0.001:
                score -= 0.05

            # RSI not oversold
            if 35 <= current_rsi <= 60:
                score -= 0.10
            elif current_rsi < 25:
                score += 0.15  # Already oversold

            # Bearish candle on rejection
            if candle_bear > 0.2:
                score -= candle_bear * 0.15

            # Trend alignment
            if ctx["direction"] == "down":
                score -= 0.10
            elif ctx["direction"] == "up" and ctx["strong_trend"]:
                score *= 0.5  # Fighting strong uptrend

            if score < -0.15:
                return max(-1.0, min(0.0, score))

        # ── Proximity plays (near VWAP bands) ────────────────
        # Price touching lower band = potential long bounce
        if not pd.isna(lower_1.iloc[-1]) and current_price <= lower_1.iloc[-1]:
            if current_rsi < 40 and candle_bull > 0.2 and vol_ratio > 1.0:
                return min(0.25, candle_bull * 0.3 + 0.1)

        # Price touching upper band = potential short fade
        if not pd.isna(upper_1.iloc[-1]) and current_price >= upper_1.iloc[-1]:
            if current_rsi > 60 and candle_bear > 0.2 and vol_ratio > 1.0:
                return max(-0.25, -(candle_bear * 0.3 + 0.1))

        return 0.0
