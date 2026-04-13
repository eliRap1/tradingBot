import pandas as pd
import ta
from indicators import vwap_bands, rvol
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.vwap_reclaim")


class VWAPReclaimStrategy:
    """
    VWAP as dynamic support/resistance — NOT a simple cross strategy.

    LONG: Price pulls back to VWAP from above with volume spike + bullish candle
          (institutions defending fair value as support)

    SHORT: Price rallies to VWAP from below with volume spike + bearish candle
           (institutions defending fair value as resistance)

    Filters:
    - Requires rvol > 1.3 AND confirming candle pattern
    - Chop detection: >3 VWAP crosses in 20 bars = score reduction
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

        pct_from_vwap = (current_price - current_vwap) / current_vwap
        above_vwap = current_price > current_vwap

        vwap_crosses = self._count_vwap_crosses(close, vwap_line, lookback=20)
        if vwap_crosses > 2:
            return 0.0  # Choppy market — skip entirely

        vol_ratio = rvol(df)
        if vol_ratio < 1.3:
            return 0.0

        pullback_to_support = self._pullback_to_vwap_from_above(close, vwap_line)
        rally_to_resistance = self._rally_to_vwap_from_below(close, vwap_line)

        ctx = get_trend_context(df)

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]

        if pullback_to_support:
            score = self._score_long(pct_from_vwap, vol_ratio, ctx, current_rsi)
            if score > 0.15:
                return max(0.0, min(1.0, score))

        if rally_to_resistance:
            score = self._score_short(pct_from_vwap, vol_ratio, ctx, current_rsi)
            if score < -0.15:
                return max(-1.0, min(0.0, score))

        if not pd.isna(lower_1.iloc[-1]) and current_price <= lower_1.iloc[-1]:
            if current_rsi < 40 and vol_ratio > 1.3:
                return 0.20

        if not pd.isna(upper_1.iloc[-1]) and current_price >= upper_1.iloc[-1]:
            if current_rsi > 60 and vol_ratio > 1.3:
                return -0.20

        return 0.0

    def _count_vwap_crosses(self, close: pd.Series, vwap: pd.Series, lookback: int) -> int:
        n = min(lookback, len(close) - 1)
        crosses = 0
        for i in range(1, n):
            prev_above = close.iloc[-(i + 1)] > vwap.iloc[-(i + 1)]
            curr_above = close.iloc[-i] > vwap.iloc[-i]
            if prev_above != curr_above:
                crosses += 1
        return crosses

    def _pullback_to_vwap_from_above(self, close: pd.Series, vwap: pd.Series) -> bool:
        lookback = min(10, len(close) - 1)
        was_above = False
        for i in range(3, lookback):
            if close.iloc[-i] > vwap.iloc[-i] * 1.002:
                was_above = True
                break

        near_vwap = abs(close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] < 0.01
        price_above_or_at = close.iloc[-1] >= vwap.iloc[-1] * 0.998

        return was_above and near_vwap and price_above_or_at

    def _rally_to_vwap_from_below(self, close: pd.Series, vwap: pd.Series) -> bool:
        lookback = min(10, len(close) - 1)
        was_below = False
        for i in range(3, lookback):
            if close.iloc[-i] < vwap.iloc[-i] * 0.998:
                was_below = True
                break

        near_vwap = abs(close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] < 0.01
        price_below_or_at = close.iloc[-1] <= vwap.iloc[-1] * 1.002

        return was_below and near_vwap and price_below_or_at

    def _score_long(self, pct_from_vwap, vol_ratio, ctx, rsi):
        score = 0.25

        if vol_ratio >= 2.0:
            score += 0.20
        elif vol_ratio >= 1.5:
            score += 0.15
        elif vol_ratio >= 1.3:
            score += 0.05

        if abs(pct_from_vwap) < 0.001:
            score += 0.10
        elif pct_from_vwap > 0.001:
            score += 0.05

        if 35 <= rsi <= 60:
            score += 0.10
        elif rsi > 75:
            score -= 0.15

        if ctx["direction"] == "up":
            score += 0.10
        elif ctx["direction"] == "down" and ctx.get("strong_trend"):
            score *= 0.5

        return score

    def _score_short(self, pct_from_vwap, vol_ratio, ctx, rsi):
        score = -0.25

        if vol_ratio >= 2.0:
            score -= 0.20
        elif vol_ratio >= 1.5:
            score -= 0.15
        elif vol_ratio >= 1.3:
            score -= 0.05

        if abs(pct_from_vwap) < 0.001:
            score -= 0.10
        elif pct_from_vwap < -0.001:
            score -= 0.05

        if 40 <= rsi <= 65:
            score -= 0.10
        elif rsi < 25:
            score += 0.15

        if ctx["direction"] == "down":
            score -= 0.10
        elif ctx["direction"] == "up" and ctx.get("strong_trend"):
            score *= 0.5

        return score
