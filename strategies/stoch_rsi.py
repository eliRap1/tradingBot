import pandas as pd
import ta
from indicators import stochastic_rsi, crossover, crossunder
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context
from utils import setup_logger

log = setup_logger("strategy.stoch_rsi")


class StochRSIStrategy:
    """
    Stochastic RSI — catches pullbacks in BOTH trends.

    LONG: Uptrend + StochRSI crosses up from oversold (buy the dip)
    SHORT: Downtrend + StochRSI crosses down from overbought (sell the rally)
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

        # Stochastic RSI
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

        # Crossovers
        k_cross_up = current_k > current_d and prev_k <= d.iloc[-2]
        k_cross_down = current_k < current_d and prev_k >= d.iloc[-2]

        cross_up_series = crossover(k, d)
        cross_down_series = crossunder(k, d)
        recent_cross_up = cross_up_series.iloc[-3:].any()
        recent_cross_down = cross_down_series.iloc[-3:].any()

        # Zones
        oversold = current_k < self.cfg["oversold"]
        overbought = current_k > self.cfg["overbought"]
        was_oversold = k.iloc[-3:].min() < self.cfg["oversold"]
        was_overbought = k.iloc[-3:].max() > self.cfg["overbought"]

        # Trend filter (50 EMA)
        ema_period = self.cfg["ema_period"]
        ema = ta.trend.EMAIndicator(close, window=ema_period).ema_indicator()
        above_ema = close.iloc[-1] > ema.iloc[-1]

        ctx = get_trend_context(df)

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]

        patterns = detect_patterns(df)
        candle_bull = bullish_score(patterns)
        candle_bear = bearish_score(patterns)

        avg_vol = volume.tail(20).mean()
        vol_ok = volume.iloc[-1] > avg_vol * 0.8

        # ── LONG: Buy pullback in uptrend ────────────────────
        if above_ema or ctx["direction"] != "down":
            long_score = 0.0

            if k_cross_up and was_oversold:
                long_score += 0.4
            elif recent_cross_up and was_oversold:
                long_score += 0.25
            elif k_cross_up and current_k < 40:
                long_score += 0.15

            if long_score > 0:
                if above_ema and ctx["direction"] == "up":
                    long_score += 0.15
                elif above_ema:
                    long_score += 0.05
                elif not above_ema:
                    long_score -= 0.1

                if 35 <= current_rsi <= 60:
                    long_score += 0.1
                elif current_rsi > 70:
                    long_score -= 0.2

                if candle_bull > 0.2:
                    long_score += candle_bull * 0.15
                elif candle_bull == 0:
                    long_score *= 0.8

                if vol_ok:
                    long_score += 0.05

                if ctx["above_ema_200"]:
                    long_score += 0.05

                if long_score > 0.15:
                    return max(0.0, min(1.0, long_score))

        # ── SHORT: Sell rally in downtrend ───────────────────
        if not above_ema or ctx["direction"] != "up":
            short_score = 0.0

            # StochRSI crosses DOWN from overbought
            if k_cross_down and was_overbought:
                short_score -= 0.4
            elif recent_cross_down and was_overbought:
                short_score -= 0.25
            elif k_cross_down and current_k > 60:
                short_score -= 0.15

            if short_score < 0:
                # Below EMA = downtrend confirmed
                if not above_ema and ctx["direction"] == "down":
                    short_score -= 0.15
                elif not above_ema:
                    short_score -= 0.05
                elif above_ema:
                    short_score += 0.1  # Above EMA = risky short

                # RSI in bearish zone
                if 40 <= current_rsi <= 65:
                    short_score -= 0.1
                elif current_rsi < 30:
                    short_score += 0.2  # Already oversold, don't short

                # Bearish candle confirmation
                if candle_bear > 0.2:
                    short_score -= candle_bear * 0.15
                elif candle_bear == 0:
                    short_score *= 0.8

                if vol_ok:
                    short_score -= 0.05

                # Below 200 EMA = long-term downtrend
                if not ctx["above_ema_200"]:
                    short_score -= 0.05

                # Already oversold = don't short
                if oversold:
                    short_score += 0.3

                if short_score < -0.15:
                    return max(-1.0, min(0.0, short_score))

        return 0.0
