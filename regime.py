"""
Market regime filter - checks SPY/broad market before allowing individual trades.
"""

import random
import time

import numpy as np
import pandas as pd
import ta

from utils import setup_logger

log = setup_logger("regime")


class RegimeFilter:
    def __init__(self, data_fetcher, universe: list[str] = None):
        self.data = data_fetcher
        self._last_regime = None
        self._universe = universe or []
        self._hmm_model = None
        self._hmm_state_map = {}
        self._hmm_last_fit: float = 0.0
        self._hmm_refit_interval: int = 3600

    def get_regime(self) -> dict:
        spy_df = self.data.get_intraday_bars("SPY", timeframe="1Day", days=250)
        if spy_df is None or len(spy_df) < 50:
            log.warning("Cannot fetch SPY data - defaulting to cautious mode")
            return self._default_regime()

        df = spy_df
        close = df["close"]
        current_price = close.iloc[-1]
        ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        ema_200_window = min(200, len(close) - 1)
        ema_200 = ta.trend.EMAIndicator(close, window=ema_200_window).ema_indicator()
        above_200 = current_price > ema_200.iloc[-1]
        ema_50_above_200 = ema_50.iloc[-1] > ema_200.iloc[-1]

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        spy_rsi = rsi.iloc[-1]
        adx = ta.trend.ADXIndicator(df["high"], df["low"], close, window=14).adx().iloc[-1]

        if above_200 and ema_50_above_200:
            regime = "bull"
            allow_longs = True
            size_mult = 1.0
            trend = "up"
            desc = f"BULL - SPY above 200 EMA, 50>200, RSI={spy_rsi:.0f}"
        elif not above_200 and not ema_50_above_200:
            regime = "bear"
            allow_longs = False
            size_mult = 0.3
            trend = "down"
            desc = f"BEAR - SPY below 200 EMA, 50<200, RSI={spy_rsi:.0f}"
        else:
            regime = "chop"
            allow_longs = True
            size_mult = 0.6
            trend = "neutral"
            desc = f"CHOP - SPY mixed signals, RSI={spy_rsi:.0f}"

        if spy_rsi > 75 and regime == "bull":
            size_mult *= 0.7
            desc += " (SPY overbought - reduce size)"
        if spy_rsi < 30 and regime == "bear":
            size_mult = 0.5
            desc += " (SPY deeply oversold - bounce likely)"
        if regime == "bear" and adx > 30:
            size_mult = 0.2
            desc += " (Strong downtrend - minimal exposure)"

        hmm = self._get_hmm_regime(df)
        if hmm and hmm["confidence"] > 0.7:
            hmm_state = hmm["state"]
            if hmm_state != regime:
                size_mult *= 0.7
                desc += f" (HMM says {hmm_state.upper()} @ {hmm['confidence']:.0%})"
            else:
                size_mult = min(size_mult * 1.1, 1.0)
                desc += f" (HMM confirms @ {hmm['confidence']:.0%})"
            if hmm_state == "bear" and hmm["confidence"] > 0.92 and regime == "bull":
                size_mult = min(size_mult, 0.25)
                desc += " [HMM CAUTION: size capped at 25%]"

        atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14).average_true_range()
        atr_pct = (atr_ind / close) * 100
        atr_pct_current = atr_pct.iloc[-1]
        atr_pct_avg = atr_pct.iloc[-60:].mean() if len(atr_pct) >= 60 else atr_pct.mean()
        if atr_pct_avg > 0 and atr_pct_current > 1.5 * atr_pct_avg:
            atr_regime = "high_vol"
            size_mult *= 0.8
            desc += " (HIGH VOL - ATR elevated, tighter stops recommended)"
        elif atr_pct_avg > 0 and atr_pct_current < 0.5 * atr_pct_avg:
            atr_regime = "low_vol"
            desc += " (LOW VOL - signals may be false breakouts)"
        else:
            atr_regime = "normal"

        breadth = self._get_market_breadth()
        breadth_pct = breadth["pct_above_50ema"]
        if breadth_pct < 30 and regime != "bear":
            size_mult *= 0.7
            desc += f" (weak breadth: {breadth_pct:.0f}% above 50EMA)"
        elif breadth_pct > 70 and regime == "bull":
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
            "hmm_regime": hmm["state"] if hmm else None,
            "hmm_confidence": hmm["confidence"] if hmm else None,
            "atr_regime": atr_regime,
            "description": desc,
        }

        if self._last_regime != regime:
            log.info(f"REGIME: {desc}")
            self._last_regime = regime

        return result

    def classify_4state(self) -> str:
        """Classify the market into bull/bear x trending/choppy."""
        spy_df = self.data.get_intraday_bars("SPY", timeframe="1Day", days=250)
        if spy_df is None or len(spy_df) < 60:
            return "bull_choppy"
        return self._classify_4state_from_df(spy_df)

    def _get_hmm_regime(self, df: pd.DataFrame) -> dict | None:
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401
        except ImportError:
            return None

        if len(df) < 100:
            return None

        try:
            close = df["close"].values
            returns = np.diff(np.log(close))
            vol_window = 20
            if len(returns) < vol_window + 10:
                return None
            vol = pd.Series(returns).rolling(vol_window).std().values
            valid = ~np.isnan(vol)
            returns_valid = returns[valid]
            vol_valid = vol[valid]
            if len(returns_valid) < 60:
                return None

            features = np.column_stack([returns_valid, vol_valid])
            needs_refit = (
                self._hmm_model is None
                or time.time() - self._hmm_last_fit > self._hmm_refit_interval
            )
            if needs_refit:
                self._fit_hmm(features)
            if self._hmm_model is None:
                return None

            state_seq = self._hmm_model.predict(features)
            current_state = state_seq[-1]
            probs = self._hmm_model.predict_proba(features[-1:].reshape(1, -1))[0]
            regime = self._hmm_state_map.get(current_state, "chop")
            confidence = float(probs[current_state])
            return {
                "state": regime,
                "confidence": confidence,
                "probs": [float(p) for p in probs],
            }
        except Exception as e:
            log.warning(f"HMM regime detection failed: {e}")
            return None

    def _fit_hmm(self, features: np.ndarray):
        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(
                n_components=3,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            model.fit(features)
            mean_returns = model.means_[:, 0]
            sorted_indices = np.argsort(mean_returns)
            self._hmm_state_map = {
                sorted_indices[0]: "bear",
                sorted_indices[1]: "chop",
                sorted_indices[2]: "bull",
            }
            self._hmm_model = model
            self._hmm_last_fit = time.time()
        except Exception as e:
            log.warning(f"HMM fit failed: {e}")
            self._hmm_model = None

    def _get_market_breadth(self) -> dict:
        if not self._universe:
            return {"pct_above_50ema": 50.0, "total": 0, "above": 0}

        try:
            sample_size = min(30, len(self._universe))
            sample = random.sample(self._universe, sample_size)
            bars = self.data.get_bars(sample, timeframe="1Day", days=80)

            above = 0
            total = 0
            for _, df in bars.items():
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
        return {
            "regime": "chop",
            "allow_longs": True,
            "size_multiplier": 0.5,
            "spy_trend": "neutral",
            "spy_rsi": 50.0,
            "breadth_pct": 50.0,
            "description": "UNKNOWN - SPY data unavailable, cautious mode",
        }

    def _classify_4state_from_df(self, daily_bars: pd.DataFrame) -> str:
        close = daily_bars["close"]
        high = daily_bars["high"]
        low = daily_bars["low"]
        adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        trending = adx > 25
        ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        bull = close.iloc[-1] > ema50.iloc[-1]
        if bull and trending:
            return "bull_trending"
        if bull and not trending:
            return "bull_choppy"
        if not bull and trending:
            return "bear_trending"
        return "bear_choppy"
