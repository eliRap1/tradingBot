"""
Market regime filter — checks SPY/broad market before allowing individual trades.

Like Pine Script's request.security("SPY", ...) to gate entries.
Professional traders NEVER trade stocks in isolation — they check the market first.

Regimes:
  BULL:   SPY above 200 EMA, 50 EMA > 200 EMA → full size, longs allowed
  BEAR:   SPY below 200 EMA, 50 EMA < 200 EMA → reduce size, avoid longs
  CHOP:   Mixed signals → reduce size, be selective

HMM layer: 3-state Hidden Markov Model on returns + volatility for
probabilistic regime transitions. Overrides EMA regime when high confidence.
"""

import pandas as pd
import numpy as np
import ta
from utils import setup_logger

log = setup_logger("regime")


class RegimeFilter:
    def __init__(self, data_fetcher, universe: list[str] = None):
        self.data = data_fetcher
        self._last_regime = None
        self._universe = universe or []
        self._hmm_model = None
        self._hmm_state_map = {}  # maps HMM state index → "bull"/"bear"/"chop"

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

        # ── HMM regime overlay ────────────────────────────────
        hmm = self._get_hmm_regime(df)
        if hmm and hmm["confidence"] > 0.7:
            hmm_state = hmm["state"]
            if hmm_state != regime:
                # HMM disagrees with EMA regime — reduce size (conflicting signals)
                size_mult *= 0.7
                desc += f" (HMM says {hmm_state.upper()} @ {hmm['confidence']:.0%})"
            else:
                # HMM confirms EMA regime — slight size boost
                size_mult = min(size_mult * 1.1, 1.0)
                desc += f" (HMM confirms @ {hmm['confidence']:.0%})"

            # HMM override: if EMA says bull but HMM strongly says bear, block longs
            if hmm_state == "bear" and hmm["confidence"] > 0.85 and regime == "bull":
                allow_longs = False
                size_mult = 0.4
                desc += " [HMM OVERRIDE: blocking longs]"

        # ── ATR volatility regime ─────────────────────────────
        atr_ind = ta.volatility.AverageTrueRange(
            df["high"], df["low"], close, window=14
        ).average_true_range()
        atr_pct = (atr_ind / close) * 100
        atr_pct_current = atr_pct.iloc[-1]
        atr_pct_avg = atr_pct.iloc[-60:].mean() if len(atr_pct) >= 60 else atr_pct.mean()

        if atr_pct_avg > 0 and atr_pct_current > 1.5 * atr_pct_avg:
            atr_regime = "high_vol"
            size_mult *= 0.8
            desc += " (HIGH VOL — ATR elevated, tighter stops recommended)"
        elif atr_pct_avg > 0 and atr_pct_current < 0.5 * atr_pct_avg:
            atr_regime = "low_vol"
            desc += " (LOW VOL — signals may be false breakouts)"
        else:
            atr_regime = "normal"

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
            "hmm_regime": hmm["state"] if hmm else None,
            "hmm_confidence": hmm["confidence"] if hmm else None,
            "atr_regime": atr_regime,
            "description": desc,
        }

        # Log regime change
        if self._last_regime != regime:
            log.info(f"REGIME: {desc}")
            self._last_regime = regime

        return result

    def _get_hmm_regime(self, df: pd.DataFrame) -> dict | None:
        """
        Fit a 3-state Gaussian HMM on SPY returns + realized volatility.

        Returns:
            {"state": "bull"|"bear"|"chop", "confidence": float, "probs": [p0, p1, p2]}
            or None if HMM fails (not enough data, library issue).

        States are identified by their mean return:
          - Highest mean return → bull
          - Lowest mean return → bear
          - Middle → chop
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            return None

        if len(df) < 100:
            return None

        try:
            close = df["close"].values
            returns = np.diff(np.log(close))  # log returns

            # Rolling 20-day realized volatility
            vol_window = 20
            if len(returns) < vol_window + 10:
                return None
            vol = pd.Series(returns).rolling(vol_window).std().values

            # Align: drop NaN from rolling vol
            valid = ~np.isnan(vol)
            returns_valid = returns[valid]
            vol_valid = vol[valid]

            if len(returns_valid) < 60:
                return None

            features = np.column_stack([returns_valid, vol_valid])

            # Fit HMM (3 states, diagonal covariance for speed)
            model = GaussianHMM(
                n_components=3,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            model.fit(features)

            # Predict current state
            state_seq = model.predict(features)
            current_state = state_seq[-1]

            # State probabilities for the last observation
            probs = model.predict_proba(features[-1:].reshape(1, -1))[0]

            # Map states by mean return: highest=bull, lowest=bear, middle=chop
            mean_returns = model.means_[:, 0]
            sorted_indices = np.argsort(mean_returns)
            state_map = {
                sorted_indices[0]: "bear",
                sorted_indices[1]: "chop",
                sorted_indices[2]: "bull",
            }

            self._hmm_model = model
            self._hmm_state_map = state_map

            regime = state_map[current_state]
            confidence = float(probs[current_state])

            return {
                "state": regime,
                "confidence": confidence,
                "probs": [float(p) for p in probs],
            }
        except Exception as e:
            log.warning(f"HMM regime detection failed: {e}")
            return None

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
