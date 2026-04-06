"""
Crypto sentiment estimator — funding rate proxy from OHLCV data.

Since Alpaca doesn't provide perpetual funding rates, we estimate
crypto-specific sentiment from:
  1. Volume-momentum divergence (high vol + declining price = capitulation)
  2. Extreme move detection (>3% in 4H = overextended)
  3. Rolling return skew (positive skew = euphoria, negative = fear)

Returns a sentiment score from -1 (extreme fear) to +1 (extreme greed).
Used to penalize longs in extreme greed and shorts in extreme fear.
"""

import numpy as np
import pandas as pd
from utils import setup_logger

log = setup_logger("crypto_sentiment")


def crypto_sentiment(df: pd.DataFrame) -> dict:
    """
    Estimate crypto market sentiment from price/volume data.

    Returns:
        {
            "sentiment": float,        # -1 (fear) to +1 (greed)
            "extreme_greed": bool,     # sentiment > 0.6
            "extreme_fear": bool,      # sentiment < -0.6
            "penalize_longs": bool,    # extreme greed = contrarian bearish
            "penalize_shorts": bool,   # extreme fear = contrarian bullish
        }
    """
    if len(df) < 50:
        return {"sentiment": 0.0, "extreme_greed": False, "extreme_fear": False,
                "penalize_longs": False, "penalize_shorts": False}

    close = df["close"].values
    volume = df["volume"].values.astype(float)

    # 1. Short-term momentum (last 12 bars vs 48 bars)
    short_ret = (close[-1] / close[-12] - 1) if close[-12] > 0 else 0
    med_ret = (close[-1] / close[-48] - 1) if len(close) >= 48 and close[-48] > 0 else 0

    # 2. Volume trend (recent volume vs average)
    recent_vol = np.mean(volume[-12:])
    avg_vol = np.mean(volume[-48:]) if len(volume) >= 48 else np.mean(volume)
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

    # 3. Return skew (last 48 bars) — positive = euphoria, negative = fear
    returns = np.diff(np.log(close[-49:])) if len(close) >= 49 else np.diff(np.log(close))
    if len(returns) > 5:
        from scipy.stats import skew as calc_skew
        ret_skew = float(calc_skew(returns))
    else:
        ret_skew = 0.0

    # 4. Extreme move detection
    pct_4h = abs(short_ret)
    extreme_move = pct_4h > 0.03  # >3% in ~1 hour on 5min bars (12 bars)

    # Composite sentiment
    sentiment = 0.0

    # Momentum contribution
    sentiment += np.clip(short_ret * 10, -0.4, 0.4)

    # Volume-momentum divergence
    if vol_ratio > 1.5 and short_ret > 0.01:
        sentiment += 0.2  # high vol + up = greed
    elif vol_ratio > 1.5 and short_ret < -0.01:
        sentiment -= 0.2  # high vol + down = capitulation/fear

    # Skew contribution
    sentiment += np.clip(ret_skew * 0.15, -0.2, 0.2)

    # Extreme move penalty
    if extreme_move:
        if short_ret > 0:
            sentiment += 0.2  # euphoric pump
        else:
            sentiment -= 0.2  # panic dump

    sentiment = float(np.clip(sentiment, -1.0, 1.0))

    return {
        "sentiment": round(sentiment, 3),
        "extreme_greed": sentiment > 0.6,
        "extreme_fear": sentiment < -0.6,
        "penalize_longs": sentiment > 0.6,   # don't long at peak greed
        "penalize_shorts": sentiment < -0.6,  # don't short at peak fear
    }
