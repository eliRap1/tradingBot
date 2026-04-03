"""
Candlestick pattern recognition — reads OHLC candles like a real trader.

Patterns detected (same logic Pine Script's ta.* candle functions use):
  Bullish: hammer, bullish_engulfing, morning_star, three_white_soldiers, dragonfly_doji, piercing_line
  Bearish: shooting_star, bearish_engulfing, evening_star, three_black_crows, gravestone_doji, dark_cloud_cover
  Neutral: doji, spinning_top
"""

import pandas as pd
import numpy as np
from utils import setup_logger

log = setup_logger("candles")


def detect_patterns(df: pd.DataFrame) -> dict[str, bool]:
    """
    Detect candlestick patterns on the last few candles of a DataFrame.
    Returns dict of pattern_name -> True/False.
    """
    if len(df) < 4:
        return {}

    o = df["open"].values if "open" in df.columns else None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # If we don't have open prices, we can't read candles properly
    if o is None:
        return {}

    patterns = {}

    # Current and previous candles (index from end)
    # -1 = current, -2 = prev, -3 = two bars ago
    patterns["doji"] = _is_doji(o, h, l, c, -1)
    patterns["hammer"] = _is_hammer(o, h, l, c, -1)
    patterns["shooting_star"] = _is_shooting_star(o, h, l, c, -1)
    patterns["dragonfly_doji"] = _is_dragonfly_doji(o, h, l, c, -1)
    patterns["gravestone_doji"] = _is_gravestone_doji(o, h, l, c, -1)
    patterns["spinning_top"] = _is_spinning_top(o, h, l, c, -1)

    # Two-candle patterns
    patterns["bullish_engulfing"] = _is_bullish_engulfing(o, h, l, c)
    patterns["bearish_engulfing"] = _is_bearish_engulfing(o, h, l, c)
    patterns["piercing_line"] = _is_piercing_line(o, h, l, c)
    patterns["dark_cloud_cover"] = _is_dark_cloud_cover(o, h, l, c)

    # Three-candle patterns
    patterns["morning_star"] = _is_morning_star(o, h, l, c)
    patterns["evening_star"] = _is_evening_star(o, h, l, c)
    patterns["three_white_soldiers"] = _is_three_white_soldiers(o, h, l, c)
    patterns["three_black_crows"] = _is_three_black_crows(o, h, l, c)

    return patterns


def bullish_score(patterns: dict[str, bool]) -> float:
    """Score from 0 to 1 based on bullish candle signals."""
    score = 0.0

    # Strong bullish patterns
    if patterns.get("bullish_engulfing"):
        score += 0.4
    if patterns.get("morning_star"):
        score += 0.45
    if patterns.get("three_white_soldiers"):
        score += 0.5
    if patterns.get("piercing_line"):
        score += 0.3

    # Moderate bullish
    if patterns.get("hammer"):
        score += 0.3
    if patterns.get("dragonfly_doji"):
        score += 0.2

    # Indecision (slight bullish bias in oversold context)
    if patterns.get("doji"):
        score += 0.05

    return min(score, 1.0)


def bearish_score(patterns: dict[str, bool]) -> float:
    """Score from 0 to 1 based on bearish candle signals."""
    score = 0.0

    # Strong bearish patterns
    if patterns.get("bearish_engulfing"):
        score += 0.4
    if patterns.get("evening_star"):
        score += 0.45
    if patterns.get("three_black_crows"):
        score += 0.5
    if patterns.get("dark_cloud_cover"):
        score += 0.3

    # Moderate bearish
    if patterns.get("shooting_star"):
        score += 0.3
    if patterns.get("gravestone_doji"):
        score += 0.2

    if patterns.get("doji"):
        score += 0.05

    return min(score, 1.0)


# ── Single candle helpers ────────────────────────────────────

def _body(o, c, i):
    return abs(c[i] - o[i])

def _upper_wick(o, h, c, i):
    return h[i] - max(o[i], c[i])

def _lower_wick(o, l, c, i):
    return min(o[i], c[i]) - l[i]

def _candle_range(h, l, i):
    return h[i] - l[i]

def _is_bullish(o, c, i):
    return c[i] > o[i]

def _is_bearish(o, c, i):
    return c[i] < o[i]


def _is_doji(o, h, l, c, i):
    """Body is tiny relative to full range — indecision."""
    r = _candle_range(h, l, i)
    if r == 0:
        return False
    return _body(o, c, i) / r < 0.1


def _is_hammer(o, h, l, c, i):
    """
    Small body at top, long lower wick (2x+ body), tiny upper wick.
    Bullish reversal at bottom of downtrend.
    """
    body = _body(o, c, i)
    lower = _lower_wick(o, l, c, i)
    upper = _upper_wick(o, h, c, i)
    r = _candle_range(h, l, i)
    if r == 0 or body == 0:
        return False
    return (lower >= body * 2 and
            upper <= body * 0.5 and
            body / r >= 0.1)


def _is_shooting_star(o, h, l, c, i):
    """
    Small body at bottom, long upper wick (2x+ body), tiny lower wick.
    Bearish reversal at top of uptrend.
    """
    body = _body(o, c, i)
    lower = _lower_wick(o, l, c, i)
    upper = _upper_wick(o, h, c, i)
    r = _candle_range(h, l, i)
    if r == 0 or body == 0:
        return False
    return (upper >= body * 2 and
            lower <= body * 0.5 and
            body / r >= 0.1)


def _is_dragonfly_doji(o, h, l, c, i):
    """Doji with long lower shadow — bullish at support."""
    r = _candle_range(h, l, i)
    if r == 0:
        return False
    body = _body(o, c, i)
    lower = _lower_wick(o, l, c, i)
    upper = _upper_wick(o, h, c, i)
    return (body / r < 0.1 and
            lower / r > 0.6 and
            upper / r < 0.1)


def _is_gravestone_doji(o, h, l, c, i):
    """Doji with long upper shadow — bearish at resistance."""
    r = _candle_range(h, l, i)
    if r == 0:
        return False
    body = _body(o, c, i)
    lower = _lower_wick(o, l, c, i)
    upper = _upper_wick(o, h, c, i)
    return (body / r < 0.1 and
            upper / r > 0.6 and
            lower / r < 0.1)


def _is_spinning_top(o, h, l, c, i):
    """Small body with roughly equal upper and lower wicks."""
    r = _candle_range(h, l, i)
    if r == 0:
        return False
    body = _body(o, c, i)
    lower = _lower_wick(o, l, c, i)
    upper = _upper_wick(o, h, c, i)
    if upper == 0 or lower == 0:
        return False
    wick_ratio = min(upper, lower) / max(upper, lower)
    return (body / r < 0.3 and
            wick_ratio > 0.5 and
            upper > body and
            lower > body)


# ── Two-candle patterns ─────────────────────────────────────

def _is_bullish_engulfing(o, h, l, c):
    """Current bullish candle body fully engulfs previous bearish candle body."""
    return (_is_bearish(o, c, -2) and
            _is_bullish(o, c, -1) and
            o[-1] <= c[-2] and
            c[-1] >= o[-2])


def _is_bearish_engulfing(o, h, l, c):
    """Current bearish candle body fully engulfs previous bullish candle body."""
    return (_is_bullish(o, c, -2) and
            _is_bearish(o, c, -1) and
            o[-1] >= c[-2] and
            c[-1] <= o[-2])


def _is_piercing_line(o, h, l, c):
    """
    Prev is bearish. Current opens below prev low, closes above prev midpoint.
    Bullish reversal.
    """
    if not _is_bearish(o, c, -2) or not _is_bullish(o, c, -1):
        return False
    prev_mid = (o[-2] + c[-2]) / 2
    return o[-1] < l[-2] and c[-1] > prev_mid and c[-1] < o[-2]


def _is_dark_cloud_cover(o, h, l, c):
    """
    Prev is bullish. Current opens above prev high, closes below prev midpoint.
    Bearish reversal.
    """
    if not _is_bullish(o, c, -2) or not _is_bearish(o, c, -1):
        return False
    prev_mid = (o[-2] + c[-2]) / 2
    return o[-1] > h[-2] and c[-1] < prev_mid and c[-1] > o[-2]


# ── Three-candle patterns ───────────────────────────────────

def _is_morning_star(o, h, l, c):
    """
    Three-bar bullish reversal:
    1. Large bearish candle
    2. Small body (gap down) — the star
    3. Large bullish candle closing into candle 1's body
    """
    if len(o) < 3:
        return False
    # Candle 1: bearish with decent body
    r1 = _candle_range(h, l, -3)
    if r1 == 0 or not _is_bearish(o, c, -3):
        return False
    body1 = _body(o, c, -3)
    if body1 / r1 < 0.4:
        return False

    # Candle 2: small body (star)
    body2 = _body(o, c, -2)
    if body2 > body1 * 0.5:
        return False

    # Candle 3: bullish, closes at least halfway into candle 1
    if not _is_bullish(o, c, -1):
        return False
    body3 = _body(o, c, -1)
    mid1 = (o[-3] + c[-3]) / 2
    return body3 > body2 and c[-1] > mid1


def _is_evening_star(o, h, l, c):
    """
    Three-bar bearish reversal:
    1. Large bullish candle
    2. Small body (gap up) — the star
    3. Large bearish candle closing into candle 1's body
    """
    if len(o) < 3:
        return False
    r1 = _candle_range(h, l, -3)
    if r1 == 0 or not _is_bullish(o, c, -3):
        return False
    body1 = _body(o, c, -3)
    if body1 / r1 < 0.4:
        return False

    body2 = _body(o, c, -2)
    if body2 > body1 * 0.5:
        return False

    if not _is_bearish(o, c, -1):
        return False
    body3 = _body(o, c, -1)
    mid1 = (o[-3] + c[-3]) / 2
    return body3 > body2 and c[-1] < mid1


def _is_three_white_soldiers(o, h, l, c):
    """Three consecutive bullish candles, each closing higher, small upper wicks."""
    if len(o) < 3:
        return False
    for i in [-3, -2, -1]:
        if not _is_bullish(o, c, i):
            return False
        r = _candle_range(h, l, i)
        if r == 0:
            return False
        body = _body(o, c, i)
        upper = _upper_wick(o, h, c, i)
        # Body should be decent and upper wick small
        if body / r < 0.4 or upper > body * 0.5:
            return False

    # Each close higher than previous, each open within previous body
    return (c[-2] > c[-3] and c[-1] > c[-2] and
            o[-2] > o[-3] and o[-2] < c[-3] and
            o[-1] > o[-2] and o[-1] < c[-2])


def _is_three_black_crows(o, h, l, c):
    """Three consecutive bearish candles, each closing lower, small lower wicks."""
    if len(o) < 3:
        return False
    for i in [-3, -2, -1]:
        if not _is_bearish(o, c, i):
            return False
        r = _candle_range(h, l, i)
        if r == 0:
            return False
        body = _body(o, c, i)
        lower = _lower_wick(o, l, c, i)
        if body / r < 0.4 or lower > body * 0.5:
            return False

    return (c[-2] < c[-3] and c[-1] < c[-2] and
            o[-2] < o[-3] and o[-2] > c[-3] and
            o[-1] < o[-2] and o[-1] > c[-2])
