"""
Trend context analysis — ADX, VWAP, and structure detection.

Tells strategies whether the market is trending or ranging,
and which direction, so they don't fight the trend.
"""

import pandas as pd
import numpy as np
import ta
from utils import setup_logger

log = setup_logger("trend")


def get_trend_context(df: pd.DataFrame) -> dict:
    """
    Analyze trend state for a single symbol's DataFrame.

    Returns:
        {
            "adx": float,           # 0-100, >25 = trending
            "trending": bool,       # ADX > 25
            "strong_trend": bool,   # ADX > 40
            "direction": str,       # "up", "down", "neutral"
            "di_plus": float,       # +DI value
            "di_minus": float,      # -DI value
            "vwap": float | None,   # VWAP if available
            "above_vwap": bool,     # price above VWAP
            "ema_200": float,       # 200-period EMA (or longest available)
            "above_ema_200": bool,  # price above 200 EMA
            "higher_highs": bool,   # last 3 swing highs ascending
            "lower_lows": bool,     # last 3 swing lows descending
        }
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    current_price = close.iloc[-1]

    result = {}

    # ── ADX + Directional Indicators ─────────────────────────
    adx_period = min(14, len(df) - 2)
    if adx_period >= 5:
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=adx_period)
        result["adx"] = adx_ind.adx().iloc[-1]
        result["di_plus"] = adx_ind.adx_pos().iloc[-1]
        result["di_minus"] = adx_ind.adx_neg().iloc[-1]
    else:
        result["adx"] = 0.0
        result["di_plus"] = 0.0
        result["di_minus"] = 0.0

    result["trending"] = result["adx"] > 25
    result["strong_trend"] = result["adx"] > 40

    if result["di_plus"] > result["di_minus"] + 5:
        result["direction"] = "up"
    elif result["di_minus"] > result["di_plus"] + 5:
        result["direction"] = "down"
    else:
        result["direction"] = "neutral"

    # ── VWAP ─────────────────────────────────────────────────
    if "vwap" in df.columns and df["vwap"].notna().any():
        result["vwap"] = df["vwap"].iloc[-1]
        result["above_vwap"] = current_price > result["vwap"] if result["vwap"] else False
    else:
        # Calculate from OHLCV if VWAP column not available
        typical_price = (high + low + close) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol
        result["vwap"] = vwap_series.iloc[-1] if not np.isnan(vwap_series.iloc[-1]) else None
        result["above_vwap"] = current_price > result["vwap"] if result["vwap"] else False

    # ── 200 EMA (or longest available) ───────────────────────
    ema_window = min(200, len(df) - 1)
    if ema_window >= 10:
        ema_200 = ta.trend.EMAIndicator(close, window=ema_window).ema_indicator()
        result["ema_200"] = ema_200.iloc[-1]
        result["above_ema_200"] = current_price > result["ema_200"]
    else:
        result["ema_200"] = current_price
        result["above_ema_200"] = True

    # ── Price structure (higher highs / lower lows) ──────────
    result["higher_highs"] = _check_higher_highs(high, window=5, count=3)
    result["lower_lows"] = _check_lower_lows(low, window=5, count=3)

    return result


def _check_higher_highs(high: pd.Series, window: int = 5, count: int = 3) -> bool:
    """Check if the last `count` local highs are ascending."""
    if len(high) < window * count:
        return False

    peaks = []
    data = high.values
    for i in range(window, len(data) - 1):
        if data[i] == max(data[i - window:i + 1]):
            peaks.append(data[i])

    if len(peaks) < count:
        return False

    recent = peaks[-count:]
    return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))


def get_weekly_trend(df: pd.DataFrame) -> dict:
    """
    Resample daily bars to weekly and compute weekly trend.
    Like Pine Script's request.security(syminfo.tickerid, "W", ...).

    Returns:
        {
            "weekly_trend_up": bool,   # weekly 21 EMA > 50 EMA
            "weekly_ema21": float,
            "weekly_ema50": float,
            "price_above_weekly_21": bool,
        }
    """
    if len(df) < 30:
        return {"weekly_trend_up": True, "weekly_ema21": 0, "weekly_ema50": 0,
                "price_above_weekly_21": True}

    # Resample to weekly (Friday close)
    weekly = df.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    if len(weekly) < 10:
        return {"weekly_trend_up": True, "weekly_ema21": 0, "weekly_ema50": 0,
                "price_above_weekly_21": True}

    close_w = weekly["close"]
    ema21 = ta.trend.EMAIndicator(close_w, window=min(21, len(close_w) - 1)).ema_indicator()
    ema50_window = min(50, len(close_w) - 1)
    ema50 = ta.trend.EMAIndicator(close_w, window=ema50_window).ema_indicator()

    current_price = df["close"].iloc[-1]

    return {
        "weekly_trend_up": ema21.iloc[-1] > ema50.iloc[-1],
        "weekly_ema21": ema21.iloc[-1],
        "weekly_ema50": ema50.iloc[-1],
        "price_above_weekly_21": current_price > ema21.iloc[-1],
    }


def _check_lower_lows(low: pd.Series, window: int = 5, count: int = 3) -> bool:
    """Check if the last `count` local lows are descending."""
    if len(low) < window * count:
        return False

    troughs = []
    data = low.values
    for i in range(window, len(data) - 1):
        if data[i] == min(data[i - window:i + 1]):
            troughs.append(data[i])

    if len(troughs) < count:
        return False

    recent = troughs[-count:]
    return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
