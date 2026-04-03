"""
Strategy selector — picks the right strategies for each stock's behavior.

A trending stock needs momentum/supertrend.
A ranging stock needs mean reversion.
A breaking-out stock needs the breakout strategy.

This is what experienced traders do: they don't apply every strategy to every
stock. They READ the chart first, figure out the market structure, then pick
the right tool.
"""

import pandas as pd
from trend import get_trend_context
from indicators import supertrend
from utils import setup_logger

log = setup_logger("selector")


def select_strategies(df: pd.DataFrame, symbol: str) -> dict:
    """
    Analyze a stock's current behavior and return which strategies to use
    and how much weight to give each one.

    Returns:
        {
            "regime": "trending" | "ranging" | "breakout" | "volatile",
            "strategies": {strategy_name: weight},
            "reason": str,
        }
    """
    if len(df) < 30:
        return _default()

    ctx = get_trend_context(df)
    adx = ctx["adx"]
    direction = ctx["direction"]
    trending = ctx["trending"]
    strong_trend = ctx["strong_trend"]

    # Volatility regime
    atr_series = df["high"] - df["low"]
    atr_recent = atr_series.tail(5).mean()
    atr_avg = atr_series.tail(20).mean()
    vol_expanding = atr_recent > atr_avg * 1.3

    # Range compression (squeeze)
    recent_range = df["high"].tail(5).max() - df["low"].tail(5).min()
    lookback_range = df["high"].tail(20).max() - df["low"].tail(20).min()
    is_squeeze = recent_range < lookback_range * 0.4

    # SuperTrend state
    _, st_dir = supertrend(df)
    st_bullish = st_dir.iloc[-1] == -1

    # Price vs key levels
    close = df["close"].iloc[-1]
    high_20 = df["high"].tail(20).max()
    low_20 = df["low"].tail(20).min()
    near_high = (high_20 - close) / close < 0.02  # within 2% of 20-day high
    near_low = (close - low_20) / close < 0.02

    # ── Classify and assign strategies ───────────────────────

    # STRONG TREND — use trend-following strategies
    if strong_trend and direction == "up":
        return {
            "regime": "trending",
            "strategies": {
                "supertrend": 0.35,
                "momentum": 0.35,
                "stoch_rsi": 0.20,
                "breakout": 0.10,
                "mean_reversion": 0.00,  # OFF — don't fight the trend
            },
            "reason": f"Strong uptrend (ADX={adx:.0f}), using trend strategies",
        }

    if strong_trend and direction == "down":
        return {
            "regime": "trending",
            "strategies": {
                "supertrend": 0.10,
                "momentum": 0.10,
                "stoch_rsi": 0.10,
                "breakout": 0.00,
                "mean_reversion": 0.00,
            },
            "reason": f"Strong downtrend (ADX={adx:.0f}), minimal exposure",
        }

    # SQUEEZE / PRE-BREAKOUT — favor breakout strategy
    if is_squeeze and not trending:
        return {
            "regime": "breakout",
            "strategies": {
                "breakout": 0.40,
                "supertrend": 0.20,
                "momentum": 0.20,
                "stoch_rsi": 0.10,
                "mean_reversion": 0.10,
            },
            "reason": f"Range squeeze detected, watching for breakout",
        }

    # NEAR HIGHS + VOLUME — breakout mode
    if near_high and vol_expanding:
        return {
            "regime": "breakout",
            "strategies": {
                "breakout": 0.35,
                "momentum": 0.25,
                "supertrend": 0.25,
                "stoch_rsi": 0.15,
                "mean_reversion": 0.00,
            },
            "reason": f"Near 20-day high with volume expansion",
        }

    # RANGING (low ADX, no trend) — mean reversion works here
    if adx < 20 and not trending:
        return {
            "regime": "ranging",
            "strategies": {
                "mean_reversion": 0.35,
                "stoch_rsi": 0.25,
                "momentum": 0.15,
                "supertrend": 0.15,
                "breakout": 0.10,
            },
            "reason": f"Ranging market (ADX={adx:.0f}), mean reversion favored",
        }

    # MODERATE TREND — balanced mix
    if trending and direction == "up":
        return {
            "regime": "trending",
            "strategies": {
                "momentum": 0.25,
                "supertrend": 0.25,
                "stoch_rsi": 0.20,
                "breakout": 0.15,
                "mean_reversion": 0.15,
            },
            "reason": f"Moderate uptrend (ADX={adx:.0f}), balanced approach",
        }

    # DEFAULT — mild bias toward trend
    return _default()


def _default():
    return {
        "regime": "mixed",
        "strategies": {
            "momentum": 0.25,
            "supertrend": 0.25,
            "stoch_rsi": 0.15,
            "breakout": 0.20,
            "mean_reversion": 0.15,
        },
        "reason": "No clear regime, using balanced weights",
    }
