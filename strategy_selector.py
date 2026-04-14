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


def select_strategies(df: pd.DataFrame, symbol: str,
                      sector_regime: dict = None) -> dict:
    """
    Analyze a stock's current behavior and return which strategies to use
    and how much weight to give each one.

    Args:
        sector_regime: Optional dict from SectorRegimeFilter.get_regime_for_sector().
            When provided, strategy weights are biased toward/away from trend-following
            based on the sector's current regime (bull dampens mean-reversion,
            bear dampens trend-following).

    Returns:
        {
            "regime": "trending" | "ranging" | "breakout" | "volatile",
            "strategies": {strategy_name: weight},
            "reason": str,
        }
    """
    result = _select_strategies_inner(df, symbol)
    result["strategies"] = _apply_sector_bias(result["strategies"], sector_regime)
    return result


def _apply_sector_bias(strategies: dict, sector_reg: dict) -> dict:
    """Scale trend vs mean-reversion weights based on sector regime."""
    if sector_reg is None:
        return strategies
    bias = sector_reg.get("regime", "chop")
    if bias == "chop":
        return strategies
    result = dict(strategies)
    trend_strats = {"supertrend", "momentum", "breakout"}
    mean_rev_strats = {"mean_reversion", "stoch_rsi"}
    scale_trend = 0.70 if bias == "bear" else 1.15
    scale_mr    = 1.20 if bias == "bear" else 0.90
    for s in trend_strats:
        if s in result:
            result[s] = result[s] * scale_trend
    for s in mean_rev_strats:
        if s in result:
            result[s] = result[s] * scale_mr
    total = sum(result.values())
    if total > 0:
        result = {k: round(v / total, 3) for k, v in result.items()}
    return result


def _select_strategies_inner(df: pd.DataFrame, symbol: str) -> dict:
    """Inner logic — classify regime and return raw strategy weights."""
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
                "supertrend": 0.25,
                "momentum": 0.25,
                "stoch_rsi": 0.10,
                "breakout": 0.10,
                "vwap_reclaim": 0.10,
                "mean_reversion": 0.00,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Strong uptrend (ADX={adx:.0f}), using trend strategies",
        }

    if strong_trend and direction == "down":
        return {
            "regime": "trending",
            "strategies": {
                "supertrend": 0.25,
                "momentum": 0.20,
                "stoch_rsi": 0.10,
                "breakout": 0.15,
                "vwap_reclaim": 0.10,
                "mean_reversion": 0.00,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Strong downtrend (ADX={adx:.0f}), short bias active",
        }

    # SQUEEZE / PRE-BREAKOUT — favor breakout strategy
    if is_squeeze and not trending:
        return {
            "regime": "breakout",
            "strategies": {
                "breakout": 0.30,
                "supertrend": 0.10,
                "momentum": 0.10,
                "stoch_rsi": 0.10,
                "vwap_reclaim": 0.10,
                "mean_reversion": 0.10,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Range squeeze detected, watching for breakout",
        }

    # NEAR HIGHS + VOLUME — breakout mode
    if near_high and vol_expanding:
        return {
            "regime": "breakout",
            "strategies": {
                "breakout": 0.25,
                "momentum": 0.15,
                "supertrend": 0.15,
                "vwap_reclaim": 0.15,
                "stoch_rsi": 0.10,
                "mean_reversion": 0.00,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Near 20-day high with volume expansion",
        }

    # NEAR LOWS + VOLUME — breakout mode (short)
    if near_low and vol_expanding:
        return {
            "regime": "breakout",
            "strategies": {
                "breakout": 0.25,
                "momentum": 0.15,
                "supertrend": 0.15,
                "vwap_reclaim": 0.15,
                "stoch_rsi": 0.10,
                "mean_reversion": 0.00,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Near 20-day low with volume expansion, breakdown setup",
        }

    # RANGING (low ADX, no trend) — mean reversion works here
    if adx < 20 and not trending:
        return {
            "regime": "ranging",
            "strategies": {
                "mean_reversion": 0.25,
                "stoch_rsi": 0.15,
                "vwap_reclaim": 0.15,
                "momentum": 0.10,
                "supertrend": 0.10,
                "breakout": 0.10,
                "gap": 0.05,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Ranging market (ADX={adx:.0f}), mean reversion favored",
        }

    # MODERATE TREND — balanced mix
    if trending and direction == "up":
        return {
            "regime": "trending",
            "strategies": {
                "momentum": 0.15,
                "supertrend": 0.20,
                "vwap_reclaim": 0.15,
                "stoch_rsi": 0.10,
                "breakout": 0.15,
                "mean_reversion": 0.05,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Moderate uptrend (ADX={adx:.0f}), balanced approach",
        }

    # MODERATE DOWNTREND — short bias
    if trending and direction == "down":
        return {
            "regime": "trending",
            "strategies": {
                "supertrend": 0.20,
                "momentum": 0.15,
                "vwap_reclaim": 0.15,
                "stoch_rsi": 0.10,
                "breakout": 0.10,
                "mean_reversion": 0.10,
                "gap": 0.10,
                "liquidity_sweep": 0.10,
            },
            "reason": f"Moderate downtrend (ADX={adx:.0f}), short bias",
        }

    # DEFAULT — mild bias toward trend
    return _default()


def _default():
    return {
        "regime": "mixed",
        "strategies": {
            "momentum": 0.15,
            "supertrend": 0.15,
            "vwap_reclaim": 0.15,
            "stoch_rsi": 0.10,
            "breakout": 0.15,
            "mean_reversion": 0.10,
            "gap": 0.10,
            "liquidity_sweep": 0.10,
        },
        "reason": "No clear regime, using balanced weights",
    }
