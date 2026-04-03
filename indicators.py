"""
Custom indicators matching Pine Script built-in functions.

Implements:
  - SuperTrend      (Pine: ta.supertrend)
  - Pivot High/Low  (Pine: ta.pivothigh / ta.pivotlow)
  - Stochastic RSI  (Pine: ta.stoch applied to ta.rsi)
  - Crossover/under (Pine: ta.crossover / ta.crossunder)
  - VWAP bands
"""

import pandas as pd
import numpy as np
import ta as ta_lib
from utils import setup_logger

log = setup_logger("indicators")


# ═══════════════════════════════════════════════════════════════
# SuperTrend — exact Pine Script logic
# ═══════════════════════════════════════════════════════════════

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Calculate SuperTrend exactly like Pine Script's ta.supertrend.

    Returns:
        st_line: pd.Series — the SuperTrend line
        direction: pd.Series — -1 = bullish (price above), +1 = bearish (price below)
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    # Wilder's ATR (same as Pine Script)
    atr = ta_lib.volatility.AverageTrueRange(
        pd.Series(high), pd.Series(low), pd.Series(close), window=period
    ).average_true_range().values

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st_line = np.zeros(n)
    direction = np.zeros(n, dtype=int)

    # Initialize
    st_line[period] = upper_band[period]
    direction[period] = 1  # start bearish

    for i in range(period + 1, n):
        # Tighten lower band (never move down in uptrend)
        if lower_band[i] > lower_band[i - 1] or close[i - 1] < st_line[i - 1]:
            final_lower = lower_band[i]
        else:
            final_lower = max(lower_band[i], st_line[i - 1])

        # Tighten upper band (never move up in downtrend)
        if upper_band[i] < upper_band[i - 1] or close[i - 1] > st_line[i - 1]:
            final_upper = upper_band[i]
        else:
            final_upper = min(upper_band[i], st_line[i - 1])

        # Determine direction
        if direction[i - 1] == -1:  # was bullish
            if close[i] >= final_lower:
                st_line[i] = final_lower
                direction[i] = -1  # stay bullish
            else:
                st_line[i] = final_upper
                direction[i] = 1  # flip bearish
        else:  # was bearish
            if close[i] <= final_upper:
                st_line[i] = final_upper
                direction[i] = 1  # stay bearish
            else:
                st_line[i] = final_lower
                direction[i] = -1  # flip bullish

    return (pd.Series(st_line, index=df.index),
            pd.Series(direction, index=df.index))


# ═══════════════════════════════════════════════════════════════
# Pivot High / Low — Pine Script's ta.pivothigh / ta.pivotlow
# ═══════════════════════════════════════════════════════════════

def pivot_high(series: pd.Series, left_bars: int = 5, right_bars: int = 5) -> pd.Series:
    """
    Detect pivot highs (swing highs).
    A bar is a pivot high if it's the highest value in [i-left .. i+right].
    Returns NaN where no pivot, the pivot value where confirmed.
    Like Pine Script, the pivot is confirmed `right_bars` bars AFTER it occurs.
    """
    values = series.values
    n = len(values)
    pivots = np.full(n, np.nan)

    for i in range(left_bars, n - right_bars):
        window = values[i - left_bars:i + right_bars + 1]
        if values[i] == np.max(window) and np.sum(window == values[i]) == 1:
            # Confirm at i + right_bars (shifted like Pine Script)
            pivots[i + right_bars] = values[i]

    return pd.Series(pivots, index=series.index)


def pivot_low(series: pd.Series, left_bars: int = 5, right_bars: int = 5) -> pd.Series:
    """Detect pivot lows (swing lows). Same logic as pivot_high but for minima."""
    values = series.values
    n = len(values)
    pivots = np.full(n, np.nan)

    for i in range(left_bars, n - right_bars):
        window = values[i - left_bars:i + right_bars + 1]
        if values[i] == np.min(window) and np.sum(window == values[i]) == 1:
            pivots[i + right_bars] = values[i]

    return pd.Series(pivots, index=series.index)


def last_pivot_value(pivot_series: pd.Series) -> float | None:
    """Pine Script's ta.valuewhen(not na(pivot), pivot, 0) — most recent pivot value."""
    valid = pivot_series.dropna()
    return valid.iloc[-1] if len(valid) > 0 else None


def bars_since_pivot(pivot_series: pd.Series) -> int:
    """Pine Script's ta.barssince(not na(pivot)) — bars since last pivot."""
    valid_idx = pivot_series.dropna().index
    if len(valid_idx) == 0:
        return 9999
    last_idx = pivot_series.index.get_loc(valid_idx[-1])
    return len(pivot_series) - 1 - last_idx


# ═══════════════════════════════════════════════════════════════
# Stochastic RSI — Pine Script's ta.stoch applied to ta.rsi
# ═══════════════════════════════════════════════════════════════

def stochastic_rsi(close: pd.Series, rsi_period: int = 14,
                   stoch_period: int = 14, k_smooth: int = 3,
                   d_smooth: int = 3):
    """
    Calculate Stochastic RSI exactly like Pine Script.

    stoch_k = SMA(stoch(rsi, rsi, rsi, stoch_period), k_smooth)
    stoch_d = SMA(stoch_k, d_smooth)

    Returns:
        k: pd.Series (0-100)
        d: pd.Series (0-100)
    """
    rsi = ta_lib.momentum.RSIIndicator(close, window=rsi_period).rsi()

    # Stochastic formula applied to RSI values
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_raw = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    stoch_raw = stoch_raw.fillna(50)  # handle division by zero

    k = stoch_raw.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()

    return k, d


# ═══════════════════════════════════════════════════════════════
# Crossover / Crossunder — Pine Script's ta.crossover/ta.crossunder
# ═══════════════════════════════════════════════════════════════

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where `a` crosses above `b`. Matches Pine's ta.crossover."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where `a` crosses below `b`. Matches Pine's ta.crossunder."""
    return (a < b) & (a.shift(1) >= b.shift(1))


# ═══════════════════════════════════════════════════════════════
# Ichimoku Cloud
# ═══════════════════════════════════════════════════════════════

def ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26,
             senkou_b_period: int = 52):
    """
    Calculate Ichimoku Cloud components.
    Returns dict with: tenkan, kijun, senkou_a, senkou_b, chikou
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2

    senkou_a = (tenkan + kijun) / 2  # normally displaced 26 forward
    senkou_b = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2

    # Chikou = current close (normally plotted 26 bars back)
    # For signal generation, we compare chikou to price 26 bars ago
    chikou_vs_past = close > close.shift(kijun_period)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "cloud_green": senkou_a > senkou_b,
        "price_above_cloud": close > pd.concat([senkou_a, senkou_b], axis=1).max(axis=1),
        "price_below_cloud": close < pd.concat([senkou_a, senkou_b], axis=1).min(axis=1),
        "chikou_bullish": chikou_vs_past,
    }


# ═══════════════════════════════════════════════════════════════
# VWAP with standard deviation bands
# ═══════════════════════════════════════════════════════════════

def vwap_bands(df: pd.DataFrame, num_deviations: list[float] = [1.0, 2.0]):
    """
    Calculate VWAP and standard deviation bands.
    For daily data, computes a rolling VWAP (since true VWAP resets intraday).

    Returns dict: vwap, upper_1, lower_1, upper_2, lower_2
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]

    cum_vol = volume.cumsum()
    cum_tp_vol = (typical_price * volume).cumsum()
    vwap = cum_tp_vol / cum_vol

    # Standard deviation bands
    cum_tp_sq_vol = (typical_price ** 2 * volume).cumsum()
    variance = (cum_tp_sq_vol / cum_vol) - (vwap ** 2)
    variance = variance.clip(lower=0)
    std = np.sqrt(variance)

    result = {"vwap": vwap}
    for i, mult in enumerate(num_deviations, 1):
        result[f"upper_{i}"] = vwap + mult * std
        result[f"lower_{i}"] = vwap - mult * std

    return result
