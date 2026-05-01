"""
Leading regime gate. Blocks new entries when broad market is in chop
(low ADX) or panic (high realized vol). Independent from PF-based
regime guard which is lagging (needs trades to fire).

Inputs are a SPY (or any benchmark) OHLC DataFrame and config. Output
is a (block, reason) tuple. Stateless — safe to call per bar in
backtester or per cycle in live coordinator.
"""
from __future__ import annotations

import pandas as pd
import ta


def is_chop_or_panic(
    spy_df: pd.DataFrame | None,
    config: dict,
) -> tuple[bool, str]:
    if spy_df is None or len(spy_df) < 50:
        return False, "insufficient SPY history"

    cfg = (config or {}).get("edge", {}).get("regime_gate", {})
    if not cfg.get("enabled", True):
        return False, "regime_gate disabled"

    adx_window = int(cfg.get("spy_adx_window", 14))
    adx_threshold = float(cfg.get("spy_adx_threshold", 18.0))
    panic_pct = float(cfg.get("spy_vol_panic_pct", 35.0))
    require_above_ema50 = bool(cfg.get("require_above_ema50", False))

    try:
        adx_series = ta.trend.ADXIndicator(
            high=spy_df["high"],
            low=spy_df["low"],
            close=spy_df["close"],
            window=adx_window,
        ).adx()
        adx = float(adx_series.iloc[-1])
    except Exception:
        return False, "ADX calc failed"

    returns = spy_df["close"].pct_change().dropna()
    if len(returns) < 20:
        return False, "insufficient returns history"
    vol_20d = float(returns.tail(20).std()) * (252 ** 0.5) * 100

    if vol_20d > panic_pct:
        return True, f"panic: SPY 20d vol={vol_20d:.1f}% > {panic_pct:.1f}%"

    if adx < adx_threshold:
        return True, f"chop: SPY ADX({adx_window})={adx:.1f} < {adx_threshold:.1f}"

    if require_above_ema50:
        ema50 = ta.trend.EMAIndicator(
            spy_df["close"], window=50
        ).ema_indicator()
        if float(spy_df["close"].iloc[-1]) < float(ema50.iloc[-1]):
            return True, "trend down: SPY below EMA50"

    return False, f"regime ok: ADX={adx:.1f} vol={vol_20d:.1f}%"
