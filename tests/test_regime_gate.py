"""Tests for edge.regime_gate (leading regime kill-switch)."""
import numpy as np
import pandas as pd
import pytest

from edge.regime_gate import is_chop_or_panic


def _make_spy_df(n: int = 80, drift: float = 0.001, vol: float = 0.008,
                 seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    closes = 400.0 * np.cumprod(1.0 + rets)
    highs = closes * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    lows = closes * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    opens = closes * (1.0 + rng.normal(0, 0.001, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 100_000_000)},
        index=idx,
    )


def _cfg(**overrides):
    base = {
        "edge": {
            "regime_gate": {
                "enabled": True,
                "spy_adx_window": 14,
                "spy_adx_threshold": 18.0,
                "spy_vol_panic_pct": 35.0,
                "require_above_ema50": False,
            }
        }
    }
    base["edge"]["regime_gate"].update(overrides)
    return base


class TestRegimeGate:
    def test_disabled_returns_false(self):
        spy = _make_spy_df(vol=0.05)  # would normally trigger panic
        blocked, _ = is_chop_or_panic(spy, _cfg(enabled=False))
        assert blocked is False

    def test_insufficient_data(self):
        spy = _make_spy_df(n=20)
        blocked, reason = is_chop_or_panic(spy, _cfg())
        assert blocked is False
        assert "insufficient" in reason

    def test_none_input(self):
        blocked, _ = is_chop_or_panic(None, _cfg())
        assert blocked is False

    def test_panic_vol_blocks(self):
        # very high realized vol -> annualized > 35%
        spy = _make_spy_df(vol=0.04, seed=1)
        blocked, reason = is_chop_or_panic(spy, _cfg())
        assert blocked is True
        assert "panic" in reason

    def test_chop_blocks(self):
        # Sideways noise -> low directional movement, ADX stays low.
        rng = np.random.default_rng(7)
        n = 80
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        # Tiny mean-reverting series (zero drift, tiny vol).
        rets = rng.normal(0.0, 0.0008, n)  # ~1.3% annualized
        closes = 400.0 * np.cumprod(1.0 + rets)
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.0008,
                "low": closes * 0.9992,
                "close": closes,
                "volume": np.full(n, 100_000_000),
            },
            index=idx,
        )
        blocked, reason = is_chop_or_panic(df, _cfg(spy_vol_panic_pct=50.0))
        assert blocked is True
        assert "chop" in reason

    def test_strong_trend_passes(self):
        # Strong steady uptrend -> ADX should be high, vol moderate
        n = 80
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        closes = 400.0 + np.arange(n) * 1.0  # +$1 per bar, no noise
        df = pd.DataFrame(
            {
                "open": closes - 0.1,
                "high": closes + 0.5,
                "low": closes - 0.5,
                "close": closes,
                "volume": np.full(n, 100_000_000),
            },
            index=idx,
        )
        blocked, reason = is_chop_or_panic(df, _cfg(spy_adx_threshold=25.0))
        assert blocked is False
        assert "ok" in reason

    def test_require_ema50_blocks_below(self):
        # Slow downtrend (low vol, low ADX would not block alone) — EMA50
        # gate should fire because price < EMA50.
        n = 80
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        # Smooth downtrend ~0.1% per bar -> ~25% annualized realized
        closes = 400.0 * np.power(0.999, np.arange(n))
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.0005,
                "low": closes * 0.9995,
                "close": closes,
                "volume": np.full(n, 100_000_000),
            },
            index=idx,
        )
        # Loosen ADX + panic so only EMA50 condition can fire
        blocked, reason = is_chop_or_panic(
            df,
            _cfg(spy_adx_threshold=0.0, spy_vol_panic_pct=200.0,
                 require_above_ema50=True),
        )
        assert blocked is True
        assert "below EMA50" in reason
