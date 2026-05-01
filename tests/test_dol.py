"""
Unit tests for DOL (Draw-on-Liquidity) strategy.

Covers: bounded output, bullish/bearish setups, flat/empty bars,
broken-OB polarity flip, HTF alignment gate.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np

from tests.helpers import (
    make_config, make_bars, make_uptrend_bars, make_downtrend_bars,
    make_ranging_bars, make_volatile_bars, make_5min_bars,
)

from strategies.dol import DOLStrategy


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def dol(config):
    return DOLStrategy(config)


# ═══════════════════════════════════════════════════════════
# 1. Bounded output on every bar fixture
# ═══════════════════════════════════════════════════════════

class TestBoundedOutput:

    @pytest.mark.parametrize("bar_fn", [
        make_uptrend_bars, make_downtrend_bars, make_ranging_bars,
        make_volatile_bars, make_5min_bars,
    ])
    def test_bounded_on_all_fixtures(self, dol, bar_fn):
        bars = bar_fn()
        signals = dol.generate_signals({"TEST": bars})
        for sym, score in signals.items():
            assert -1.0 <= score <= 1.0, \
                f"DOL out of bounds: {score} on {bar_fn.__name__}"

    def test_returns_dict(self, dol):
        bars = make_uptrend_bars(100)
        result = dol.generate_signals({"AAPL": bars, "MSFT": bars})
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════
# 2. Bullish OB + FVG setup
# ═══════════════════════════════════════════════════════════

def _make_bullish_dol_bars(n: int = 60) -> pd.DataFrame:
    """Base ranging bars, then craft a down candle + displacement-up + bull FVG."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1D")
    close = np.full(n, 100.0)
    # Small noise to avoid zero ATR
    rng = np.random.RandomState(7)
    close = close + rng.normal(0, 0.3, n)

    open_ = close.copy()
    high = close + 0.4
    low = close - 0.4

    # Build a bullish OB zone at bar i=30
    i = 30
    # Bar i: down candle (open > close), range = [98, 101]
    open_[i] = 101.0
    close[i] = 98.0
    high[i] = 101.2
    low[i]  = 97.8

    # Bar i+1: big up displacement — close above prior 5-bar high
    prior_high = max(high[i - 5:i].max(), high[i])
    open_[i + 1] = 99.0
    close[i + 1] = prior_high + 3.0  # strong break
    high[i + 1]  = close[i + 1] + 0.5
    low[i + 1]   = 98.7

    # Bar i+2: creates bull FVG (high[i] < low[i+2])
    open_[i + 2] = close[i + 1] + 0.2
    close[i + 2] = close[i + 1] + 1.0
    low[i + 2]   = open_[i + 2]  # > high[i]=101.2, gap confirmed
    high[i + 2]  = close[i + 2] + 0.3

    # Price hovers above the OB zone on final bars
    for k in range(i + 3, n):
        close[k] = close[i + 2] + rng.normal(0, 0.2)
        open_[k] = close[k - 1]
        high[k] = max(open_[k], close[k]) + 0.2
        low[k]  = min(open_[k], close[k]) - 0.2

    vol = np.full(n, 1_000_000)
    vwap = pd.Series(close).rolling(20, min_periods=1).mean().values

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol, "vwap": vwap,
    }, index=dates)


def _make_bearish_dol_bars(n: int = 60) -> pd.DataFrame:
    """Mirror of bullish: up candle + displacement-down + bear FVG."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1D")
    rng = np.random.RandomState(11)
    close = np.full(n, 100.0) + rng.normal(0, 0.3, n)
    open_ = close.copy()
    high = close + 0.4
    low  = close - 0.4

    i = 30
    # Bar i: up candle, zone [99, 102]
    open_[i] = 99.0
    close[i] = 102.0
    high[i]  = 102.2
    low[i]   = 98.8

    prior_low = min(low[i - 5:i].min(), low[i])
    open_[i + 1] = 101.0
    close[i + 1] = prior_low - 3.0
    low[i + 1]   = close[i + 1] - 0.5
    high[i + 1]  = 101.3

    # Bear FVG: low[i]=98.8 > high[i+2]
    open_[i + 2] = close[i + 1] - 0.2
    close[i + 2] = close[i + 1] - 1.0
    high[i + 2]  = open_[i + 2]
    low[i + 2]   = close[i + 2] - 0.3

    for k in range(i + 3, n):
        close[k] = close[i + 2] + rng.normal(0, 0.2)
        open_[k] = close[k - 1]
        high[k] = max(open_[k], close[k]) + 0.2
        low[k]  = min(open_[k], close[k]) - 0.2

    vol = np.full(n, 1_000_000)
    vwap = pd.Series(close).rolling(20, min_periods=1).mean().values

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol, "vwap": vwap,
    }, index=dates)


class TestDirectionalSetups:

    def test_bullish_setup_scores_long_or_zero(self, config):
        """Bullish OB + FVG, HTF not forced down — expect score ≥ 0."""
        cfg = make_config()
        cfg["strategies"]["dol"]["require_htf_align"] = False
        cfg["strategies"]["dol"]["min_verdict"] = 0.05
        strat = DOLStrategy(cfg)
        bars = _make_bullish_dol_bars(60)
        signals = strat.generate_signals({"TEST": bars})
        score = signals.get("TEST", 0.0)
        assert score >= 0.0, f"Bullish setup scored negative: {score}"

    def test_bearish_setup_scores_short_or_zero(self, config):
        cfg = make_config()
        cfg["strategies"]["dol"]["require_htf_align"] = False
        cfg["strategies"]["dol"]["min_verdict"] = 0.05
        strat = DOLStrategy(cfg)
        bars = _make_bearish_dol_bars(60)
        signals = strat.generate_signals({"TEST": bars})
        score = signals.get("TEST", 0.0)
        assert score <= 0.0, f"Bearish setup scored positive: {score}"


# ═══════════════════════════════════════════════════════════
# 3. Flat / empty / short bars → zero or silent
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_flat_ohlc_returns_no_signal(self, dol):
        """All OHLC equal → ATR = 0 → return 0 (skipped from signals dict)."""
        n = 50
        dates = pd.date_range("2025-01-01", periods=n, freq="1D")
        bars = pd.DataFrame({
            "open":  np.full(n, 100.0),
            "high":  np.full(n, 100.0),
            "low":   np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1_000_000),
        }, index=dates)
        signals = dol.generate_signals({"TEST": bars})
        # ATR=0 aborts → no entry
        assert "TEST" not in signals or signals["TEST"] == 0.0

    def test_short_data_skipped(self, dol):
        """< 40 bars → skip symbol entirely."""
        bars = make_bars(n=20)
        signals = dol.generate_signals({"TEST": bars})
        assert "TEST" not in signals

    def test_empty_df_no_crash(self, dol):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signals = dol.generate_signals({"TEST": empty})
        assert isinstance(signals, dict)
        assert "TEST" not in signals


# ═══════════════════════════════════════════════════════════
# 4. Broken OB flips polarity (demand → supply)
# ═══════════════════════════════════════════════════════════

class TestBreakerFlip:

    def test_broken_demand_zone_marked_supply(self):
        """Build a bullish OB, then close below the zone. Expect polarity flip."""
        cfg = make_config()
        cfg["strategies"]["dol"]["require_htf_align"] = False
        cfg["strategies"]["dol"]["min_verdict"] = 0.0
        strat = DOLStrategy(cfg)

        n = 80
        dates = pd.date_range("2025-01-01", periods=n, freq="1D")
        rng = np.random.RandomState(3)
        close = np.full(n, 100.0) + rng.normal(0, 0.3, n)
        open_ = close.copy()
        high = close + 0.4
        low  = close - 0.4

        # Bullish OB at i=30
        i = 30
        open_[i] = 101.0
        close[i] = 98.0
        high[i]  = 101.2
        low[i]   = 97.8

        prior_high = high[i - 5:i].max()
        open_[i + 1] = 99.0
        close[i + 1] = prior_high + 3.0
        high[i + 1]  = close[i + 1] + 0.5
        low[i + 1]   = 98.7

        # Bars drift up a bit
        for k in range(i + 2, 60):
            close[k] = close[i + 1] + rng.normal(0, 0.3)
            open_[k] = close[k - 1]
            high[k] = max(open_[k], close[k]) + 0.2
            low[k]  = min(open_[k], close[k]) - 0.2

        # Then bars 60+ close BELOW the OB zone_low (=98)
        for k in range(60, n):
            close[k] = 96.0 + rng.normal(0, 0.2)
            open_[k] = close[k - 1]
            high[k] = max(open_[k], close[k]) + 0.3
            low[k]  = min(open_[k], close[k]) - 0.3

        vol = np.full(n, 1_000_000)
        df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": vol,
        }, index=dates)

        # Directly inspect the primitive output
        atr_series = (df["high"] - df["low"]).rolling(14).mean()
        obs = strat._detect_order_blocks(df, atr_series, start=20)
        strat._apply_breakers(obs, df)

        # Find OB formed around bar 30
        ob30 = [lv for lv in obs if lv["bar_formed"] == i]
        assert len(ob30) >= 1
        flipped = [lv for lv in ob30 if lv.get("broken")]
        assert len(flipped) >= 1, "Expected broken OB to be marked"
        assert flipped[0]["side"] == "supply", \
            f"Broken demand OB should flip to supply, got {flipped[0]['side']}"


# ═══════════════════════════════════════════════════════════
# 5. HTF alignment gate
# ═══════════════════════════════════════════════════════════

class TestHTFGate:

    def test_htf_gate_blocks_counter_trend(self, monkeypatch):
        """Forced HTF direction='down' must zero any bullish verdict when gate=True."""
        import strategies.dol as dol_mod

        def fake_ctx_down(df):
            return {
                "direction": "down",
                "vwap": float(df["close"].iloc[-1]),
                "above_vwap": False,
                "adx": 30.0,
                "trending": True,
                "strong_trend": False,
            }

        monkeypatch.setattr(dol_mod, "get_trend_context", fake_ctx_down)

        cfg_on = make_config()
        cfg_on["strategies"]["dol"]["require_htf_align"] = True
        cfg_on["strategies"]["dol"]["min_verdict"] = 0.05
        strat_on = DOLStrategy(cfg_on)

        cfg_off = make_config()
        cfg_off["strategies"]["dol"]["require_htf_align"] = False
        cfg_off["strategies"]["dol"]["min_verdict"] = 0.05
        strat_off = DOLStrategy(cfg_off)

        # Bullish-setup bars, but HTF forced down.
        bars = _make_bullish_dol_bars(60)
        sig_on = strat_on.generate_signals({"TEST": bars}).get("TEST", 0.0)
        sig_off = strat_off.generate_signals({"TEST": bars}).get("TEST", 0.0)

        # Gate ON: any long verdict must be zeroed.
        assert sig_on <= 0.0, f"HTF gate failed to block long: {sig_on}"
        # Gate OFF: bullish setup can show through.
        assert -1.0 <= sig_off <= 1.0
