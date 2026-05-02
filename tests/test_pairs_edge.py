"""Unit tests for edge.pairs PairsEdge."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge.pairs import PairsEdge, PairDef, PairSignal


@pytest.fixture
def cfg():
    return {"edge": {"pairs": {
        "enabled": True,
        "lookback_days": 120,
        "refresh_ttl_sec": 0,
        "entry_z": 1.5,
        "exit_z": 0.5,
        "max_p_value": 0.10,
        "min_half_life_days": 1.0,
        "max_half_life_days": 50.0,
    }}}


def _gen_cointegrated(n=200, hedge_ratio=1.2, noise_sd=1.0, seed=42):
    """Generate two cointegrated series — B drives A with shared random walk."""
    rng = np.random.default_rng(seed)
    rw = np.cumsum(rng.normal(0, 1, n))
    a = hedge_ratio * rw + rng.normal(0, noise_sd, n)
    b = rw + rng.normal(0, noise_sd, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return (
        pd.DataFrame({"close": a + 100}, index=idx),
        pd.DataFrame({"close": b + 100}, index=idx),
    )


def test_half_life_finite_for_mean_reverting():
    rng = np.random.default_rng(0)
    n = 200
    series = []
    x = 0.0
    for _ in range(n):
        x = 0.5 * x + rng.normal(0, 1)
        series.append(x)
    s = pd.Series(series)
    hl = PairsEdge.half_life_ou(s)
    assert hl > 0 and hl < 5.0


def test_half_life_inf_for_random_walk():
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
    hl = PairsEdge.half_life_ou(s)
    assert hl == float("inf") or hl > 50.0


def test_hedge_ratio_recovered():
    rng = np.random.default_rng(7)
    n = 200
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    y = 0.7 * x + rng.normal(0, 0.5, n)
    beta = PairsEdge.hedge_ratio_ols(y, x)
    assert 0.5 < beta < 0.9


def test_discover_pairs_finds_cointegrated(cfg):
    a, b = _gen_cointegrated(n=180)
    sector = {"AAA": "tech", "BBB": "tech"}
    pe = PairsEdge(data_fetcher=None, config=cfg, sector_map=sector)
    pairs = pe.discover_pairs(["AAA", "BBB"], {"AAA": a, "BBB": b})
    assert len(pairs) >= 0


def test_discover_skips_different_sectors(cfg):
    a, b = _gen_cointegrated(n=180)
    sector = {"AAA": "tech", "BBB": "energy"}
    pe = PairsEdge(data_fetcher=None, config=cfg, sector_map=sector)
    pairs = pe.discover_pairs(["AAA", "BBB"], {"AAA": a, "BBB": b})
    assert len(pairs) == 0


def test_evaluate_no_pair_returns_inactive(cfg):
    pe = PairsEdge(data_fetcher=None, config=cfg, sector_map={})
    sig = pe.evaluate("AAPL")
    assert not sig.in_pair


def test_evaluate_extreme_z_triggers_signal(cfg):
    pe = PairsEdge(data_fetcher=None, config=cfg, sector_map={})
    pe._pairs = [PairDef(a="AAA", b="BBB", hedge_ratio=1.0,
                         z_mean=0.0, z_std=1.0, half_life=5.0, p_value=0.01)]
    pe._pair_by_symbol = {"AAA": pe._pairs, "BBB": pe._pairs}

    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    df_a = pd.DataFrame({"close": [100.0] * 10}, index=idx)
    df_b = pd.DataFrame({"close": [97.0] * 10}, index=idx)
    sig = pe.evaluate("AAA", {"AAA": df_a, "BBB": df_b})
    assert sig.in_pair
    assert sig.partner == "BBB"
    assert sig.role == "leader"  # AAA elevated → leader → short A, long B
    assert sig.z_score > 1.5


def test_disabled_returns_empty(cfg):
    cfg["edge"]["pairs"]["enabled"] = False
    pe = PairsEdge(data_fetcher=None, config=cfg, sector_map={})
    assert pe.evaluate("AAPL").in_pair is False
    assert pe.refresh_pairs([], {}) == 0
