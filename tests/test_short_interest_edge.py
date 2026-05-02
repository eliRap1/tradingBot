"""Unit tests for edge.short_interest ShortInterestEdge."""
from __future__ import annotations

import json
import pandas as pd
import pytest

from edge.short_interest import ShortInterestEdge, SqueezeSignal


@pytest.fixture
def cfg(tmp_path):
    cache = tmp_path / "si_cache.json"
    return {"edge": {"short_interest": {
        "enabled": True,
        "si_threshold_pct": 20.0,
        "dtc_threshold_days": 5.0,
        "atr_expansion_min": 1.3,
        "size_mult": 1.20,
        "block_short_on_candidate": True,
        "cache_path": str(cache),
    }}}


def _bars_with_atr_expansion(expansion=1.5, n=30):
    base_range = 1.0
    rng_recent = base_range * expansion
    highs = []
    lows = []
    closes = []
    for i in range(n - 5):
        highs.append(101.0 + base_range)
        lows.append(101.0)
        closes.append(101.0 + base_range / 2)
    for i in range(5):
        highs.append(102.0 + rng_recent)
        lows.append(102.0)
        closes.append(102.0 + rng_recent / 2)
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def test_disabled(cfg):
    cfg["edge"]["short_interest"]["enabled"] = False
    e = ShortInterestEdge(cfg)
    assert e.evaluate("AAPL").candidate is False


def test_no_si_data(cfg):
    e = ShortInterestEdge(cfg)
    sig = e.evaluate("AAPL")
    assert "no SI data" in sig.reason


def test_low_si_no_squeeze(cfg):
    e = ShortInterestEdge(cfg)
    e.update_cache("AAPL", short_pct_float=10.0, days_to_cover=2.0)
    sig = e.evaluate("AAPL")
    assert not sig.candidate
    assert sig.short_pct_float == 10.0


def test_high_si_with_squeeze_and_strong_rs(cfg):
    e = ShortInterestEdge(cfg)
    e.update_cache("XYZ", short_pct_float=35.0, days_to_cover=8.0)
    bars = _bars_with_atr_expansion(expansion=1.6)
    sig = e.evaluate("XYZ", daily_bars=bars, rs_bucket="strong")
    assert sig.candidate is True
    assert sig.block_short is True
    assert sig.long_size_mult == 1.20
    assert "squeeze" in sig.reason


def test_high_si_no_squeeze_blocks_shorts(cfg):
    e = ShortInterestEdge(cfg)
    e.update_cache("XYZ", short_pct_float=35.0, days_to_cover=8.0)
    bars = _bars_with_atr_expansion(expansion=1.0)
    sig = e.evaluate("XYZ", daily_bars=bars, rs_bucket="weak")
    assert not sig.candidate
    assert sig.block_short is True


def test_caches_to_disk(cfg, tmp_path):
    e = ShortInterestEdge(cfg)
    e.update_cache("XYZ", 25.0, 6.0)
    cache_file = tmp_path / "si_cache.json"
    saved = json.loads(cache_file.read_text())
    assert saved["XYZ"]["short_pct_float"] == 25.0


def test_inline_si_data_overrides_cache(cfg):
    e = ShortInterestEdge(cfg)
    e.update_cache("XYZ", short_pct_float=10.0, days_to_cover=2.0)
    si_data = {"XYZ": {"short_pct_float": 30.0, "days_to_cover": 7.0}}
    bars = _bars_with_atr_expansion(expansion=1.5)
    sig = e.evaluate("XYZ", daily_bars=bars, rs_bucket="strong", si_data=si_data)
    assert sig.candidate is True
