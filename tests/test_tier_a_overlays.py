"""Tier A overlays: TOM seasonality, Fed days, intermarket ratios,
analyst revisions."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.market_calendar import MarketCalendar
from edge.analyst_revisions import AnalystRevisionsEdge
from edge.cross_asset import CrossAssetEngine


_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _et_dt(year, month, day, hour=10):
    return datetime(year, month, day, hour, 0, tzinfo=_ET).astimezone(_UTC)


@pytest.fixture
def cal_cfg():
    return {"edge": {
        "market_calendar": True,
        "tom": {"enabled": True, "size_mult": 1.20, "pre_days": 4, "post_days": 3},
        "fed_days": ["2026-05-06"],
        "fed_size_mult": 0.5,
    }}


def test_tom_window_first_day_of_month(cal_cfg):
    mc = MarketCalendar(cal_cfg)
    sig = mc.evaluate(_et_dt(2026, 5, 1))
    assert sig.in_tom_window
    assert sig.tom_size_mult == 1.20


def test_tom_window_last_day_of_month(cal_cfg):
    mc = MarketCalendar(cal_cfg)
    sig = mc.evaluate(_et_dt(2026, 5, 29))
    assert sig.in_tom_window


def test_tom_window_excluded_mid_month(cal_cfg):
    mc = MarketCalendar(cal_cfg)
    sig = mc.evaluate(_et_dt(2026, 5, 15))
    assert not sig.in_tom_window


def test_fed_day_flag(cal_cfg):
    mc = MarketCalendar(cal_cfg)
    sig = mc.evaluate(_et_dt(2026, 5, 6))
    assert sig.is_fed_day


def test_holiday_overrides(cal_cfg):
    mc = MarketCalendar(cal_cfg)
    sig = mc.evaluate(_et_dt(2026, 12, 25))
    assert sig.is_holiday


# ── Analyst revisions ──────────────────────────────────────────────


@pytest.fixture
def ar_cfg(tmp_path):
    return {"edge": {"analyst_revisions": {
        "enabled": True,
        "window_days": 30,
        "strong_threshold": 3,
        "positive_threshold": 1,
        "negative_threshold": -1,
        "weak_threshold": -3,
        "size_mult_strong": 1.20,
        "size_mult_weak": 0.70,
        "cache_path": str(tmp_path / "analyst.json"),
    }}}


def test_analyst_disabled(ar_cfg):
    ar_cfg["edge"]["analyst_revisions"]["enabled"] = False
    ar = AnalystRevisionsEdge(ar_cfg)
    assert ar.evaluate("AAPL").bucket == "neutral"


def test_analyst_strong_bucket(ar_cfg):
    ar = AnalystRevisionsEdge(ar_cfg)
    ar.update_cache("AAPL", upgrades=4, downgrades=0)
    sig = ar.evaluate("AAPL")
    assert sig.bucket == "strong"
    assert sig.size_mult == 1.20
    assert sig.block_short is True


def test_analyst_weak_bucket_blocks_long(ar_cfg):
    ar = AnalystRevisionsEdge(ar_cfg)
    ar.update_cache("XYZ", upgrades=0, downgrades=4)
    sig = ar.evaluate("XYZ")
    assert sig.bucket == "weak"
    assert sig.block_long is True
    assert sig.size_mult == 0.70


def test_analyst_neutral_bucket(ar_cfg):
    ar = AnalystRevisionsEdge(ar_cfg)
    ar.update_cache("AAPL", upgrades=0, downgrades=0)
    sig = ar.evaluate("AAPL")
    assert sig.bucket == "neutral"
    assert sig.size_mult == 1.0


# ── Intermarket ratios ─────────────────────────────────────────────


def _ratio_df(values):
    return pd.DataFrame({"close": values, "open": values, "high": values, "low": values})


def test_credit_spread_widening_signal():
    eng = CrossAssetEngine(data_fetcher=None)
    # HYG falling vs IEF stable -> ratio widening
    hyg = _ratio_df([100.0] * 10 + [98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0])
    ief = _ratio_df([100.0] * 21)
    sig = eng._credit_spread_signal(hyg, ief)
    assert sig == "widening"


def test_credit_spread_tightening_signal():
    eng = CrossAssetEngine(data_fetcher=None)
    hyg = _ratio_df([100.0] * 10 + [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0])
    ief = _ratio_df([100.0] * 21)
    sig = eng._credit_spread_signal(hyg, ief)
    assert sig == "tightening"


def test_cyclical_defensive_risk_off():
    eng = CrossAssetEngine(data_fetcher=None)
    xly = _ratio_df([100.0] * 10 + [98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 82.0, 80.0, 78.0])
    xlp = _ratio_df([100.0] * 21)
    sig = eng._cyclical_defensive_signal(xly, xlp)
    assert sig == "risk_off"
