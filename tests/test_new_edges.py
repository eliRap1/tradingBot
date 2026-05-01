"""Unit tests for econ_calendar, gap_filter, and risk ADV cap."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.econ_calendar import EconCalendar
from edge.gap_filter import GapFilter
from risk import RiskManager


_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


# ── EconCalendar ─────────────────────────────────────────────


class TestEconCalendar:
    def test_disabled_returns_none(self):
        cal = EconCalendar({"edge": {"econ_calendar": False}})
        assert cal.is_blackout() is None

    def test_fomc_blackout_window(self):
        cal = EconCalendar({
            "edge": {
                "econ_calendar": True,
                "fomc_dates": ["2026-04-30T14:00:00"],
                "econ_pre_block_min": 30,
                "econ_post_block_min": 60,
            }
        })
        # Inside window: 14:15 ET on 2026-04-30
        inside = datetime(2026, 4, 30, 14, 15, tzinfo=_ET).astimezone(_UTC)
        ev = cal.is_blackout(inside)
        assert ev is not None
        assert ev.name == "FOMC"

        # Outside window: 14:00 ET a week later
        outside = datetime(2026, 5, 7, 14, 0, tzinfo=_ET).astimezone(_UTC)
        assert cal.is_blackout(outside) is None

    def test_cpi_second_tuesday(self):
        cal = EconCalendar({"edge": {"econ_calendar": True}})
        # 2026-04-14 is second Tuesday of April
        at_event = datetime(2026, 4, 14, 8, 30, tzinfo=_ET).astimezone(_UTC)
        ev = cal.is_blackout(at_event)
        assert ev is not None
        assert ev.name == "CPI"

    def test_nfp_first_friday(self):
        cal = EconCalendar({"edge": {"econ_calendar": True}})
        # 2026-05-01 is first Friday of May
        at_event = datetime(2026, 5, 1, 8, 30, tzinfo=_ET).astimezone(_UTC)
        ev = cal.is_blackout(at_event)
        assert ev is not None
        assert ev.name == "NFP"

    def test_next_event_in_future(self):
        cal = EconCalendar({"edge": {"econ_calendar": True}})
        quiet = datetime(2026, 4, 20, 12, 0, tzinfo=_ET).astimezone(_UTC)
        nxt = cal.next_event(quiet)
        assert nxt is not None
        assert nxt.minutes_to > 0


# ── GapFilter ────────────────────────────────────────────────


def _bars(prev_close: float, today_open: float) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [100.0, today_open],
        "close": [prev_close, today_open],
        "high": [prev_close, today_open],
        "low": [prev_close, today_open],
        "volume": [1_000_000, 1_000_000],
    })


class TestGapFilter:
    def test_disabled(self):
        gf = GapFilter({"edge": {"gap_filter": False}})
        sig = gf.evaluate(_bars(100.0, 110.0), side="buy")
        assert sig.category == "flat"
        assert not sig.block

    def test_flat_gap(self):
        gf = GapFilter({"edge": {"gap_filter": True}})
        sig = gf.evaluate(_bars(100.0, 100.3), side="buy")
        assert sig.category == "flat"
        assert sig.size_mult == 1.0

    def test_normal_gap(self):
        gf = GapFilter({"edge": {"gap_filter": True}})
        sig = gf.evaluate(_bars(100.0, 102.0), side="buy")
        assert sig.category == "normal"
        assert sig.size_mult == 1.0
        assert not sig.block

    def test_large_gap_chase_penalty(self):
        gf = GapFilter({"edge": {"gap_filter": True}})
        # +4% gap, long entry = chasing
        sig = gf.evaluate(_bars(100.0, 104.0), side="buy")
        assert sig.category == "large"
        assert sig.size_mult < 1.0
        assert not sig.block

    def test_large_gap_counter_not_penalized(self):
        gf = GapFilter({"edge": {"gap_filter": True}})
        # +4% gap, short entry = counter-move; no penalty
        sig = gf.evaluate(_bars(100.0, 104.0), side="sell")
        assert sig.category == "large"
        assert sig.size_mult == 1.0

    def test_extreme_gap_blocks(self):
        gf = GapFilter({"edge": {"gap_filter": True}})
        sig = gf.evaluate(_bars(100.0, 108.0), side="buy")
        assert sig.category == "extreme"
        assert sig.block

    def test_missing_bars(self):
        gf = GapFilter({"edge": {"gap_filter": True}})
        assert gf.evaluate(None).category == "flat"
        assert gf.evaluate(pd.DataFrame()).category == "flat"


# ── Risk ADV cap ─────────────────────────────────────────────


class TestADVCap:
    def _risk_manager(self, max_pct_of_adv: float = 0.01):
        return RiskManager({
            "risk": {
                "max_position_pct": 0.08,
                "max_portfolio_risk_pct": 0.015,
                "stop_loss_atr_mult": 2.0,
                "take_profit_atr_mult": 5.0,
                "min_risk_reward": 2.5,
                "max_drawdown_pct": 0.10,
                "max_pct_of_adv": max_pct_of_adv,
                "sizing_method": "volatility_adjusted",
            },
            "screener": {"crypto_risk": {}},
        })

    def test_adv_cap_reduces_qty_on_thin_name(self):
        rm = self._risk_manager(0.01)
        # 20 rows: avg dollar vol = 100 * 1000 = 100k
        df = pd.DataFrame({
            "close": [100.0] * 20,
            "volume": [1000] * 20,
        })
        # Cap: 100k * 0.01 / 100 = 10 shares
        cap_qty = rm._adv_cap_qty(df, price=100.0, cap_pct=0.01)
        assert cap_qty == pytest.approx(10.0)

    def test_adv_cap_none_on_missing_volume(self):
        rm = self._risk_manager()
        df = pd.DataFrame({"close": [100.0] * 20})
        assert rm._adv_cap_qty(df, 100.0, 0.01) is None

    def test_adv_cap_none_on_short_history(self):
        rm = self._risk_manager()
        df = pd.DataFrame({"close": [100.0] * 5, "volume": [1000] * 5})
        assert rm._adv_cap_qty(df, 100.0, 0.01) is None
