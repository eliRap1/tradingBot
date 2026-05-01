"""Unit tests for market_calendar + volume_gate edges."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from edge.market_calendar import MarketCalendar
from edge.volume_gate import VolumeGate


_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


# ── MarketCalendar ───────────────────────────────────────────


class TestMarketCalendar:
    def test_disabled(self):
        cal = MarketCalendar({"edge": {"market_calendar": False}})
        sig = cal.evaluate()
        assert not sig.is_holiday
        assert not sig.is_half_day

    def test_holiday_detected(self):
        cal = MarketCalendar({"edge": {"market_calendar": True}})
        # 2026-12-25 = Christmas
        at = datetime(2026, 12, 25, 10, 0, tzinfo=_ET).astimezone(_UTC)
        sig = cal.evaluate(at)
        assert sig.is_holiday
        assert sig.session_name == "holiday"

    def test_half_day_before_cutoff(self):
        cal = MarketCalendar({
            "edge": {
                "market_calendar": True,
                "half_day_cutoff_hour": 12,
            }
        })
        # 2026-12-24 Christmas Eve, 10:00 ET
        at = datetime(2026, 12, 24, 10, 0, tzinfo=_ET).astimezone(_UTC)
        sig = cal.evaluate(at)
        assert sig.is_half_day
        assert not sig.early_close

    def test_half_day_past_cutoff(self):
        cal = MarketCalendar({
            "edge": {
                "market_calendar": True,
                "half_day_cutoff_hour": 12,
            }
        })
        # 2026-12-24 Christmas Eve, 12:30 ET
        at = datetime(2026, 12, 24, 12, 30, tzinfo=_ET).astimezone(_UTC)
        sig = cal.evaluate(at)
        assert sig.is_half_day
        assert sig.early_close

    def test_regular_day(self):
        cal = MarketCalendar({"edge": {"market_calendar": True}})
        at = datetime(2026, 4, 20, 10, 0, tzinfo=_ET).astimezone(_UTC)
        sig = cal.evaluate(at)
        assert not sig.is_holiday
        assert not sig.is_half_day
        assert sig.session_name == "regular"

    def test_custom_holidays_merged(self):
        cal = MarketCalendar({
            "edge": {
                "market_calendar": True,
                "holidays": ["2027-01-15"],
            }
        })
        at = datetime(2027, 1, 15, 10, 0, tzinfo=_ET).astimezone(_UTC)
        assert cal.evaluate(at).is_holiday


# ── VolumeGate ───────────────────────────────────────────────


def _bars_with_vol(volumes: list[float]) -> pd.DataFrame:
    n = len(volumes)
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "close": [100.0] * n,
        "volume": volumes,
    })


class TestVolumeGate:
    def test_disabled(self):
        vg = VolumeGate({"edge": {"volume_gate": False}})
        sig = vg.evaluate(_bars_with_vol([1000] * 25))
        assert sig.size_mult == 1.0
        assert not sig.block

    def test_surge_detected(self):
        vg = VolumeGate({"edge": {"volume_gate": True}})
        # 20 bars of 1000, last bar 3000 → ratio 3.0 → surge
        sig = vg.evaluate(_bars_with_vol([1000] * 20 + [3000]))
        assert sig.bucket == "surge"
        assert sig.size_mult == 1.15

    def test_normal(self):
        vg = VolumeGate({"edge": {"volume_gate": True}})
        # ratio 1.5 → normal
        sig = vg.evaluate(_bars_with_vol([1000] * 20 + [1500]))
        assert sig.bucket == "normal"
        assert sig.size_mult == 1.0

    def test_below_avg(self):
        vg = VolumeGate({"edge": {"volume_gate": True}})
        # ratio 0.8 → below
        sig = vg.evaluate(_bars_with_vol([1000] * 20 + [800]))
        assert sig.bucket == "below"
        assert sig.size_mult == 0.80

    def test_weak_blocks(self):
        vg = VolumeGate({"edge": {"volume_gate": True}})
        # ratio 0.3 → weak → block
        sig = vg.evaluate(_bars_with_vol([1000] * 20 + [300]))
        assert sig.bucket == "weak"
        assert sig.block

    def test_weak_no_block_when_disabled(self):
        vg = VolumeGate({"edge": {"volume_gate": True, "volume_block_on_weak": False}})
        sig = vg.evaluate(_bars_with_vol([1000] * 20 + [300]))
        assert sig.bucket == "weak"
        assert not sig.block

    def test_short_history(self):
        vg = VolumeGate({"edge": {"volume_gate": True}})
        sig = vg.evaluate(_bars_with_vol([1000] * 5))
        assert sig.bucket == "normal"
        assert sig.size_mult == 1.0
