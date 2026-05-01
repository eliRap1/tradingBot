"""Market calendar gate — holidays + half-day early closes.

NYSE/NASDAQ closed days + scheduled 1:00 PM ET closes. Block new
entries on half-days after configurable cutoff (default 12:00 ET) to
avoid thin late-session liquidity.

User-maintained list of holidays and half-days via config. Pre-seeded
with 2026 US market schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo


_ET = ZoneInfo("America/New_York")


# US market holidays 2026 (full-day closures)
_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}

# Half-days (1:00 PM ET close) — day before/after major holidays
_HALF_DAYS_2026 = {
    "2026-07-02",  # day before July 4 (Thursday)
    "2026-11-27",  # day after Thanksgiving
    "2026-12-24",  # Christmas Eve
}


@dataclass
class CalendarSignal:
    is_holiday: bool = False
    is_half_day: bool = False
    early_close: bool = False  # half-day AND past cutoff
    session_name: str = "regular"


class MarketCalendar:
    def __init__(self, config: dict):
        cfg = config.get("edge", {}) or {}
        self.enabled: bool = bool(cfg.get("market_calendar", True))
        # Past this ET hour on half-days, block new entries
        self.half_day_cutoff_hour: int = int(cfg.get("half_day_cutoff_hour", 12))
        self.holidays: set[str] = set(_HOLIDAYS_2026) | set(cfg.get("holidays", []))
        self.half_days: set[str] = set(_HALF_DAYS_2026) | set(cfg.get("half_days", []))

    def evaluate(self, now_utc: Optional[datetime] = None) -> CalendarSignal:
        if not self.enabled:
            return CalendarSignal()
        now_et = (now_utc or datetime.now(tz=ZoneInfo("UTC"))).astimezone(_ET)
        date_str = now_et.date().isoformat()

        if date_str in self.holidays:
            return CalendarSignal(is_holiday=True, session_name="holiday")

        if date_str in self.half_days:
            past_cutoff = now_et.time() >= time(self.half_day_cutoff_hour, 0)
            return CalendarSignal(
                is_half_day=True,
                early_close=past_cutoff,
                session_name="half_day",
            )

        return CalendarSignal(session_name="regular")
