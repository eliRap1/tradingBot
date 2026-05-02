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
    in_tom_window: bool = False  # turn-of-month: last 4 + first 3 trading days
    tom_size_mult: float = 1.0
    is_fed_day: bool = False     # FOMC announcement (size dampener)


class MarketCalendar:
    def __init__(self, config: dict):
        cfg = config.get("edge", {}) or {}
        self.enabled: bool = bool(cfg.get("market_calendar", True))
        # Past this ET hour on half-days, block new entries
        self.half_day_cutoff_hour: int = int(cfg.get("half_day_cutoff_hour", 12))
        self.holidays: set[str] = set(_HOLIDAYS_2026) | set(cfg.get("holidays", []))
        self.half_days: set[str] = set(_HALF_DAYS_2026) | set(cfg.get("half_days", []))
        # TOM (turn-of-month) seasonality
        tom_cfg = cfg.get("tom", {}) or {}
        self.tom_enabled: bool = bool(tom_cfg.get("enabled", True))
        self.tom_size_mult: float = float(tom_cfg.get("size_mult", 1.20))
        self.tom_pre_days: int = int(tom_cfg.get("pre_days", 4))   # last N trading days
        self.tom_post_days: int = int(tom_cfg.get("post_days", 3)) # first N trading days
        # Fed/FOMC announcement days — size dampener (event vol)
        self.fed_days: set[str] = set(cfg.get("fed_days", []))
        self.fed_size_mult: float = float(cfg.get("fed_size_mult", 0.50))

    def evaluate(self, now_utc: Optional[datetime] = None) -> CalendarSignal:
        if not self.enabled:
            return CalendarSignal()
        now_et = (now_utc or datetime.now(tz=ZoneInfo("UTC"))).astimezone(_ET)
        date_str = now_et.date().isoformat()

        if date_str in self.holidays:
            return CalendarSignal(is_holiday=True, session_name="holiday")

        # Always compute TOM and Fed-day flags (compose with half-day handling)
        is_tom = self._in_tom_window(now_et)
        tom_mult = self.tom_size_mult if (is_tom and self.tom_enabled) else 1.0
        is_fed = date_str in self.fed_days

        if date_str in self.half_days:
            past_cutoff = now_et.time() >= time(self.half_day_cutoff_hour, 0)
            return CalendarSignal(
                is_half_day=True,
                early_close=past_cutoff,
                session_name="half_day",
                in_tom_window=is_tom,
                tom_size_mult=tom_mult,
                is_fed_day=is_fed,
            )

        return CalendarSignal(
            session_name="regular",
            in_tom_window=is_tom,
            tom_size_mult=tom_mult,
            is_fed_day=is_fed,
        )

    def _in_tom_window(self, now_et: datetime) -> bool:
        """Return True if today is within last `pre_days` or first `post_days`
        trading days of the calendar month (skipping weekends + holidays)."""
        if not self.tom_enabled:
            return False
        d = now_et.date()
        # Days from month start (counting only weekdays, skipping holidays)
        post_count = 0
        cur = d.replace(day=1)
        while cur <= d:
            if cur.weekday() < 5 and cur.isoformat() not in self.holidays:
                post_count += 1
            cur = cur.fromordinal(cur.toordinal() + 1)
        if post_count <= self.tom_post_days:
            return True
        # Days until month end
        next_month = (d.replace(day=28) + (d.replace(day=28) - d.replace(day=1))).replace(day=1) \
            if False else None  # placeholder
        # Simpler: walk forward to first day of next month
        from datetime import date as _date
        if d.month == 12:
            first_next = _date(d.year + 1, 1, 1)
        else:
            first_next = _date(d.year, d.month + 1, 1)
        pre_count = 0
        cur = d
        while cur < first_next:
            if cur.weekday() < 5 and cur.isoformat() not in self.holidays:
                pre_count += 1
            cur = cur.fromordinal(cur.toordinal() + 1)
        return pre_count <= self.tom_pre_days
