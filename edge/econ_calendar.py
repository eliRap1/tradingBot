"""Economic calendar gate.

Blocks new positions around high-impact scheduled events:
- CPI (second Tuesday of month, 08:30 ET)
- NFP / jobs report (first Friday of month, 08:30 ET)
- FOMC rate decision (from config list — user-maintained quarterly)

Block window defaults: 30 min before to 60 min after event time.
All times US/Eastern; converted to UTC for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


_ET = ZoneInfo("America/New_York")


@dataclass
class EconEvent:
    name: str
    event_time_utc: datetime
    minutes_to: float  # negative if past, positive if future


class EconCalendar:
    """Returns whether current time is inside a blackout window for any
    high-impact macro event.
    """

    def __init__(self, config: dict):
        edge_cfg = config.get("edge", {}) or {}
        self.pre_min = int(edge_cfg.get("econ_pre_block_min", 30))
        self.post_min = int(edge_cfg.get("econ_post_block_min", 60))
        # User-populated list of FOMC datetimes (ISO strings, ET assumed)
        self.fomc_dates: list[str] = edge_cfg.get("fomc_dates", []) or []
        self.enabled: bool = bool(edge_cfg.get("econ_calendar", True))

    # ── public ────────────────────────────────────────────────

    def is_blackout(self, now_utc: Optional[datetime] = None) -> Optional[EconEvent]:
        """If `now_utc` is inside the blackout window of any event, return
        that event. Else return None.
        """
        if not self.enabled:
            return None
        now = now_utc or datetime.now(tz=_ET).astimezone(ZoneInfo("UTC"))
        events = self._upcoming_and_recent(now)
        for ev in events:
            lower = ev.event_time_utc - timedelta(minutes=self.pre_min)
            upper = ev.event_time_utc + timedelta(minutes=self.post_min)
            if lower <= now <= upper:
                ev.minutes_to = (ev.event_time_utc - now).total_seconds() / 60.0
                return ev
        return None

    def next_event(self, now_utc: Optional[datetime] = None) -> Optional[EconEvent]:
        now = now_utc or datetime.now(tz=_ET).astimezone(ZoneInfo("UTC"))
        events = [ev for ev in self._upcoming_and_recent(now) if ev.event_time_utc >= now]
        if not events:
            return None
        events.sort(key=lambda e: e.event_time_utc)
        nxt = events[0]
        nxt.minutes_to = (nxt.event_time_utc - now).total_seconds() / 60.0
        return nxt

    # ── helpers ───────────────────────────────────────────────

    def _upcoming_and_recent(self, now_utc: datetime) -> list[EconEvent]:
        """Return events in [now-2d, now+14d] window."""
        events: list[EconEvent] = []
        today_et = now_utc.astimezone(_ET).date()
        # scan +/- 2 calendar months
        for month_offset in (-1, 0, 1):
            year = today_et.year
            month = today_et.month + month_offset
            while month < 1:
                month += 12
                year -= 1
            while month > 12:
                month -= 12
                year += 1
            events.append(self._cpi_event(year, month))
            events.append(self._nfp_event(year, month))

        for iso in self.fomc_dates:
            ev = self._parse_fomc(iso)
            if ev is not None:
                events.append(ev)

        # Filter to +/- 2 week window for relevance
        lo = now_utc - timedelta(days=2)
        hi = now_utc + timedelta(days=14)
        return [ev for ev in events if lo <= ev.event_time_utc <= hi]

    def _cpi_event(self, year: int, month: int) -> EconEvent:
        """CPI: second Tuesday of month, 08:30 ET (approximation).
        Actual BLS schedule varies; treat this as conservative default.
        """
        day = self._nth_weekday(year, month, weekday=1, n=2)  # 1 = Tue
        dt_et = datetime.combine(day, time(8, 30), tzinfo=_ET)
        return EconEvent(
            name="CPI",
            event_time_utc=dt_et.astimezone(ZoneInfo("UTC")),
            minutes_to=0.0,
        )

    def _nfp_event(self, year: int, month: int) -> EconEvent:
        """NFP: first Friday of month, 08:30 ET."""
        day = self._nth_weekday(year, month, weekday=4, n=1)  # 4 = Fri
        dt_et = datetime.combine(day, time(8, 30), tzinfo=_ET)
        return EconEvent(
            name="NFP",
            event_time_utc=dt_et.astimezone(ZoneInfo("UTC")),
            minutes_to=0.0,
        )

    def _parse_fomc(self, iso: str) -> Optional[EconEvent]:
        """Parse FOMC datetime from ISO. Assume ET if no tz info.
        Default time 14:00 ET (rate decision release)."""
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError:
            return None
        if dt.tzinfo is None:
            if dt.hour == 0 and dt.minute == 0:
                dt = dt.replace(hour=14)
            dt = dt.replace(tzinfo=_ET)
        return EconEvent(
            name="FOMC",
            event_time_utc=dt.astimezone(ZoneInfo("UTC")),
            minutes_to=0.0,
        )

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int):
        """Return date of the nth occurrence of `weekday` (0=Mon..6=Sun)
        in year/month."""
        from datetime import date

        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return date(year, month, 1 + offset + 7 * (n - 1))
