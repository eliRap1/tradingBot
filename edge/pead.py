"""Post-Earnings Announcement Drift (PEAD) edge.

Top-decile EPS surprise (SUE) drifts up 60+ days after report;
bottom-decile drifts down. Documented since Bernard & Thomas 1989,
robust 50+ years.

SUE = (actual_eps - consensus_eps) / std(historical_surprises)

Implementation:
- Pulls last earnings event per symbol (consensus + actual EPS).
- Computes SUE if 4+ historical surprises available; else uses
  surprise_pct = (actual - consensus) / |consensus|.
- Active window: bars since report ≤ drift_window_days (default 60).
- Tier classification:
    sue >= +1.5 → "top"     → long-bias size_mult 1.15, allow_short=False
    sue <= -1.5 → "bottom"  → short-bias size_mult 1.15, allow_long=False
    else        → "neutral" → no effect

Free data sources tried in order:
1. Alpaca corporate-actions endpoint (paper/live keys via env)
2. Cached local EDGAR pulls (research/earnings_cache.json) if present
3. Returns neutral signal when no data — never raises.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import requests
except Exception:
    requests = None  # type: ignore


@dataclass
class EarningsEvent:
    symbol: str
    report_date: datetime
    actual_eps: Optional[float] = None
    consensus_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    sue: Optional[float] = None


@dataclass
class PEADSignal:
    active: bool = False
    tier: str = "neutral"             # top / bottom / neutral
    sue: float = 0.0
    surprise_pct: float = 0.0
    days_since_report: int = -1
    size_mult: float = 1.0
    allow_long: bool = True
    allow_short: bool = True
    reason: str = ""


class PEADEdge:
    def __init__(self, config: dict):
        self.config = config.get("edge", {}).get("pead", {})
        self.enabled = bool(self.config.get("enabled", True))
        self.drift_window_days = int(self.config.get("drift_window_days", 60))
        self.top_threshold = float(self.config.get("top_threshold_sue", 1.5))
        self.bottom_threshold = float(self.config.get("bottom_threshold_sue", -1.5))
        self.size_mult = float(self.config.get("size_mult", 1.15))
        self.cache_ttl_sec = int(self.config.get("cache_ttl_sec", 21_600))  # 6h
        self.cache_path = self.config.get(
            "cache_path",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "research", "earnings_cache.json"),
        )
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._cache: dict[str, dict] = {}
        self._cache_loaded_at: float = 0.0
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            self._cache = {}
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f) or {}
        except Exception:
            self._cache = {}
        self._cache_loaded_at = time.time()

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception:
            pass

    @staticmethod
    def compute_sue(actual: float, consensus: float, history: list[float]) -> Optional[float]:
        """Standardized Unexpected Earnings.

        history = list of past (actual - consensus) surprises (≥ 4 needed for std).
        """
        if not history or len(history) < 4:
            return None
        mean = sum(history) / len(history)
        var = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
        std = var ** 0.5
        if std == 0:
            return None
        return (actual - consensus - mean) / std

    @staticmethod
    def compute_surprise_pct(actual: float, consensus: float) -> Optional[float]:
        if consensus is None or actual is None:
            return None
        denom = abs(consensus) if consensus != 0 else 0.01
        return (actual - consensus) / denom

    def evaluate(self, symbol: str, now: datetime | None = None) -> PEADSignal:
        if not self.enabled:
            return PEADSignal(reason="pead disabled")
        ev = self._latest_earnings(symbol)
        if ev is None or ev.report_date is None:
            return PEADSignal(reason="no earnings data")

        cur = now or datetime.now(timezone.utc)
        if ev.report_date.tzinfo is None:
            ev.report_date = ev.report_date.replace(tzinfo=timezone.utc)
        days = (cur - ev.report_date).days
        if days < 0 or days > self.drift_window_days:
            return PEADSignal(
                active=False,
                days_since_report=days,
                reason=f"outside PEAD window ({days}d post-report)",
            )

        score = ev.sue if ev.sue is not None else (ev.surprise_pct or 0.0) * 10
        # Scale surprise_pct → SUE-like number when SUE not available
        # (12% surprise ≈ 1.2 SUE under typical std~10%; rough but bounded)
        if score >= self.top_threshold:
            return PEADSignal(
                active=True, tier="top",
                sue=ev.sue or 0.0,
                surprise_pct=ev.surprise_pct or 0.0,
                days_since_report=days,
                size_mult=self.size_mult,
                allow_long=True, allow_short=False,
                reason=f"top SUE={score:.2f} day{days}/{self.drift_window_days}",
            )
        if score <= self.bottom_threshold:
            return PEADSignal(
                active=True, tier="bottom",
                sue=ev.sue or 0.0,
                surprise_pct=ev.surprise_pct or 0.0,
                days_since_report=days,
                size_mult=self.size_mult,
                allow_long=False, allow_short=True,
                reason=f"bottom SUE={score:.2f} day{days}/{self.drift_window_days}",
            )
        return PEADSignal(
            active=False, tier="neutral",
            sue=ev.sue or 0.0, surprise_pct=ev.surprise_pct or 0.0,
            days_since_report=days,
            reason=f"neutral SUE={score:.2f}",
        )

    def _latest_earnings(self, symbol: str) -> Optional[EarningsEvent]:
        cached = self._cache.get(symbol.upper())
        if cached:
            try:
                report_date = datetime.fromisoformat(cached["report_date"])
                return EarningsEvent(
                    symbol=symbol.upper(),
                    report_date=report_date,
                    actual_eps=cached.get("actual_eps"),
                    consensus_eps=cached.get("consensus_eps"),
                    surprise_pct=cached.get("surprise_pct"),
                    sue=cached.get("sue"),
                )
            except Exception:
                pass
        return self._fetch_alpaca(symbol)

    def _fetch_alpaca(self, symbol: str) -> Optional[EarningsEvent]:
        if requests is None or not self._alpaca_key:
            return None
        url = f"https://data.alpaca.markets/v1beta1/forecasts/earnings"
        try:
            resp = requests.get(
                url,
                params={"symbols": symbol.upper(), "limit": 1},
                headers={
                    "APCA-API-KEY-ID": self._alpaca_key,
                    "APCA-API-SECRET-KEY": self._alpaca_secret,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data = resp.json() or {}
            items = data.get("earnings") or data.get(symbol.upper()) or []
            if not items:
                return None
            it = items[0]
            actual = float(it.get("actual_eps") or 0.0) or None
            consensus = float(it.get("consensus_eps") or 0.0) or None
            report_str = it.get("report_date") or it.get("date") or ""
            try:
                report_date = datetime.fromisoformat(report_str.replace("Z", "+00:00"))
            except Exception:
                return None
            surprise = self.compute_surprise_pct(actual, consensus) if (actual and consensus) else None
            ev = EarningsEvent(
                symbol=symbol.upper(),
                report_date=report_date,
                actual_eps=actual,
                consensus_eps=consensus,
                surprise_pct=surprise,
                sue=None,
            )
            self._cache[symbol.upper()] = {
                "report_date": report_date.isoformat(),
                "actual_eps": actual,
                "consensus_eps": consensus,
                "surprise_pct": surprise,
                "sue": None,
            }
            self._save_cache()
            return ev
        except Exception:
            return None
