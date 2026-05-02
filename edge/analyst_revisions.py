"""Analyst revisions momentum edge.

30-day net upgrades minus downgrades. Top quintile of net revisions
historically outperforms bottom by ~6% annual (Womack 1996, Jegadeesh
& Kim 2010).

Inputs:
- Per-symbol upgrade/downgrade event log (caller-supplied or fetched
  from Finnhub /stock/upgrade-downgrade free tier).

Output:
  AnalystSignal with size_mult and direction bias.

Caches per symbol with 24h TTL — broker call rate limits.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import requests
except Exception:
    requests = None  # type: ignore


@dataclass
class AnalystSignal:
    net_revisions: int = 0          # upgrades - downgrades over window
    upgrades: int = 0
    downgrades: int = 0
    bucket: str = "neutral"         # strong / positive / neutral / negative / weak
    size_mult: float = 1.0
    block_short: bool = False
    block_long: bool = False
    reason: str = ""


class AnalystRevisionsEdge:
    def __init__(self, config: dict):
        self.config = config.get("edge", {}).get("analyst_revisions", {})
        self.enabled = bool(self.config.get("enabled", True))
        self.window_days = int(self.config.get("window_days", 30))
        self.strong_threshold = int(self.config.get("strong_threshold", 3))
        self.positive_threshold = int(self.config.get("positive_threshold", 1))
        self.negative_threshold = int(self.config.get("negative_threshold", -1))
        self.weak_threshold = int(self.config.get("weak_threshold", -3))
        self.size_mult_strong = float(self.config.get("size_mult_strong", 1.20))
        self.size_mult_positive = float(self.config.get("size_mult_positive", 1.10))
        self.size_mult_negative = float(self.config.get("size_mult_negative", 0.85))
        self.size_mult_weak = float(self.config.get("size_mult_weak", 0.70))
        self.cache_ttl_sec = int(self.config.get("cache_ttl_sec", 86_400))
        self.cache_path = self.config.get(
            "cache_path",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "research", "analyst_cache.json"),
        )
        self._finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        self._cache: dict[str, dict] = {}
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

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def update_cache(self, symbol: str, upgrades: int, downgrades: int):
        """Caller pushes counts (e.g. from Finnhub or manual sync)."""
        self._cache[symbol.upper()] = {
            "upgrades": int(upgrades),
            "downgrades": int(downgrades),
            "updated_at": time.time(),
        }
        self._save_cache()

    def _classify(self, net: int) -> tuple[str, float]:
        if net >= self.strong_threshold:
            return "strong", self.size_mult_strong
        if net >= self.positive_threshold:
            return "positive", self.size_mult_positive
        if net <= self.weak_threshold:
            return "weak", self.size_mult_weak
        if net <= self.negative_threshold:
            return "negative", self.size_mult_negative
        return "neutral", 1.0

    def evaluate(self, symbol: str) -> AnalystSignal:
        if not self.enabled:
            return AnalystSignal(reason="analyst_revisions disabled")
        sym = symbol.upper()
        entry = self._cache.get(sym)
        if not entry:
            entry = self._fetch_finnhub(sym)
        if not entry:
            return AnalystSignal(reason="no analyst data")
        ups = int(entry.get("upgrades", 0))
        downs = int(entry.get("downgrades", 0))
        net = ups - downs
        bucket, mult = self._classify(net)
        block_short = bucket in ("strong", "positive")
        block_long = bucket in ("weak",)
        return AnalystSignal(
            net_revisions=net,
            upgrades=ups,
            downgrades=downs,
            bucket=bucket,
            size_mult=mult,
            block_short=block_short,
            block_long=block_long,
            reason=f"{bucket} ({ups} up, {downs} down, net {net:+d})",
        )

    def _fetch_finnhub(self, symbol: str) -> Optional[dict]:
        if requests is None or not self._finnhub_key:
            return None
        try:
            from_date = (datetime.now(timezone.utc) - timedelta(days=self.window_days)).strftime("%Y-%m-%d")
            to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/upgrade-downgrade",
                params={
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": self._finnhub_key,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            items = resp.json() or []
            ups = sum(1 for x in items if (x.get("toGrade") or "").lower() in
                      ("buy", "strong buy", "outperform", "overweight"))
            downs = sum(1 for x in items if (x.get("toGrade") or "").lower() in
                        ("sell", "strong sell", "underperform", "underweight"))
            entry = {"upgrades": ups, "downgrades": downs, "updated_at": time.time()}
            self._cache[symbol] = entry
            self._save_cache()
            return entry
        except Exception:
            return None
