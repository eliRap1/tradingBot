"""Insider Form 4 cluster buy detector.

Queries SEC EDGAR Atom feed for Form 4 filings per symbol. Detects
clusters (2+ distinct insider buys in last 30 days) which historically
outperform by ~6-7% annualized (Lakonishok & Lee 2001; replicated in
Cohen, Malloy, Pomorski 2012).

Cache: per-symbol 24h (filings are daily-at-best).
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests


_EDGAR_ATOM = "https://www.sec.gov/cgi-bin/browse-edgar"
_USER_AGENT = "tradingBot insider-flow-edge contact@example.com"
# Transaction codes in Form 4: P = open-market purchase, S = sale
_BUY_CODES = {"P"}
_SELL_CODES = {"S"}
_NAMESPACES = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class InsiderSignal:
    buys: int = 0
    sells: int = 0
    cluster: bool = False
    size_mult: float = 1.0      # applied to long size
    block_short: bool = False   # True when strong insider-buy cluster


class InsiderFlow:
    def __init__(self, config: dict):
        cfg = config.get("edge", {}) or {}
        self.enabled: bool = bool(cfg.get("insider_flow", True))
        self.lookback_days: int = int(cfg.get("insider_lookback_days", 30))
        self.min_cluster: int = int(cfg.get("insider_min_cluster", 2))
        self.boost_mult: float = float(cfg.get("insider_boost_mult", 1.20))
        self.cache_ttl: int = int(cfg.get("insider_cache_ttl_sec", 86400))
        self._cache: dict[str, tuple[float, InsiderSignal]] = {}
        self._symbol_cik: dict[str, str] = {}  # symbol → CIK lookup
        self._cik_loaded = False

    def evaluate(self, symbol: str) -> InsiderSignal:
        if not self.enabled:
            return InsiderSignal()
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]

        cik = self._get_cik(symbol)
        if not cik:
            sig = InsiderSignal()
            self._cache[symbol] = (now, sig)
            return sig

        sig = self._fetch_and_classify(cik)
        self._cache[symbol] = (now, sig)
        return sig

    # ── helpers ────────────────────────────────────────────

    def _get_cik(self, symbol: str) -> Optional[str]:
        if not self._cik_loaded:
            self._load_cik_map()
        return self._symbol_cik.get(symbol.upper())

    def _load_cik_map(self):
        """Fetch SEC company_tickers.json; build {TICKER: CIK_zero_padded}."""
        self._cik_loaded = True
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": _USER_AGENT},
                timeout=15,
            )
            if not resp.ok:
                return
            data = resp.json()
            for entry in data.values():
                ticker = str(entry.get("ticker", "")).upper()
                cik = str(entry.get("cik_str", ""))
                if ticker and cik:
                    self._symbol_cik[ticker] = cik.zfill(10)
        except Exception:
            pass

    def _fetch_and_classify(self, cik: str) -> InsiderSignal:
        try:
            resp = requests.get(
                _EDGAR_ATOM,
                params={
                    "action": "getcompany",
                    "CIK": cik,
                    "type": "4",
                    "dateb": "",
                    "owner": "include",
                    "count": "40",
                    "output": "atom",
                },
                headers={"User-Agent": _USER_AGENT},
                timeout=15,
            )
            if not resp.ok:
                return InsiderSignal()
            return self._parse_atom(resp.text, cik=cik)
        except Exception:
            return InsiderSignal()

    def _parse_atom(self, xml_text: str, cik: str = "") -> InsiderSignal:
        """Parse Atom feed, then fetch inner Form 4 XML for each in-window
        filing to extract transaction codes (P = purchase, S = sale).

        Fetching inner XML is rate-limited (SEC allows 10 req/sec) but cheap
        since each primary_doc.xml is ~5KB and per-symbol cache is 24h.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return InsiderSignal()

        cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)
        buys = 0
        sells = 0

        # Collect in-window filing accession numbers
        acc_nos: list[str] = []
        for entry in root.findall("atom:entry", _NAMESPACES):
            updated = entry.find("atom:updated", _NAMESPACES)
            if updated is None or not updated.text:
                continue
            try:
                dt = datetime.strptime(updated.text[:10], "%Y-%m-%d")
            except ValueError:
                continue
            if dt < cutoff:
                continue

            summary = entry.find("atom:summary", _NAMESPACES)
            if summary is None or not summary.text:
                continue
            m = re.search(r"AccNo:\s*</b>\s*([\d\-]+)", summary.text)
            if m:
                acc_nos.append(m.group(1).strip())

        for acc in acc_nos[:30]:  # cap per-symbol fetches
            b, s = self._count_codes_from_filing(cik, acc)
            buys += b
            sells += s

        cluster = buys >= self.min_cluster and buys > sells
        return InsiderSignal(
            buys=buys,
            sells=sells,
            cluster=cluster,
            size_mult=self.boost_mult if cluster else 1.0,
            block_short=cluster,
        )

    def _count_codes_from_filing(self, cik: str, acc_no: str) -> tuple[int, int]:
        """Fetch the Form 4 XML for a filing and count P / S codes.

        Filing dir: /Archives/edgar/data/{cik_int}/{acc_nodash}/
        XML filename varies by filer (`primary_doc.xml`, `form4.xml`,
        `ownership.xml`, etc). Use the filing's `index.json` to find any
        `.xml` that isn't the FilingSummary.
        """
        try:
            cik_int = str(int(cik))  # strip leading zeros
            acc_nodash = acc_no.replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"

            idx = requests.get(
                f"{base}/index.json",
                headers={"User-Agent": _USER_AGENT},
                timeout=10,
            )
            if not idx.ok:
                return 0, 0
            items = idx.json().get("directory", {}).get("item", [])
            xml_name = None
            for it in items:
                name = it.get("name", "")
                if name.endswith(".xml") and "FilingSummary" not in name:
                    xml_name = name
                    break
            if not xml_name:
                return 0, 0

            resp = requests.get(
                f"{base}/{xml_name}",
                headers={"User-Agent": _USER_AGENT},
                timeout=10,
            )
            if not resp.ok:
                return 0, 0
            buys = sells = 0
            for m in re.finditer(
                r"<transactionCode>\s*([A-Z])\s*</transactionCode>",
                resp.text,
            ):
                code = m.group(1)
                if code in _BUY_CODES:
                    buys += 1
                elif code in _SELL_CODES:
                    sells += 1
            return buys, sells
        except Exception:
            return 0, 0
