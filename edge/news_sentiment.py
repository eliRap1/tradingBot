from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import requests


class NewsSentimentEngine:
    """Earnings blocker plus optional headline sentiment.

    Uses Alpaca news API (unlimited for paper accounts). Per-symbol TTL cache
    (default 30 min) to avoid redundant calls. Falls back to NEWSAPI_KEY if
    Alpaca creds absent.
    """

    _ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, config: dict):
        self.config = config.get("edge", {})
        self._blocked_cache: set[str] = set()
        self._blocked_at = 0.0
        # symbol → nearest earnings date (absolute date, not relative)
        self._earnings_dates: dict[str, datetime] = {}
        # symbol → (ts, score) — per-symbol sentiment cache
        self._sentiment_cache: dict[str, tuple[float, float]] = {}
        self._sentiment_ttl_sec = int(self.config.get("news_cache_ttl_sec", 1800))
        self._news_key = os.getenv("NEWSAPI_KEY", "")
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._analyzer = None  # lazy-init vader

    def get_blocked_symbols(self, symbols: list[str] | None = None) -> set[str]:
        if time.time() - self._blocked_at < 3600:
            return set(self._blocked_cache)

        symbols = symbols or []
        blocked = set()
        if self.config.get("earnings_avoidance", True) and symbols:
            blocked |= self._fetch_earnings_blocks(symbols)

        self._blocked_cache = blocked
        self._blocked_at = time.time()
        return set(blocked)

    def get_days_to_earnings(self, symbol: str) -> int:
        """Return calendar days to the nearest known earnings event.

        Returns 99 if no earnings data is available for the symbol.
        A value of 0 means earnings are today; negative values mean
        earnings were in the past window (still within block period).
        """
        dt = self._earnings_dates.get(symbol)
        if dt is None:
            return 99
        delta = (dt.date() - datetime.utcnow().date()).days
        return abs(delta)

    def score_symbol_news(self, symbol: str) -> float:
        """Composite sentiment score for recent headlines on `symbol`.

        Returns value in [-1.0, +1.0] (vader compound average). Zero means
        neutral or no data. Cached per symbol for `news_cache_ttl_sec`.
        """
        if not self.config.get("news_sentiment", False):
            return 0.0
        if not (self._alpaca_key or self._news_key):
            return 0.0

        now = time.time()
        cached = self._sentiment_cache.get(symbol)
        if cached and now - cached[0] < self._sentiment_ttl_sec:
            return cached[1]

        articles = self._fetch_articles(symbol)
        if not articles:
            score = 0.0
        else:
            analyzer = self._get_analyzer()
            if analyzer is None:
                return 0.0
            scores = [analyzer.polarity_scores(text)["compound"] for text in articles]
            score = sum(scores) / len(scores)

        self._sentiment_cache[symbol] = (now, score)
        return score

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._analyzer = SentimentIntensityAnalyzer()
            except Exception:
                self._analyzer = None
        return self._analyzer

    def _fetch_articles(self, symbol: str) -> list[str]:
        """Fetch recent headline+summary text for `symbol`. Alpaca preferred."""
        hours = int(self.config.get("news_lookback_hours", 4))
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

        if self._alpaca_key and self._alpaca_secret:
            try:
                headers = {
                    "APCA-API-KEY-ID": self._alpaca_key,
                    "APCA-API-SECRET-KEY": self._alpaca_secret,
                }
                resp = requests.get(
                    self._ALPACA_NEWS_URL,
                    params={
                        "symbols": symbol,
                        "start": since,
                        "limit": 20,
                        "sort": "desc",
                    },
                    headers=headers,
                    timeout=10,
                )
                data = resp.json() if resp.ok else {}
                items = data.get("news", [])
                return [
                    f"{item.get('headline', '')}. {item.get('summary', '')}"
                    for item in items
                    if item.get("headline") or item.get("summary")
                ]
            except Exception:
                pass  # fall through to NewsAPI

        if self._news_key:
            try:
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": symbol,
                        "from": since,
                        "sortBy": "publishedAt",
                        "language": "en",
                        "apiKey": self._news_key,
                        "pageSize": 10,
                    },
                    timeout=10,
                )
                articles = resp.json().get("articles", []) if resp.ok else []
                return [
                    f"{a.get('title', '')}. {a.get('description', '')}"
                    for a in articles
                ]
            except Exception:
                return []

        return []

    def _fetch_earnings_blocks(self, symbols: list[str]) -> set[str]:
        if not self._alpaca_key or not self._alpaca_secret:
            return set()
        try:
            headers = {
                "APCA-API-KEY-ID": self._alpaca_key,
                "APCA-API-SECRET-KEY": self._alpaca_secret,
            }
            today = datetime.utcnow().date()
            start = (today - timedelta(days=2)).isoformat()
            end = (today + timedelta(days=1)).isoformat()
            resp = requests.get(
                "https://data.alpaca.markets/v1beta1/corporate-actions/announcements",
                params={"ca_types": "earnings", "since": start, "until": end, "symbols": ",".join(symbols)},
                headers=headers,
                timeout=10,
            )
            data = resp.json()
            blocked = set()
            for item in data.get("announcements", data if isinstance(data, list) else []):
                symbol = item.get("symbol")
                if not symbol:
                    continue
                blocked.add(symbol)
                # Store the earnings date for days_to_earnings calculation
                date_str = item.get("announcement_date") or item.get("ex_date") or item.get("date")
                if date_str:
                    try:
                        self._earnings_dates[symbol] = datetime.fromisoformat(str(date_str)[:10])
                    except (ValueError, TypeError):
                        pass
            return blocked
        except Exception:
            return set()
