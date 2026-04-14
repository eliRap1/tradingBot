from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import requests


class NewsSentimentEngine:
    """Earnings blocker plus optional headline sentiment."""

    def __init__(self, config: dict):
        self.config = config.get("edge", {})
        self._blocked_cache: set[str] = set()
        self._blocked_at = 0.0
        # symbol → nearest earnings date (absolute date, not relative)
        self._earnings_dates: dict[str, datetime] = {}
        self._news_key = os.getenv("NEWSAPI_KEY", "")
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")

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
        if not self.config.get("news_sentiment", False) or not self._news_key:
            return 0.0
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            hours = int(self.config.get("news_lookback_hours", 4))
            since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
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
            articles = resp.json().get("articles", [])
            if not articles:
                return 0.0
            scores = []
            for article in articles:
                text = f"{article.get('title', '')}. {article.get('description', '')}"
                scores.append(analyzer.polarity_scores(text)["compound"])
            return sum(scores) / len(scores)
        except Exception:
            return 0.0

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
