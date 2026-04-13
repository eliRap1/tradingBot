"""Alpaca data fetcher implementation of BaseDataFetcher."""
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from typing import Optional
from utils import setup_logger
from base_data import BaseDataFetcher

log = setup_logger("alpaca_data")

CRYPTO_SYMBOLS = {
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD",
    "BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD", "DOGEUSD",
}


def _normalize_crypto(symbol: str) -> str:
    """Normalize crypto symbol to slash format (ETHUSD → ETH/USD) for Alpaca data API."""
    if symbol in CRYPTO_SYMBOLS and "/" not in symbol:
        return symbol[:-3] + "/" + symbol[-3:]
    return symbol


class AlpacaDataFetcher(BaseDataFetcher):
    """Data fetching with rate limiting and exponential backoff.

    Alpaca free tier: ~60 requests/min.
    Global semaphore limits concurrent requests across all watcher threads.
    """

    # Global semaphore — max 2 concurrent API calls
    _global_semaphore = threading.Semaphore(2)
    # Minimum gap between any two API calls (seconds)
    _min_interval = 2.0
    _last_call_time = 0.0
    _call_lock = threading.Lock()

    def __init__(self, broker, requests_per_minute: int = 40):
        # Accept either AlpacaBroker or the old Broker (both have .api)
        self.api = broker.api
        self._rate_limit = requests_per_minute
        # Intraday bar cache: {(symbol, timeframe): (timestamp, DataFrame)}
        self._intraday_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 360  # 6 minutes — slightly longer than the 5min cycle

    def _wait_for_rate_limit(self):
        """Enforce minimum interval between API calls globally."""
        with AlpacaDataFetcher._call_lock:
            now = time.time()
            gap = now - AlpacaDataFetcher._last_call_time
            if gap < AlpacaDataFetcher._min_interval:
                time.sleep(AlpacaDataFetcher._min_interval - gap)
            AlpacaDataFetcher._last_call_time = time.time()

    def _api_call_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Execute API call with global semaphore + exponential backoff."""
        with AlpacaDataFetcher._global_semaphore:
            for attempt in range(max_retries):
                self._wait_for_rate_limit()
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate" in error_str or "429" in error_str or "too many" in error_str:
                        backoff = min(60, 2 ** attempt * 10)
                        log.warning(f"Rate limit hit, backing off {backoff}s: {e}")
                        time.sleep(backoff)
                    elif "500" in error_str or "502" in error_str or "503" in error_str:
                        backoff = 2 ** attempt
                        log.warning(f"Server error, retry in {backoff}s: {e}")
                        time.sleep(backoff)
                    else:
                        raise e
            self._wait_for_rate_limit()
            return func(*args, **kwargs)

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        """Fetch historical bars for multiple symbols with rate limiting."""
        symbols = [_normalize_crypto(s) for s in symbols if isinstance(s, str)]
        end = datetime.now()
        start = end - timedelta(days=days)

        stock_symbols = [s for s in symbols if s not in CRYPTO_SYMBOLS]
        crypto_symbols = [s for s in symbols if s in CRYPTO_SYMBOLS]

        bars = {}
        time_fmt = "%Y-%m-%dT%H:%M:%SZ" if "Min" in timeframe else "%Y-%m-%d"

        # Fetch stock bars in batches of 30 with delays to avoid rate limits
        batch_size = 30
        for i in range(0, len(stock_symbols), batch_size):
            batch = stock_symbols[i:i + batch_size]
            try:
                raw = self._api_call_with_retry(
                    self.api.get_bars,
                    batch,
                    timeframe,
                    start=start.strftime(time_fmt),
                    end=end.strftime(time_fmt),
                    adjustment="split",
                    feed="iex"
                )
                for bar in raw:
                    sym = bar.S
                    if sym not in bars:
                        bars[sym] = []
                    bars[sym].append({
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": int(bar.v),
                        "vwap": float(bar.vw) if hasattr(bar, "vw") else None,
                    })
                if i + batch_size < len(stock_symbols):
                    time.sleep(3)
            except Exception as e:
                log.error(f"Failed to fetch bars for batch {batch}: {e}")

        # Fetch crypto bars
        for sym in crypto_symbols:
            try:
                raw = self._api_call_with_retry(
                    self.api.get_crypto_bars,
                    sym,
                    timeframe,
                    start=start.strftime(time_fmt),
                    end=end.strftime(time_fmt),
                )
                bars[sym] = []
                for bar in raw:
                    bars[sym].append({
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": float(bar.v),
                        "vwap": float(bar.vw) if hasattr(bar, "vw") else None,
                    })
            except Exception as e:
                log.error(f"Failed to fetch crypto bars for {sym}: {e}")

        result = {}
        for sym, data in bars.items():
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                result[sym] = df

        log.info(f"Fetched {timeframe} bars for {len(result)}/{len(symbols)} symbols")
        return result

    # Per-timeframe cache TTLs
    _CACHE_TTLS = {"5Min": 360, "1Hour": 3600, "1Day": 14400}

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> Optional[pd.DataFrame]:
        """Fetch bars, returning cached data if fresh. TTL varies by timeframe."""
        ttl = self._CACHE_TTLS.get(timeframe, self._cache_ttl)
        key = (symbol, timeframe)
        with self._cache_lock:
            if key in self._intraday_cache:
                cached_time, cached_df = self._intraday_cache[key]
                if time.time() - cached_time < ttl:
                    return cached_df

        result = self.get_bars([symbol], timeframe=timeframe, days=days)
        df = result.get(symbol)
        if df is not None:
            with self._cache_lock:
                self._intraday_cache[key] = (time.time(), df)
        return df

    def prime_intraday_cache(self, symbols: list[str], timeframe: str = "5Min",
                              days: int = 5):
        """Bulk-fetch bars for all symbols at once and populate cache.

        Call this once per coordinator cycle instead of letting each watcher fetch independently.
        """
        ttl = self._CACHE_TTLS.get(timeframe, self._cache_ttl)
        now = time.time()
        with self._cache_lock:
            stale = [s for s in symbols
                     if (s, timeframe) not in self._intraday_cache
                     or now - self._intraday_cache[(s, timeframe)][0] >= ttl]
        if not stale:
            return
        result = self.get_bars(stale, timeframe=timeframe, days=days)
        now = time.time()
        with self._cache_lock:
            for sym, df in result.items():
                self._intraday_cache[(sym, timeframe)] = (now, df)
        log.info(f"Cache primed ({timeframe}): {len(result)}/{len(stale)} symbols")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            symbol = _normalize_crypto(symbol)
            if symbol in CRYPTO_SYMBOLS:
                try:
                    quotes = self._api_call_with_retry(
                        self.api.get_latest_crypto_quotes, [symbol]
                    )
                    if symbol in quotes:
                        q = quotes[symbol]
                        bid = float(q.bp) if hasattr(q, 'bp') else 0
                        ask = float(q.ap) if hasattr(q, 'ap') else 0
                        if bid > 0 and ask > 0:
                            return (bid + ask) / 2
                        return ask or bid
                except AttributeError:
                    pass
                try:
                    bars = self._api_call_with_retry(
                        self.api.get_crypto_bars, symbol, "1Min", limit=1
                    )
                    for bar in bars:
                        return float(bar.c)
                except Exception:
                    pass
                try:
                    bar = self._api_call_with_retry(
                        self.api.get_latest_crypto_bar, symbol
                    )
                    if bar:
                        return float(bar.c)
                except AttributeError:
                    pass
                return None
            else:
                trade = self._api_call_with_retry(self.api.get_latest_trade, symbol, feed="iex")
                return float(trade.price)
        except Exception as e:
            log.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        prices = {}
        for sym in symbols:
            price = self.get_latest_price(sym)
            if price:
                prices[sym] = price
        return prices

    def get_snapshot(self, symbol: str):
        try:
            return self._api_call_with_retry(self.api.get_snapshot, symbol, feed="iex")
        except Exception as e:
            log.error(f"Failed to get snapshot for {symbol}: {e}")
            return None
