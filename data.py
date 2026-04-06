import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from utils import setup_logger

log = setup_logger("data")

CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}


def _normalize_crypto(symbol: str) -> str:
    """Normalize crypto symbol to slash format (ETHUSD → ETH/USD) for Alpaca data API."""
    if symbol in CRYPTO_SYMBOLS and "/" not in symbol:
        return symbol[:-3] + "/" + symbol[-3:]
    return symbol


class DataFetcher:
    """Data fetching with rate limiting and exponential backoff.
    
    Alpaca limits: 200 requests/min for free tier, higher for paid.
    We implement a token bucket rate limiter + exponential backoff on failures.
    """
    
    def __init__(self, broker, requests_per_minute: int = 150):
        self.api = broker.api
        
        # Rate limiter: token bucket
        self._rate_limit = requests_per_minute
        self._tokens = requests_per_minute
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def _wait_for_rate_limit(self):
        """Block until a request token is available."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            
            # Refill tokens based on elapsed time
            refill = elapsed * (self._rate_limit / 60.0)
            self._tokens = min(self._rate_limit, self._tokens + refill)
            self._last_refill = now
            
            if self._tokens < 1:
                # Wait until we have a token
                wait_time = (1 - self._tokens) * (60.0 / self._rate_limit)
                time.sleep(wait_time)
                self._tokens = 1
            
            self._tokens -= 1
    
    def _api_call_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Execute API call with exponential backoff on failure."""
        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                
                # Rate limit error - back off significantly
                if "rate" in error_str or "429" in error_str or "too many" in error_str:
                    backoff = min(60, 2 ** attempt * 5)  # 5s, 10s, 20s, max 60s
                    log.warning(f"Rate limit hit, backing off {backoff}s: {e}")
                    time.sleep(backoff)
                # Server error - retry with backoff
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    backoff = 2 ** attempt
                    log.warning(f"Server error, retry in {backoff}s: {e}")
                    time.sleep(backoff)
                # Other errors - don't retry
                else:
                    raise e
        
        # Final attempt without catching
        self._wait_for_rate_limit()
        return func(*args, **kwargs)

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        """Fetch historical bars for multiple symbols with rate limiting."""
        # Normalize crypto symbols to slash format for Alpaca API
        symbols = [_normalize_crypto(s) for s in symbols]
        end = datetime.now()
        start = end - timedelta(days=days)

        # Separate crypto and stock symbols
        stock_symbols = [s for s in symbols if s not in CRYPTO_SYMBOLS]
        crypto_symbols = [s for s in symbols if s in CRYPTO_SYMBOLS]

        bars = {}
        time_fmt = "%Y-%m-%dT%H:%M:%SZ" if "Min" in timeframe else "%Y-%m-%d"

        # Fetch stock bars with rate limiting
        batch_size = 20
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
            except Exception as e:
                log.error(f"Failed to fetch bars for {batch}: {e}")

        # Fetch crypto bars with rate limiting
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

        # Convert to DataFrames
        result = {}
        for sym, data in bars.items():
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                result[sym] = df

        log.info(f"Fetched {timeframe} bars for {len(result)}/{len(symbols)} symbols")
        return result

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> pd.DataFrame | None:
        """Fetch intraday bars for a single symbol."""
        result = self.get_bars([symbol], timeframe=timeframe, days=days)
        return result.get(symbol)

    def get_latest_price(self, symbol: str) -> float | None:
        try:
            symbol = _normalize_crypto(symbol)
            if symbol in CRYPTO_SYMBOLS:
                # Try multiple methods for crypto price (API versions vary)
                # Method 1: get_latest_crypto_quotes (plural, newer API)
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
                
                # Method 2: get_crypto_bars (fallback)
                try:
                    bars = self._api_call_with_retry(
                        self.api.get_crypto_bars, symbol, "1Min", limit=1
                    )
                    for bar in bars:
                        return float(bar.c)
                except Exception:
                    pass
                
                # Method 3: get_latest_bar (another fallback)
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
