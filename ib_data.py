"""IB data fetcher — historical bars for ALL asset types via ib_insync.

Handles stocks, futures, and crypto in one fetcher.
No Alpaca dependency.

Threading / asyncio:
  ib_insync uses asyncio internally. In Python 3.10+, worker threads have
  no event loop by default — calling ib methods from them raises
  "There is no current event loop in thread ...". We fix this by creating
  a fresh event loop for each worker thread on first use (_ensure_event_loop).

IB Pacing rules (TWS hard limits):
  - 60 reqHistoricalData calls per 10-minute window
  - Identical contract/barSize/duration: wait 15 s between requests
  - Violations return error code 162 — we back off and retry

Market hours (IB data availability):
  - Stocks  : useRTH=True  (TRADES, regular hours only)
  - Futures : useRTH=False (TRADES, 23-hour CME Globex sessions)
  - Crypto  : useRTH=False (MIDPOINT, near 24/7 on PAXOS)
"""
import asyncio
import time
import threading
import pandas as pd
from typing import Optional
from utils import setup_logger
from base_data import BaseDataFetcher

log = setup_logger("ib_data")

IB_CRYPTO_SYMBOLS: frozenset[str] = frozenset({
    "BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD",
})

TIMEFRAME_MAP = {
    "1Min":  "1 min",
    "5Min":  "5 mins",
    "15Min": "15 mins",
    "1Hour": "1 hour",
    "1Day":  "1 day",
}

CACHE_TTLS = {
    "1Min":  60,
    "5Min":  300,
    "15Min": 600,
    "1Hour": 3600,
    "1Day":  14400,
}

INTER_REQUEST_DELAY = 0.6   # seconds between IB historical data requests (pacing)
MAX_PACING_RETRIES = 3
PACING_BACKOFF_BASE = 15    # seconds — IB recommends 15 s wait after pacing violation

# Per-asset lenient bar-age limits (minutes) for freshness validation
BAR_AGE_LIMITS = {
    "stock":   {"1Min": 3, "5Min": 10, "15Min": 20, "1Hour": 90,  "1Day": 1440},
    "futures": {"1Min": 3, "5Min": 15, "15Min": 30, "1Hour": 120, "1Day": 1440},
    "crypto":  {"1Min": 5, "5Min": 25, "15Min": 50, "1Hour": 200, "1Day": 1440},
}


def _ensure_event_loop():
    """
    Guarantee the current thread has an asyncio event loop.

    Python 3.10+ raises RuntimeError in non-main threads that have no loop.
    ib_insync's synchronous wrappers (reqHistoricalData, qualifyContracts, etc.)
    internally call asyncio.get_event_loop(), so this must be called before
    any ib_insync method in worker threads.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


class IBDataFetcher(BaseDataFetcher):
    """Fetches historical and real-time data from IB Gateway for all asset types."""

    # Exchanges to try (in order) when SMART qualification fails for a stock
    _STOCK_EXCHANGE_HINTS = ("NYSE", "NASDAQ", "ARCA", "BATS", "IEX")

    # Hard-coded primary exchange for symbols that IB won't qualify via SMART
    _SYMBOL_EXCHANGE: dict[str, str] = {
        # ── Fintech / growth (NASDAQ) ──────────────────────────────────
        "HOOD": "NASDAQ",
        "RIVN": "NASDAQ",
        "LCID": "NASDAQ",
        "SOFI": "NASDAQ",
        "DKNG": "NASDAQ",
        "OPEN": "NASDAQ",
        "COIN": "NASDAQ",
        "AFRM": "NASDAQ",
        "GTLB": "NASDAQ",
        "MNST": "NASDAQ",
        "ON":   "NASDAQ",
        # ── Symbols that need explicit NYSE routing ────────────────────
        "UWMC": "NYSE",
        "BILL": "NYSE",
        "PATH": "NYSE",
        # ── NYSE large-caps that fail SMART qualification ──────────────
        "OXY":  "NYSE",
        "MO":   "NYSE",
        "SNOW": "NYSE",
        "NKE":  "NYSE",
        "MPC":  "NYSE",
        "NET":  "NYSE",
        "HPQ":  "NYSE",
        "NOW":  "NYSE",
        "COF":  "NYSE",
        "SO":   "NYSE",
        "LOW":  "NYSE",
        "CMG":  "NYSE",
        "PM":   "NYSE",
        "V":    "NYSE",
        "ZS":   "NASDAQ",
        "OKTA": "NASDAQ",
        # ── NASDAQ large-caps that fail SMART qualification ────────────
        "MELI": "NASDAQ",
        "CSX":  "NASDAQ",
        "PEP":  "NASDAQ",
        "ZM":   "NASDAQ",
        "AMAT": "NASDAQ",
        "IDXX": "NASDAQ",
        # ── ETFs used for regime/breadth checks ───────────────────────
        "SPY":  "ARCA",
        "QQQ":  "NASDAQ",
        "IWM":  "ARCA",
        "DIA":  "ARCA",
        "TLT":  "NASDAQ",
        "GLD":  "ARCA",
        "SLV":  "ARCA",
        "UUP":  "ARCA",
        "XLK":  "ARCA",
        "XLF":  "ARCA",
        "XLV":  "ARCA",
        "XLE":  "ARCA",
        "XLY":  "ARCA",
        "XLP":  "ARCA",
        "XLI":  "ARCA",
        "XLRE": "ARCA",
        "XLU":  "ARCA",
        "RSP":  "ARCA",
    }

    def __init__(self, ib, contract_manager, config: dict):
        self._ib = ib
        self._cm = contract_manager
        self._config = config
        # Allow delayed data when live market data subscription is missing
        self._ib.reqMarketDataType(3)  # 3 = delayed-frozen (live if available, else delayed)
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()
        self._pacing_lock = threading.Lock()
        self._last_request_ts: float = 0.0
        # Symbols that failed all contract qualification attempts.
        # TTL: symbol → retry-after timestamp (30 min for transient failures).
        # Permanent symbols (truly not in IB, e.g. SQ) get a 24-hour TTL.
        self._bad_contracts: dict[str, float] = {}
        # Symbols where data is temporarily unavailable (e.g. CL no subscription)
        # TTL: symbol → expiry timestamp. Cleared after 1 hour.
        self._no_data_cache: dict[str, float] = {}

        from instrument_classifier import InstrumentClassifier
        self._clf = InstrumentClassifier(config)

    # ── BaseDataFetcher public interface ──────────────────────────────────────

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        result = {}
        for symbol in symbols:
            df = self._fetch_with_retry(symbol, timeframe, days)
            if df is not None and not df.empty:
                result[symbol] = df
        return result

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5, cache_only: bool = False) -> Optional[pd.DataFrame]:
        key = (symbol, timeframe)
        ttl = CACHE_TTLS.get(timeframe, 360)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and time.time() - cached[0] < ttl:
                return cached[1]
            # Even if stale, return cached data in cache-only mode (better than None).
            stale_cached = cached[1] if cached else None
        if cache_only:
            return stale_cached
        df = self._fetch_with_retry(symbol, timeframe, days)
        if df is not None:
            with self._cache_lock:
                self._cache[key] = (time.time(), df)
        return df

    def prime_intraday_cache(self, symbols: list[str], timeframe: str = "5Min",
                              days: int = 5):
        ttl = CACHE_TTLS.get(timeframe, 360)
        now = time.time()
        with self._cache_lock:
            stale = [s for s in symbols
                     if (s, timeframe) not in self._cache
                     or now - self._cache[(s, timeframe)][0] >= ttl]
        if not stale:
            return
        fetched = 0
        for sym in stale:
            df = self._fetch_with_retry(sym, timeframe, days)
            if df is not None:
                with self._cache_lock:
                    self._cache[(sym, timeframe)] = (time.time(), df)
                fetched += 1
        log.info(f"IB cache primed ({timeframe}): {fetched}/{len(stale)} symbols")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        _ensure_event_loop()
        try:
            contract = self._resolve_contract(symbol)
            if not contract:
                return None
            ticker = self._ib.reqMktData(contract, "", True, False)
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if (ticker.last and ticker.last > 0) or (ticker.bid and ticker.bid > 0):
                    break
                time.sleep(0.1)
            for attr in ("last", "close", "bid"):
                val = getattr(ticker, attr, None)
                if val and float(val) > 0:
                    return float(val)
        except Exception as e:
            log.error(f"IB latest price failed for {symbol}: {e}")
        return None

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        result = {}
        for sym in symbols:
            price = self.get_latest_price(sym)
            if price:
                result[sym] = price
        return result

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        """Return bid/ask/last snapshot — drop-in for Alpaca get_snapshot."""
        _ensure_event_loop()
        try:
            contract = self._resolve_contract(symbol)
            if not contract:
                return None
            ticker = self._ib.reqMktData(contract, "", True, False)
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if (ticker.bid and ticker.bid > 0) or (ticker.last and ticker.last > 0):
                    break
                time.sleep(0.1)
            return {
                "bid":   float(ticker.bid)   if ticker.bid   and ticker.bid   > 0 else None,
                "ask":   float(ticker.ask)   if ticker.ask   and ticker.ask   > 0 else None,
                "last":  float(ticker.last)  if ticker.last  and ticker.last  > 0 else None,
                "close": float(ticker.close) if ticker.close and ticker.close > 0 else None,
            }
        except Exception as e:
            log.error(f"IB snapshot failed for {symbol}: {e}")
        return None

    def bar_age_limit(self, symbol: str, timeframe: str) -> int:
        """Return max acceptable bar age in minutes for this symbol/timeframe."""
        asset = self._clf.classify(symbol)
        return BAR_AGE_LIMITS.get(asset, BAR_AGE_LIMITS["stock"]).get(timeframe, 60)

    def invalidate_cache(self, symbol: str = None):
        with self._cache_lock:
            if symbol:
                for k in [k for k in self._cache if k[0] == symbol]:
                    del self._cache[k]
            else:
                self._cache.clear()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _fetch_with_retry(self, symbol: str, timeframe: str,
                          days: int) -> Optional[pd.DataFrame]:
        _ensure_event_loop()          # ensure loop exists in this thread
        bar_size = TIMEFRAME_MAP.get(timeframe, "1 day")
        if days >= 365:
            duration = f"{days // 365 + 1} Y"
        elif days >= 30:
            duration = f"{(days // 7) + 1} W"
        else:
            duration = f"{days} D"

        for attempt in range(MAX_PACING_RETRIES + 1):
            try:
                return self._do_fetch(symbol, bar_size, duration)
            except _PacingError:
                wait = PACING_BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    f"IB pacing violation: {symbol} "
                    f"(attempt {attempt + 1}/{MAX_PACING_RETRIES + 1}) "
                    f"— waiting {wait}s"
                )
                time.sleep(wait)
            except Exception as e:
                log.error(f"IB bars failed {symbol} @ {timeframe}: {e}")
                return None

        log.error(f"IB bars: gave up on {symbol} after pacing retries")
        return None

    def _do_fetch(self, symbol: str, bar_size: str,
                  duration: str) -> Optional[pd.DataFrame]:
        # Skip symbols whose data was unavailable recently (1-hour cooldown)
        no_data_expiry = self._no_data_cache.get(symbol, 0.0)
        if time.time() < no_data_expiry:
            return None

        contract = self._resolve_contract(symbol)
        if contract is None:
            log.warning(f"Cannot resolve IB contract for {symbol}")
            return None

        asset = self._clf.classify(symbol)
        # Crypto: MIDPOINT (TRADES not always available on PAXOS)
        # Stocks/futures: TRADES
        what_to_show = "MIDPOINT" if asset == "crypto" else "TRADES"
        # Stocks: regular hours only; futures/crypto: 24-hour sessions
        use_rth = (asset == "stock")

        bars = self._request_historical(contract, bar_size, duration,
                                        what_to_show, use_rth, symbol)

        # ── CONTFUT fallback ────────────────────────────────────────────────────
        # reqHistoricalData is unreliable with secType='CONTFUT' on some products
        # (e.g. CL on NYMEX). If we get nothing back and the contract is CONTFUT,
        # try to rebuild it as an explicit FUT with the same expiry.
        if bars is None and asset == "futures":
            sec_type = getattr(contract, "secType", "")
            if sec_type == "CONTFUT":
                log.warning(
                    f"{symbol}: CONTFUT reqHistoricalData returned nothing — "
                    f"retrying as explicit FUT"
                )
                bars = self._contfut_fallback(
                    symbol, contract, bar_size, duration, what_to_show, use_rth
                )
            # Both attempts failed — suppress retries for 1 hour
            if not bars:
                self._no_data_cache[symbol] = time.time() + 3600
                log.warning(
                    f"{symbol}: no historical data available (IB subscription/permissions). "
                    f"Suppressing retries for 1 hour."
                )

        if not bars:
            log.debug(f"Empty bar response: {symbol} ({bar_size})")
            return None

        rows = []
        for bar in bars:
            rows.append({
                "timestamp": pd.Timestamp(bar.date),
                "open":   float(bar.open),
                "high":   float(bar.high),
                "low":    float(bar.low),
                "close":  float(bar.close),
                "volume": float(bar.volume),
                "vwap":   float(bar.average) if hasattr(bar, "average") and bar.average else None,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        log.debug(f"IB bars {symbol}: {len(df)} rows @ {bar_size}")
        return df

    def _resolve_contract(self, symbol: str):
        """Return the correct ib_insync Contract for any asset type.

        For stocks: tries SMART qualification first, then falls back to explicit
        primaryExch hints (NYSE, NASDAQ, ARCA …).  Symbols that fail all attempts
        are added to _bad_contracts and skipped on subsequent calls to avoid
        flooding the logs with repeated Error 200s.
        """
        # Fast-path: skip symbols that recently failed qualification (TTL-based)
        retry_after = self._bad_contracts.get(symbol, 0.0)
        if time.time() < retry_after:
            return None

        asset = self._clf.classify(symbol)

        if asset == "futures":
            return self._cm.get_contract(symbol)

        if asset == "stock":
            from ib_insync import Stock
            _ensure_event_loop()

            # 0th attempt: hard-coded override with SMART routing + primary exchange hint.
            # This keeps qualification stable without forcing direct exchange routing.
            if symbol in self._SYMBOL_EXCHANGE:
                exch = self._SYMBOL_EXCHANGE[symbol]
                try:
                    c = Stock(symbol, "SMART", "USD")
                    c.primaryExch = exch
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} via override primaryExch={exch}")
                        return qualified[0]
                    log.warning(f"qualify {symbol} override {exch} returned empty")
                except Exception as e:
                    log.warning(f"qualify {symbol} override {exch} exception: {e}")

            # 1st attempt: plain SMART routing (works for most liquid US equities)
            contract = Stock(symbol, "SMART", "USD")
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
                log.warning(f"qualify {symbol} SMART returned empty")
            except Exception as e:
                log.warning(f"qualify {symbol} SMART exception: {e}")

            # 2nd attempt: primaryExch hints (SMART + hint)
            for exch in self._STOCK_EXCHANGE_HINTS:
                try:
                    c = Stock(symbol, "SMART", "USD")
                    c.primaryExch = exch
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} with primaryExch={exch}")
                        return qualified[0]
                except Exception:
                    pass

            # 3rd attempt: direct exchange routing (no SMART)
            for exch in self._STOCK_EXCHANGE_HINTS:
                try:
                    c = Stock(symbol, exch, "USD")
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} via direct exchange={exch}")
                        return qualified[0]
                except Exception:
                    pass

            # All attempts failed — suppress retries for 30 min (transient failures
            # e.g. IB blip, asyncio glitch) so the log doesn't flood every cycle.
            # Truly unavailable symbols (Error 200) will consistently fail and stay
            # suppressed; valid symbols recover automatically after the TTL.
            log.warning(
                f"IB: cannot qualify contract for {symbol} "
                f"(tried override + SMART + primaryExch + direct) — skipping for 30 min"
            )
            self._bad_contracts[symbol] = time.time() + 1800  # 30-minute cooldown
            return None

        if asset == "crypto":
            from ib_insync import Crypto
            _ensure_event_loop()
            base = symbol.split("/")[0] if "/" in symbol else symbol[:-3]
            contract = Crypto(base, "PAXOS", "USD")
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
            except Exception:
                pass
            return contract

        log.warning(f"Unknown asset type for {symbol} — treating as stock")
        from ib_insync import Stock
        return Stock(symbol, "SMART", "USD")


    def _request_historical(self, contract, bar_size: str, duration: str,
                             what_to_show: str, use_rth: bool,
                             symbol: str):
        """Single serialized reqHistoricalData call with pacing + error detection.

        Returns list of bars or None.
        Raises _PacingError if IB returns error 162 (pacing violation).
        Logs subscription/permission errors (error 354, 10197) and returns None.
        """
        with self._pacing_lock:
            elapsed = time.time() - self._last_request_ts
            if elapsed < INTER_REQUEST_DELAY:
                time.sleep(INTER_REQUEST_DELAY - elapsed)
            try:
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
            except Exception as e:
                err_str = str(e)
                if "162" in err_str or "pacing" in err_str.lower():
                    raise _PacingError(err_str)
                # Subscription / permission errors — log clearly and return None
                if any(code in err_str for code in ("354", "10197", "10090")):
                    log.warning(
                        f"{symbol}: IB data subscription error — "
                        f"check market data permissions for this product. ({e})"
                    )
                    return None
                raise
            finally:
                self._last_request_ts = time.time()

        return bars if bars else None

    def _contfut_fallback(self, symbol: str, contfut_contract,
                          bar_size: str, duration: str,
                          what_to_show: str, use_rth: bool):
        """Try reqHistoricalData with an explicit FUT contract derived from a CONTFUT.

        Called when a CONTFUT reqHistoricalData returns None/timeout.
        Builds a proper Future using the expiry already resolved in the CONTFUT.
        """
        from ib_insync import Future
        expiry = getattr(contfut_contract, "lastTradeDateOrContractMonth", "")
        exchange = getattr(contfut_contract, "exchange", "")
        currency = getattr(contfut_contract, "currency", "USD")
        multiplier = getattr(contfut_contract, "multiplier", "")
        root = getattr(contfut_contract, "symbol", symbol)

        if not expiry or not exchange:
            log.warning(f"{symbol}: CONTFUT fallback — missing expiry/exchange, cannot retry")
            return None

        try:
            fut = Future(
                symbol=root,
                lastTradeDateOrContractMonth=expiry,
                exchange=exchange,
                currency=currency,
                multiplier=multiplier,
            )
            _ensure_event_loop()
            qualified = self._ib.qualifyContracts(fut)
            if qualified:
                fut = qualified[0]

            bars = self._request_historical(fut, bar_size, duration,
                                             what_to_show, use_rth, symbol)
            if bars:
                log.info(
                    f"{symbol}: CONTFUT fallback succeeded using explicit FUT "
                    f"{getattr(fut, 'localSymbol', expiry)}"
                )
                # Update contract manager cache so next fetch uses the FUT directly
                self._cm.invalidate(root)
            return bars
        except Exception as e:
            log.error(f"{symbol}: CONTFUT fallback failed: {e}")
            return None


class _PacingError(Exception):
    """Raised when IB returns a pacing violation (error 162)."""
