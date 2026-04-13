"""IB data fetcher — historical bars for futures via ib_insync reqHistoricalData.

Only handles futures. Stocks and crypto are handled by AlpacaDataFetcher.
"""
import time
import threading
import pandas as pd
from typing import Optional
from utils import setup_logger
from base_data import BaseDataFetcher

log = setup_logger("ib_data")

# Map our timeframe strings to IB barSizeSetting strings
TIMEFRAME_MAP = {
    "1Min":  "1 min",
    "5Min":  "5 mins",
    "15Min": "15 mins",
    "1Hour": "1 hour",
    "1Day":  "1 day",
}

# Cache TTLs by timeframe (seconds)
CACHE_TTLS = {"5Min": 360, "1Hour": 3600, "1Day": 14400}


class IBDataFetcher(BaseDataFetcher):
    """Fetches historical bar data from IB Gateway for futures contracts."""

    def __init__(self, ib, contract_manager, config: dict):
        """
        Args:
            ib: connected ib_insync.IB instance
            contract_manager: ContractManager for resolving futures contracts
            config: full config dict
        """
        self._ib = ib
        self._cm = contract_manager
        self._config = config
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        result = {}
        bar_size = TIMEFRAME_MAP.get(timeframe, "1 day")
        duration = f"{days} D"

        for symbol in symbols:
            try:
                contract = self._cm.get_contract(symbol)
                if not contract:
                    log.warning(f"Cannot resolve IB contract for {symbol}")
                    continue

                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                if not bars:
                    log.warning(f"No IB historical data for {symbol}")
                    continue

                rows = []
                for bar in bars:
                    rows.append({
                        "timestamp": pd.Timestamp(bar.date),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "vwap": float(bar.average) if hasattr(bar, "average") else None,
                    })
                df = pd.DataFrame(rows)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                result[symbol] = df
                log.info(f"IB bars for {symbol}: {len(df)} bars @ {timeframe}")

            except Exception as e:
                log.error(f"Failed to fetch IB bars for {symbol}: {e}")

        return result

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> Optional[pd.DataFrame]:
        ttl = CACHE_TTLS.get(timeframe, 360)
        key = (symbol, timeframe)
        with self._cache_lock:
            if key in self._cache:
                cached_ts, cached_df = self._cache[key]
                if time.time() - cached_ts < ttl:
                    return cached_df

        result = self.get_bars([symbol], timeframe=timeframe, days=days)
        df = result.get(symbol)
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
        result = self.get_bars(stale, timeframe=timeframe, days=days)
        now = time.time()
        with self._cache_lock:
            for sym, df in result.items():
                self._cache[(sym, timeframe)] = (now, df)
        log.info(f"IB cache primed ({timeframe}): {len(result)}/{len(stale)} symbols")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price via IB snapshot (last or close)."""
        try:
            contract = self._cm.get_contract(symbol)
            if not contract:
                return None
            ticker = self._ib.reqMktData(contract, "", True, False)
            self._ib.sleep(1)
            self._ib.cancelMktData(contract)
            if ticker.last and ticker.last > 0:
                return float(ticker.last)
            if ticker.close and ticker.close > 0:
                return float(ticker.close)
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
