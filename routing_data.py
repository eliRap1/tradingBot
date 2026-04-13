"""RoutingDataFetcher — dispatches data requests by asset type.

Routing table:
  futures → IBDataFetcher
  stocks  → AlpacaDataFetcher
  crypto  → AlpacaDataFetcher
"""
from typing import Optional
import pandas as pd
from utils import setup_logger
from base_data import BaseDataFetcher

log = setup_logger("routing_data")


class RoutingDataFetcher(BaseDataFetcher):
    def __init__(self, alpaca_fetcher: BaseDataFetcher, ib_fetcher: BaseDataFetcher,
                 classifier):
        """
        Args:
            alpaca_fetcher: AlpacaDataFetcher (stocks + crypto)
            ib_fetcher: IBDataFetcher (futures)
            classifier: InstrumentClassifier
        """
        self._alpaca = alpaca_fetcher
        self._ib = ib_fetcher
        self._clf = classifier

    def _fetcher_for(self, symbol: str) -> BaseDataFetcher:
        if self._clf.is_futures(symbol):
            return self._ib
        return self._alpaca

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        # Split by asset type
        futures_syms = [s for s in symbols if self._clf.is_futures(s)]
        other_syms = [s for s in symbols if not self._clf.is_futures(s)]

        result = {}
        if other_syms:
            result.update(self._alpaca.get_bars(other_syms, timeframe, days))
        if futures_syms:
            result.update(self._ib.get_bars(futures_syms, timeframe, days))
        return result

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> Optional[pd.DataFrame]:
        return self._fetcher_for(symbol).get_intraday_bars(symbol, timeframe, days)

    def prime_intraday_cache(self, symbols: list[str], timeframe: str = "5Min",
                              days: int = 5):
        futures_syms = [s for s in symbols if self._clf.is_futures(s)]
        other_syms = [s for s in symbols if not self._clf.is_futures(s)]
        if other_syms:
            self._alpaca.prime_intraday_cache(other_syms, timeframe, days)
        if futures_syms:
            self._ib.prime_intraday_cache(futures_syms, timeframe, days)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self._fetcher_for(symbol).get_latest_price(symbol)

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        result = {}
        for sym in symbols:
            price = self.get_latest_price(sym)
            if price:
                result[sym] = price
        return result

    # Pass-through for Alpaca-specific methods used by coordinator
    def get_snapshot(self, symbol: str):
        return self._alpaca.get_snapshot(symbol)
