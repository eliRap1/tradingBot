"""Abstract data fetcher interface."""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseDataFetcher(ABC):
    """Abstract data fetcher — coordinator and watchers use only this interface."""

    @abstractmethod
    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]: ...

    @abstractmethod
    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> Optional[pd.DataFrame]: ...

    @abstractmethod
    def prime_intraday_cache(self, symbols: list[str], timeframe: str = "5Min",
                              days: int = 5): ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]: ...

    @abstractmethod
    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]: ...
