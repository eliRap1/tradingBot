"""Abstract broker interface and shared order/position dataclasses."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    symbol: str
    qty: float          # positive = long, negative = short
    avg_price: float
    market_value: float
    unrealized_pl: float
    side: str           # "long" | "short"


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: str           # "buy" | "sell"
    order_type: str     # "market" | "limit" | "stop" | "stop_limit" | "bracket"
    status: str         # "new" | "filled" | "canceled" | "rejected" | "submitted"
    filled_avg_price: Optional[float] = None
    filled_qty: Optional[float] = None


@dataclass
class OrderRequest:
    symbol: str
    qty: float
    side: str                    # "buy" | "sell"
    order_type: str = "market"
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    notional: Optional[float] = None  # USD notional — required for IB crypto BUY orders


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    mid: float = field(init=False)

    def __post_init__(self):
        self.mid = (self.bid + self.ask) / 2


@dataclass
class Clock:
    is_open: bool
    next_open: Optional[object]   # datetime
    next_close: Optional[object]  # datetime


class BrokerConnectionError(Exception):
    """Raised when the broker connection is lost or cannot be established."""


class BaseBroker(ABC):
    """Abstract broker — all coordinator code uses only this interface."""

    @abstractmethod
    def get_account(self) -> dict: ...

    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_buying_power(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]: ...

    @abstractmethod
    def get_open_orders(self) -> list[Order]: ...

    @abstractmethod
    def submit_order(self, req: OrderRequest) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str): ...

    @abstractmethod
    def cancel_all_orders(self): ...

    @abstractmethod
    def close_position(self, symbol: str): ...

    @abstractmethod
    def close_all_positions(self): ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]: ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def get_clock(self) -> Clock: ...

    @abstractmethod
    def asset_type(self, symbol: str) -> str:
        """Return 'stock', 'crypto', or 'futures'."""
        ...
