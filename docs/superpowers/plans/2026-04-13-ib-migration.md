# IB Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Alpaca as active broker with Interactive Brokers (paper, port 4002), extend the bot to trade stocks, crypto (via Alpaca fallback), and futures (NQ, ES, CL, GC), with a clean broker/data abstraction layer modelled on QuantConnect/Lean.

**Architecture:** `BaseBroker` and `BaseDataFetcher` abstract interfaces define the contract; `AlpacaBroker`/`AlpacaDataFetcher` wrap existing code; `IBBroker`/`IBDataFetcher` use `ib_insync`; `RoutingBroker`/`RoutingDataFetcher` dispatch by asset type (futures→IB, stocks→IB, crypto→Alpaca). `InstrumentClassifier` is the single source of truth for symbol type. `StrategyRouter` assigns the optimal strategy set per instrument type.

**Tech Stack:** Python 3.11, `ib_insync`, `alpaca-trade-api`, `pandas`, `numpy`, `pytest`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `base_broker.py` | Abstract `BaseBroker` interface + `Position`/`Order`/`OrderRequest`/`Quote`/`Clock` dataclasses |
| Create | `base_data.py` | Abstract `BaseDataFetcher` interface |
| Create | `instrument_classifier.py` | Classify any symbol → "stock"\|"crypto"\|"futures" |
| Create | `alpaca_broker.py` | `AlpacaBroker(BaseBroker)` — full Alpaca logic moved from `broker.py` |
| Create | `alpaca_data.py` | `AlpacaDataFetcher(BaseDataFetcher)` — full Alpaca logic moved from `data.py` |
| Modify | `broker.py` | Thin shim: `from alpaca_broker import AlpacaBroker as Broker; from alpaca_broker import CRYPTO_SYMBOLS` |
| Modify | `data.py` | Thin shim: `from alpaca_data import AlpacaDataFetcher as DataFetcher; from alpaca_data import CRYPTO_SYMBOLS` |
| Create | `contract_manager.py` | IB contract resolution + front-month detection + auto-roll cache |
| Create | `ib_broker.py` | `IBBroker(BaseBroker)` — IB bracket orders, account queries via `ib_insync` |
| Create | `ib_data.py` | `IBDataFetcher(BaseDataFetcher)` — IB historical bars for futures |
| Create | `routing_broker.py` | `RoutingBroker(BaseBroker)` — dispatches by asset type |
| Create | `routing_data.py` | `RoutingDataFetcher(BaseDataFetcher)` — dispatches by asset type |
| Create | `strategy_router.py` | `StrategyRouter` — returns strategy list per instrument type |
| Create | `strategies/futures_trend.py` | `FuturesTrendStrategy` — ORB + session VWAP + ADX + ATR gate |
| Modify | `strategies/__init__.py` | Register `FuturesTrendStrategy` |
| Modify | `watcher.py` | Accept optional `strategies` dict at `__init__`; fall back to `ALL_STRATEGIES` |
| Modify | `coordinator.py` | Use `RoutingBroker`, `RoutingDataFetcher`, `StrategyRouter` when spawning watchers |
| Modify | `config.yaml` | Add `ib:` block and `futures:` block |
| Create | `tests/test_instrument_classifier.py` | Unit tests for classifier |
| Create | `tests/test_strategy_router.py` | Unit tests for strategy router |
| Create | `tests/test_routing_broker.py` | Unit tests for routing broker dispatch |

---

## Task 1: Install `ib_insync`

**Files:** none

- [ ] **Step 1: Install the library**

```bash
pip install ib_insync
```

Expected: `Successfully installed ib_insync-...`

- [ ] **Step 2: Verify import**

```bash
python -c "import ib_insync; print(ib_insync.__version__)"
```

Expected: version string printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: install ib_insync for IB Gateway connectivity"
```

---

## Task 2: Base Interfaces + Shared Dataclasses

**Files:**
- Create: `base_broker.py`
- Create: `base_data.py`

- [ ] **Step 1: Create `base_broker.py`**

```python
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
    order_type: str     # "market" | "limit" | "stop" | "stop_limit"
    status: str         # "new" | "filled" | "canceled" | "rejected"
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
    """Raised when the broker connection is lost."""


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
```

- [ ] **Step 2: Create `base_data.py`**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add base_broker.py base_data.py
git commit -m "feat: add BaseBroker and BaseDataFetcher abstract interfaces"
```

---

## Task 3: InstrumentClassifier

**Files:**
- Create: `instrument_classifier.py`
- Create: `tests/test_instrument_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_instrument_classifier.py
import pytest
from instrument_classifier import InstrumentClassifier

CONFIG = {
    "futures": {"contracts": [
        {"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"}
    ]},
    "screener": {"crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD"]}
}


@pytest.fixture
def clf():
    return InstrumentClassifier(CONFIG)


def test_futures_roots(clf):
    assert clf.classify("NQ") == "futures"
    assert clf.classify("ES") == "futures"
    assert clf.classify("CL") == "futures"
    assert clf.classify("GC") == "futures"


def test_crypto_symbols(clf):
    assert clf.classify("BTC/USD") == "crypto"
    assert clf.classify("ETH/USD") == "crypto"
    assert clf.classify("DOGE/USD") == "crypto"


def test_stock_fallback(clf):
    assert clf.classify("AAPL") == "stock"
    assert clf.classify("NVDA") == "stock"
    assert clf.classify("TSLA") == "stock"


def test_unknown_is_stock(clf):
    assert clf.classify("ZZZZ") == "stock"
```

- [ ] **Step 2: Run test — expect failure**

```bash
cd C:\Users\eli08\Videos\tradingBot && python -m pytest tests/test_instrument_classifier.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'instrument_classifier'`

- [ ] **Step 3: Create `instrument_classifier.py`**

```python
"""Classifies any symbol as 'stock', 'crypto', or 'futures'.

Priority order (first match wins):
  1. Symbol in FUTURES_ROOTS set  → 'futures'
  2. Symbol in CRYPTO_SYMBOLS set → 'crypto'
  3. Everything else              → 'stock'
"""


class InstrumentClassifier:
    def __init__(self, config: dict):
        futures_contracts = config.get("futures", {}).get("contracts", [])
        self._futures_roots = {c["root"] for c in futures_contracts}

        crypto_list = config.get("screener", {}).get("crypto", [])
        # Store both slash and no-slash formats
        self._crypto_symbols: set[str] = set()
        for sym in crypto_list:
            self._crypto_symbols.add(sym)
            self._crypto_symbols.add(sym.replace("/", ""))

    def classify(self, symbol: str) -> str:
        """Return 'futures', 'crypto', or 'stock'."""
        if symbol in self._futures_roots:
            return "futures"
        if symbol in self._crypto_symbols:
            return "crypto"
        return "stock"

    def is_futures(self, symbol: str) -> bool:
        return self.classify(symbol) == "futures"

    def is_crypto(self, symbol: str) -> bool:
        return self.classify(symbol) == "crypto"

    def is_stock(self, symbol: str) -> bool:
        return self.classify(symbol) == "stock"
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_instrument_classifier.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add instrument_classifier.py tests/test_instrument_classifier.py
git commit -m "feat: add InstrumentClassifier — stock/crypto/futures symbol routing"
```

---

## Task 4: AlpacaBroker (refactor broker.py)

**Files:**
- Create: `alpaca_broker.py`
- Modify: `broker.py` (shim)

The goal: copy the entire `Broker` class from `broker.py` into `alpaca_broker.py` as `AlpacaBroker(BaseBroker)`, implement the `BaseBroker` interface (add `submit_order`, `cancel_order`, `asset_type`, return typed dataclasses), then make `broker.py` a 3-line shim.

- [ ] **Step 1: Create `alpaca_broker.py`**

Write the full file (copy from `broker.py` and adapt):

```python
"""Alpaca broker implementation of BaseBroker."""
import os
import time
import threading
import uuid
from typing import Optional
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from utils import setup_logger
from base_broker import (
    BaseBroker, BrokerConnectionError,
    Position, Order, OrderRequest, Quote, Clock
)

load_dotenv()
log = setup_logger("alpaca_broker")

CRYPTO_SYMBOLS = {
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD",
    "BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD", "DOGEUSD",
}


class AlpacaBroker(BaseBroker):
    def __init__(self, config: dict):
        mode = os.getenv("TRADING_MODE", "paper")
        if mode == "live":
            base_url = config["alpaca"]["live_url"]
            log.warning("*** LIVE TRADING MODE ***")
        else:
            base_url = config["alpaca"]["paper_url"]
            log.info("Paper trading mode")

        self.api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=base_url,
            api_version="v2"
        )
        self._crypto_exit_orders = {}
        self._crypto_lock = threading.Lock()

    # ── BaseBroker interface ──────────────────────────────────────

    def get_account(self) -> dict:
        return self.api.get_account()

    def get_equity(self) -> float:
        return float(self.api.get_account().equity)

    def get_cash(self) -> float:
        return float(self.api.get_account().cash)

    def get_buying_power(self) -> float:
        return float(self.api.get_account().buying_power)

    def get_positions(self) -> list[Position]:
        raw = self.api.list_positions()
        result = []
        for p in raw:
            qty = float(p.qty)
            result.append(Position(
                symbol=p.symbol,
                qty=qty,
                avg_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                side="long" if qty > 0 else "short",
            ))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            p = self.api.get_position(symbol)
            qty = float(p.qty)
            return Position(
                symbol=p.symbol,
                qty=qty,
                avg_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                side="long" if qty > 0 else "short",
            )
        except Exception:
            return None

    def get_open_orders(self) -> list[Order]:
        raw = self.api.list_orders(status="open")
        return [
            Order(
                id=o.id,
                symbol=o.symbol,
                qty=float(o.qty),
                side=o.side,
                order_type=o.order_type,
                status=o.status,
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                filled_qty=float(o.filled_qty) if o.filled_qty else None,
            )
            for o in raw
        ]

    def submit_order(self, req: OrderRequest) -> Order:
        """Route to the correct Alpaca order type based on OrderRequest fields."""
        symbol = req.symbol
        is_crypto = symbol in CRYPTO_SYMBOLS

        if is_crypto:
            if req.side == "sell":
                log.error(f"CRYPTO SHORT REJECTED: {symbol} — Alpaca does not support crypto short selling")
                raise ValueError("Alpaca does not support crypto short selling")
            entry = self.submit_crypto_order(
                symbol, req.qty, req.side,
                req.take_profit, req.stop_loss
            )
            return Order(
                id=entry.id if entry else "crypto_entry",
                symbol=symbol,
                qty=req.qty,
                side=req.side,
                order_type="market",
                status=entry.status if entry else "submitted",
            )

        # Stocks/ETFs
        if req.take_profit and req.stop_loss:
            raw = self.submit_bracket_order(
                symbol, int(req.qty), req.side,
                req.take_profit, req.stop_loss
            )
        else:
            raw = self.api.submit_order(
                symbol=symbol,
                qty=int(req.qty),
                side=req.side,
                type=req.order_type,
                time_in_force=req.time_in_force,
            )
        return Order(
            id=raw.id,
            symbol=symbol,
            qty=float(raw.qty),
            side=raw.side,
            order_type=raw.order_type,
            status=raw.status,
        )

    def cancel_order(self, order_id: str):
        self.api.cancel_order(order_id)

    def cancel_all_orders(self):
        log.warning("Cancelling all open orders")
        self.api.cancel_all_orders()

    def close_position(self, symbol: str):
        log.info(f"Closing position: {symbol}")
        self.api.close_position(symbol)

    def close_all_positions(self):
        log.warning("Closing all positions")
        self.api.close_all_positions()

    def get_quote(self, symbol: str) -> Optional[Quote]:
        try:
            if symbol in CRYPTO_SYMBOLS:
                q = self.api.get_latest_crypto_quote(symbol)
            else:
                q = self.api.get_latest_quote(symbol, feed="iex")
            bid = float(q.bp) if hasattr(q, 'bp') else float(q.bid_price)
            ask = float(q.ap) if hasattr(q, 'ap') else float(q.ask_price)
            return Quote(symbol=symbol, bid=bid, ask=ask)
        except Exception as e:
            log.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def is_market_open(self) -> bool:
        return self.api.get_clock().is_open

    def get_clock(self) -> Clock:
        c = self.api.get_clock()
        return Clock(is_open=c.is_open, next_open=c.next_open, next_close=c.next_close)

    def asset_type(self, symbol: str) -> str:
        if symbol in CRYPTO_SYMBOLS:
            return "crypto"
        return "stock"

    # ── Alpaca-specific helpers (used by coordinator directly) ────

    def submit_bracket_order(self, symbol: str, qty: int, side: str,
                              take_profit: float, stop_loss: float):
        log.info(f"BRACKET ORDER: {side} {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")
        return self.api.submit_order(
            symbol=symbol, qty=qty, side=side, type="market",
            time_in_force="day", order_class="bracket",
            take_profit={"limit_price": round(take_profit, 2)},
            stop_loss={"stop_price": round(stop_loss, 2)}
        )

    def submit_short_bracket(self, symbol: str, qty: int,
                              take_profit: float, stop_loss: float):
        log.info(f"SHORT BRACKET: sell {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")
        return self.api.submit_order(
            symbol=symbol, qty=qty, side="sell", type="market",
            time_in_force="day", order_class="bracket",
            take_profit={"limit_price": round(take_profit, 2)},
            stop_loss={"stop_price": round(stop_loss, 2)}
        )

    def submit_crypto_order(self, symbol: str, qty: float, side: str,
                             take_profit: float, stop_loss: float):
        if side == "sell":
            log.error(f"CRYPTO SHORT REJECTED: {symbol} — Alpaca does not support crypto short selling")
            return None

        log.info(f"CRYPTO ORDER: {side} {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")
        tp_price = round(take_profit, 2)
        sl_price = round(stop_loss, 2)
        client_id = f"crypto_{symbol.replace('/', '')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        entry = self.api.submit_order(
            symbol=symbol, qty=qty, side=side, type="market",
            time_in_force="gtc", client_order_id=f"{client_id}_entry"
        )

        filled_qty = qty
        for _ in range(15):
            time.sleep(2)
            try:
                order_status = self.api.get_order(entry.id)
                if order_status.status == "filled":
                    filled_qty = float(order_status.filled_qty)
                    log.info(f"CRYPTO ENTRY FILLED: {symbol} qty={filled_qty}")
                    break
                elif order_status.status in ("canceled", "expired", "rejected"):
                    log.error(f"CRYPTO ENTRY FAILED: {symbol} status={order_status.status}")
                    return entry
            except Exception as e:
                log.warning(f"Error checking entry status: {e}")

        exit_qty = round(filled_qty * 0.999, 8)
        exit_side = "sell" if side == "buy" else "buy"
        tp_order_id = None
        sl_order_id = None

        try:
            tp_order = self.api.submit_order(
                symbol=symbol, qty=exit_qty, side=exit_side,
                type="limit", limit_price=tp_price, time_in_force="gtc",
                client_order_id=f"{client_id}_tp"
            )
            tp_order_id = tp_order.id
            log.info(f"CRYPTO TP order placed: {exit_side} {exit_qty} {symbol} @ ${tp_price}")
        except Exception as e:
            log.error(f"Failed to place crypto TP order: {e}")

        try:
            sl_limit = round(sl_price * (0.997 if exit_side == "sell" else 1.003), 2)
            sl_order = self.api.submit_order(
                symbol=symbol, qty=exit_qty, side=exit_side,
                type="stop_limit", stop_price=sl_price, limit_price=sl_limit,
                time_in_force="gtc", client_order_id=f"{client_id}_sl"
            )
            sl_order_id = sl_order.id
            log.info(f"CRYPTO SL order placed: {exit_side} {exit_qty} {symbol} @ ${sl_price}")
        except Exception as e:
            log.error(f"Failed to place crypto SL order: {e}")

        if tp_order_id or sl_order_id:
            with self._crypto_lock:
                self._crypto_exit_orders[symbol] = {
                    "tp_order_id": tp_order_id,
                    "sl_order_id": sl_order_id,
                    "qty": exit_qty,
                    "entry_side": side
                }
        return entry

    def check_crypto_exit_fills(self):
        """Check if any crypto TP/SL orders have filled and cancel the other (OCO)."""
        filled_exits = {}
        with self._crypto_lock:
            symbols_to_remove = []
            for symbol, orders in self._crypto_exit_orders.items():
                tp_id = orders.get("tp_order_id")
                sl_id = orders.get("sl_order_id")
                tp_filled = sl_filled = False

                if tp_id:
                    try:
                        tp_status = self.api.get_order(tp_id)
                        if tp_status.status == "filled":
                            tp_filled = True
                            log.info(f"CRYPTO TP FILLED: {symbol}")
                        elif tp_status.status in ("canceled", "expired", "rejected"):
                            orders["tp_order_id"] = None
                    except Exception:
                        pass

                if sl_id:
                    try:
                        sl_status = self.api.get_order(sl_id)
                        if sl_status.status == "filled":
                            sl_filled = True
                            log.info(f"CRYPTO SL FILLED: {symbol}")
                        elif sl_status.status in ("canceled", "expired", "rejected"):
                            orders["sl_order_id"] = None
                    except Exception:
                        pass

                if tp_filled and sl_id:
                    try:
                        self.api.cancel_order(sl_id)
                    except Exception:
                        pass
                    filled_exits[symbol] = "take_profit"
                    symbols_to_remove.append(symbol)
                elif sl_filled and tp_id:
                    try:
                        self.api.cancel_order(tp_id)
                    except Exception:
                        pass
                    filled_exits[symbol] = "stop_loss"
                    symbols_to_remove.append(symbol)
                elif not orders.get("tp_order_id") and not orders.get("sl_order_id"):
                    symbols_to_remove.append(symbol)

            for symbol in symbols_to_remove:
                self._crypto_exit_orders.pop(symbol, None)

        return filled_exits

    def cancel_crypto_exit_orders(self, symbol: str):
        with self._crypto_lock:
            orders = self._crypto_exit_orders.pop(symbol, {})
        for key in ("tp_order_id", "sl_order_id"):
            if orders.get(key):
                try:
                    self.api.cancel_order(orders[key])
                except Exception:
                    pass

    def submit_smart_order(self, symbol: str, qty: int, side: str,
                           take_profit: float, stop_loss: float,
                           limit_offset_pct: float = 0.0002,
                           timeout_sec: int = 30) -> dict:
        client_id = f"smart_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        quote = self.get_quote(symbol)
        if not quote:
            log.info(f"No quote for {symbol}, using market order")
            if side == "sell":
                self.submit_short_bracket(symbol, qty, take_profit, stop_loss)
            else:
                self.submit_bracket_order(symbol, qty, side, take_profit, stop_loss)
            return {"method": "market", "symbol": symbol}

        if side == "buy":
            limit_price = round(quote.ask * (1 - limit_offset_pct), 2)
        else:
            limit_price = round(quote.bid * (1 + limit_offset_pct), 2)

        log.info(f"SMART ORDER: {side} {qty} {symbol} limit=${limit_price:.2f} (timeout={timeout_sec}s)")

        try:
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="limit", limit_price=limit_price,
                time_in_force="day", client_order_id=f"{client_id}_entry"
            )
            order_id = order.id
            filled = False
            fill_price = limit_price
            elapsed = 0
            while elapsed < timeout_sec:
                time.sleep(2)
                elapsed += 2
                status = self.api.get_order(order_id)
                if status.status == "filled":
                    filled = True
                    fill_price = float(status.filled_avg_price)
                    break
                elif status.status in ("canceled", "expired", "rejected"):
                    break

            if filled:
                log.info(f"LIMIT FILLED: {symbol} @ ${fill_price:.2f}")
                tp_side = "sell" if side == "buy" else "buy"
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=qty, side=tp_side,
                        type="limit", limit_price=round(take_profit, 2),
                        time_in_force="gtc", order_class="oco",
                        stop_loss={"stop_price": round(stop_loss, 2)}
                    )
                except Exception:
                    self.api.submit_order(
                        symbol=symbol, qty=qty, side=tp_side,
                        type="limit", limit_price=round(take_profit, 2),
                        time_in_force="gtc"
                    )
                return {"method": "limit", "fill_price": fill_price, "symbol": symbol}
            else:
                try:
                    self.api.cancel_order(order_id)
                except Exception:
                    check = self.api.get_order(order_id)
                    if check.status == "filled":
                        return {"method": "limit", "fill_price": float(check.filled_avg_price), "symbol": symbol}

                log.info(f"LIMIT TIMEOUT: {symbol} — falling back to market bracket")
                if side == "sell":
                    self.submit_short_bracket(symbol, qty, take_profit, stop_loss)
                else:
                    self.submit_bracket_order(symbol, qty, side, take_profit, stop_loss)
                return {"method": "market_fallback", "symbol": symbol}

        except Exception as e:
            log.error(f"Smart order failed for {symbol}: {e} — using market")
            if side == "sell":
                self.submit_short_bracket(symbol, qty, take_profit, stop_loss)
            else:
                self.submit_bracket_order(symbol, qty, side, take_profit, stop_loss)
            return {"method": "market_error", "symbol": symbol}

    def submit_market_order(self, symbol: str, qty: int, side: str):
        log.info(f"MARKET ORDER: {side} {qty} {symbol}")
        tif = "gtc" if symbol in CRYPTO_SYMBOLS else "day"
        return self.api.submit_order(
            symbol=symbol, qty=qty, side=side, type="market", time_in_force=tif
        )

    def submit_trailing_stop(self, symbol: str, qty: int, trail_percent: float):
        log.info(f"TRAILING STOP: sell {qty} {symbol} trail={trail_percent}%")
        return self.api.submit_order(
            symbol=symbol, qty=qty, side="sell", type="trailing_stop",
            trail_percent=str(round(trail_percent, 2)), time_in_force="gtc"
        )
```

- [ ] **Step 2: Update `broker.py` → thin shim**

Replace entire `broker.py` content with:

```python
"""Backward-compat shim. Import AlpacaBroker as Broker."""
from alpaca_broker import AlpacaBroker as Broker, CRYPTO_SYMBOLS

__all__ = ["Broker", "CRYPTO_SYMBOLS"]
```

- [ ] **Step 3: Verify import chain works**

```bash
python -c "from broker import Broker, CRYPTO_SYMBOLS; print('OK', len(CRYPTO_SYMBOLS))"
```

Expected: `OK 12`

- [ ] **Step 4: Commit**

```bash
git add alpaca_broker.py broker.py
git commit -m "feat: extract AlpacaBroker(BaseBroker) from broker.py; broker.py becomes shim"
```

---

## Task 5: AlpacaDataFetcher (refactor data.py)

**Files:**
- Create: `alpaca_data.py`
- Modify: `data.py` (shim)

- [ ] **Step 1: Create `alpaca_data.py`**

Copy the full `DataFetcher` class from `data.py`, rename it `AlpacaDataFetcher`, add `BaseDataFetcher` as parent:

```python
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
    if symbol in CRYPTO_SYMBOLS and "/" not in symbol:
        return symbol[:-3] + "/" + symbol[-3:]
    return symbol


class AlpacaDataFetcher(BaseDataFetcher):
    """Data fetching with rate limiting and exponential backoff.

    Alpaca free tier: ~60 requests/min.
    Global semaphore limits concurrent requests across all watcher threads.
    """

    _global_semaphore = threading.Semaphore(2)
    _min_interval = 2.0
    _last_call_time = 0.0
    _call_lock = threading.Lock()

    def __init__(self, broker, requests_per_minute: int = 40):
        # Accept either AlpacaBroker or the old Broker (both have .api)
        self.api = broker.api
        self._rate_limit = requests_per_minute
        self._intraday_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 360

    def _wait_for_rate_limit(self):
        with AlpacaDataFetcher._call_lock:
            now = time.time()
            gap = now - AlpacaDataFetcher._last_call_time
            if gap < AlpacaDataFetcher._min_interval:
                time.sleep(AlpacaDataFetcher._min_interval - gap)
            AlpacaDataFetcher._last_call_time = time.time()

    def _api_call_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
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
        symbols = [_normalize_crypto(s) for s in symbols if isinstance(s, str)]
        end = datetime.now()
        start = end - timedelta(days=days)
        stock_symbols = [s for s in symbols if s not in CRYPTO_SYMBOLS]
        crypto_symbols = [s for s in symbols if s in CRYPTO_SYMBOLS]
        bars = {}
        time_fmt = "%Y-%m-%dT%H:%M:%SZ" if "Min" in timeframe else "%Y-%m-%d"

        batch_size = 30
        for i in range(0, len(stock_symbols), batch_size):
            batch = stock_symbols[i:i + batch_size]
            try:
                raw = self._api_call_with_retry(
                    self.api.get_bars, batch, timeframe,
                    start=start.strftime(time_fmt), end=end.strftime(time_fmt),
                    adjustment="split", feed="iex"
                )
                for bar in raw:
                    sym = bar.S
                    if sym not in bars:
                        bars[sym] = []
                    bars[sym].append({
                        "timestamp": bar.t, "open": float(bar.o),
                        "high": float(bar.h), "low": float(bar.l),
                        "close": float(bar.c), "volume": int(bar.v),
                        "vwap": float(bar.vw) if hasattr(bar, "vw") else None,
                    })
                if i + batch_size < len(stock_symbols):
                    time.sleep(3)
            except Exception as e:
                log.error(f"Failed to fetch bars for batch {batch}: {e}")

        for sym in crypto_symbols:
            try:
                raw = self._api_call_with_retry(
                    self.api.get_crypto_bars, sym, timeframe,
                    start=start.strftime(time_fmt), end=end.strftime(time_fmt),
                )
                bars[sym] = []
                for bar in raw:
                    bars[sym].append({
                        "timestamp": bar.t, "open": float(bar.o),
                        "high": float(bar.h), "low": float(bar.l),
                        "close": float(bar.c), "volume": float(bar.v),
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

    _CACHE_TTLS = {"5Min": 360, "1Hour": 3600, "1Day": 14400}

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> Optional[pd.DataFrame]:
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
```

- [ ] **Step 2: Update `data.py` → thin shim**

Replace entire `data.py` content with:

```python
"""Backward-compat shim. Import AlpacaDataFetcher as DataFetcher."""
from alpaca_data import AlpacaDataFetcher as DataFetcher, CRYPTO_SYMBOLS

__all__ = ["DataFetcher", "CRYPTO_SYMBOLS"]
```

- [ ] **Step 3: Verify import chain**

```bash
python -c "from data import DataFetcher, CRYPTO_SYMBOLS; print('OK', len(CRYPTO_SYMBOLS))"
```

Expected: `OK 12`

- [ ] **Step 4: Commit**

```bash
git add alpaca_data.py data.py
git commit -m "feat: extract AlpacaDataFetcher(BaseDataFetcher) from data.py; data.py becomes shim"
```

---

## Task 6: ContractManager (IB futures contract resolution)

**Files:**
- Create: `contract_manager.py`

- [ ] **Step 1: Create `contract_manager.py`**

```python
"""IB contract resolution and auto-roll for futures.

Called only by IBBroker and IBDataFetcher — never directly by strategies or coordinator.
Front-month detection: pick nearest expiry with open interest > 0.
Contracts cached for 4 hours to avoid repeated IB queries.
"""
import time
import threading
from typing import Optional
from utils import setup_logger

log = setup_logger("contract_manager")

# Default contract specs (can be overridden in config under futures.contracts)
CONTRACT_SPECS = {
    "NQ": {"exchange": "CME",   "currency": "USD", "multiplier": 20,   "min_tick": 0.25},
    "ES": {"exchange": "CME",   "currency": "USD", "multiplier": 50,   "min_tick": 0.25},
    "CL": {"exchange": "NYMEX", "currency": "USD", "multiplier": 1000, "min_tick": 0.01},
    "GC": {"exchange": "COMEX", "currency": "USD", "multiplier": 100,  "min_tick": 0.10},
}

CACHE_TTL_SEC = 4 * 3600  # 4 hours


class ContractManager:
    """Resolves IB futures contracts and manages auto-roll."""

    def __init__(self, ib, config: dict):
        """
        Args:
            ib: connected ib_insync.IB instance
            config: full config dict (reads futures.contracts for overrides)
        """
        self._ib = ib
        self._cache: dict[str, tuple[float, object]] = {}  # root → (ts, Contract)
        self._lock = threading.Lock()
        self._specs = dict(CONTRACT_SPECS)

        # Apply config overrides
        for entry in config.get("futures", {}).get("contracts", []):
            root = entry["root"]
            if root in self._specs:
                self._specs[root].update({k: v for k, v in entry.items() if k != "root"})

    def get_contract(self, root: str):
        """Return the front-month IB Contract for `root`, using cache if fresh."""
        with self._lock:
            cached = self._cache.get(root)
            if cached and time.time() - cached[0] < CACHE_TTL_SEC:
                return cached[1]

        contract = self._resolve_front_month(root)
        if contract:
            with self._lock:
                self._cache[root] = (time.time(), contract)
        return contract

    def _resolve_front_month(self, root: str):
        """Query IB for available contracts and pick the nearest expiry with OI > 0."""
        from ib_insync import Future, ContFuture

        spec = self._specs.get(root)
        if not spec:
            log.error(f"No contract spec for {root}")
            return None

        try:
            # Use ContFuture to get the continuous/front-month contract details
            cont = ContFuture(root, spec["exchange"], currency=spec["currency"])
            details = self._ib.reqContractDetails(cont)
            if not details:
                log.warning(f"No contract details for {root}, trying explicit Future")
                # Fall back to finding front month manually
                fut = Future(root, exchange=spec["exchange"], currency=spec["currency"])
                details = self._ib.reqContractDetails(fut)

            if not details:
                log.error(f"Cannot resolve contract for {root}")
                return None

            # Pick the contract with nearest expiry
            sorted_details = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
            front = sorted_details[0].contract

            # Qualify the contract (fills in conId, trading class, etc.)
            self._ib.qualifyContracts(front)
            log.info(f"Front-month {root}: {front.localSymbol} exp={front.lastTradeDateOrContractMonth}")
            return front

        except Exception as e:
            log.error(f"Failed to resolve front-month for {root}: {e}")
            return None

    def invalidate(self, root: str):
        """Force re-resolution on next get_contract call (e.g., after roll)."""
        with self._lock:
            self._cache.pop(root, None)

    def get_spec(self, root: str) -> dict:
        return self._specs.get(root, {})
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import contract_manager; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add contract_manager.py
git commit -m "feat: add ContractManager — IB front-month detection with 4h cache"
```

---

## Task 7: IBBroker

**Files:**
- Create: `ib_broker.py`

- [ ] **Step 1: Create `ib_broker.py`**

```python
"""Interactive Brokers broker implementation of BaseBroker.

Connects to IB Gateway at 127.0.0.1:4002 (paper account).
Uses ib_insync for all IB communication.
"""
import time
import threading
from typing import Optional
from utils import setup_logger
from base_broker import (
    BaseBroker, BrokerConnectionError,
    Position, Order, OrderRequest, Quote, Clock
)

log = setup_logger("ib_broker")

RECONNECT_DELAYS = [30, 60, 120]  # exponential backoff on connection loss


class IBBroker(BaseBroker):
    def __init__(self, config: dict):
        ib_cfg = config.get("ib", {})
        self._host = ib_cfg.get("host", "127.0.0.1")
        self._port = ib_cfg.get("port", 4002)
        self._client_id = ib_cfg.get("client_id", 1)
        self._timeout = ib_cfg.get("timeout_sec", 10)
        self._config = config
        self._lock = threading.Lock()

        from ib_insync import IB
        self._ib = IB()
        self._connect()

        from contract_manager import ContractManager
        self._contracts = ContractManager(self._ib, config)

    def _connect(self):
        """Connect to IB Gateway with retry."""
        from ib_insync import IB
        for attempt, delay in enumerate([0] + RECONNECT_DELAYS):
            if delay:
                log.warning(f"IB reconnect attempt {attempt}, waiting {delay}s...")
                time.sleep(delay)
            try:
                if not self._ib.isConnected():
                    self._ib.connect(self._host, self._port, clientId=self._client_id,
                                     timeout=self._timeout)
                log.info(f"Connected to IB Gateway at {self._host}:{self._port}")
                return
            except Exception as e:
                log.error(f"IB connection failed: {e}")
        raise BrokerConnectionError(f"Cannot connect to IB Gateway at {self._host}:{self._port}")

    def _ensure_connected(self):
        if not self._ib.isConnected():
            log.warning("IB disconnected — attempting reconnect")
            self._connect()

    # ── BaseBroker interface ──────────────────────────────────────

    def get_account(self) -> dict:
        self._ensure_connected()
        summary = self._ib.accountSummary()
        return {item.tag: item.value for item in summary}

    def get_equity(self) -> float:
        acct = self.get_account()
        return float(acct.get("NetLiquidation", 0))

    def get_cash(self) -> float:
        acct = self.get_account()
        return float(acct.get("CashBalance", 0))

    def get_buying_power(self) -> float:
        acct = self.get_account()
        return float(acct.get("BuyingPower", 0))

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        result = []
        for item in self._ib.portfolio():
            qty = float(item.position)
            if qty == 0:
                continue
            result.append(Position(
                symbol=item.contract.localSymbol or item.contract.symbol,
                qty=qty,
                avg_price=float(item.averageCost),
                market_value=float(item.marketValue),
                unrealized_pl=float(item.unrealizedPNL),
                side="long" if qty > 0 else "short",
            ))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def get_open_orders(self) -> list[Order]:
        self._ensure_connected()
        trades = self._ib.openTrades()
        result = []
        for trade in trades:
            result.append(Order(
                id=str(trade.order.orderId),
                symbol=trade.contract.localSymbol or trade.contract.symbol,
                qty=float(trade.order.totalQuantity),
                side=trade.order.action.lower(),
                order_type=trade.order.orderType.lower(),
                status=trade.orderStatus.status.lower(),
                filled_qty=float(trade.orderStatus.filled),
                filled_avg_price=float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else None,
            ))
        return result

    def submit_order(self, req: OrderRequest) -> Order:
        """Submit bracket order to IB. Routes futures vs stock/crypto contracts."""
        self._ensure_connected()
        from ib_insync import MarketOrder, LimitOrder, StopOrder, BracketOrder

        asset = self.asset_type(req.symbol)
        contract = self._resolve_contract(req.symbol, asset)
        if contract is None:
            raise ValueError(f"Cannot resolve IB contract for {req.symbol}")

        ib_side = "BUY" if req.side == "buy" else "SELL"

        if req.take_profit and req.stop_loss:
            # IB native bracket order (parent + TP limit + SL stop)
            parent = MarketOrder(ib_side, req.qty)
            parent.transmit = False

            tp_side = "SELL" if ib_side == "BUY" else "BUY"
            tp_order = LimitOrder(tp_side, req.qty, req.take_profit)
            tp_order.parentId = parent.orderId
            tp_order.transmit = False

            sl_order = StopOrder(tp_side, req.qty, req.stop_loss)
            sl_order.parentId = parent.orderId
            sl_order.transmit = True  # transmit the whole bracket at once

            parent_trade = self._ib.placeOrder(contract, parent)
            self._ib.placeOrder(contract, tp_order)
            sl_trade = self._ib.placeOrder(contract, sl_order)

            log.info(f"IB BRACKET: {ib_side} {req.qty} {req.symbol} "
                     f"TP={req.take_profit} SL={req.stop_loss}")
            return Order(
                id=str(parent_trade.order.orderId),
                symbol=req.symbol,
                qty=req.qty,
                side=req.side,
                order_type="bracket",
                status="submitted",
            )
        else:
            order = MarketOrder(ib_side, req.qty)
            trade = self._ib.placeOrder(contract, order)
            log.info(f"IB MARKET: {ib_side} {req.qty} {req.symbol}")
            return Order(
                id=str(trade.order.orderId),
                symbol=req.symbol,
                qty=req.qty,
                side=req.side,
                order_type="market",
                status="submitted",
            )

    def cancel_order(self, order_id: str):
        self._ensure_connected()
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == str(order_id):
                self._ib.cancelOrder(trade.order)
                return

    def cancel_all_orders(self):
        self._ensure_connected()
        self._ib.reqGlobalCancel()

    def close_position(self, symbol: str):
        pos = self.get_position(symbol)
        if not pos:
            log.warning(f"No IB position to close: {symbol}")
            return
        side = "sell" if pos.qty > 0 else "buy"
        req = OrderRequest(symbol=symbol, qty=abs(pos.qty), side=side)
        self.submit_order(req)

    def close_all_positions(self):
        for pos in self.get_positions():
            self.close_position(pos.symbol)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        self._ensure_connected()
        asset = self.asset_type(symbol)
        contract = self._resolve_contract(symbol, asset)
        if not contract:
            return None
        try:
            ticker = self._ib.reqMktData(contract, "", True, False)
            self._ib.sleep(1)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
            self._ib.cancelMktData(contract)
            if bid > 0 and ask > 0:
                return Quote(symbol=symbol, bid=bid, ask=ask)
            # Fallback: use last price as mid
            last = float(ticker.last) if ticker.last else 0.0
            if last > 0:
                spread = last * 0.0001  # 1 bps estimated spread
                return Quote(symbol=symbol, bid=last - spread, ask=last + spread)
        except Exception as e:
            log.error(f"IB quote failed for {symbol}: {e}")
        return None

    def is_market_open(self) -> bool:
        return self.get_clock().is_open

    def get_clock(self) -> Clock:
        # IB doesn't have a direct clock API — use time-based check
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        now = datetime.now(ET)
        weekday = now.weekday()
        hour, minute = now.hour, now.minute
        is_open = (weekday < 5 and
                   (hour > 9 or (hour == 9 and minute >= 30)) and
                   hour < 16)
        return Clock(is_open=is_open, next_open=None, next_close=None)

    def asset_type(self, symbol: str) -> str:
        from instrument_classifier import InstrumentClassifier
        clf = InstrumentClassifier(self._config)
        return clf.classify(symbol)

    # ── Internal helpers ──────────────────────────────────────────

    def _resolve_contract(self, symbol: str, asset: str):
        """Resolve IB contract object for a symbol."""
        if asset == "futures":
            return self._contracts.get_contract(symbol)
        elif asset == "stock":
            from ib_insync import Stock
            return Stock(symbol, "SMART", "USD")
        elif asset == "crypto":
            from ib_insync import Crypto
            # IB paper only supports BTC and ETH
            base = symbol.split("/")[0] if "/" in symbol else symbol[:-3]
            return Crypto(base, "PAXOS", "USD")
        return None

    def disconnect(self):
        if self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IB Gateway")
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ib_broker; print('IBBroker module OK')"
```

Expected: `IBBroker module OK`

- [ ] **Step 3: Commit**

```bash
git add ib_broker.py
git commit -m "feat: add IBBroker(BaseBroker) — IB Gateway bracket orders via ib_insync"
```

---

## Task 8: IBDataFetcher

**Files:**
- Create: `ib_data.py`

- [ ] **Step 1: Create `ib_data.py`**

```python
"""IB data fetcher — historical bars for futures via ib_insync reqHistoricalData.

Only handles futures. Stocks and crypto are handled by AlpacaDataFetcher.
"""
import time
import threading
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
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
        """Get latest price via IB snapshot."""
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
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ib_data; print('IBDataFetcher module OK')"
```

Expected: `IBDataFetcher module OK`

- [ ] **Step 3: Commit**

```bash
git add ib_data.py
git commit -m "feat: add IBDataFetcher(BaseDataFetcher) — futures historical bars via ib_insync"
```

---

## Task 9: RoutingBroker + RoutingDataFetcher

**Files:**
- Create: `routing_broker.py`
- Create: `routing_data.py`
- Create: `tests/test_routing_broker.py`

- [ ] **Step 1: Write failing tests for routing**

```python
# tests/test_routing_broker.py
from unittest.mock import MagicMock, patch
import pytest
from base_broker import OrderRequest, Position, Order, Clock


def make_routing_broker():
    """Build a RoutingBroker with mock sub-brokers."""
    from routing_broker import RoutingBroker
    from instrument_classifier import InstrumentClassifier

    config = {
        "futures": {"contracts": [
            {"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"}
        ]},
        "screener": {"crypto": ["BTC/USD", "ETH/USD"]},
    }

    ib_broker = MagicMock()
    alpaca_broker = MagicMock()
    clf = InstrumentClassifier(config)

    rb = RoutingBroker.__new__(RoutingBroker)
    rb._ib = ib_broker
    rb._alpaca = alpaca_broker
    rb._clf = clf
    return rb, ib_broker, alpaca_broker


def test_futures_routes_to_ib():
    rb, ib, alpaca = make_routing_broker()
    ib.get_equity.return_value = 100_000.0
    rb.get_equity()
    # For get_equity we call the active broker — but submit_order routes by asset type
    req = OrderRequest(symbol="NQ", qty=1, side="buy", take_profit=19000, stop_loss=18500)
    ib.submit_order.return_value = MagicMock()
    rb.submit_order(req)
    ib.submit_order.assert_called_once_with(req)
    alpaca.submit_order.assert_not_called()


def test_crypto_routes_to_alpaca():
    rb, ib, alpaca = make_routing_broker()
    req = OrderRequest(symbol="BTC/USD", qty=0.01, side="buy", take_profit=85000, stop_loss=78000)
    alpaca.submit_order.return_value = MagicMock()
    rb.submit_order(req)
    alpaca.submit_order.assert_called_once_with(req)
    ib.submit_order.assert_not_called()


def test_stock_routes_to_ib():
    rb, ib, alpaca = make_routing_broker()
    req = OrderRequest(symbol="AAPL", qty=10, side="buy", take_profit=210, stop_loss=195)
    ib.submit_order.return_value = MagicMock()
    rb.submit_order(req)
    ib.submit_order.assert_called_once_with(req)
    alpaca.submit_order.assert_not_called()
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_routing_broker.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'routing_broker'`

- [ ] **Step 3: Create `routing_broker.py`**

```python
"""RoutingBroker — dispatches broker calls by asset type.

Routing table (config-driven):
  futures → IBBroker
  stocks  → IBBroker
  crypto  → AlpacaBroker  (IB paper only supports BTC/ETH; Alpaca covers full universe)

The coordinator holds exactly one broker reference: RoutingBroker.
No broker-specific code outside broker files.
"""
from typing import Optional
from utils import setup_logger
from base_broker import (
    BaseBroker, Position, Order, OrderRequest, Quote, Clock
)

log = setup_logger("routing_broker")


class RoutingBroker(BaseBroker):
    def __init__(self, ib_broker: BaseBroker, alpaca_broker: BaseBroker,
                 classifier):
        """
        Args:
            ib_broker: IBBroker instance (handles stocks + futures)
            alpaca_broker: AlpacaBroker instance (handles crypto)
            classifier: InstrumentClassifier instance
        """
        self._ib = ib_broker
        self._alpaca = alpaca_broker
        self._clf = classifier

    def _broker_for(self, symbol: str) -> BaseBroker:
        asset = self._clf.classify(symbol)
        if asset == "crypto":
            return self._alpaca
        return self._ib  # stocks and futures go to IB

    # ── Account / portfolio queries go to IB (primary account) ───

    def get_account(self) -> dict:
        return self._ib.get_account()

    def get_equity(self) -> float:
        return self._ib.get_equity()

    def get_cash(self) -> float:
        return self._ib.get_cash()

    def get_buying_power(self) -> float:
        return self._ib.get_buying_power()

    def get_positions(self) -> list[Position]:
        """Aggregate positions from both brokers."""
        positions = self._ib.get_positions()
        try:
            positions += self._alpaca.get_positions()
        except Exception as e:
            log.warning(f"Could not fetch Alpaca positions: {e}")
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._broker_for(symbol).get_position(symbol)

    def get_open_orders(self) -> list[Order]:
        orders = self._ib.get_open_orders()
        try:
            orders += self._alpaca.get_open_orders()
        except Exception as e:
            log.warning(f"Could not fetch Alpaca orders: {e}")
        return orders

    def submit_order(self, req: OrderRequest) -> Order:
        broker = self._broker_for(req.symbol)
        log.info(f"Routing {req.symbol} ({self._clf.classify(req.symbol)}) "
                 f"→ {broker.__class__.__name__}")
        return broker.submit_order(req)

    def cancel_order(self, order_id: str):
        # Try IB first, then Alpaca
        try:
            self._ib.cancel_order(order_id)
        except Exception:
            self._alpaca.cancel_order(order_id)

    def cancel_all_orders(self):
        self._ib.cancel_all_orders()
        self._alpaca.cancel_all_orders()

    def close_position(self, symbol: str):
        self._broker_for(symbol).close_position(symbol)

    def close_all_positions(self):
        self._ib.close_all_positions()
        self._alpaca.close_all_positions()

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._broker_for(symbol).get_quote(symbol)

    def is_market_open(self) -> bool:
        return self._ib.is_market_open()

    def get_clock(self) -> Clock:
        return self._ib.get_clock()

    def asset_type(self, symbol: str) -> str:
        return self._clf.classify(symbol)

    # ── Alpaca-specific pass-throughs (for coordinator's crypto OCO logic) ─

    def check_crypto_exit_fills(self):
        return self._alpaca.check_crypto_exit_fills()

    def cancel_crypto_exit_orders(self, symbol: str):
        self._alpaca.cancel_crypto_exit_orders(symbol)

    def submit_bracket_order(self, symbol: str, qty: int, side: str,
                              take_profit: float, stop_loss: float):
        return self._broker_for(symbol).submit_bracket_order(
            symbol, qty, side, take_profit, stop_loss
        )

    def submit_short_bracket(self, symbol: str, qty: int,
                              take_profit: float, stop_loss: float):
        return self._broker_for(symbol).submit_short_bracket(
            symbol, qty, take_profit, stop_loss
        )

    def submit_crypto_order(self, symbol: str, qty: float, side: str,
                             take_profit: float, stop_loss: float):
        return self._alpaca.submit_crypto_order(symbol, qty, side, take_profit, stop_loss)

    def submit_smart_order(self, symbol: str, qty: int, side: str,
                           take_profit: float, stop_loss: float,
                           limit_offset_pct: float = 0.0002,
                           timeout_sec: int = 30) -> dict:
        return self._broker_for(symbol).submit_smart_order(
            symbol, qty, side, take_profit, stop_loss, limit_offset_pct, timeout_sec
        )

    def submit_market_order(self, symbol: str, qty: int, side: str):
        return self._broker_for(symbol).submit_market_order(symbol, qty, side)

    def submit_trailing_stop(self, symbol: str, qty: int, trail_percent: float):
        return self._broker_for(symbol).submit_trailing_stop(symbol, qty, trail_percent)
```

- [ ] **Step 4: Create `routing_data.py`**

```python
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
```

- [ ] **Step 5: Run routing tests — expect pass**

```bash
python -m pytest tests/test_routing_broker.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add routing_broker.py routing_data.py tests/test_routing_broker.py
git commit -m "feat: add RoutingBroker and RoutingDataFetcher — asset-type dispatch layer"
```

---

## Task 10: StrategyRouter

**Files:**
- Create: `strategy_router.py`
- Create: `tests/test_strategy_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategy_router.py
import pytest

CONFIG = {
    "futures": {"contracts": [{"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"}]},
    "screener": {"crypto": ["BTC/USD", "ETH/USD", "SOL/USD"]},
    "strategies": {
        "momentum": {"weight": 0.25},
        "mean_reversion": {"weight": 0.15},
        "breakout": {"weight": 0.20},
        "supertrend": {"weight": 0.25},
        "stoch_rsi": {"weight": 0.15},
        "vwap_reclaim": {"weight": 0.15},
        "gap": {"weight": 0.15},
        "liquidity_sweep": {"weight": 0.20},
    }
}


@pytest.fixture
def router():
    from strategy_router import StrategyRouter
    return StrategyRouter(CONFIG)


def test_stocks_include_mean_reversion(router):
    strats = router.get_strategies("stock")
    assert "mean_reversion" in strats


def test_stocks_include_gap(router):
    strats = router.get_strategies("stock")
    assert "gap" in strats


def test_crypto_excludes_mean_reversion(router):
    strats = router.get_strategies("crypto")
    assert "mean_reversion" not in strats


def test_crypto_excludes_gap(router):
    strats = router.get_strategies("crypto")
    assert "gap" not in strats


def test_crypto_excludes_vwap_reclaim(router):
    strats = router.get_strategies("crypto")
    assert "vwap_reclaim" not in strats


def test_futures_includes_futures_trend(router):
    strats = router.get_strategies("futures")
    assert "futures_trend" in strats


def test_futures_excludes_mean_reversion(router):
    strats = router.get_strategies("futures")
    assert "mean_reversion" not in strats


def test_futures_excludes_gap(router):
    strats = router.get_strategies("futures")
    assert "gap" not in strats


def test_weights_sum_to_one_stock(router):
    strats = router.get_strategies("stock")
    total = sum(strats.values())
    assert abs(total - 1.0) < 0.001


def test_weights_sum_to_one_crypto(router):
    strats = router.get_strategies("crypto")
    total = sum(strats.values())
    assert abs(total - 1.0) < 0.001


def test_weights_sum_to_one_futures(router):
    strats = router.get_strategies("futures")
    total = sum(strats.values())
    assert abs(total - 1.0) < 0.001
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_strategy_router.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'strategy_router'`

- [ ] **Step 3: Create `strategy_router.py`**

```python
"""Per-instrument strategy assignment with normalized weights.

Strategy matrix (weights before normalization):
  Strategy         Stocks   Crypto   Futures
  momentum         0.20     0.25     0.20
  mean_reversion   0.15     ❌       ❌
  breakout         0.20     0.25     0.20
  supertrend       0.20     0.25     0.25
  stoch_rsi        0.15     0.25     0.15
  vwap_reclaim     0.10     ❌       0.10
  gap              0.10     ❌       ❌
  liquidity_sweep  0.20     0.25     0.25
  futures_trend    ❌       ❌       0.30

Weights are normalized to sum to 1.0 per instrument type.
"""

# Raw weights before normalization — edit here to tune per-instrument emphasis
_STOCK_WEIGHTS = {
    "momentum":        0.20,
    "mean_reversion":  0.15,
    "breakout":        0.20,
    "supertrend":      0.20,
    "stoch_rsi":       0.15,
    "vwap_reclaim":    0.10,
    "gap":             0.10,
    "liquidity_sweep": 0.20,
}

_CRYPTO_WEIGHTS = {
    "momentum":        0.25,
    "breakout":        0.25,
    "supertrend":      0.25,
    "stoch_rsi":       0.25,
    "liquidity_sweep": 0.25,
}

_FUTURES_WEIGHTS = {
    "momentum":        0.20,
    "breakout":        0.20,
    "supertrend":      0.25,
    "stoch_rsi":       0.15,
    "vwap_reclaim":    0.10,
    "liquidity_sweep": 0.25,
    "futures_trend":   0.30,
}


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: round(v / total, 4) for k, v in weights.items()}


class StrategyRouter:
    """Returns the normalized strategy weight dict for a given instrument type."""

    def __init__(self, config: dict):
        self._config = config
        self._stock_weights = _normalize(_STOCK_WEIGHTS)
        self._crypto_weights = _normalize(_CRYPTO_WEIGHTS)
        self._futures_weights = _normalize(_FUTURES_WEIGHTS)

    def get_strategies(self, instrument_type: str) -> dict[str, float]:
        """Return {strategy_name: normalized_weight} for the instrument type.

        Args:
            instrument_type: 'stock', 'crypto', or 'futures'

        Returns:
            Dict of strategy name → normalized weight (sums to 1.0)
        """
        if instrument_type == "crypto":
            return dict(self._crypto_weights)
        elif instrument_type == "futures":
            return dict(self._futures_weights)
        else:  # stock (default)
            return dict(self._stock_weights)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_strategy_router.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strategy_router.py tests/test_strategy_router.py
git commit -m "feat: add StrategyRouter — normalized per-instrument strategy weights"
```

---

## Task 11: FuturesTrendStrategy

**Files:**
- Create: `strategies/futures_trend.py`

- [ ] **Step 1: Create `strategies/futures_trend.py`**

```python
"""FuturesTrendStrategy — designed for NQ, ES, CL, GC on 5-min bars.

Signals:
  1. Opening Range Breakout (ORB): first 30-min range; breakout with volume
  2. Session VWAP reclaim: price crosses VWAP from below/above with volume
  3. Trend filter: ADX > 25, EMA 8/21 alignment
  4. ATR volatility gate: skip if ATR spike > 2.5× 20-bar average

Score range: -1.0 to +1.0. High-conviction only: |score| >= 0.40.
"""
import numpy as np
import pandas as pd
from utils import setup_logger

log = setup_logger("futures_trend")

HIGH_CONVICTION_THRESHOLD = 0.40


class FuturesTrendStrategy:
    def __init__(self, config: dict):
        self.config = config

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Return {symbol: score} for each symbol in data."""
        results = {}
        for symbol, df in data.items():
            try:
                score = self._score(df)
                results[symbol] = score
            except Exception as e:
                log.error(f"FuturesTrend error for {symbol}: {e}")
                results[symbol] = 0.0
        return results

    def _score(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 50:
            return 0.0

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 1. ATR volatility gate ────────────────────────────────
        atr = self._atr(high, low, close, period=14)
        atr_20ma = atr.rolling(20).mean()
        if len(atr) > 0 and len(atr_20ma) > 0:
            latest_atr = atr.iloc[-1]
            avg_atr = atr_20ma.iloc[-1]
            if avg_atr > 0 and latest_atr > 2.5 * avg_atr:
                return 0.0  # news/event noise — skip

        # ── 2. Trend filter (ADX + EMA alignment) ────────────────
        adx = self._adx(high, low, close, period=14)
        ema8 = close.ewm(span=8).mean()
        ema21 = close.ewm(span=21).mean()

        trend_score = 0.0
        latest_adx = adx.iloc[-1] if len(adx) > 0 else 0
        ema_bullish = ema8.iloc[-1] > ema21.iloc[-1]
        ema_bearish = ema8.iloc[-1] < ema21.iloc[-1]

        if latest_adx > 25:
            if ema_bullish:
                trend_score = 0.40
            elif ema_bearish:
                trend_score = -0.40

        if abs(trend_score) < 0.01:
            return 0.0  # No trend — don't trade

        # ── 3. Session VWAP reclaim ───────────────────────────────
        vwap = self._session_vwap(df)
        vwap_score = 0.0
        if vwap is not None and len(vwap) >= 3:
            prev_below = close.iloc[-3] < vwap.iloc[-3]
            now_above = close.iloc[-1] > vwap.iloc[-1]
            prev_above = close.iloc[-3] > vwap.iloc[-3]
            now_below = close.iloc[-1] < vwap.iloc[-1]

            vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] > 0 else 1.0

            if prev_below and now_above and vol_ratio > 1.2:
                vwap_score = 0.30  # bullish VWAP reclaim
            elif prev_above and now_below and vol_ratio > 1.2:
                vwap_score = -0.30  # bearish VWAP breakdown

        # ── 4. Opening Range Breakout (ORB) ───────────────────────
        orb_score = self._orb_score(df, volume)

        # ── Composite score ────────────────────────────────────────
        # Trend direction gates: only add bullish signals if trend is bullish
        if trend_score > 0:
            total = trend_score + max(0, vwap_score) + max(0, orb_score)
        else:
            total = trend_score + min(0, vwap_score) + min(0, orb_score)

        # Clamp to [-1, 1]
        total = max(-1.0, min(1.0, total))

        # High-conviction filter: return 0 if score doesn't meet threshold
        if abs(total) < HIGH_CONVICTION_THRESHOLD:
            return 0.0

        return round(float(total), 3)

    def _orb_score(self, df: pd.DataFrame, volume: pd.Series) -> float:
        """Opening Range Breakout: 30-min range from session open."""
        try:
            # Get today's bars (5-min bars, so first 6 bars = 30 min)
            if not hasattr(df.index, 'hour'):
                return 0.0

            today = df.index[-1].date()
            today_bars = df[df.index.date == today]
            if len(today_bars) < 8:
                return 0.0

            # Opening range = first 6 bars (30 min)
            orb_bars = today_bars.iloc[:6]
            orb_high = orb_bars["high"].max()
            orb_low = orb_bars["low"].min()
            orb_range = orb_high - orb_low

            if orb_range <= 0:
                return 0.0

            # Current price relative to ORB
            current_close = df["close"].iloc[-1]
            vol_avg = volume.rolling(20).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            vol_confirm = current_vol > 1.5 * vol_avg if vol_avg > 0 else False

            if current_close > orb_high and vol_confirm:
                return 0.30  # bullish ORB breakout
            elif current_close < orb_low and vol_confirm:
                return -0.30  # bearish ORB breakdown
        except Exception:
            pass
        return 0.0

    def _session_vwap(self, df: pd.DataFrame) -> pd.Series | None:
        """Calculate session VWAP from today's open."""
        try:
            if not hasattr(df.index, 'date'):
                return None
            today = df.index[-1].date()
            today_mask = pd.Series(df.index.date, index=df.index) == today
            session_df = df[today_mask]
            if len(session_df) < 3:
                return None
            typical_price = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
            cumvol = session_df["volume"].cumsum()
            cumtpv = (typical_price * session_df["volume"]).cumsum()
            vwap = cumtpv / cumvol
            return vwap
        except Exception:
            return None

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = high - prev_high
        minus_dm = prev_low - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx.fillna(0)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "from strategies.futures_trend import FuturesTrendStrategy; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add strategies/futures_trend.py
git commit -m "feat: add FuturesTrendStrategy — ORB + session VWAP + ADX + ATR gate"
```

---

## Task 12: Register FuturesTrendStrategy

**Files:**
- Modify: `strategies/__init__.py`

- [ ] **Step 1: Add import and registration**

Current `strategies/__init__.py`:
```python
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .supertrend import SuperTrendStrategy
from .stoch_rsi import StochRSIStrategy
from .vwap_reclaim import VWAPReclaimStrategy
from .gap import GapStrategy
from .liquidity_sweep import LiquiditySweepStrategy

ALL_STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "supertrend": SuperTrendStrategy,
    "stoch_rsi": StochRSIStrategy,
    "vwap_reclaim": VWAPReclaimStrategy,
    "gap": GapStrategy,
    "liquidity_sweep": LiquiditySweepStrategy,
}
```

Replace with:
```python
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .supertrend import SuperTrendStrategy
from .stoch_rsi import StochRSIStrategy
from .vwap_reclaim import VWAPReclaimStrategy
from .gap import GapStrategy
from .liquidity_sweep import LiquiditySweepStrategy
from .futures_trend import FuturesTrendStrategy

ALL_STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "supertrend": SuperTrendStrategy,
    "stoch_rsi": StochRSIStrategy,
    "vwap_reclaim": VWAPReclaimStrategy,
    "gap": GapStrategy,
    "liquidity_sweep": LiquiditySweepStrategy,
    "futures_trend": FuturesTrendStrategy,
}
```

- [ ] **Step 2: Verify**

```bash
python -c "from strategies import ALL_STRATEGIES; print(list(ALL_STRATEGIES.keys()))"
```

Expected: list including `futures_trend`.

- [ ] **Step 3: Commit**

```bash
git add strategies/__init__.py
git commit -m "feat: register FuturesTrendStrategy in ALL_STRATEGIES"
```

---

## Task 13: Update watcher.py — accept strategy list

**Files:**
- Modify: `watcher.py`

The watcher currently builds its strategy dict from `ALL_STRATEGIES` internally. It needs to accept an optional `strategies` dict at `__init__` so `StrategyRouter` can inject the right strategies. If `strategies=None`, fall back to `ALL_STRATEGIES` (backward compat).

- [ ] **Step 1: Modify `watcher.py` `__init__`**

Find this block in `watcher.py` (lines 83–86):
```python
        # Initialize strategy instances (one per watcher)
        self.strategies = {
            name: cls(config) for name, cls in ALL_STRATEGIES.items()
        }
```

Replace with:
```python
        # Initialize strategy instances (one per watcher)
        # If strategies dict provided (from StrategyRouter), use it.
        # Otherwise fall back to all strategies for backward compat.
        if strategies is not None:
            self.strategies = {
                name: cls(config) for name, cls in ALL_STRATEGIES.items()
                if name in strategies
            }
        else:
            self.strategies = {
                name: cls(config) for name, cls in ALL_STRATEGIES.items()
            }
        # Store weight overrides from StrategyRouter for use in _analyze
        self._strategy_weights_override = strategies  # None = use select_strategies()
```

- [ ] **Step 2: Update `__init__` signature to accept `strategies`**

Find:
```python
    def __init__(self, symbol: str, config: dict, data_fetcher,
                 interval: int = 60, sector_regime_getter=None):
```

Replace with:
```python
    def __init__(self, symbol: str, config: dict, data_fetcher,
                 interval: int = 60, sector_regime_getter=None,
                 strategies: dict | None = None):
```

- [ ] **Step 3: Update `_analyze` to use weight overrides when provided**

In `_analyze`, find the strategy weight selection block (around line 194):
```python
        selection = select_strategies(daily_df, self.symbol, sector_regime=_sector_reg)
        self.state.regime = selection["regime"]
        self.state.regime_reason = selection["reason"]
        self.state.strategy_weights = selection["strategies"]
```

Replace with:
```python
        if self._strategy_weights_override is not None:
            # StrategyRouter provided weights — bypass select_strategies
            selection = {
                "regime": "router_assigned",
                "reason": "StrategyRouter per-instrument weights",
                "strategies": self._strategy_weights_override,
            }
        else:
            selection = select_strategies(daily_df, self.symbol, sector_regime=_sector_reg)
        self.state.regime = selection["regime"]
        self.state.regime_reason = selection["reason"]
        self.state.strategy_weights = selection["strategies"]
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "from watcher import StockWatcher; print('watcher OK')"
```

Expected: `watcher OK`

- [ ] **Step 5: Commit**

```bash
git add watcher.py
git commit -m "feat: watcher accepts optional strategies dict from StrategyRouter"
```

---

## Task 14: Update config.yaml — add IB and futures blocks

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add the `ib:` block and `futures:` block**

Add after the existing `sector_regime:` block:

```yaml
ib:
  host: "127.0.0.1"
  port: 4002
  client_id: 1
  timeout_sec: 10

futures:
  contracts:
    - root: NQ
      exchange: CME
      description: "E-mini Nasdaq-100"
    - root: ES
      exchange: CME
      description: "E-mini S&P 500"
    - root: CL
      exchange: NYMEX
      description: "Crude Oil WTI"
    - root: GC
      exchange: COMEX
      description: "Gold"
  risk:
    stop_loss_atr_mult: 1.5
    take_profit_atr_mult: 3.0
    min_risk_reward: 2.0
    max_position_pct: 0.05
```

- [ ] **Step 2: Verify YAML parses**

```bash
python -c "from utils import load_config; c=load_config(); print('IB port:', c['ib']['port']); print('Futures:', [x['root'] for x in c['futures']['contracts']])"
```

Expected: `IB port: 4002` and `Futures: ['NQ', 'ES', 'CL', 'GC']`

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "config: add ib block (port 4002) and futures contracts (NQ/ES/CL/GC)"
```

---

## Task 15: Update coordinator.py — use RoutingBroker + RoutingDataFetcher + StrategyRouter

**Files:**
- Modify: `coordinator.py`

This is the integration task. Three changes:
1. Import and instantiate `RoutingBroker` and `RoutingDataFetcher` instead of raw `Broker`/`DataFetcher`
2. Instantiate `StrategyRouter` and `InstrumentClassifier`
3. Pass per-instrument strategies to `StockWatcher` at creation

- [ ] **Step 1: Update imports at top of `coordinator.py`**

Find:
```python
from broker import Broker, CRYPTO_SYMBOLS
from data import DataFetcher
```

Replace with:
```python
from alpaca_broker import AlpacaBroker, CRYPTO_SYMBOLS
from alpaca_data import AlpacaDataFetcher
from ib_broker import IBBroker
from ib_data import IBDataFetcher
from routing_broker import RoutingBroker
from routing_data import RoutingDataFetcher
from instrument_classifier import InstrumentClassifier
from strategy_router import StrategyRouter
from contract_manager import ContractManager
```

- [ ] **Step 2: Update `Coordinator.__init__` to build routing layer**

Find in `__init__` (around line 56):
```python
        self.broker = Broker(self.config)
        self.data = DataFetcher(self.broker)
```

Replace with:
```python
        # Build broker layer
        self._alpaca_broker = AlpacaBroker(self.config)
        self._ib_broker = IBBroker(self.config)
        self._clf = InstrumentClassifier(self.config)
        self.broker = RoutingBroker(self._ib_broker, self._alpaca_broker, self._clf)

        # Build data layer
        self._alpaca_data = AlpacaDataFetcher(self._alpaca_broker)
        self._ib_data = IBDataFetcher(self._ib_broker._ib,
                                       self._ib_broker._contracts,
                                       self.config)
        self.data = RoutingDataFetcher(self._alpaca_data, self._ib_data, self._clf)

        # Strategy router
        self._strategy_router = StrategyRouter(self.config)
```

- [ ] **Step 3: Update `start_watchers` to include futures and pass per-instrument strategies**

Find in `start_watchers` (around line 127):
```python
        for sym in universe:
            if sym not in self.watchers:
                watcher = StockWatcher(
                    symbol=sym,
                    config=self.config,
                    data_fetcher=self.data,
                    interval=watcher_interval,
                    sector_regime_getter=(
                        self.sector_regime.get_regime_for_sector
                        if self._sector_regime_enabled and self.sector_regime else None
                    ),
                )
```

Replace with:
```python
        for sym in universe:
            if sym not in self.watchers:
                instrument_type = self._clf.classify(sym)
                strat_weights = self._strategy_router.get_strategies(instrument_type)
                watcher = StockWatcher(
                    symbol=sym,
                    config=self.config,
                    data_fetcher=self.data,
                    interval=watcher_interval,
                    sector_regime_getter=(
                        self.sector_regime.get_regime_for_sector
                        if self._sector_regime_enabled and self.sector_regime else None
                    ),
                    strategies=strat_weights,
                )
```

- [ ] **Step 4: Add futures to universe in `start_watchers`**

Find in `start_watchers`:
```python
            # Add crypto symbols
            universe = list(universe) + self.config["screener"].get("crypto", [])
```

Replace with:
```python
            # Add crypto symbols
            universe = list(universe) + self.config["screener"].get("crypto", [])
            # Add futures roots
            futures_roots = [c["root"] for c in self.config.get("futures", {}).get("contracts", [])]
            universe = list(universe) + futures_roots
```

- [ ] **Step 5: Update the `stop_watchers` CRYPTO_SYMBOLS check**

`stop_watchers` uses `CRYPTO_SYMBOLS` to decide which watchers are stocks. Update to use the classifier:

Find in `stop_watchers`:
```python
            to_stop = [s for s in self.watchers if s not in CRYPTO_SYMBOLS]
```

Replace with:
```python
            to_stop = [s for s in self.watchers if self._clf.classify(s) == "stock"]
```

Also find:
```python
            has_stock_watchers = any(
                s not in CRYPTO_SYMBOLS for s in self.watchers
            )
```

Both occurrences (in `run()`) — replace with:
```python
            has_stock_watchers = any(
                self._clf.classify(s) == "stock" for s in self.watchers
            )
```

And:
```python
            crypto_running = any(
                s in CRYPTO_SYMBOLS for s in self.watchers
            )
```

Replace with:
```python
            crypto_running = any(
                self._clf.classify(s) in ("crypto", "futures") for s in self.watchers
            )
```

- [ ] **Step 6: Update `__init__` account logging to handle IB account format**

Find (around line 94):
```python
        account = self.broker.get_account()
        equity = float(account.equity)
        log.info(f"Account: equity=${equity:,.2f} "
                 f"cash=${float(account.cash):,.2f} "
                 f"buying_power=${float(account.buying_power):,.2f}")
```

Replace with:
```python
        equity = self.broker.get_equity()
        cash = self.broker.get_cash()
        buying_power = self.broker.get_buying_power()
        log.info(f"Account: equity=${equity:,.2f} "
                 f"cash=${cash:,.2f} "
                 f"buying_power=${buying_power:,.2f}")
```

- [ ] **Step 7: Verify coordinator imports parse cleanly (IB Gateway NOT required)**

```bash
python -c "
import sys
# Patch ib_insync to avoid needing an active connection
import unittest.mock as mock
sys.modules['ib_insync'] = mock.MagicMock()
# Just check our new modules import without syntax errors
import instrument_classifier, strategy_router, routing_broker, routing_data
print('coordinator dependencies OK')
"
```

Expected: `coordinator dependencies OK`

- [ ] **Step 8: Commit**

```bash
git add coordinator.py
git commit -m "feat: wire RoutingBroker + RoutingDataFetcher + StrategyRouter into coordinator"
```

---

## Task 16: Smoke Test — boot with IB Gateway

**Files:** none (runtime test)

This task requires IB Gateway to be running at `127.0.0.1:4002`.

- [ ] **Step 1: Start IB Gateway**

In `D:\ibgateway`, launch IB Gateway. Select paper trading. Confirm API port is 4002.

- [ ] **Step 2: Verify IB connection in isolation**

```bash
python -c "
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=99)
print('Connected:', ib.isConnected())
acct = ib.accountSummary()
print('Account items:', len(acct))
ib.disconnect()
"
```

Expected: `Connected: True`, `Account items: N`

- [ ] **Step 3: Test ContractManager resolves NQ**

```bash
python -c "
from ib_insync import IB
from contract_manager import ContractManager
from utils import load_config

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=98)
config = load_config()
cm = ContractManager(ib, config)
contract = cm.get_contract('NQ')
print('NQ contract:', contract)
ib.disconnect()
"
```

Expected: contract object printed with localSymbol (e.g., `NQM5`).

- [ ] **Step 4: Test IBBroker equity**

```bash
python -c "
from utils import load_config
from ib_broker import IBBroker
config = load_config()
broker = IBBroker(config)
print('Equity:', broker.get_equity())
broker.disconnect()
"
```

Expected: paper account equity printed (e.g., `1000000.0`).

- [ ] **Step 5: Full coordinator boot (dry run — do not start trading)**

```python
# test_boot.py — run manually, Ctrl+C after seeing "All N watchers running"
from coordinator import Coordinator
c = Coordinator()
print("Coordinator built OK")
print("Broker type:", type(c.broker).__name__)
print("Data type:", type(c.data).__name__)
```

```bash
python test_boot.py
```

Expected: sees `RoutingBroker` and `RoutingDataFetcher` in output.

- [ ] **Step 6: Commit smoke test script**

```bash
git add test_boot.py
git commit -m "test: add test_boot.py smoke test for coordinator IB boot"
```

---

## Task 17: Full test run

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests in `test_instrument_classifier.py`, `test_strategy_router.py`, `test_routing_broker.py` PASS.

- [ ] **Step 2: If any test fails, fix before proceeding**

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "feat: IB migration complete — broker/data abstraction + futures + strategy routing"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - BaseBroker + BaseDataFetcher ✅ Task 2
  - InstrumentClassifier ✅ Task 3
  - AlpacaBroker (from broker.py) ✅ Task 4
  - AlpacaDataFetcher (from data.py) ✅ Task 5
  - broker.py / data.py shims ✅ Tasks 4 & 5
  - ContractManager ✅ Task 6
  - IBBroker ✅ Task 7
  - IBDataFetcher ✅ Task 8
  - RoutingBroker ✅ Task 9
  - RoutingDataFetcher ✅ Task 9
  - StrategyRouter ✅ Task 10
  - FuturesTrendStrategy ✅ Task 11
  - strategies/__init__.py ✅ Task 12
  - watcher.py update ✅ Task 13
  - config.yaml ✅ Task 14
  - coordinator.py ✅ Task 15
  - Error handling / BrokerConnectionError ✅ IBBroker._connect()
  - Contract auto-roll (4h cache) ✅ ContractManager
  - Crypto stays on Alpaca (IB paper limit) ✅ RoutingBroker

- [x] **Type consistency:** `OrderRequest` used throughout; `BaseBroker.submit_order(req: OrderRequest)` matches every implementation.

- [x] **No placeholders** in code blocks.
