# IB Integration Test Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live integration test suite that validates the full IB pipeline — contract resolution, strategy signals, order submission, and resilience — against IB paper trading.

**Architecture:** Single test file `tests/test_ib_live.py` with a shared pytest fixture that manages one IB connection across all tests. Four test classes map to the four spec phases. A `conftest.py` registers the `live` marker. Tests run with `pytest -m live -v`.

**Tech Stack:** pytest, ib_insync, nest_asyncio, existing project modules (IBBroker, IBDataFetcher, StrategyRouter, all 9 strategies)

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/test_ib_live.py` | All live integration tests (4 phases, ~24 test methods) |
| `tests/conftest.py` | Register `live` marker, shared IB session fixture |
| `pytest.ini` | Add marker registration (if not already present) |

---

### Task 1: Pytest Configuration and Shared IB Fixture

**Files:**
- Create: `tests/conftest.py`
- Create or Modify: `pytest.ini`

- [ ] **Step 1: Create `pytest.ini` with live marker**

Create `pytest.ini` in the project root (if it doesn't exist, create it; if it does, add the marker):

```ini
[pytest]
markers =
    live: requires live IB Gateway/TWS connection on port 4002
```

- [ ] **Step 2: Create `tests/conftest.py` with shared IB session fixture**

```python
"""Shared fixtures for live IB integration tests."""

import sys
import os
import time
import pytest
import nest_asyncio

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

nest_asyncio.apply()

import yaml
from ib_broker import IBBroker
from ib_data import IBDataFetcher
from instrument_classifier import InstrumentClassifier

SYMBOLS = {
    "stocks": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "TSLA", "AMD", "PLTR", "SNOW", "CRWD",
        "PEP", "LOW", "SO", "PM", "COF",
    ],
    "etfs": ["GLD", "SPY", "QQQ"],
    "sector_etfs": ["XLF", "XLE", "XLK"],
    "futures": ["NQ", "ES"],
    "crypto": ["BTC/USD", "ETH/USD"],
}
ALL_SYMBOLS = [s for group in SYMBOLS.values() for s in group]


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def ib_session():
    """Connect to IB paper, yield (broker, data, config), clean up after."""
    config = _load_config()
    broker = IBBroker(config)
    assert broker._ib.isConnected(), "Failed to connect to IB Gateway"

    data = IBDataFetcher(broker._ib, broker._contracts, config)

    yield broker, data, config

    # Teardown: cancel open orders, close test positions
    try:
        broker.cancel_all_orders()
        time.sleep(1)
        broker.close_all_positions()
    except Exception:
        pass
    broker._ib.disconnect()
```

- [ ] **Step 3: Verify fixture loads by creating a minimal test**

Create `tests/test_ib_live.py` with just:

```python
"""Live IB integration tests — requires TWS/Gateway on port 4002."""

import pytest

pytestmark = pytest.mark.live


class TestPhase1Connectivity:
    def test_connection_health(self, ib_session):
        broker, data, config = ib_session
        assert broker._ib.isConnected()
        equity = broker.get_equity()
        assert equity > 0, f"Expected positive equity, got {equity}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_ib_live.py -m live -v -k "test_connection_health"`
Expected: PASS — connects to IB, reads equity.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/conftest.py tests/test_ib_live.py
git commit -m "test: add live IB test harness scaffold with connection health check"
```

---

### Task 2: Phase 1 — Contract Resolution Tests

**Files:**
- Modify: `tests/test_ib_live.py`

- [ ] **Step 1: Add contract resolution test for all 25 symbols**

Append to `TestPhase1Connectivity` class in `tests/test_ib_live.py`:

```python
    def test_resolve_all_contracts(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        broker, data, config = ib_session

        resolved = {}
        failed = []
        for sym in ALL_SYMBOLS:
            asset = broker.asset_type(sym)
            contract = broker._resolve_contract(sym, asset)
            if contract:
                resolved[sym] = {
                    "conId": getattr(contract, "conId", None),
                    "exchange": getattr(contract, "exchange", None),
                    "primaryExch": getattr(contract, "primaryExchange",
                                           getattr(contract, "primaryExch", None)),
                }
            else:
                failed.append(sym)

        print(f"\nResolved {len(resolved)}/{len(ALL_SYMBOLS)} contracts:")
        for sym, info in resolved.items():
            print(f"  {sym}: conId={info['conId']} exchange={info['exchange']} primary={info['primaryExch']}")
        if failed:
            print(f"FAILED: {failed}")

        assert len(failed) == 0, f"Failed to resolve: {failed}"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_ib_live.py -m live -v -k "test_resolve_all_contracts" -s`
Expected: PASS — all 25 symbols resolve. `-s` shows the resolution log.

- [ ] **Step 3: Add quote test for all 25 symbols**

Append to `TestPhase1Connectivity`:

```python
    def test_get_quotes(self, ib_session):
        import time
        from tests.conftest import ALL_SYMBOLS
        broker, data, config = ib_session

        got_price = {}
        no_price = []
        for sym in ALL_SYMBOLS:
            quote = broker.get_quote(sym)
            if quote and quote.mid > 0:
                got_price[sym] = quote.mid
            else:
                # Try data fetcher fallback
                price = data.get_latest_price(sym)
                if price and price > 0:
                    got_price[sym] = price
                else:
                    no_price.append(sym)

        print(f"\nGot prices for {len(got_price)}/{len(ALL_SYMBOLS)}:")
        for sym, price in got_price.items():
            print(f"  {sym}: ${price:,.2f}")
        if no_price:
            print(f"No price: {no_price}")

        assert len(got_price) >= 20, (
            f"Expected at least 20 symbols with prices, got {len(got_price)}. "
            f"No price: {no_price}"
        )
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_ib_live.py -m live -v -k "test_get_quotes" -s`
Expected: PASS — at least 20 of 25 return a price.

- [ ] **Step 5: Add bad symbol and pacing tests**

Append to `TestPhase1Connectivity`:

```python
    def test_bad_symbol(self, ib_session):
        broker, data, config = ib_session

        # Clear any previous cache entry
        broker._bad_contracts.pop("FAKESYM123", None)
        data._bad_contracts.pop("FAKESYM123", None)

        contract = broker._resolve_contract("FAKESYM123", "stock")
        assert contract is None, "Expected None for fake symbol"
        assert "FAKESYM123" in broker._bad_contracts, "Should be in bad_contracts cache"
        assert broker._bad_contracts["FAKESYM123"] > time.time(), "TTL should be in the future"

        # Clear and verify it retries
        broker._bad_contracts.pop("FAKESYM123")
        assert "FAKESYM123" not in broker._bad_contracts

    def test_pacing_rapid_resolution(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        broker, data, config = ib_session

        # Clear bad contracts so all are retried fresh
        broker._bad_contracts.clear()

        start = time.time()
        results = {}
        for sym in ALL_SYMBOLS:
            asset = broker.asset_type(sym)
            contract = broker._resolve_contract(sym, asset)
            results[sym] = contract is not None
        elapsed = time.time() - start

        resolved_count = sum(1 for v in results.values() if v)
        print(f"\nRapid resolution: {resolved_count}/{len(ALL_SYMBOLS)} in {elapsed:.1f}s")
        assert resolved_count == len(ALL_SYMBOLS), (
            f"Some symbols failed rapid resolution: "
            f"{[s for s, ok in results.items() if not ok]}"
        )
```

- [ ] **Step 6: Run all Phase 1 tests**

Run: `pytest tests/test_ib_live.py -m live -v -k "Phase1" -s`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_ib_live.py
git commit -m "test: add Phase 1 — contract resolution, quotes, bad symbol, pacing"
```

---

### Task 3: Phase 2 — Strategy Signal Validation Tests

**Files:**
- Modify: `tests/test_ib_live.py`

- [ ] **Step 1: Add bar fetching test**

Add new class after `TestPhase1Connectivity` in `tests/test_ib_live.py`:

```python
class TestPhase2Strategies:
    def test_fetch_bars_daily(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        broker, data, config = ib_session

        fetched = {}
        failed = []
        bars_dict = data.get_bars(ALL_SYMBOLS, timeframe="1 day", days=30)

        for sym in ALL_SYMBOLS:
            df = bars_dict.get(sym)
            if df is not None and not df.empty:
                fetched[sym] = len(df)
                # Validate columns
                for col in ["open", "high", "low", "close", "volume"]:
                    assert col in df.columns, f"{sym}: missing column {col}"
                # No NaN in OHLC
                for col in ["open", "high", "low", "close"]:
                    assert df[col].notna().all(), f"{sym}: NaN in {col}"
            else:
                failed.append(sym)

        print(f"\nDaily bars fetched for {len(fetched)}/{len(ALL_SYMBOLS)}:")
        for sym, count in fetched.items():
            print(f"  {sym}: {count} rows")
        if failed:
            print(f"FAILED: {failed}")

        # Allow some failures (crypto/futures may need subscriptions)
        assert len(fetched) >= 20, f"Expected 20+ symbols with bars, got {len(fetched)}. Failed: {failed}"

    def test_fetch_bars_intraday(self, ib_session):
        from tests.conftest import SYMBOLS
        broker, data, config = ib_session

        # Test intraday on a subset (5 stocks + 1 ETF)
        test_symbols = SYMBOLS["stocks"][:5] + ["SPY"]
        fetched = {}
        failed = []

        for sym in test_symbols:
            df = data.get_intraday_bars(sym, timeframe="5 mins", days=2)
            if df is not None and not df.empty:
                fetched[sym] = len(df)
                for col in ["open", "high", "low", "close", "volume"]:
                    assert col in df.columns, f"{sym}: missing column {col}"
            else:
                failed.append(sym)

        print(f"\nIntraday bars fetched for {len(fetched)}/{len(test_symbols)}:")
        for sym, count in fetched.items():
            print(f"  {sym}: {count} rows")

        assert len(fetched) >= 4, f"Expected 4+ symbols with intraday bars. Failed: {failed}"
```

- [ ] **Step 2: Run bar fetching tests**

Run: `pytest tests/test_ib_live.py -m live -v -k "test_fetch_bars" -s`
Expected: PASS — bars fetched for most symbols.

- [ ] **Step 3: Add strategy signal tests**

Append to `TestPhase2Strategies`:

```python
    def test_strategy_signals(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        from strategies import ALL_STRATEGIES
        broker, data, config = ib_session

        # Fetch bars for all symbols
        bars_dict = data.get_bars(ALL_SYMBOLS, timeframe="1 day", days=60)
        assert len(bars_dict) > 0, "No bars fetched — cannot test strategies"

        errors = []
        signal_counts = {name: 0 for name in ALL_STRATEGIES}

        for strategy_name, StrategyCls in ALL_STRATEGIES.items():
            try:
                strategy = StrategyCls(config)
            except Exception as e:
                errors.append(f"{strategy_name}: failed to instantiate: {e}")
                continue

            try:
                signals = strategy.generate_signals(bars_dict)
            except Exception as e:
                errors.append(f"{strategy_name}: generate_signals crashed: {e}")
                continue

            assert isinstance(signals, dict), (
                f"{strategy_name}: expected dict, got {type(signals)}"
            )

            for sym, score in signals.items():
                assert isinstance(score, (int, float)), (
                    f"{strategy_name}/{sym}: score is {type(score)}, expected float"
                )
                assert -1.0 <= score <= 1.0, (
                    f"{strategy_name}/{sym}: score {score} out of [-1, 1] range"
                )
                signal_counts[strategy_name] += 1

        print(f"\nStrategy signal counts:")
        for name, count in signal_counts.items():
            print(f"  {name}: {count} signals")
        if errors:
            print(f"\nErrors:\n" + "\n".join(f"  {e}" for e in errors))

        assert len(errors) == 0, f"Strategy errors:\n" + "\n".join(errors)

    def test_strategy_router(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        from strategy_router import StrategyRouter
        broker, data, config = ib_session

        router = StrategyRouter(config)

        for sym in ALL_SYMBOLS:
            asset = broker.asset_type(sym)
            weights = router.get_strategies(asset)
            assert isinstance(weights, dict), f"{sym}: weights is not a dict"
            assert len(weights) > 0, f"{sym}: no strategies assigned"
            total = sum(weights.values())
            # Normalized weights should sum to ~1.0
            assert 0.99 <= total <= 1.01, (
                f"{sym} ({asset}): weights sum to {total}, expected ~1.0"
            )
            for name, w in weights.items():
                assert 0.0 < w <= 1.0, f"{sym}: strategy {name} weight {w} out of range"

        print(f"\nRouter validated for {len(ALL_SYMBOLS)} symbols across all asset types")
```

- [ ] **Step 4: Run strategy tests**

Run: `pytest tests/test_ib_live.py -m live -v -k "Phase2" -s`
Expected: PASS — all 9 strategies run on fetched bars without crashes, all scores in [-1, 1].

- [ ] **Step 5: Commit**

```bash
git add tests/test_ib_live.py
git commit -m "test: add Phase 2 — bar fetching, strategy signals, router validation"
```

---

### Task 4: Phase 3 — Order Pipeline Tests

**Files:**
- Modify: `tests/test_ib_live.py`

- [ ] **Step 1: Add market order tests (stock, ETF, futures, crypto)**

Add new class in `tests/test_ib_live.py`:

```python
class TestPhase3Orders:
    """Order tests submit real orders to IB paper — minimal size to keep costs near zero."""

    def test_market_order_stock(self, ib_session):
        broker, data, config = ib_session
        from base_broker import OrderRequest

        req = OrderRequest(symbol="AMD", qty=1, side="buy")
        order = broker.submit_order(req)

        assert order is not None, "submit_order returned None"
        assert order.id is not None, "order has no id"
        assert order.status in ("submitted", "filled", "new"), (
            f"Unexpected order status: {order.status}"
        )
        print(f"\nStock order: AMD qty=1 id={order.id} status={order.status}")

        # Wait for fill
        time.sleep(2)
        positions = broker.get_positions()
        amd_pos = [p for p in positions if p.symbol == "AMD"]
        print(f"AMD positions after buy: {amd_pos}")

    def test_market_order_etf(self, ib_session):
        broker, data, config = ib_session
        from base_broker import OrderRequest

        req = OrderRequest(symbol="SPY", qty=1, side="buy")
        order = broker.submit_order(req)

        assert order is not None
        assert order.id is not None
        assert order.status in ("submitted", "filled", "new")
        print(f"\nETF order: SPY qty=1 id={order.id} status={order.status}")
        time.sleep(2)

    def test_market_order_futures(self, ib_session):
        broker, data, config = ib_session
        from base_broker import OrderRequest

        try:
            req = OrderRequest(symbol="NQ", qty=1, side="buy")
            order = broker.submit_order(req)
            assert order is not None
            print(f"\nFutures order: NQ qty=1 id={order.id} status={order.status}")
            time.sleep(2)
        except Exception as e:
            err = str(e)
            if "margin" in err.lower() or "insufficient" in err.lower():
                pytest.skip(f"Futures margin insufficient: {e}")
            raise

    def test_market_order_crypto(self, ib_session):
        broker, data, config = ib_session
        from base_broker import OrderRequest

        try:
            req = OrderRequest(symbol="BTC/USD", qty=0.00001, side="buy", notional=1.0)
            order = broker.submit_order(req)
            assert order is not None
            print(f"\nCrypto order: BTC/USD cashQty=$1 id={order.id} status={order.status}")
            time.sleep(2)
        except Exception as e:
            err = str(e)
            if "regulatory" in err.lower() or "not allowed" in err.lower():
                pytest.skip(f"Crypto not enabled on this account: {e}")
            raise
```

- [ ] **Step 2: Run order tests**

Run: `pytest tests/test_ib_live.py -m live -v -k "test_market_order" -s`
Expected: Stock and ETF orders fill. Futures may skip on margin. Crypto may skip on regulatory restriction.

- [ ] **Step 3: Add bracket, cancel, and close tests**

Append to `TestPhase3Orders`:

```python
    def test_bracket_order(self, ib_session):
        broker, data, config = ib_session

        # Get AAPL price for realistic TP/SL
        quote = broker.get_quote("AAPL")
        if not quote or quote.mid <= 0:
            pytest.skip("Cannot get AAPL quote for bracket test")

        price = quote.mid
        tp = round(price * 1.05, 2)  # +5%
        sl = round(price * 0.97, 2)  # -3%

        order = broker.submit_bracket_order("AAPL", 1, "buy", tp, sl)

        assert order is not None, "Bracket order returned None"
        assert order.id is not None, "Bracket order has no id"
        print(f"\nBracket order: AAPL id={order.id} TP={tp} SL={sl}")

        # Verify parent and children exist in open trades
        time.sleep(2)
        open_trades = broker._ib.openTrades()
        parent_id = int(order.id)
        children = [t for t in open_trades if t.order.parentId == parent_id]
        print(f"Open trades with parentId={parent_id}: {len(children)}")
        # Parent should have filled or be submitted; children should exist
        assert len(children) >= 1, (
            f"Expected child orders for parentId={parent_id}, found {len(children)}"
        )

    def test_cancel_order(self, ib_session):
        broker, data, config = ib_session
        from base_broker import OrderRequest

        # Submit a limit buy at $1 — will never fill
        req = OrderRequest(
            symbol="MSFT", qty=1, side="buy",
            order_type="limit", limit_price=1.0
        )
        # Use submit_order which handles limit via LimitOrder
        from ib_insync import LimitOrder
        asset = broker.asset_type("MSFT")
        contract = broker._resolve_contract("MSFT", asset)
        assert contract is not None, "Cannot resolve MSFT"

        limit_order = LimitOrder("BUY", 1, 1.0)
        trade = broker._ib.placeOrder(contract, limit_order)
        order_id = str(trade.order.orderId)
        print(f"\nLimit order placed: MSFT @ $1.00 id={order_id}")

        time.sleep(1)
        broker.cancel_order(order_id)
        time.sleep(1)

        # Verify cancelled
        status = trade.orderStatus.status
        print(f"Order {order_id} status after cancel: {status}")
        assert status in ("Cancelled", "Inactive", "ApiCancelled"), (
            f"Expected cancelled status, got {status}"
        )

    def test_close_positions(self, ib_session):
        broker, data, config = ib_session

        time.sleep(2)
        positions = broker.get_positions()
        if not positions:
            print("\nNo positions to close")
            return

        for pos in positions:
            print(f"Closing: {pos.symbol} qty={pos.qty}")
            broker.close_position(pos.symbol)

        time.sleep(3)
        remaining = broker.get_positions()
        remaining_syms = [p.symbol for p in remaining if abs(p.qty) > 0]
        print(f"Remaining positions after close: {remaining_syms}")
        assert len(remaining_syms) == 0, f"Failed to close: {remaining_syms}"
```

- [ ] **Step 4: Run all Phase 3 tests**

Run: `pytest tests/test_ib_live.py -m live -v -k "Phase3" -s`
Expected: Stock/ETF orders fill, bracket has children, cancel works, positions closed. Futures/crypto may skip.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ib_live.py
git commit -m "test: add Phase 3 — market orders, bracket, cancel, close positions"
```

---

### Task 5: Phase 4 — Resilience & Edge Case Tests

**Files:**
- Modify: `tests/test_ib_live.py`

- [ ] **Step 1: Add resilience tests**

Add new class in `tests/test_ib_live.py`:

```python
class TestPhase4Resilience:

    def test_bad_contract_cache_ttl(self, ib_session):
        broker, data, config = ib_session

        # Ensure clean state
        broker._bad_contracts.pop("FAKESYM123", None)

        # Trigger bad contract
        contract = broker._resolve_contract("FAKESYM123", "stock")
        assert contract is None
        assert "FAKESYM123" in broker._bad_contracts

        ttl = broker._bad_contracts["FAKESYM123"]
        expected_min = time.time() + 1700  # ~28 min (allow some slack)
        expected_max = time.time() + 1900  # ~32 min
        assert expected_min < ttl < expected_max, (
            f"TTL {ttl} not in expected range [{expected_min}, {expected_max}]"
        )

        # Clear and verify retry works
        broker._bad_contracts.pop("FAKESYM123")
        contract2 = broker._resolve_contract("FAKESYM123", "stock")
        assert contract2 is None  # Still fails (not a real symbol)
        assert "FAKESYM123" in broker._bad_contracts  # Re-cached
        print("\nBad contract cache TTL: OK")

        # Cleanup
        broker._bad_contracts.pop("FAKESYM123", None)

    def test_duplicate_orders(self, ib_session):
        broker, data, config = ib_session
        from base_broker import OrderRequest

        req1 = OrderRequest(symbol="AAPL", qty=1, side="buy")
        req2 = OrderRequest(symbol="AAPL", qty=1, side="buy")

        order1 = broker.submit_order(req1)
        order2 = broker.submit_order(req2)

        assert order1.id != order2.id, "Duplicate orders should have unique IDs"
        print(f"\nDuplicate orders: id1={order1.id} id2={order2.id}")

        # Cleanup: close AAPL position (2 shares)
        time.sleep(2)
        broker.close_position("AAPL")
        time.sleep(1)

    def test_event_loop_safety(self, ib_session):
        import threading
        import concurrent.futures
        broker, data, config = ib_session

        errors = []

        def _run_from_thread():
            """Simulate Discord bot context — call broker methods from a worker thread."""
            try:
                quote = broker.get_quote("SPY")
                if quote is None:
                    errors.append("get_quote returned None")
                elif quote.mid <= 0:
                    errors.append(f"get_quote returned mid={quote.mid}")
            except Exception as e:
                errors.append(f"get_quote error: {e}")

            try:
                price = data.get_latest_price("SPY")
                if price is None or price <= 0:
                    errors.append(f"get_latest_price returned {price}")
            except Exception as e:
                errors.append(f"get_latest_price error: {e}")

            try:
                asset = broker.asset_type("SPY")
                contract = broker._resolve_contract("SPY", asset)
                if contract is None:
                    errors.append("_resolve_contract returned None from thread")
            except Exception as e:
                errors.append(f"_resolve_contract error: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_from_thread)
            future.result(timeout=15)

        if errors:
            print(f"\nEvent loop errors:\n" + "\n".join(f"  {e}" for e in errors))
        assert len(errors) == 0, f"Event loop safety failures:\n" + "\n".join(errors)
        print("\nEvent loop safety: all calls succeeded from worker thread")

    def test_reconnection(self, ib_session):
        broker, data, config = ib_session

        # Force disconnect
        broker._ib.disconnect()
        assert not broker._ib.isConnected(), "Should be disconnected"
        print("\nDisconnected from IB")

        # Call a method that triggers _ensure_connected
        equity = broker.get_equity()
        assert broker._ib.isConnected(), "Should have reconnected"
        assert equity > 0, f"Equity after reconnect: {equity}"
        print(f"Reconnected successfully, equity=${equity:,.2f}")

    def test_pacing_limits_bar_fetch(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        broker, data, config = ib_session

        # Clear data caches to force fresh fetches
        data._cache.clear()
        data._bad_contracts.clear()
        data._no_data_cache.clear()

        start = time.time()
        bars_dict = data.get_bars(ALL_SYMBOLS, timeframe="1 day", days=30)
        elapsed = time.time() - start

        print(f"\nFetched bars for {len(bars_dict)}/{len(ALL_SYMBOLS)} in {elapsed:.1f}s")
        assert len(bars_dict) >= 18, (
            f"Expected 18+ symbols with bars, got {len(bars_dict)}"
        )

    def test_fractional_qty_handling(self, ib_session):
        broker, data, config = ib_session

        # _format_order_qty should keep fractional for crypto, integer for stocks
        stock_qty = broker._format_order_qty("stock", 0.5)
        assert stock_qty == 0.5, "Stock qty should pass through as-is from _format_order_qty"

        crypto_qty = broker._format_order_qty("crypto", 0.00001234)
        assert crypto_qty == "0.00001234", f"Crypto qty should be formatted string, got {crypto_qty}"
        print(f"\nFractional qty: stock={stock_qty}, crypto={crypto_qty}")

    def test_missing_data_no_crash(self, ib_session):
        broker, data, config = ib_session

        # Clear caches so it actually tries
        data._bad_contracts.pop("ZZZZZ", None)
        data._no_data_cache.pop("ZZZZZ", None)

        # Request bars for a non-existent symbol
        result = data.get_bars(["ZZZZZ"], timeframe="1 day", days=30)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "ZZZZZ" not in result or result.get("ZZZZZ") is None or result["ZZZZZ"].empty, (
            "Expected no data for fake symbol"
        )
        print("\nMissing data handled cleanly — no crash")
```

- [ ] **Step 2: Run all Phase 4 tests**

Run: `pytest tests/test_ib_live.py -m live -v -k "Phase4" -s`
Expected: All PASS. Reconnection works, no event loop errors, bad contracts cached properly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ib_live.py
git commit -m "test: add Phase 4 — resilience, event loop safety, reconnection, pacing"
```

---

### Task 6: Add Missing Imports and Run Full Suite

**Files:**
- Modify: `tests/test_ib_live.py`

- [ ] **Step 1: Ensure all imports are at the top of test_ib_live.py**

Add at the top of `tests/test_ib_live.py`, after the docstring:

```python
"""Live IB integration tests — requires TWS/Gateway on port 4002."""

import time
import pytest

pytestmark = pytest.mark.live
```

All other imports (`from tests.conftest import ...`, `from base_broker import ...`, etc.) are inline in the test methods to avoid import-order issues.

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/test_ib_live.py -m live -v -s`
Expected: All Phase 1-4 tests run. Stock/ETF orders fill. Futures/crypto may skip. All resilience tests pass.

- [ ] **Step 3: Fix any bugs found**

For each failure:
1. Read the exact error message
2. Fix the bug in the source module (ib_broker.py, ib_data.py, strategies/, etc.)
3. Re-run only the failing test: `pytest tests/test_ib_live.py -m live -v -k "test_name" -s`
4. Re-run full suite to check for regressions

- [ ] **Step 4: Final commit**

```bash
git add tests/test_ib_live.py tests/conftest.py
git add -u  # any source files fixed during bug fixing
git commit -m "test: complete live IB integration test harness — all phases passing"
```

---

### Task 7: Summary Report

- [ ] **Step 1: Run full suite with verbose output and capture results**

Run: `pytest tests/test_ib_live.py -m live -v -s 2>&1 | tail -30`

- [ ] **Step 2: Document results**

Print a summary table:

| Phase | Tests | Pass | Fail | Skip |
|---|---|---|---|---|
| Phase 1 — Connectivity | 5 | ? | ? | ? |
| Phase 2 — Strategies | 4 | ? | ? | ? |
| Phase 3 — Orders | 7 | ? | ? | ? |
| Phase 4 — Resilience | 7 | ? | ? | ? |

List any bugs found and fixed during the run.
