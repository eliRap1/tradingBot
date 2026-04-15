# IB Integration Test Harness Design

## Goal

A comprehensive live integration test harness that connects to IB paper trading and validates the full pipeline: contract resolution, strategy signals, order submission, and resilience. All tests require a live IB Gateway/TWS connection.

## Architecture

Single test file (`tests/test_ib_live.py`) that runs sequentially through four phases. Uses pytest with a shared IB connection fixture. Requires `pytest -m live` marker to avoid running in CI or offline environments.

A curated 25-symbol universe covers all asset types (stocks, ETFs, sector ETFs, futures, crypto) and includes previously-problematic symbols to catch regressions.

## Tech Stack

- pytest (test runner + fixtures + markers)
- ib_insync (IB connection)
- Existing project modules: IBBroker, IBDataFetcher, StrategyRouter, all 9 strategies
- nest_asyncio (event loop compatibility)

---

## Test Symbol Universe (25 symbols)

| Category | Symbols | Why |
|---|---|---|
| Mega-cap tech | AAPL, MSFT, NVDA, GOOGL, META | Core universe, high liquidity |
| Growth/volatile | TSLA, AMD, PLTR, SNOW, CRWD | Volatile names, SNOW previously failed qualification |
| Value/dividend | PEP, LOW, SO, PM, COF | Previously failed contract resolution (SO, PM, LOW, COF, PEP) |
| ETFs | GLD, SPY, QQQ | GLD had quote/qualification issues, SPY/QQQ are core |
| Sector ETFs | XLF, XLE, XLK | Sector momentum edge layer depends on these |
| Futures | NQ, ES | Primary futures targets |
| Crypto | BTC/USD, ETH/USD | PAXOS routing, cashQty, IOC tif requirements |

---

## Phase 1: Connectivity & Contract Resolution

### 1.1 Connection Health
- Verify `_ib.isConnected()` returns True
- Verify `get_equity()` returns a positive float
- Verify `reqMarketDataType(3)` was set (delayed-frozen fallback)

### 1.2 Contract Resolution (all 25 symbols)
- Call `_resolve_contract()` for each symbol
- Assert returns a valid contract object (not None) for all 25
- Log: symbol, conId, exchange, primaryExchange
- Track resolution method (override, SMART, primaryExch hint, direct)
- Fail the test if any symbol returns None

### 1.3 Price Quotes (all 25 symbols)
- Call `reqMktData` for each symbol with delayed-frozen fallback
- Assert at least 20 of 25 return a price (live or delayed) — some may be outside market hours
- Log: symbol, bid, ask, last, data type (live vs delayed)
- Track which symbols return no data — these become known gaps

### 1.4 Bad Symbol Handling
- Attempt to resolve `FAKESYM123`
- Assert it returns None (no crash)
- Assert it enters `_bad_contracts` cache with a future timestamp
- Clear it from cache, verify it's gone

### 1.5 Pacing
- Resolve all 25 symbols in rapid succession (no sleep between calls)
- Assert no IB pacing errors (Error 162)
- Assert all 25 resolve successfully

---

## Phase 2: Strategy Signal Validation

### 2.1 Bar Fetching (all 25 symbols, both timeframes)
- Fetch 30 days of 1d bars via `ib_data.get_bars(symbol, "1d")`
- Fetch 2 days of 5m bars via `ib_data.get_bars(symbol, "5m")`
- Assert each result is a non-empty DataFrame
- Assert columns include: open, high, low, close, volume
- Assert no NaN in OHLC columns
- Log: symbol, timeframe, row count, date range

### 2.2 Individual Strategy Signals (9 strategies x 25 symbols)
- Strategies: momentum, breakout, mean_reversion, supertrend, stoch_rsi, liquidity_sweep, vwap_reclaim, gap, futures_trend
- For each strategy, call `generate_signals(bars)` with the fetched bars
- Assert:
  - Returns a dict
  - All values are floats
  - All scores between -1.0 and +1.0
  - No unhandled exceptions
- Log: symbol, strategy, score (or "no signal")

### 2.3 Strategy Router Aggregation
- For each symbol, run all applicable strategies through `StrategyRouter`
- Assert composite score is a valid float
- Assert weighted aggregation produces reasonable output (not all zeros unless no signals fired)

### 2.4 Edge Cases
- Run strategies on futures bars (NQ, ES) — different bar structure, fewer data points
- Run strategies on crypto bars (BTC/USD, ETH/USD) — 24h market, different volume profile
- Run strategies on a symbol with very low volume if one exists in the set
- Assert no crashes on any edge case

---

## Phase 3: Order Pipeline

All orders use minimal size to keep costs near zero. Each test logs the full IB Trade response.

### 3.1 Market Order — Stock
- Submit: 1 share of cheapest stock in universe (e.g. AMD or PLTR)
- Assert: IB accepts the order (status is PendingSubmit, Submitted, or Filled — not Cancelled/Inactive)
- Log: orderId, status, fill price

### 3.2 Market Order — ETF
- Submit: 1 share of SPY
- Assert: same as 3.1

### 3.3 Market Order — Futures
- Submit: 1 MNQ (Micro NQ) or 1 ES micro
- If margin insufficient: log as expected skip, don't fail
- Assert: order accepted or margin-rejection logged cleanly

### 3.4 Market Order — Crypto
- Submit: $1 cashQty BTC/USD with tif=IOC
- If regulatory restriction: log as expected skip, don't fail
- Assert: order uses cashQty (not totalQuantity), tif is IOC

### 3.5 Bracket Order
- Submit: bracket for 1 share of AAPL (parent + TP at +5% + SL at -3%)
- Assert:
  - Parent order has transmit=False initially
  - Parent gets a valid orderId from IB
  - TP order has parentId matching parent's orderId
  - SL order has parentId matching parent's orderId
  - SL has transmit=True (releases all three)
  - All three orders appear in `openOrders()` or `openTrades()`

### 3.6 Cancel Order
- Submit: limit buy for 1 share of MSFT at $1.00 (far below market — won't fill)
- Wait 1 second
- Cancel via `cancel_order(order_id)`
- Assert: order status becomes Cancelled

### 3.7 Close Positions
- After tests 3.1-3.3 fill, close each position via `close_position(symbol)`
- Assert: position qty goes to 0 for each

---

## Phase 4: Resilience & Edge Cases

### 4.1 Bad Contract Cache TTL
- Resolve `FAKESYM123` — enters bad_contracts
- Verify `_bad_contracts["FAKESYM123"]` has timestamp ~30 min in future
- Clear via `_bad_contracts.pop("FAKESYM123")`
- Resolve again — verify it retries (not cached)

### 4.2 Duplicate Orders
- Submit the same order (1 share AAPL) twice in <1 second
- Assert: no crash, both orders get unique orderIds
- Clean up: cancel/close both

### 4.3 Event Loop Safety
- From a separate thread (simulating Discord bot context):
  - Call `get_quote(symbol)`
  - Call `get_latest_price(symbol)`
  - Call `_resolve_contract(symbol)`
- Assert: no "event loop already running" errors
- Assert: all calls return valid results
- Uses `nest_asyncio.apply()` to match production Discord setup

### 4.4 Reconnection
- Call `_ib.disconnect()` to force disconnect
- Call `get_equity()` (which should trigger `_ensure_connected` -> `_connect()`)
- Assert: reconnects successfully
- Assert: subsequent calls work normally

### 4.5 Pacing Limits (Bar Fetching)
- Fetch 1d bars for all 25 symbols with no delay between requests
- Assert: IB pacing limiter in `ib_data` prevents Error 162
- Assert: all 25 return valid data (possibly with small delays from pacing)

### 4.6 Fractional Quantity Rejection
- Create an OrderRequest with qty=0.5 for a stock
- Assert: `_format_order_qty` or submit_order handles it (rounds to 1 or rejects cleanly)

### 4.7 Missing Data
- Request bars for a symbol with very thin data (or a weird timeframe)
- Assert: returns empty DataFrame or None, no crash

---

## Test Structure

```
tests/test_ib_live.py
```

```python
import pytest

# Shared fixture — one IB connection for all tests
@pytest.fixture(scope="module")
def ib_session():
    """Connect to IB paper, yield broker + data, disconnect after all tests."""
    # Setup: connect broker, data fetcher
    # Yield: (broker, data, config)
    # Teardown: close positions, disconnect

SYMBOLS = {
    "stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "META",
               "TSLA", "AMD", "PLTR", "SNOW", "CRWD",
               "PEP", "LOW", "SO", "PM", "COF"],
    "etfs": ["GLD", "SPY", "QQQ"],
    "sector_etfs": ["XLF", "XLE", "XLK"],
    "futures": ["NQ", "ES"],
    "crypto": ["BTC/USD", "ETH/USD"],
}
ALL_SYMBOLS = [s for group in SYMBOLS.values() for s in group]

@pytest.mark.live
class TestPhase1Connectivity:
    def test_connection_health(self, ib_session): ...
    def test_resolve_all_contracts(self, ib_session): ...
    def test_get_quotes(self, ib_session): ...
    def test_bad_symbol(self, ib_session): ...
    def test_pacing(self, ib_session): ...

@pytest.mark.live
class TestPhase2Strategies:
    def test_fetch_bars(self, ib_session): ...
    def test_strategy_signals(self, ib_session): ...
    def test_strategy_router(self, ib_session): ...
    def test_edge_cases(self, ib_session): ...

@pytest.mark.live
class TestPhase3Orders:
    def test_market_order_stock(self, ib_session): ...
    def test_market_order_etf(self, ib_session): ...
    def test_market_order_futures(self, ib_session): ...
    def test_market_order_crypto(self, ib_session): ...
    def test_bracket_order(self, ib_session): ...
    def test_cancel_order(self, ib_session): ...
    def test_close_positions(self, ib_session): ...

@pytest.mark.live
class TestPhase4Resilience:
    def test_bad_contract_cache(self, ib_session): ...
    def test_duplicate_orders(self, ib_session): ...
    def test_event_loop_safety(self, ib_session): ...
    def test_reconnection(self, ib_session): ...
    def test_pacing_limits(self, ib_session): ...
    def test_fractional_qty(self, ib_session): ...
    def test_missing_data(self, ib_session): ...
```

## Running the Tests

```bash
# Run all live IB tests (requires TWS/Gateway running on port 4002)
pytest tests/test_ib_live.py -m live -v

# Run only connectivity tests
pytest tests/test_ib_live.py -m live -k "Phase1" -v

# Run only strategy tests
pytest tests/test_ib_live.py -m live -k "Phase2" -v

# Run only order tests
pytest tests/test_ib_live.py -m live -k "Phase3" -v
```

## Pass/Fail Criteria

| Phase | Pass Criteria |
|---|---|
| Phase 1 | All 25 symbols resolve. 20+ return a price. Bad symbol handled cleanly. |
| Phase 2 | All 9 strategies run on all 25 symbols without crashes. All scores in [-1, +1]. Bars fetched for all symbols. |
| Phase 3 | Stock + ETF orders fill. Bracket structure correct. Cancel works. Crypto/futures log status (pass or expected restriction). All positions closed at end. |
| Phase 4 | No event loop errors. Reconnection works. Pacing respected. Bad contracts cached/cleared properly. |

## Bug Fixing Process

When a test fails:
1. Log the exact IB error code and message
2. Fix the bug in the relevant module
3. Re-run only the failing test to verify the fix
4. Re-run the full suite to check for regressions
