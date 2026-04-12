# IB Migration Design — 2026-04-13

## Goal

Replace Alpaca as the active broker with Interactive Brokers (IB Gateway, paper,
port 4002) while keeping all Alpaca code intact as a dormant fallback. Extend the
bot to trade stocks, crypto, and futures (NQ, ES, CL, GC) with per-instrument
strategy routing and elite broker/data abstraction modelled on institutional
algo-trading frameworks (QuantConnect/Lean pattern).

---

## 1. Broker Abstraction Layer

### Interface: `BaseBroker` (`base_broker.py`)

Abstract class every broker must implement. The coordinator and all downstream
code only ever reference `BaseBroker` — zero broker-specific imports outside
broker files.

```
BaseBroker (abstract)
├── get_account() → dict
├── get_equity() → float
├── get_cash() → float
├── get_buying_power() → float
├── get_positions() → list[Position]
├── get_position(symbol) → Position | None
├── get_open_orders() → list[Order]
├── submit_order(OrderRequest) → Order
├── cancel_order(order_id)
├── cancel_all_orders()
├── close_position(symbol)
├── close_all_positions()
├── get_quote(symbol) → Quote
├── is_market_open() → bool
├── get_clock() → Clock
└── asset_type(symbol) → "stock" | "crypto" | "futures"
```

Orders use a typed `OrderRequest` dataclass (symbol, qty, side, order_type,
take_profit, stop_loss, time_in_force) so each broker handles its own
translation internally.

### `AlpacaBroker(BaseBroker)` (`alpaca_broker.py`)

Current `broker.py` refactored to implement `BaseBroker`. All existing logic
preserved — bracket orders, crypto OCO, smart orders, trailing stops.
Dormant when IB is the active router.

### `IBBroker(BaseBroker)` (`ib_broker.py`)

New. Uses `ib_insync` connected to IB Gateway at `127.0.0.1:4002`.

Key responsibilities:
- Translate `OrderRequest` → IB parent/child bracket orders (IB native OCA groups)
- Futures: resolve front-month contract via `ContractManager` before any order
- Stocks/crypto: use `Stock` / `Crypto` contract objects
- Positions mapped from IB `Portfolio` items to a unified `Position` dataclass
- Account data from `reqAccountSummary`

### `RoutingBroker(BaseBroker)` (`routing_broker.py`)

The coordinator's single broker reference. Routes by asset type:

```
RoutingBroker
├── futures  → IBBroker
├── stocks   → IBBroker
└── crypto   → AlpacaBroker  (IB paper only supports BTC/ETH; Alpaca covers full universe)
```

Routing table is config-driven. Switching crypto to IB later is one config line.
`asset_type(symbol)` is determined by `InstrumentClassifier` (see below).

---

## 2. Data Abstraction Layer

### Interface: `BaseDataFetcher` (`base_data.py`)

```
BaseDataFetcher (abstract)
├── get_bars(symbols, timeframe, days) → dict[str, DataFrame]
├── get_intraday_bars(symbol, timeframe, days) → DataFrame | None
├── prime_intraday_cache(symbols, timeframe, days)
├── get_latest_price(symbol) → float | None
└── get_latest_prices(symbols) → dict[str, float]
```

### `AlpacaDataFetcher(BaseDataFetcher)` (`alpaca_data.py`)

Current `data.py` refactored. Handles stocks + crypto. Unchanged logic.

### `IBDataFetcher(BaseDataFetcher)` (`ib_data.py`)

New. Uses `ib_insync` `reqHistoricalData` for futures bars.
Maps IB bar objects to the same OHLCV DataFrame schema as Alpaca.
Handles continuous contract stitching across rolls via `ContractManager`.

### `RoutingDataFetcher(BaseDataFetcher)` (`routing_data.py`)

Routes by asset type:
```
RoutingDataFetcher
├── futures  → IBDataFetcher
├── stocks   → AlpacaDataFetcher
└── crypto   → AlpacaDataFetcher
```

Coordinator calls `RoutingDataFetcher` — source is invisible.

---

## 3. Contract Manager (`contract_manager.py`)

Handles all IB contract resolution. Never called directly by strategies or
coordinator — only called internally by `IBBroker` and `IBDataFetcher`.

Responsibilities:
- **Front-month detection**: for each futures root (NQ, ES, CL, GC), query IB
  for available contracts, pick the nearest expiry with open interest > 0
- **Auto-roll**: re-resolve front-month at bot startup and once daily; swap
  contracts transparently
- **Contract cache**: resolved contracts cached (TTL = 4 hours) to avoid
  repeated IB queries
- **Contract specs** (hardcoded defaults, overridable in config):

| Root | Exchange | Currency | Multiplier | Tick |
|------|----------|----------|-----------|------|
| NQ   | CME      | USD      | 20        | 0.25 |
| ES   | CME      | USD      | 50        | 0.25 |
| CL   | NYMEX    | USD      | 1000      | 0.01 |
| GC   | COMEX    | USD      | 100       | 0.10 |

---

## 4. Instrument Classifier (`instrument_classifier.py`)

Central, stateless lookup used by `RoutingBroker`, `RoutingDataFetcher`, and
`StrategyRouter`. Returns `"stock"`, `"crypto"`, or `"futures"` for any symbol.

Rules (in priority order):
1. Symbol in `FUTURES_ROOTS` set (NQ, ES, CL, GC) → `"futures"`
2. Symbol in `CRYPTO_SYMBOLS` set (BTC/USD, ETH/USD, etc.) → `"crypto"`
3. Everything else → `"stock"`

`FUTURES_ROOTS` and `CRYPTO_SYMBOLS` are populated from `config.yaml` at startup.
No magic string matching — explicit membership sets only.

---

## 5. Strategy Router (`strategy_router.py`)

Each watcher is assigned a strategy set at creation time based on its
instrument type. No strategy runs on an instrument it wasn't designed for.

### Strategy matrix

| Strategy        | Stocks | Crypto | Futures |
|----------------|--------|--------|---------|
| Momentum        | ✅ 0.20 | ✅ 0.25 | ✅ 0.20 |
| Mean Reversion  | ✅ 0.15 | ❌      | ❌      |
| Breakout        | ✅ 0.20 | ✅ 0.25 | ✅ 0.20 |
| SuperTrend      | ✅ 0.20 | ✅ 0.25 | ✅ 0.25 |
| StochRSI        | ✅ 0.15 | ✅ 0.25 | ✅ 0.15 |
| VWAP Reclaim    | ✅ 0.10 | ❌      | ✅ 0.10 |
| Gap             | ✅ 0.10 | ❌      | ❌      |
| Liquidity Sweep | ✅ 0.20 | ✅ 0.25 | ✅ 0.25 |
| Futures Trend   | ❌      | ❌      | ✅ 0.30 |

Weights are normalized to 1.0 per instrument type inside `StrategyRouter`.
`StrategyRouter.get_strategies(instrument_type)` returns the instantiated
strategy list for that type. Watcher receives strategies at init, never changes.

---

## 6. New Strategy: `FuturesTrendStrategy` (`strategies/futures_trend.py`)

Designed specifically for NQ, ES, CL, GC on 5-min bars.

**Signals:**

1. **Opening Range Breakout (ORB)**: first 30-min range established at session
   open (09:30 ET); breakout above/below range with volume = strong signal
2. **Session VWAP reclaim**: intraday VWAP calculated from session open (not
   daily reset); price reclaims VWAP from above/below with volume confirmation
3. **Trend filter**: ADX > 25 required; EMA 8/21 alignment gating
4. **ATR volatility gate**: skip signal if ATR spike > 2.5× 20-bar average
   (news/event noise filter)

Returns score −1.0 to +1.0. High-conviction only: score ≥ 0.40 or ≤ −0.40
before contributing to composite.

---

## 7. Futures Universe (added to `config.yaml`)

```yaml
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
    stop_loss_atr_mult: 1.5     # futures ATR stops tighter (tick-based)
    take_profit_atr_mult: 3.0
    min_risk_reward: 2.0
    max_position_pct: 0.05      # 5% per futures position (higher notional)

ib:
  host: "127.0.0.1"
  port: 4002
  client_id: 1
  timeout_sec: 10
```

---

## 8. Files Created / Modified

### New files
| File | Purpose |
|------|---------|
| `base_broker.py` | Abstract broker interface |
| `alpaca_broker.py` | Alpaca implementation (renamed from broker.py) |
| `ib_broker.py` | IB implementation |
| `routing_broker.py` | Asset-type router |
| `base_data.py` | Abstract data interface |
| `alpaca_data.py` | Alpaca implementation (renamed from data.py) |
| `ib_data.py` | IB implementation (futures) |
| `routing_data.py` | Asset-type router |
| `contract_manager.py` | IB contract resolution + auto-roll |
| `instrument_classifier.py` | Classifies any symbol as stock/crypto/futures |
| `strategy_router.py` | Per-instrument strategy assignment |
| `strategies/futures_trend.py` | ORB + session VWAP + trend filter |

### Modified files
| File | Change |
|------|--------|
| `broker.py` | Becomes thin shim that re-exports `AlpacaBroker` as `Broker` for backward compat |
| `data.py` | Becomes thin shim that re-exports `AlpacaDataFetcher` as `DataFetcher` |
| `coordinator.py` | Use `RoutingBroker` + `RoutingDataFetcher`; call `StrategyRouter` when creating watchers |
| `watcher.py` | Accept strategy list at init instead of building internally |
| `config.yaml` | Add `ib` block + `futures` block |
| `strategies/__init__.py` | Register `FuturesTrendStrategy` |

### Untouched
All existing strategies (momentum, breakout, supertrend, stoch_rsi, vwap_reclaim,
gap, liquidity_sweep, mean_reversion) — no logic changes, only weight assignments
change per instrument type via `StrategyRouter`.

---

## 9. Error Handling & Fallback

- IB Gateway connection loss: `IBBroker` raises `BrokerConnectionError`;
  coordinator catches, pauses cycle, retries with exponential backoff (30s, 60s, 120s)
- Contract resolution failure: skip that futures symbol for the cycle, log warning
- IB data failure: `RoutingDataFetcher` falls back to cached bars if within TTL;
  if stale, skips the symbol
- Alpaca broker/data kept fully intact — re-enabling is a config change only

---

## 10. Out of Scope

- Live account (paper only for now)
- Options trading
- FIX protocol
- Level 2 / order flow data
- COT (Commitments of Traders) signal
- Backtesting futures (separate task — needs continuous contract data)
