# Design Spec: SQLite State Migration + Critical Bug Fixes
**Date:** 2026-04-13  
**Status:** Approved  
**Scope:** Full SQLite state migration, three missing PortfolioManager methods, IB bracket order fix, CRYPTO_SYMBOLS fix, ML + regime hardening

---

## Background

The bot has three missing methods in `PortfolioManager` (`set_position_risk`, `_save_meta`, `_save_watermarks`) that are called throughout the coordinator but never defined. This causes:
- `initial_risk` always 0.0 → partial exits (1.2R / 2.5R) never fire
- `position_meta` changes never persisted to disk → all position state lost on restart
- Watermarks reset on restart → chandelier trailing stop restarts from scratch
- Strategy attribution always `[]` → ML model has no training signal

Additionally, the IB `bracketOrder` call has incorrect parameter usage, and `CRYPTO_SYMBOLS` is imported from the Alpaca-era module (includes SOL/AVAX/LINK/DOGE not available on IB PAXOS).

---

## Architecture

### New File: `state_db.py`

Single `StateDB` class. One `bot_state.db` SQLite file. All writes go through a `threading.Lock()`. Replaces `state.json`, `trades.json`, `watcher_pending.json`.

**Tables:**

```sql
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    entry_price REAL,
    initial_risk REAL DEFAULT 0.0,
    stop_loss REAL,
    take_profit REAL,
    strategies TEXT,        -- JSON list e.g. '["momentum","supertrend"]'
    opened_at TEXT,
    original_qty REAL,
    side TEXT,
    partial_done INTEGER DEFAULT 0,
    second_partial_done INTEGER DEFAULT 0,
    breakeven_armed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS watermarks (
    symbol TEXT PRIMARY KEY,
    high_watermark REAL,
    low_watermark REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    side TEXT,
    qty REAL,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    reason TEXT,
    risk_dollars REAL,
    strategies TEXT,        -- JSON list
    opened_at TEXT,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT             -- JSON-encoded value
);

CREATE TABLE IF NOT EXISTS pending_signals (
    symbol TEXT PRIMARY KEY,
    prev_signal INTEGER DEFAULT 0
);
```

**Key methods:**
- `upsert_position(symbol, **fields)` — insert or update a position row
- `get_position(symbol) -> dict | None`
- `delete_position(symbol)`
- `get_all_positions() -> list[dict]`
- `upsert_watermark(symbol, high=None, low=None)`
- `get_watermarks() -> dict[str, dict]`
- `record_trade(**fields)`
- `get_trades() -> list[dict]`
- `set_state(key, value)` — for peak_equity, starting_equity etc.
- `get_state(key, default=None)`
- `set_pending_signal(symbol, prev_signal: bool)`
- `get_pending_signal(symbol) -> bool`
- `migrate_from_json()` — reads legacy JSON files, imports data, renames originals to `.bak`

### Migration Strategy

On `StateDB.__init__`, if `state.json` or `trades.json` or `watcher_pending.json` exist:
1. Read each file
2. Import data into corresponding SQLite tables
3. Rename files to `.json.bak` (not deleted — safety net)
4. Log a one-time migration summary

---

## Changes by File

### `state_db.py` (new)
Full `StateDB` implementation as described above.

### `portfolio.py`
- Import `StateDB` from `state_db`; instantiate as `self.db = StateDB()`
- Add `set_position_risk(symbol, entry_price, stop_loss, qty)`:
  ```python
  initial_risk = abs(entry_price - stop_loss) * qty
  self.db.upsert_position(symbol, initial_risk=initial_risk, entry_price=entry_price, original_qty=qty)
  self.position_meta.setdefault(symbol, {})["initial_risk"] = initial_risk
  ```
- Add `_save_meta()`: upserts all `self.position_meta` entries to `positions` table
- Add `_save_watermarks()`: upserts all `self.high_watermarks` / `self.low_watermarks` to `watermarks` table
- `__init__`: load `position_meta`, `high_watermarks`, `low_watermarks` from SQLite instead of `state.json`
- `execute_exits` → `record_trade` writes to `trades` table via `self.db.record_trade(...)`
- Replace `from broker import CRYPTO_SYMBOLS` with IB-specific constant

### `ib_broker.py`
Fix `submit_order` bracket logic. Replace `self._ib.bracketOrder(...)` with manual market + OCO children using correct `parentId` chaining:
```python
from ib_insync import MarketOrder, LimitOrder, StopOrder

tp_side = "SELL" if ib_side == "BUY" else "BUY"

# 1. Parent market order — do NOT transmit yet
parent = MarketOrder(ib_side, req.qty)
parent.transmit = False
parent_trade = self._ib.placeOrder(contract, parent)
parent_id = parent_trade.order.orderId

# 2. Take-profit limit leg
tp_order = LimitOrder(tp_side, req.qty, req.take_profit)
tp_order.parentId = parent_id
tp_order.transmit = False
self._ib.placeOrder(contract, tp_order)

# 3. Stop-loss leg — transmit=True sends all three atomically
sl_order = StopOrder(tp_side, req.qty, req.stop_loss)
sl_order.parentId = parent_id
sl_order.transmit = True
self._ib.placeOrder(contract, sl_order)
```

### `ib_data.py` + `instrument_classifier.py`
Add IB-specific `IB_CRYPTO_SYMBOLS` constant:
```python
IB_CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}
```
Export it from `ib_data.py`. Update `instrument_classifier.py` to use it.

### `watcher.py`
Replace `from data import CRYPTO_SYMBOLS` with `from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS`.

### `portfolio.py`, `risk.py`
Replace `from broker import CRYPTO_SYMBOLS` with `from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS`.

### `tracker.py`
Update `record_trade` and `get_trades` to read/write from `StateDB` instead of `trades.json`.

### `state.py`
Keep `load_state`/`save_state` for backward compat but have them delegate to `StateDB.get_state`/`set_state` for `peak_equity` and similar scalar values.

### `watcher.py` — pending signal persistence
Replace `_load_pending_state` / `_save_pending_state` JSON file functions with `StateDB.get_pending_signal` / `set_pending_signal`.

### `ml_model.py`
Reduce `min_trades` threshold from 50 → 20.

### `strategy_selector.py`
Add bear-regime veto: when regime is `bear` and ADX > 30, zero out `momentum`, `gap`, `breakout` weights for long signals.

---

## Data Flow

```
Coordinator places order
  → ib_broker.submit_order() — fixed bracket via manual OCO
  → portfolio.set_position_risk() — NEW: computes initial_risk, writes to SQLite
  → position_meta.update() — in-memory + SQLite via _save_meta()

Each coordinator cycle
  → portfolio.get_current_positions() — reads broker, merges with SQLite meta
  → portfolio.check_trailing_stops() — partial exits use initial_risk from meta
  → portfolio._save_meta() / _save_watermarks() — persist to SQLite

Position closed
  → portfolio.execute_exits() → tracker.record_trade() → StateDB.record_trade()
  → position deleted from positions table

Bot restart
  → StateDB.__init__() — migrates JSON if present, loads SQLite
  → portfolio.__init__() — restores position_meta, watermarks from SQLite
  → watcher.__init__() — restores prev_signal from SQLite
```

---

## Testing

- Unit test `StateDB`: upsert, get, delete, migration from JSON fixtures
- Unit test `set_position_risk`: verify `initial_risk = |entry - stop| * qty`
- Unit test partial exit trigger: with `initial_risk > 0`, verify 1.2R fires `partial_done`
- Integration smoke test: `test_boot.py` must still pass (coordinator boots, IB connects)
- Verify `bracketOrder` fix: paper account bracket order places correctly with TP and SL legs

---

## What Does NOT Change

- Signal generation logic (all strategies unchanged)
- Regime detection (HMM, breadth, sector layer)
- IB connection / contract resolution
- Confluence filtering (3-strategy minimum, 2-cycle confirmation)
- Order rate limiter, drawdown halt
- Dashboard, alerts, Discord bot
