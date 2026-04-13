# Design Spec: SQLite State Migration + Full Bug Fix Pass
**Date:** 2026-04-13  
**Status:** Approved  
**Scope:** Full SQLite state migration, all missing PortfolioManager methods, IB bracket order fix,
CRYPTO_SYMBOLS fix, ML feature/leakage fix, HMM caching, breadth sampling, bear veto (correct location),
dead-code removal, regime label fix, live_trading crypto set.

---

## Complete Bug Inventory

| Priority | Bug | File | Impact |
|----------|-----|------|--------|
| P0 | `set_position_risk` not defined | `portfolio.py` | `initial_risk=0` → partials never fire |
| P0 | `_save_meta` not defined | `portfolio.py` | Meta lost on every restart |
| P0 | `_save_watermarks` not defined | `portfolio.py` | Watermarks reset → trailing stop restarts |
| P0 | `bracketOrder` missing `takeProfitPrice` | `ib_broker.py` | TP leg malformed / TypeError silently caught |
| P0 | ML train/predict feature mismatch + leakage | `ml_model.py` | Model produces garbage predictions |
| P1 | `CRYPTO_SYMBOLS` from Alpaca (SOL/AVAX/LINK/DOGE) | `portfolio.py`, `watcher.py`, `risk.py` | Wrong crypto set for IB |
| P1 | Bear veto in wrong file (bypassed by StrategyRouter) | `strategy_selector.py` | Veto never fires |
| P1 | HMM refit every 5-minute cycle | `regime.py` | Slow + non-deterministic results |
| P1 | Breadth sample always first 20 symbols (all tech) | `regime.py` | Misleading breadth signal |
| P2 | `SmartFilters.filter_confirmed` dead code | `filters.py` | Wasted disk writes each cycle |
| P2 | `_classify_symbol` includes non-IB crypto | `live_trading.py` | Wrong tradability classification |
| P2 | `"breakdown"` regime value undocumented | `strategy_selector.py` | Inconsistent regime string |
| P3 | Duplicate `"regime"` key in dict | `coordinator.py:891` | Cosmetic — second overwrites first |

---

## Architecture

### New File: `state_db.py`

Single `StateDB` class. One `bot_state.db` SQLite file. All writes go through a `threading.Lock()`.
Replaces `state.json`, `trades.json`, `watcher_pending.json`.

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
    value TEXT              -- JSON-encoded value
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

---

### `portfolio.py` — P0 fixes (3 missing methods)

- Import `StateDB` from `state_db`; instantiate as `self.db = StateDB()`
- Add `set_position_risk(symbol, entry_price, stop_loss, qty)`:
  ```python
  initial_risk = abs(entry_price - stop_loss) * qty
  self.db.upsert_position(symbol, initial_risk=initial_risk,
                          entry_price=entry_price, original_qty=qty)
  self.position_meta.setdefault(symbol, {})["initial_risk"] = initial_risk
  ```
- Add `_save_meta()`: iterates `self.position_meta`, calls `self.db.upsert_position` for each entry
- Add `_save_watermarks()`: iterates `self.high_watermarks` / `self.low_watermarks`, calls `self.db.upsert_watermark`
- `__init__`: load `position_meta`, `high_watermarks`, `low_watermarks` from SQLite (via `StateDB`) instead of `state.json`
- `execute_exits` → `record_trade` writes to `trades` table via `self.db.record_trade(...)`
- Replace `from broker import CRYPTO_SYMBOLS` → `from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS`

---

### `ib_broker.py` — P0 bracket order fix

Replace `self._ib.bracketOrder(...)` with manual market + OCO children using correct `parentId` chaining.
The current call passes `limitPrice=req.take_profit` as the parent's entry price (wrong) and omits the
required `takeProfitPrice` positional argument (raises TypeError caught silently by coordinator's
`except Exception`).

Correct implementation:
```python
from ib_insync import MarketOrder, LimitOrder, StopOrder

tp_side = "SELL" if ib_side == "BUY" else "BUY"

# 1. Parent market order — transmit=False holds it until children are attached
parent = MarketOrder(ib_side, req.qty)
parent.transmit = False
parent_trade = self._ib.placeOrder(contract, parent)
parent_id = parent_trade.order.orderId

# 2. Take-profit limit leg
tp_order = LimitOrder(tp_side, req.qty, req.take_profit)
tp_order.parentId = parent_id
tp_order.transmit = False
self._ib.placeOrder(contract, tp_order)

# 3. Stop-loss leg — transmit=True releases all three atomically
sl_order = StopOrder(tp_side, req.qty, req.stop_loss)
sl_order.parentId = parent_id
sl_order.transmit = True
self._ib.placeOrder(contract, sl_order)
```

---

### `ml_model.py` — P0 feature mismatch + leakage fix

**Current bugs:**
- `_trade_to_features` uses binary 0/1 strategy flags; `_build_feature_vector` uses float scores — different feature spaces, model is miscalibrated
- `r_multiple` is used as a training feature but is only known after the trade closes (look-ahead bias / data leakage)

**Fix:** Unify both methods to use identical entry-time features only:
```
[strategy_score_0..7 (float), num_agreeing (int), composite_score (float)]
```
10 features total, all known at entry time, same type in both training and prediction.

For training: strategy scores are not stored in trade records currently. After the
`position_meta.strategies` fix lands, store scores too. Until then, use binary
presence flags for both methods (consistent, no leakage). Remove `r_multiple` from training features entirely.

---

### `ib_data.py` — P1 IB-specific crypto set

Add at module level:
```python
IB_CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}
```
Export from `ib_data.py`. This is the authoritative set for IB PAXOS paper + live.

---

### `watcher.py` — P1 CRYPTO_SYMBOLS + bear veto

- Replace `from data import CRYPTO_SYMBOLS` → `from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS`
- **Bear veto** goes here (not in `strategy_selector.py`), because `StrategyRouter` bypasses
  `strategy_selector` entirely. After `selection["strategies"]` is determined, apply veto:
  ```python
  # Bear veto: in macro bear regime with strong trend, suppress long-only strategies
  if macro_regime == "bear" and ctx["adx"] > 30:
      for strat in ("momentum", "gap", "breakout"):
          selection["strategies"][strat] = 0.0
      # Re-normalize
      total = sum(selection["strategies"].values())
      if total > 0:
          selection["strategies"] = {k: round(v/total, 4)
                                     for k, v in selection["strategies"].items()}
  ```
  `macro_regime` is passed in from coordinator via a new optional `regime_getter` argument, or read
  from a shared state. Simplest: pass it as a parameter to `_analyze()`.

---

### `risk.py` — P1 CRYPTO_SYMBOLS

Replace `from broker import CRYPTO_SYMBOLS` → `from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS`

---

### `regime.py` — P1 HMM caching + P1 breadth fix

**HMM caching:** Cache the fitted model and last-predict timestamp. Only refit when:
- More than 60 minutes have elapsed since last fit, OR
- A significant regime change is suspected (EMA regime changed)

```python
self._hmm_last_fit: float = 0.0
self._hmm_refit_interval: int = 3600  # refit at most once per hour

def _get_hmm_regime(self, df):
    now = time.time()
    needs_refit = (
        self._hmm_model is None or
        now - self._hmm_last_fit > self._hmm_refit_interval
    )
    if needs_refit:
        # fit model, update self._hmm_model, self._hmm_last_fit = now
        ...
    else:
        # just predict on existing model (fast path)
        ...
```

**Breadth fix:** Replace `sample = self._universe[:20]` with a stratified sample:
```python
import random
sample_size = min(30, len(self._universe))
sample = random.sample(self._universe, sample_size)
```
This ensures sectors beyond tech are represented.

---

### `filters.py` — P2 dead code removal

Remove `filter_confirmed()` method and the `CONFIRMATION_FILE = "pending_signals.json"` constant.
The coordinator never calls this method — signal confirmation is handled by `watcher.state.confirmed`.
Remove the corresponding `_load_pending` / `_save_pending` methods and the `self._pending_signals` dict.

---

### `live_trading.py` — P2 non-IB crypto fix

Replace `_CRYPTO_SUFFIXES` in `_classify_symbol` with IB-only set:
```python
_CRYPTO_SUFFIXES = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}
```

---

### `strategy_selector.py` — P2 regime label fix

Change `"regime": "breakdown"` (line 176) to `"regime": "breakout"` to match the documented
values: `"trending" | "ranging" | "breakout" | "volatile"`.

---

### `tracker.py`
- Update `record_trade` and `_save` to write to `StateDB` instead of `trades.json`
- Update `_load` to read from `StateDB.get_trades()` on init
- Keep `TRADES_FILE` path for the migration step only

---

### `state.py`
Delegate `load_state`/`save_state` to `StateDB.get_state`/`set_state` for scalar values
(`peak_equity`, `starting_equity`). Remove direct JSON file access.

---

### `watcher.py` — pending signal persistence
Replace `_load_pending_state` / `_save_pending_state` (currently writing to `watcher_pending.json`)
with `StateDB.get_pending_signal` / `set_pending_signal`.

---

### `coordinator.py` — P3 cosmetic
Remove duplicate `"regime": s.regime` key at line 891-892 in `get_all_watcher_states`.

---

## Data Flow

```
Coordinator places order
  → ib_broker.submit_order() — fixed bracket via manual OCO (parent + TP + SL)
  → portfolio.set_position_risk() — NEW: computes initial_risk, writes to SQLite
  → position_meta.setdefault().update() — in-memory, then _save_meta() persists to SQLite

Each coordinator cycle
  → portfolio.get_current_positions() — reads broker, merges with SQLite meta
  → portfolio.check_trailing_stops() — partial exits now work (initial_risk > 0)
  → portfolio._save_meta() / _save_watermarks() — persist to SQLite

Position closed (trailing stop / SL / TP)
  → portfolio.execute_exits() → tracker.record_trade() → StateDB.record_trade()
  → position row deleted from SQLite positions table

Bot restart
  → StateDB.__init__() — migrates JSON if present, loads SQLite
  → portfolio.__init__() — restores position_meta, watermarks from SQLite
  → watcher.__init__() — restores prev_signal from SQLite pending_signals table
  → ML model loaded from ml_model.bin (if exists)

ML training cycle (every 50 coordinator cycles)
  → ml_model.train(tracker.trades) — uses unified feature vector (no leakage)
  → model saved to ml_model.bin
```

---

## Testing

- Unit test `StateDB`: upsert, get, delete, migration from JSON fixtures
- Unit test `set_position_risk`: verify `initial_risk = |entry - stop| * qty` and SQLite write
- Unit test partial exit trigger: with `initial_risk > 0`, verify 1.2R fires `partial_done=True`
- Unit test ML features: `_trade_to_features` and `_build_feature_vector` must return same-shape vectors
- Unit test HMM: refit only fires after TTL, fast-path returns same regime label on second call
- Unit test breadth: sample is not always the first N symbols of the universe
- Integration smoke test: `test_boot.py` must still pass after all changes
- Manual verify on paper: confirm bracket order places with TP and SL legs visible in IB TWS

---

## What Does NOT Change

- All strategy signal generation logic (momentum, supertrend, breakout, mean_reversion, etc.)
- IB connection / reconnect / contract resolution
- Confluence filtering (3-strategy min, 2-cycle confirmation, hourly bias gate)
- Order rate limiter (20/hour), drawdown halt
- Dashboard, alerts, Discord bot
- Sector regime layer (SectorRegimeFilter)
- Correlation filter logic
