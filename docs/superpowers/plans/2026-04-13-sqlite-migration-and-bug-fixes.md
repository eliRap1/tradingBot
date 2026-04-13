# SQLite Migration + Full Bug Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 13 bugs (3 missing methods, broken bracket order, ML leakage, HMM cost, and more) and migrate all JSON state files to SQLite for thread-safe persistence.

**Architecture:** A new `StateDB` class wraps all SQLite access behind a single lock. `PortfolioManager`, `TradeTracker`, and watcher pending-signal code all delegate to it. Everything else (IB bracket, ML, regime, CRYPTO_SYMBOLS) is a surgical single-file fix.

**Tech Stack:** Python 3.10+, sqlite3 (stdlib), ib_insync, lightgbm, hmmlearn, pandas, ta

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `state_db.py` | **Create** | All SQLite access — positions, watermarks, trades, bot_state, pending_signals |
| `tests/test_state_db.py` | **Create** | Unit tests for StateDB |
| `portfolio.py` | **Modify** | Add `set_position_risk`, `_save_meta`, `_save_watermarks`; wire to StateDB |
| `tracker.py` | **Modify** | Read/write trades via StateDB |
| `state.py` | **Modify** | Delegate `load_state`/`save_state` to StateDB |
| `watcher.py` | **Modify** | Pending signal → StateDB; CRYPTO_SYMBOLS fix; bear veto |
| `ib_broker.py` | **Modify** | Replace `bracketOrder` with manual market+OCO chain |
| `ib_data.py` | **Modify** | Add `IB_CRYPTO_SYMBOLS` constant |
| `risk.py` | **Modify** | Import `IB_CRYPTO_SYMBOLS` |
| `ml_model.py` | **Modify** | Unify train/predict feature vectors; remove `r_multiple` leakage |
| `regime.py` | **Modify** | Cache HMM model; stratified breadth sample |
| `filters.py` | **Modify** | Remove dead `filter_confirmed` + related code |
| `live_trading.py` | **Modify** | IB-only crypto set in `_classify_symbol` |
| `strategy_selector.py` | **Modify** | `"breakdown"` → `"breakout"` regime label |
| `coordinator.py` | **Modify** | Remove duplicate `"regime"` key |

---

## Task 1: Create `state_db.py` — SQLite state layer

**Files:**
- Create: `state_db.py`
- Create: `tests/test_state_db.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_state_db.py`:

```python
import json
import os
import pytest
import tempfile

# Override DB path for tests
os.environ["BOT_STATE_DB"] = ":memory:"

from state_db import StateDB


@pytest.fixture
def db():
    d = StateDB(":memory:")
    return d


def test_upsert_and_get_position(db):
    db.upsert_position("AAPL", entry_price=150.0, initial_risk=200.0,
                       stop_loss=148.0, take_profit=156.0,
                       strategies='["momentum"]', opened_at="2026-04-13T10:00:00",
                       original_qty=10.0, side="buy",
                       partial_done=0, second_partial_done=0, breakeven_armed=0)
    pos = db.get_position("AAPL")
    assert pos is not None
    assert pos["initial_risk"] == 200.0
    assert pos["side"] == "buy"


def test_upsert_position_updates_existing(db):
    db.upsert_position("AAPL", entry_price=150.0, initial_risk=0.0)
    db.upsert_position("AAPL", initial_risk=250.0)
    pos = db.get_position("AAPL")
    assert pos["initial_risk"] == 250.0


def test_delete_position(db):
    db.upsert_position("MSFT", entry_price=300.0)
    db.delete_position("MSFT")
    assert db.get_position("MSFT") is None


def test_get_all_positions(db):
    db.upsert_position("AAPL", entry_price=150.0)
    db.upsert_position("MSFT", entry_price=300.0)
    positions = db.get_all_positions()
    symbols = [p["symbol"] for p in positions]
    assert "AAPL" in symbols
    assert "MSFT" in symbols


def test_watermarks(db):
    db.upsert_watermark("AAPL", high=155.0, low=148.0)
    wm = db.get_watermarks()
    assert "AAPL" in wm
    assert wm["AAPL"]["high_watermark"] == 155.0
    assert wm["AAPL"]["low_watermark"] == 148.0


def test_watermarks_partial_update(db):
    db.upsert_watermark("AAPL", high=155.0, low=148.0)
    db.upsert_watermark("AAPL", high=158.0)  # only update high
    wm = db.get_watermarks()
    assert wm["AAPL"]["high_watermark"] == 158.0
    assert wm["AAPL"]["low_watermark"] == 148.0


def test_record_and_get_trades(db):
    db.record_trade(symbol="AAPL", side="buy", qty=10, entry_price=150.0,
                    exit_price=155.0, pnl=50.0, reason="trailing_stop",
                    risk_dollars=200.0, strategies='["momentum"]',
                    opened_at="2026-04-13T10:00:00", closed_at="2026-04-13T15:00:00")
    trades = db.get_trades()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "AAPL"
    assert trades[0]["pnl"] == 50.0


def test_bot_state(db):
    db.set_state("peak_equity", 105000.0)
    val = db.get_state("peak_equity")
    assert val == 105000.0


def test_bot_state_default(db):
    val = db.get_state("missing_key", default=99.0)
    assert val == 99.0


def test_pending_signals(db):
    db.set_pending_signal("AAPL", True)
    assert db.get_pending_signal("AAPL") is True
    db.set_pending_signal("AAPL", False)
    assert db.get_pending_signal("AAPL") is False


def test_pending_signal_default_false(db):
    assert db.get_pending_signal("UNKNOWN") is False


def test_migration_from_json(tmp_path):
    """Migration reads state.json and trades.json, renames to .bak."""
    state_file = tmp_path / "state.json"
    trades_file = tmp_path / "trades.json"
    state_file.write_text(json.dumps({"peak_equity": 102000.0, "high_watermarks": {"AAPL": 155.0}}))
    trades_file.write_text(json.dumps([
        {"symbol": "AAPL", "side": "buy", "qty": 10, "entry_price": 150.0,
         "exit_price": 155.0, "pnl": 50.0, "reason": "trailing_stop",
         "risk_dollars": 200.0, "strategies": ["momentum"],
         "closed_at": "2026-04-13T15:00:00"}
    ]))

    db = StateDB(":memory:", base_dir=str(tmp_path))
    db.migrate_from_json()

    assert db.get_state("peak_equity") == 102000.0
    wm = db.get_watermarks()
    assert wm.get("AAPL", {}).get("high_watermark") == 155.0
    trades = db.get_trades()
    assert len(trades) == 1
    assert trades[0]["pnl"] == 50.0

    # originals renamed to .bak
    assert (tmp_path / "state.json.bak").exists()
    assert (tmp_path / "trades.json.bak").exists()
    assert not state_file.exists()
    assert not trades_file.exists()
```

- [ ] **Step 2: Run tests — verify they all fail**

```
pytest tests/test_state_db.py -v
```
Expected: `ImportError: No module named 'state_db'` or similar.

- [ ] **Step 3: Create `state_db.py`**

```python
"""SQLite-backed state storage.

Replaces state.json, trades.json, and watcher_pending.json.
All writes are serialized through a single threading.Lock().

DB path defaults to bot_state.db in the project root.
Override with BOT_STATE_DB env var (use ':memory:' for tests).
"""
import json
import os
import sqlite3
import threading
from typing import Any, Optional
from utils import setup_logger

log = setup_logger("state_db")

_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "bot_state.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    entry_price REAL,
    initial_risk REAL DEFAULT 0.0,
    stop_loss REAL,
    take_profit REAL,
    strategies TEXT,
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
    strategies TEXT,
    opened_at TEXT,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS pending_signals (
    symbol TEXT PRIMARY KEY,
    prev_signal INTEGER DEFAULT 0
);
"""


class StateDB:
    def __init__(self, db_path: str = None, base_dir: str = None):
        if db_path is None:
            db_path = os.environ.get("BOT_STATE_DB", _DEFAULT_DB_PATH)
        self._path = db_path
        self._base_dir = base_dir or os.path.dirname(__file__)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_SQL)
        self._conn.commit()

    # ── Positions ────────────────────────────────────────────

    def upsert_position(self, symbol: str, **fields):
        """Insert or update fields for a position row."""
        with self._lock:
            # Build dynamic upsert: only update provided fields
            existing = self._conn.execute(
                "SELECT * FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            if existing is None:
                cols = ["symbol"] + list(fields.keys())
                placeholders = ", ".join("?" * len(cols))
                vals = [symbol] + list(fields.values())
                self._conn.execute(
                    f"INSERT INTO positions ({', '.join(cols)}) VALUES ({placeholders})",
                    vals
                )
            else:
                if fields:
                    set_clause = ", ".join(f"{k} = ?" for k in fields)
                    vals = list(fields.values()) + [symbol]
                    self._conn.execute(
                        f"UPDATE positions SET {set_clause} WHERE symbol = ?", vals
                    )
            self._conn.commit()

    def get_position(self, symbol: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        return dict(row) if row else None

    def delete_position(self, symbol: str):
        with self._lock:
            self._conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            self._conn.commit()

    def get_all_positions(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]

    # ── Watermarks ───────────────────────────────────────────

    def upsert_watermark(self, symbol: str, high: float = None, low: float = None):
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM watermarks WHERE symbol = ?", (symbol,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO watermarks (symbol, high_watermark, low_watermark) VALUES (?,?,?)",
                    (symbol, high, low)
                )
            else:
                if high is not None:
                    self._conn.execute(
                        "UPDATE watermarks SET high_watermark = ? WHERE symbol = ?",
                        (high, symbol)
                    )
                if low is not None:
                    self._conn.execute(
                        "UPDATE watermarks SET low_watermark = ? WHERE symbol = ?",
                        (low, symbol)
                    )
            self._conn.commit()

    def get_watermarks(self) -> dict[str, dict]:
        rows = self._conn.execute("SELECT * FROM watermarks").fetchall()
        return {r["symbol"]: dict(r) for r in rows}

    # ── Trades ───────────────────────────────────────────────

    def record_trade(self, symbol: str, side: str, qty: float,
                     entry_price: float, exit_price: float, pnl: float,
                     reason: str = "", risk_dollars: float = 0.0,
                     strategies = None, opened_at: str = "",
                     closed_at: str = ""):
        strats_json = json.dumps(strategies) if isinstance(strategies, list) else (strategies or "[]")
        with self._lock:
            self._conn.execute(
                """INSERT INTO trades
                   (symbol,side,qty,entry_price,exit_price,pnl,reason,
                    risk_dollars,strategies,opened_at,closed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, side, qty, entry_price, exit_price, pnl, reason,
                 risk_dollars, strats_json, opened_at, closed_at)
            )
            self._conn.commit()

    def get_trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY id"
        ).fetchall()
        result = []
        for r in rows:
            t = dict(r)
            t["strategies"] = json.loads(t.get("strategies") or "[]")
            result.append(t)
        return result

    # ── Bot state (scalar key/value) ─────────────────────────

    def set_state(self, key: str, value: Any):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )
            self._conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM bot_state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    # ── Pending signals ──────────────────────────────────────

    def set_pending_signal(self, symbol: str, prev_signal: bool):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pending_signals (symbol, prev_signal) VALUES (?,?)",
                (symbol, int(prev_signal))
            )
            self._conn.commit()

    def get_pending_signal(self, symbol: str) -> bool:
        row = self._conn.execute(
            "SELECT prev_signal FROM pending_signals WHERE symbol = ?", (symbol,)
        ).fetchone()
        return bool(row["prev_signal"]) if row else False

    # ── Migration from JSON ──────────────────────────────────

    def migrate_from_json(self):
        """One-time migration from legacy JSON files to SQLite.

        Reads state.json, trades.json, watcher_pending.json if present.
        Renames originals to .bak after successful import.
        """
        base = self._base_dir
        state_path = os.path.join(base, "state.json")
        trades_path = os.path.join(base, "trades.json")
        pending_path = os.path.join(base, "watcher_pending.json")

        migrated = []

        # state.json → bot_state + watermarks
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = json.load(f)
                if "peak_equity" in state:
                    self.set_state("peak_equity", state["peak_equity"])
                for sym, hw in state.get("high_watermarks", {}).items():
                    self.upsert_watermark(sym, high=float(hw))
                for sym, lw in state.get("low_watermarks", {}).items():
                    self.upsert_watermark(sym, low=float(lw))
                for sym, meta in state.get("position_meta", {}).items():
                    self.upsert_position(sym, **{
                        k: v for k, v in meta.items()
                        if k in ("entry_price","initial_risk","stop_loss","take_profit",
                                 "strategies","opened_at","original_qty","side",
                                 "partial_done","second_partial_done","breakeven_armed")
                    })
                os.rename(state_path, state_path + ".bak")
                migrated.append("state.json")
            except Exception as e:
                log.error(f"Migration of state.json failed: {e}")

        # trades.json → trades table
        if os.path.exists(trades_path):
            try:
                with open(trades_path) as f:
                    trades = json.load(f)
                for t in trades:
                    self.record_trade(
                        symbol=t.get("symbol",""),
                        side=t.get("side","buy"),
                        qty=float(t.get("qty",0)),
                        entry_price=float(t.get("entry_price",0)),
                        exit_price=float(t.get("exit_price",0)),
                        pnl=float(t.get("pnl",0)),
                        reason=t.get("reason",""),
                        risk_dollars=float(t.get("risk_dollars") or 0),
                        strategies=t.get("strategies",[]),
                        opened_at=t.get("opened_at",""),
                        closed_at=t.get("closed_at",""),
                    )
                os.rename(trades_path, trades_path + ".bak")
                migrated.append("trades.json")
            except Exception as e:
                log.error(f"Migration of trades.json failed: {e}")

        # watcher_pending.json → pending_signals table
        if os.path.exists(pending_path):
            try:
                with open(pending_path) as f:
                    pending = json.load(f)
                for sym, val in pending.items():
                    prev = val if isinstance(val, bool) else bool(val.get("prev_signal", False))
                    self.set_pending_signal(sym, prev)
                os.rename(pending_path, pending_path + ".bak")
                migrated.append("watcher_pending.json")
            except Exception as e:
                log.error(f"Migration of watcher_pending.json failed: {e}")

        if migrated:
            log.info(f"SQLite migration complete: {', '.join(migrated)} → bot_state.db")
```

- [ ] **Step 4: Run tests — verify they all pass**

```
pytest tests/test_state_db.py -v
```
Expected: All green. If migration test fails check `tmp_path` fixture and file renaming.

- [ ] **Step 5: Commit**

```bash
git add state_db.py tests/test_state_db.py
git commit -m "feat: add StateDB — SQLite state layer replacing all JSON state files"
```

---

## Task 2: Wire `portfolio.py` to StateDB — add 3 missing methods

**Files:**
- Modify: `portfolio.py`
- Modify: `tests/test_strategies.py` (add partial exit test)

The three methods `set_position_risk`, `_save_meta`, `_save_watermarks` are called throughout
the coordinator but were never defined. Every order placement threw an `AttributeError` that was
silently caught, leaving `initial_risk=0.0` forever.

- [ ] **Step 1: Write the failing test for `set_position_risk` + partial exit**

Add to `tests/test_strategies.py` (or create `tests/test_portfolio.py`):

```python
import os
os.environ["BOT_STATE_DB"] = ":memory:"

import pytest
from unittest.mock import MagicMock, patch
from portfolio import PortfolioManager


def make_portfolio():
    config = {
        "risk": {
            "trailing_stop_pct": 0.025,
            "chandelier_atr_mult": 3.0,
            "partial_exit_enabled": True,
            "partial_exit_r": 1.2,
            "partial_exit_pct": 0.40,
            "second_partial_enabled": True,
            "second_partial_r": 2.5,
            "second_partial_pct": 0.30,
        },
        "screener": {"crypto_risk": {"trailing_stop_pct": 0.05}}
    }
    broker = MagicMock()
    broker.get_positions.return_value = []
    pm = PortfolioManager(config, broker)
    return pm


def test_set_position_risk_stores_initial_risk():
    pm = make_portfolio()
    pm.set_position_risk("AAPL", entry_price=150.0, stop_loss=148.0, qty=10)
    meta = pm.position_meta.get("AAPL", {})
    # initial_risk = |150 - 148| * 10 = 20.0
    assert meta["initial_risk"] == pytest.approx(20.0)


def test_set_position_risk_persists_to_db():
    pm = make_portfolio()
    pm.set_position_risk("AAPL", entry_price=150.0, stop_loss=148.0, qty=10)
    pos = pm.db.get_position("AAPL")
    assert pos is not None
    assert pos["initial_risk"] == pytest.approx(20.0)


def test_partial_exit_fires_at_1_2R():
    pm = make_portfolio()
    # Entry at 100, stop at 90 → risk/share = 10, initial_risk = 100
    pm.set_position_risk("TSLA", entry_price=100.0, stop_loss=90.0, qty=10)
    pm.position_meta["TSLA"]["opened_at"] = "2026-04-13T10:00:00"

    positions = {
        "TSLA": {
            "qty": 10, "entry_price": 100.0, "current_price": 112.1,
            "market_value": 1121.0, "unrealized_pl": 121.0,
            "unrealized_plpc": 0.121, "side": "long"
        }
    }
    prices = {"TSLA": 112.1}
    to_close, partials = pm.check_trailing_stops(positions, prices)
    # 112.1 - 100 = 12.1 / 10 = 1.21R ≥ 1.2R threshold
    assert any(p["symbol"] == "TSLA" for p in partials), \
        "Expected partial exit at 1.2R but none fired"
```

- [ ] **Step 2: Run test — verify it fails**

```
pytest tests/test_portfolio.py -v
```
Expected: `AttributeError: 'PortfolioManager' object has no attribute 'set_position_risk'`

- [ ] **Step 3: Add imports and `StateDB` to `portfolio.py` `__init__`**

At the top of `portfolio.py`, replace:
```python
from state import load_state, save_state
from broker import CRYPTO_SYMBOLS
```
with:
```python
from state_db import StateDB
from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS
```

In `PortfolioManager.__init__`, replace the block that loads from `load_state()`:
```python
# OLD (remove these lines):
state = load_state()
self.high_watermarks = state.get("high_watermarks", {})
self.low_watermarks = state.get("low_watermarks", {})
self.position_meta = state.get("position_meta", {})
migrated_meta = {}
for k, v in self.position_meta.items():
    migrated_meta[_normalize_symbol(k)] = v
self.position_meta = migrated_meta
```
```python
# NEW:
self.db = StateDB()
self.db.migrate_from_json()  # no-op after first run

# Restore watermarks
wm = self.db.get_watermarks()
self.high_watermarks = {sym: d.get("high_watermark", 0.0) or 0.0
                        for sym, d in wm.items() if d.get("high_watermark") is not None}
self.low_watermarks  = {sym: d.get("low_watermark", 0.0) or 0.0
                        for sym, d in wm.items() if d.get("low_watermark") is not None}

# Restore position meta
self.position_meta = {}
for pos in self.db.get_all_positions():
    sym = _normalize_symbol(pos["symbol"])
    strats = pos.get("strategies")
    if isinstance(strats, str):
        import json as _json
        try:
            strats = _json.loads(strats)
        except Exception:
            strats = []
    self.position_meta[sym] = {
        "opened_at":           pos.get("opened_at", ""),
        "entry_price":         pos.get("entry_price", 0.0),
        "initial_risk":        pos.get("initial_risk", 0.0),
        "stop_loss":           pos.get("stop_loss"),
        "take_profit":         pos.get("take_profit"),
        "strategies":          strats or [],
        "original_qty":        pos.get("original_qty"),
        "side":                pos.get("side", "buy"),
        "partial_done":        bool(pos.get("partial_done", 0)),
        "second_partial_done": bool(pos.get("second_partial_done", 0)),
        "breakeven_armed":     bool(pos.get("breakeven_armed", 0)),
    }
```

- [ ] **Step 4: Add the three missing methods to `PortfolioManager`**

Add these three methods anywhere inside the `PortfolioManager` class (e.g., after `get_position`):

```python
def set_position_risk(self, symbol: str, entry_price: float,
                      stop_loss: float, qty: float):
    """Compute and persist initial_risk = |entry - stop| * qty.

    Called immediately after an order is placed so that partial exits
    (1.2R, 2.5R) have a valid reference risk to measure against.
    """
    sym = _normalize_symbol(symbol)
    initial_risk = abs(entry_price - stop_loss) * qty
    self.db.upsert_position(sym,
                            entry_price=entry_price,
                            initial_risk=initial_risk,
                            original_qty=qty)
    meta = self.position_meta.setdefault(sym, {})
    meta["initial_risk"] = initial_risk
    meta["entry_price"]  = entry_price
    meta["original_qty"] = qty
    log.info(f"Position risk set: {sym} initial_risk=${initial_risk:.2f}")

def _save_meta(self):
    """Persist all in-memory position_meta entries to SQLite."""
    import json as _json
    for sym, meta in self.position_meta.items():
        strats = meta.get("strategies", [])
        strats_json = _json.dumps(strats) if isinstance(strats, list) else (strats or "[]")
        self.db.upsert_position(sym,
            entry_price         = meta.get("entry_price"),
            initial_risk        = meta.get("initial_risk", 0.0),
            stop_loss           = meta.get("stop_loss"),
            take_profit         = meta.get("take_profit"),
            strategies          = strats_json,
            opened_at           = meta.get("opened_at", ""),
            original_qty        = meta.get("original_qty"),
            side                = meta.get("side", "buy"),
            partial_done        = int(meta.get("partial_done", False)),
            second_partial_done = int(meta.get("second_partial_done", False)),
            breakeven_armed     = int(meta.get("breakeven_armed", False)),
        )

def _save_watermarks(self):
    """Persist all in-memory high/low watermarks to SQLite."""
    all_syms = set(self.high_watermarks) | set(self.low_watermarks)
    for sym in all_syms:
        self.db.upsert_watermark(
            sym,
            high=self.high_watermarks.get(sym),
            low=self.low_watermarks.get(sym),
        )
```

- [ ] **Step 5: Update `execute_exits` to use `db.delete_position` after recording**

In `execute_exits`, after `self.position_meta.pop(sym, None)`, add:
```python
self.db.delete_position(sym)
```

- [ ] **Step 6: Run tests — verify they pass**

```
pytest tests/test_portfolio.py -v
```
Expected: All green.

- [ ] **Step 7: Commit**

```bash
git add portfolio.py tests/test_portfolio.py
git commit -m "fix: add set_position_risk, _save_meta, _save_watermarks to PortfolioManager; wire to SQLite"
```

---

## Task 3: Migrate `tracker.py` to SQLite

**Files:**
- Modify: `tracker.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tracker.py` (create if needed):

```python
import os
os.environ["BOT_STATE_DB"] = ":memory:"

from tracker import TradeTracker


def test_record_trade_persists():
    t = TradeTracker()
    t.record_trade("AAPL", "buy", 10, 150.0, 155.0,
                   reason="trailing_stop", risk_dollars=20.0,
                   strategies=["momentum"])
    assert len(t.trades) == 1
    assert t.trades[0]["pnl"] == pytest.approx(50.0)
    # Reload from DB — should survive restart
    t2 = TradeTracker()
    assert len(t2.trades) == 1
    assert t2.trades[0]["strategies"] == ["momentum"]
```

- [ ] **Step 2: Run test — verify it fails**

```
pytest tests/test_tracker.py::test_record_trade_persists -v
```
Expected: FAIL — `TradeTracker` still uses `trades.json`.

- [ ] **Step 3: Update `tracker.py`**

Replace the `_load` and `_save` methods and `__init__`:

```python
# At top of tracker.py, replace the TRADES_FILE constant and json/os imports:
from state_db import StateDB

# In __init__:
def __init__(self):
    self.db = StateDB()
    self.trades = self._load()
    if self.trades:
        log.info(f"Loaded {len(self.trades)} historical trades from SQLite")

def _load(self) -> list[dict]:
    return self.db.get_trades()

def _save(self):
    pass  # writes happen in record_trade directly
```

In `record_trade`, replace `self.trades.append(trade)` and `self._save()` with:
```python
self.trades.append(trade)
self.db.record_trade(
    symbol=symbol,
    side=side,
    qty=qty,
    entry_price=entry_price,
    exit_price=exit_price,
    pnl=round(pnl, 2),
    reason=reason,
    risk_dollars=risk_dollars if risk_dollars > 0 else 0.0,
    strategies=strategies or [],
    opened_at="",  # not stored at trade time currently
    closed_at=datetime.now().isoformat(),
)
```

- [ ] **Step 4: Run test — verify it passes**

```
pytest tests/test_tracker.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tracker.py tests/test_tracker.py
git commit -m "fix: migrate TradeTracker from trades.json to SQLite"
```

---

## Task 4: Migrate `state.py` and `watcher.py` pending signals to SQLite

**Files:**
- Modify: `state.py`
- Modify: `watcher.py`

- [ ] **Step 1: Update `state.py`**

Replace the entire file content with a thin delegation layer:

```python
"""Thin compatibility shim — delegates to StateDB.

Callers that still use load_state()/save_state() (e.g. risk.py) continue to
work unchanged. The underlying storage is now SQLite via StateDB.
"""
from state_db import StateDB

_db = StateDB()

_STATE_KEYS = ("peak_equity", "starting_equity", "high_watermarks",
               "low_watermarks", "position_meta")


def load_state() -> dict:
    """Return a dict of all known state keys from SQLite."""
    return {k: _db.get_state(k) for k in _STATE_KEYS
            if _db.get_state(k) is not None}


def save_state(state: dict):
    """Persist scalar values from state dict to SQLite."""
    for k, v in state.items():
        if k in _STATE_KEYS and not isinstance(v, (dict, list)):
            _db.set_state(k, v)
```

- [ ] **Step 2: Update `watcher.py` pending signal persistence**

Find the functions `_load_pending_state` and `_save_pending_state` near the bottom of `watcher.py`.
Replace them entirely:

```python
# ── Pending signal persistence (SQLite) ──────────────────

def _load_pending_state(symbol: str) -> bool:
    from state_db import StateDB
    return StateDB().get_pending_signal(symbol)


def _save_pending_state(symbol: str, has_signal: bool):
    from state_db import StateDB
    StateDB().set_pending_signal(symbol, has_signal)
```

Also in `watcher.py`, replace:
```python
from data import CRYPTO_SYMBOLS
```
with:
```python
from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS
```

- [ ] **Step 3: Run smoke test**

```
python test_boot.py
```
Expected: coordinator boots, IB connects, no `AttributeError` or `ImportError`.

- [ ] **Step 4: Commit**

```bash
git add state.py watcher.py
git commit -m "fix: migrate state.py and watcher pending signals to SQLite; fix watcher CRYPTO_SYMBOLS"
```

---

## Task 5: Fix `ib_broker.py` bracket order

**Files:**
- Modify: `ib_broker.py`

The current call `self._ib.bracketOrder(ib_side, req.qty, limitPrice=req.take_profit, stopLossPrice=req.stop_loss)`
omits the required `takeProfitPrice` positional argument (TypeError silently caught) and misuses
`limitPrice` as the TP price.

- [ ] **Step 1: Write a unit test**

Create `tests/test_ib_broker.py`:

```python
from unittest.mock import MagicMock, call, patch
from base_broker import OrderRequest


def make_broker():
    config = {
        "ib": {"host": "127.0.0.1", "port": 4002, "client_id": 1, "timeout_sec": 10},
        "screener": {"universe": [], "crypto": []},
        "futures": {"contracts": []},
    }
    with patch("ib_insync.IB") as MockIB:
        ib_instance = MagicMock()
        MockIB.return_value = ib_instance
        ib_instance.isConnected.return_value = True

        # Suppress ContractManager resolution during test
        with patch("contract_manager.ContractManager"):
            from ib_broker import IBBroker
            broker = IBBroker.__new__(IBBroker)
            broker._ib = ib_instance
            broker._config = config
            broker._lock = __import__("threading").Lock()
            broker._bad_contracts = {}
            broker._contracts = MagicMock()
    return broker, ib_instance


def test_submit_bracket_order_places_three_orders():
    """submit_order with take_profit and stop_loss must place exactly 3 orders:
    parent market, TP limit, SL stop — in that order."""
    broker, ib = make_broker()

    parent_trade = MagicMock()
    parent_trade.order.orderId = 101
    ib.placeOrder.return_value = parent_trade

    # Resolve contract to a dummy
    broker._resolve_contract = MagicMock(return_value=MagicMock())

    req = OrderRequest(symbol="AAPL", qty=10, side="buy",
                       take_profit=156.0, stop_loss=148.0)
    broker.submit_order(req)

    assert ib.placeOrder.call_count == 3, \
        f"Expected 3 placeOrder calls (parent+TP+SL), got {ib.placeOrder.call_count}"

    calls = ib.placeOrder.call_args_list
    # First call = parent (MarketOrder)
    from ib_insync import MarketOrder
    parent_order = calls[0][0][1]
    assert isinstance(parent_order, MarketOrder)
    assert parent_order.transmit is False

    # Second call = TP (LimitOrder)
    from ib_insync import LimitOrder
    tp_order = calls[1][0][1]
    assert isinstance(tp_order, LimitOrder)
    assert tp_order.lmtPrice == 156.0
    assert tp_order.parentId == 101
    assert tp_order.transmit is False

    # Third call = SL (StopOrder), transmit=True
    from ib_insync import StopOrder
    sl_order = calls[2][0][1]
    assert isinstance(sl_order, StopOrder)
    assert sl_order.auxPrice == 148.0
    assert sl_order.parentId == 101
    assert sl_order.transmit is True
```

- [ ] **Step 2: Run test — verify it fails**

```
pytest tests/test_ib_broker.py -v
```
Expected: FAIL — either TypeError from bracketOrder or wrong call count.

- [ ] **Step 3: Replace the bracket block in `ib_broker.py`**

In `submit_order`, find the block:
```python
if req.take_profit and req.stop_loss:
    # Use ib.bracketOrder() ...
    bracket = self._ib.bracketOrder(...)
    ...
```

Replace the entire `if req.take_profit and req.stop_loss:` block with:

```python
if req.take_profit and req.stop_loss:
    from ib_insync import MarketOrder, LimitOrder, StopOrder
    tp_side = "SELL" if ib_side == "BUY" else "BUY"

    # 1. Parent market entry — hold until children are attached
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

    log.info(
        f"IB BRACKET: {ib_side} {req.qty} {req.symbol} "
        f"TP={req.take_profit} SL={req.stop_loss} "
        f"parentId={parent_id}"
    )
    return Order(
        id=str(parent_id),
        symbol=req.symbol,
        qty=req.qty,
        side=req.side,
        order_type="bracket",
        status="submitted",
    )
```

- [ ] **Step 4: Run test — verify it passes**

```
pytest tests/test_ib_broker.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ib_broker.py tests/test_ib_broker.py
git commit -m "fix: replace broken bracketOrder with manual market+TP+SL parentId chain"
```

---

## Task 6: Add `IB_CRYPTO_SYMBOLS` and fix all `CRYPTO_SYMBOLS` imports

**Files:**
- Modify: `ib_data.py`
- Modify: `risk.py`
- Modify: `live_trading.py`

- [ ] **Step 1: Add constant to `ib_data.py`**

After the existing imports, add at module level:

```python
# IB PAXOS supports only BTC and ETH (paper + live accounts)
# SOL, AVAX, LINK, DOGE are NOT available — do not add them here
IB_CRYPTO_SYMBOLS: frozenset[str] = frozenset({
    "BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD",
})
```

- [ ] **Step 2: Fix `risk.py`**

Replace:
```python
from broker import CRYPTO_SYMBOLS
```
with:
```python
from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS
```
(appears twice in `risk.py` — fix both)

- [ ] **Step 3: Fix `live_trading.py`**

In `_classify_symbol`, replace:
```python
_CRYPTO_SUFFIXES = {
    "BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD",
    "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD",
    "SOLUSD", "AVAXUSD", "LINKUSD", "DOGEUSD",
}
```
with:
```python
_CRYPTO_SUFFIXES = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}
```

- [ ] **Step 4: Run tests**

```
pytest tests/ -v -k "crypto or classifier"
```
Expected: All pass. Verify no imports from `broker` or `data` remain for CRYPTO_SYMBOLS.

- [ ] **Step 5: Commit**

```bash
git add ib_data.py risk.py live_trading.py
git commit -m "fix: add IB_CRYPTO_SYMBOLS constant; replace Alpaca-era CRYPTO_SYMBOLS in risk, live_trading"
```

---

## Task 7: Fix ML model feature mismatch and data leakage

**Files:**
- Modify: `ml_model.py`
- Modify: `tests/test_new_features.py` (or create `tests/test_ml_model.py`)

The model trains on binary flags + `r_multiple` (leakage) but predicts using float scores.
Fix: both methods use identical float strategy scores, no outcome-based features.

- [ ] **Step 1: Write tests**

Create `tests/test_ml_model.py`:

```python
from ml_model import MLMetaModel, STRATEGY_NAMES


def test_feature_vectors_same_length():
    """Training and prediction feature vectors must have identical length."""
    model = MLMetaModel()

    trade = {
        "strategies": ["momentum", "supertrend"],
        "pnl": 50.0,
        "strategy_scores": {"momentum": 0.6, "supertrend": 0.4},
        "num_agreeing": 2,
        "composite_score": 0.5,
    }
    live_features = {
        "strategy_scores": {"momentum": 0.6, "supertrend": 0.4},
        "num_agreeing": 2,
        "composite_score": 0.5,
    }

    train_vec = model._trade_to_features(trade)
    pred_vec  = model._build_feature_vector(live_features)

    assert train_vec is not None
    assert pred_vec is not None
    assert len(train_vec) == len(pred_vec), \
        f"Feature length mismatch: train={len(train_vec)}, predict={len(pred_vec)}"


def test_feature_vectors_same_type_range():
    """Both vectors should use float scores in [0, 1], not binary flags."""
    model = MLMetaModel()
    trade = {
        "strategies": ["momentum"],
        "strategy_scores": {"momentum": 0.7},
        "num_agreeing": 1,
        "composite_score": 0.7,
        "pnl": 10.0,
    }
    vec = model._trade_to_features(trade)
    # First 8 features = strategy scores — must be in [0, 1], not just 0 or 1
    assert vec[0] == 0.7  # momentum score, not binary 1


def test_no_r_multiple_in_training_features():
    """r_multiple must NOT appear in training features (look-ahead leakage)."""
    model = MLMetaModel()
    trade = {
        "strategies": ["momentum"],
        "strategy_scores": {"momentum": 0.7},
        "num_agreeing": 1,
        "composite_score": 0.7,
        "pnl": 100.0,
        "r_multiple": 5.0,  # This should NOT leak into training features
    }
    vec1 = model._trade_to_features(trade)
    trade["r_multiple"] = 0.5  # Change r_multiple — feature vec must NOT change
    vec2 = model._trade_to_features(trade)
    assert vec1 == vec2, "r_multiple is leaking into training features"
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_ml_model.py -v
```
Expected: FAIL on all three tests.

- [ ] **Step 3: Fix `ml_model.py`**

Replace `_trade_to_features` and `_build_feature_vector` with unified implementations:

```python
def _trade_to_features(self, trade: dict) -> list | None:
    """Convert a closed trade record to a feature vector.

    Uses the same feature space as _build_feature_vector so the trained
    model is valid at prediction time. No outcome-based features.
    """
    scores = trade.get("strategy_scores", {})
    # If strategy_scores not stored, fall back to binary presence
    strats = set(trade.get("strategies", []))
    if not strats and not scores:
        return None

    vec = []
    for s in STRATEGY_NAMES:
        # Prefer float score; fall back to binary presence flag
        if s in scores:
            vec.append(float(scores[s]))
        elif s in strats:
            vec.append(1.0)
        else:
            vec.append(0.0)

    vec.append(float(trade.get("num_agreeing", len(strats))))
    vec.append(float(trade.get("composite_score", 0.0)))
    return vec

def _build_feature_vector(self, features: dict) -> list | None:
    """Build feature vector from live signal data.

    Must produce identical shape and semantics as _trade_to_features.
    """
    vec = []
    scores = features.get("strategy_scores", {})
    for s in STRATEGY_NAMES:
        vec.append(float(scores.get(s, 0.0)))
    vec.append(float(features.get("num_agreeing", 0)))
    vec.append(float(features.get("composite_score", 0.0)))
    return vec
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_ml_model.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ml_model.py tests/test_ml_model.py
git commit -m "fix: unify ML train/predict feature vectors; remove r_multiple look-ahead leakage"
```

---

## Task 8: Fix `regime.py` — HMM caching + breadth sampling

**Files:**
- Modify: `regime.py`

- [ ] **Step 1: Write tests**

Add to `tests/test_regime.py`:

```python
import time
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from regime import RegimeFilter


def make_spy_df(n=250):
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 450 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "open": close - 0.5, "high": close + 1,
        "low": close - 1, "close": close, "volume": 1e6
    }, index=dates)


def test_hmm_not_refit_within_ttl():
    """HMM should only refit once per TTL window, not every call."""
    data = MagicMock()
    data.get_intraday_bars.return_value = make_spy_df()
    data.get_bars.return_value = {}

    rf = RegimeFilter(data)
    rf._hmm_refit_interval = 3600  # 1 hour TTL

    with patch.object(rf, '_fit_hmm', wraps=rf._fit_hmm if hasattr(rf, '_fit_hmm') else lambda df: None) as mock_fit:
        rf.get_regime()
        rf.get_regime()  # second call within TTL
        # HMM fit should only happen once
        assert mock_fit.call_count <= 1, \
            f"HMM refitted {mock_fit.call_count} times — expected at most 1 within TTL"


def test_breadth_sample_not_always_first_20():
    """Breadth sample must use random sampling, not universe[:20]."""
    universe = [f"SYM{i:03d}" for i in range(100)]  # 100 symbols
    data = MagicMock()
    data.get_intraday_bars.return_value = make_spy_df()
    data.get_bars.return_value = {}

    rf = RegimeFilter(data, universe=universe)

    samples_seen = set()
    for _ in range(5):
        rf._get_market_breadth()
        # Inspect which symbols were passed to get_bars
        if data.get_bars.called:
            call_args = data.get_bars.call_args[0][0]
            samples_seen.update(call_args)

    # If always slicing [:20], only SYM000-SYM019 would ever appear
    high_index_symbols = {s for s in samples_seen if int(s[3:]) >= 20}
    assert len(high_index_symbols) > 0, \
        "Breadth sample always picks first 20 symbols — random sampling not working"
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_regime.py::test_hmm_not_refit_within_ttl tests/test_regime.py::test_breadth_sample_not_always_first_20 -v
```

- [ ] **Step 3: Apply HMM caching to `regime.py`**

In `RegimeFilter.__init__`, add:
```python
self._hmm_last_fit: float = 0.0
self._hmm_refit_interval: int = 3600   # refit at most once per hour
self._hmm_last_features = None          # cached feature array for fast predict
```

Replace `_get_hmm_regime` to split into `_fit_hmm` (expensive) and fast-path predict:

```python
def _get_hmm_regime(self, df: pd.DataFrame) -> dict | None:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None

    if len(df) < 100:
        return None

    try:
        import time as _time
        close = df["close"].values
        returns = np.diff(np.log(close))
        vol_window = 20
        if len(returns) < vol_window + 10:
            return None
        vol = pd.Series(returns).rolling(vol_window).std().values
        valid = ~np.isnan(vol)
        returns_valid = returns[valid]
        vol_valid = vol[valid]
        if len(returns_valid) < 60:
            return None

        features = np.column_stack([returns_valid, vol_valid])
        self._hmm_last_features = features

        needs_refit = (
            self._hmm_model is None
            or _time.time() - self._hmm_last_fit > self._hmm_refit_interval
        )
        if needs_refit:
            self._fit_hmm(features)

        if self._hmm_model is None:
            return None

        state_seq = self._hmm_model.predict(features)
        current_state = state_seq[-1]
        probs = self._hmm_model.predict_proba(features[-1:].reshape(1, -1))[0]
        regime = self._hmm_state_map.get(current_state, "chop")
        confidence = float(probs[current_state])
        return {"state": regime, "confidence": confidence,
                "probs": [float(p) for p in probs]}
    except Exception as e:
        log.warning(f"HMM regime detection failed: {e}")
        return None

def _fit_hmm(self, features: np.ndarray):
    """Fit (or refit) the HMM model. Called at most once per TTL."""
    import time as _time
    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=3, covariance_type="diag",
                            n_iter=50, random_state=42)
        model.fit(features)
        mean_returns = model.means_[:, 0]
        sorted_indices = np.argsort(mean_returns)
        self._hmm_state_map = {
            sorted_indices[0]: "bear",
            sorted_indices[1]: "chop",
            sorted_indices[2]: "bull",
        }
        self._hmm_model = model
        self._hmm_last_fit = _time.time()
        log.info("HMM model refit complete")
    except Exception as e:
        log.warning(f"HMM fit failed: {e}")
```

- [ ] **Step 4: Apply breadth fix in `_get_market_breadth`**

Replace:
```python
sample = self._universe[:20]
```
with:
```python
import random as _random
sample_size = min(30, len(self._universe))
sample = _random.sample(self._universe, sample_size)
```

- [ ] **Step 5: Run tests — verify they pass**

```
pytest tests/test_regime.py -v
```

- [ ] **Step 6: Commit**

```bash
git add regime.py tests/test_regime.py
git commit -m "fix: cache HMM model (refit max 1/hour); stratify breadth sample across universe"
```

---

## Task 9: Add bear veto to `watcher.py`

**Files:**
- Modify: `watcher.py`
- Modify: `coordinator.py` (pass regime to watcher)

The bear veto must go in `watcher.py` because `StrategyRouter` bypasses `strategy_selector.py` entirely. The coordinator passes the current macro regime to each watcher via a shared attribute.

- [ ] **Step 1: Add `_macro_regime` attribute to `StockWatcher`**

In `StockWatcher.__init__`, add:
```python
self._macro_regime: str = "bull"   # updated by coordinator each cycle
```

- [ ] **Step 2: Apply bear veto inside `_analyze` after strategy selection**

After the block that sets `selection["strategies"]` (around line 215 in watcher.py), add:

```python
# ── Bear veto: suppress long-only strategies in strong bear regime ──
# StrategyRouter provides fixed weights; this is the only place to apply
# regime-specific dampening for the stock universe.
macro = getattr(self, "_macro_regime", "bull")
if macro == "bear":
    import pandas as _pd
    adx_now = ctx.get("adx", 0)
    if adx_now > 30:
        for strat in ("momentum", "gap", "breakout"):
            if strat in selection["strategies"]:
                selection["strategies"][strat] = 0.0
        _total = sum(selection["strategies"].values())
        if _total > 0:
            selection["strategies"] = {
                k: round(v / _total, 4)
                for k, v in selection["strategies"].items()
            }
        self.log.debug(
            f"Bear veto applied (ADX={adx_now:.0f}): "
            f"momentum/gap/breakout weights zeroed"
        )
```

- [ ] **Step 3: Update coordinator to broadcast regime to watchers**

In `coordinator.py`, in `_coordinator_cycle`, after `regime = self.regime.get_regime()` (around line 396), add:

```python
# Broadcast macro regime to all watchers for bear veto
for w in self.watchers.values():
    w._macro_regime = regime["regime"]
```

- [ ] **Step 4: Write test**

```python
def test_bear_veto_zeroes_momentum_gap_breakout():
    """In bear regime with ADX>30, momentum/gap/breakout weights must be 0."""
    import pandas as pd
    import numpy as np
    from watcher import StockWatcher

    config = {
        "strategies": {
            "momentum": {"rsi_period":14,"rsi_overbought":70,"rsi_oversold":30,
                         "roc_period":10,"ema_fast":8,"ema_slow":21,"weight":0.25},
            "supertrend": {"atr_period":10,"multiplier":3.0,"weight":0.25},
        },
        "signals": {"min_composite_score":0.25,"min_agreeing_strategies":3,
                    "entry_timeframe":"5Min","trend_timeframe":"1Day",
                    "intraday_lookback_days":5,"min_crypto_agreeing":2,
                    "min_crypto_score":0.15},
        "screener": {"crypto":[],"crypto_risk":{}},
        "futures": {"contracts":[]},
    }
    data = MagicMock()
    watcher = StockWatcher("AAPL", config, data, strategies={"momentum":0.5,"supertrend":0.5})
    watcher._macro_regime = "bear"

    # Build a fake daily df with high ADX
    n = 60
    close = 150 + np.cumsum(np.random.randn(n) - 0.1)
    df = pd.DataFrame({
        "open": close-0.5, "high": close+1, "low": close-1,
        "close": close, "volume": np.ones(n)*1e6
    }, index=pd.date_range("2025-01-01", periods=n, freq="B"))

    # Patch get_trend_context to return bear+high ADX
    with patch("watcher.get_trend_context", return_value={
        "adx": 35, "direction": "down", "trending": True, "strong_trend": True,
        "above_ema_200": False, "above_vwap": False, "di_plus": 10, "di_minus": 30
    }):
        selection = {"strategies": {"momentum":0.5,"supertrend":0.5}, "regime":"trending", "reason":"test"}
        watcher._apply_bear_veto(selection, {"adx":35})  # call if extracted to method
        assert selection["strategies"].get("momentum", 0) == 0.0
```

Note: if bear veto logic is inline (not a separate method), test it via `_analyze` with a mocked data fetcher.

- [ ] **Step 5: Run tests**

```
pytest tests/ -v -k "bear_veto"
```

- [ ] **Step 6: Commit**

```bash
git add watcher.py coordinator.py
git commit -m "feat: add bear veto in watcher._analyze — zero momentum/gap/breakout when bear+ADX>30"
```

---

## Task 10: Remove dead code in `filters.py`, fix minor issues

**Files:**
- Modify: `filters.py`
- Modify: `strategy_selector.py`
- Modify: `coordinator.py`

- [ ] **Step 1: Remove dead code from `filters.py`**

Delete the following from `filters.py`:
- The constant: `CONFIRMATION_FILE = os.path.join(os.path.dirname(__file__), "pending_signals.json")`
- The method: `filter_confirmed(self, opportunities, bars)` (lines ~119-146)
- The method: `_load_pending(self)` (lines ~361-368)
- The method: `_save_pending(self, data)` (lines ~370-375)
- In `__init__`, remove: `self._pending_signals = self._load_pending()`

Also remove the `import json` and `import os` lines if they are no longer used (check — `json` may still be used elsewhere in the file; `os` is used for `time` and path operations so keep it).

- [ ] **Step 2: Fix `strategy_selector.py` regime label**

Find line ~176:
```python
"regime": "breakdown",
```
Change to:
```python
"regime": "breakout",
```

- [ ] **Step 3: Fix duplicate key in `coordinator.py`**

Find in `get_all_watcher_states` (around line 891-892):
```python
"regime": s.regime,
"regime": s.regime,
```
Remove the duplicate line so only one `"regime": s.regime` remains.

- [ ] **Step 4: Verify no test regressions**

```
pytest tests/ -v
```
Expected: All existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add filters.py strategy_selector.py coordinator.py
git commit -m "fix: remove dead filter_confirmed code; fix breakdown→breakout regime label; remove duplicate dict key"
```

---

## Task 11: Full regression test + smoke test

**Files:** None modified — verification only.

- [ ] **Step 1: Run full test suite**

```
pytest tests/ -v --tb=short 2>&1 | tail -40
```
Expected: All tests green. Note any failures and fix before proceeding.

- [ ] **Step 2: Run boot smoke test**

```
python test_boot.py
```
Expected: Coordinator initializes, IB connects (or times out cleanly), no `AttributeError`,
no `ImportError`, no `TypeError`.

- [ ] **Step 3: Verify SQLite file created**

```
python -c "from state_db import StateDB; db = StateDB(); print('Tables:', db._conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())"
```
Expected output: `Tables: [('positions',), ('watermarks',), ('trades',), ('bot_state',), ('pending_signals',)]`

- [ ] **Step 4: Verify no remaining references to old JSON state**

```
grep -rn "state\.json\|trades\.json\|watcher_pending\.json" --include="*.py" .
```
Expected: Only `state_db.py` migration method references these (for reading during migration).
No other Python file should open them.

- [ ] **Step 5: Verify no remaining Alpaca CRYPTO_SYMBOLS imports**

```
grep -rn "from broker import CRYPTO_SYMBOLS\|from data import CRYPTO_SYMBOLS" --include="*.py" .
```
Expected: Zero results.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: full regression verified — all 13 bugs fixed, SQLite migration complete"
```

---

## Summary of All Commits

| Task | Commit message |
|------|----------------|
| 1 | `feat: add StateDB — SQLite state layer replacing all JSON state files` |
| 2 | `fix: add set_position_risk, _save_meta, _save_watermarks to PortfolioManager; wire to SQLite` |
| 3 | `fix: migrate TradeTracker from trades.json to SQLite` |
| 4 | `fix: migrate state.py and watcher pending signals to SQLite; fix watcher CRYPTO_SYMBOLS` |
| 5 | `fix: replace broken bracketOrder with manual market+TP+SL parentId chain` |
| 6 | `fix: add IB_CRYPTO_SYMBOLS constant; replace Alpaca-era CRYPTO_SYMBOLS in risk, live_trading` |
| 7 | `fix: unify ML train/predict feature vectors; remove r_multiple look-ahead leakage` |
| 8 | `fix: cache HMM model (refit max 1/hour); stratify breadth sample across universe` |
| 9 | `feat: add bear veto in watcher._analyze — zero momentum/gap/breakout when bear+ADX>30` |
| 10 | `fix: remove dead filter_confirmed code; fix breakdown→breakout regime label; remove duplicate dict key` |
| 11 | `chore: full regression verified — all 13 bugs fixed, SQLite migration complete` |
