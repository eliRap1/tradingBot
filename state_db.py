"""SQLite-backed state storage for bot state, trades, and pending signals."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from utils import setup_logger

log = setup_logger("state_db")

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
    breakeven_armed INTEGER DEFAULT 0,
    check_count INTEGER DEFAULT 0
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
    pnl_pct REAL,
    reason TEXT,
    risk_dollars REAL,
    r_multiple REAL,
    strategies TEXT,
    opened_at TEXT,
    closed_at TEXT,
    edge_snapshot TEXT
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

_POSITION_FIELDS = {
    "entry_price",
    "initial_risk",
    "stop_loss",
    "take_profit",
    "strategies",
    "opened_at",
    "original_qty",
    "side",
    "partial_done",
    "second_partial_done",
    "breakeven_armed",
    "check_count",
}


class StateDB:
    def __init__(self, db_path: str | None = None, base_dir: str | None = None):
        self._base_dir = base_dir or os.path.dirname(__file__)
        self._path = db_path or os.environ.get("BOT_STATE_DB") or os.path.join(self._base_dir, "bot_state.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_SQL)
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        cols = [r["name"] for r in self._conn.execute("PRAGMA table_info(trades)").fetchall()]
        if "edge_snapshot" not in cols:
            self._conn.execute("ALTER TABLE trades ADD COLUMN edge_snapshot TEXT")

    def _encode_json(self, value: Any) -> str:
        return json.dumps(value)

    def _decode_json(self, value: Any, default: Any = None) -> Any:
        if value in (None, ""):
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def _normalize_position_row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        result["strategies"] = self._decode_json(result.get("strategies"), [])
        for key in ("partial_done", "second_partial_done", "breakeven_armed"):
            result[key] = bool(result.get(key, 0))
        result["check_count"] = int(result.get("check_count") or 0)
        return result

    def upsert_position(self, symbol: str, **fields):
        clean_fields = {k: v for k, v in fields.items() if k in _POSITION_FIELDS}
        if "strategies" in clean_fields and isinstance(clean_fields["strategies"], list):
            clean_fields["strategies"] = self._encode_json(clean_fields["strategies"])

        with self._lock:
            existing = self._conn.execute(
                "SELECT symbol FROM positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            if existing is None:
                cols = ["symbol"] + list(clean_fields.keys())
                vals = [symbol] + list(clean_fields.values())
                placeholders = ", ".join("?" for _ in cols)
                self._conn.execute(
                    f"INSERT INTO positions ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
            elif clean_fields:
                set_clause = ", ".join(f"{key} = ?" for key in clean_fields)
                vals = list(clean_fields.values()) + [symbol]
                self._conn.execute(
                    f"UPDATE positions SET {set_clause} WHERE symbol = ?",
                    vals,
                )
            self._conn.commit()

    def replace_positions(self, position_meta: dict[str, dict]):
        with self._lock:
            self._conn.execute("DELETE FROM positions")
            for symbol, meta in position_meta.items():
                clean_meta = {k: v for k, v in meta.items() if k in _POSITION_FIELDS}
                if "strategies" in clean_meta and isinstance(clean_meta["strategies"], list):
                    clean_meta["strategies"] = self._encode_json(clean_meta["strategies"])
                cols = ["symbol"] + list(clean_meta.keys())
                vals = [symbol] + list(clean_meta.values())
                placeholders = ", ".join("?" for _ in cols)
                self._conn.execute(
                    f"INSERT INTO positions ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
            self._conn.commit()

    def get_position(self, symbol: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM positions WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        return self._normalize_position_row(row)

    def delete_position(self, symbol: str):
        with self._lock:
            self._conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            self._conn.commit()

    def get_all_positions(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM positions ORDER BY symbol").fetchall()
        return [self._normalize_position_row(row) for row in rows]

    def upsert_watermark(self, symbol: str, high: float | None = None, low: float | None = None):
        with self._lock:
            existing = self._conn.execute(
                "SELECT symbol FROM watermarks WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO watermarks (symbol, high_watermark, low_watermark) VALUES (?, ?, ?)",
                    (symbol, high, low),
                )
            else:
                if high is not None:
                    self._conn.execute(
                        "UPDATE watermarks SET high_watermark = ? WHERE symbol = ?",
                        (high, symbol),
                    )
                if low is not None:
                    self._conn.execute(
                        "UPDATE watermarks SET low_watermark = ? WHERE symbol = ?",
                        (low, symbol),
                    )
            self._conn.commit()

    def replace_watermarks(self, high_watermarks: dict[str, float], low_watermarks: dict[str, float]):
        symbols = set(high_watermarks) | set(low_watermarks)
        with self._lock:
            self._conn.execute("DELETE FROM watermarks")
            for symbol in symbols:
                self._conn.execute(
                    "INSERT INTO watermarks (symbol, high_watermark, low_watermark) VALUES (?, ?, ?)",
                    (symbol, high_watermarks.get(symbol), low_watermarks.get(symbol)),
                )
            self._conn.commit()

    def get_watermarks(self) -> dict[str, dict]:
        rows = self._conn.execute("SELECT * FROM watermarks").fetchall()
        return {row["symbol"]: dict(row) for row in rows}

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str = "",
        risk_dollars: float = 0.0,
        strategies=None,
        opened_at: str = "",
        closed_at: str = "",
        pnl_pct: float | None = None,
        r_multiple: float | None = None,
        edge_snapshot=None,
    ):
        strategies_json = strategies if isinstance(strategies, str) else self._encode_json(strategies or [])
        edge_json = edge_snapshot if isinstance(edge_snapshot, str) else (
            self._encode_json(edge_snapshot) if edge_snapshot is not None else None
        )
        if pnl_pct is None:
            is_long = side in ("buy", "long")
            pnl_pct = ((exit_price - entry_price) / entry_price) if is_long and entry_price else (
                (entry_price - exit_price) / entry_price if entry_price else 0.0
            )
        if r_multiple is None:
            r_multiple = (pnl / risk_dollars) if risk_dollars else None

        with self._lock:
            self._conn.execute(
                """INSERT INTO trades (
                    symbol, side, qty, entry_price, exit_price, pnl, pnl_pct, reason,
                    risk_dollars, r_multiple, strategies, opened_at, closed_at,
                    edge_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol,
                    side,
                    qty,
                    entry_price,
                    exit_price,
                    pnl,
                    pnl_pct,
                    reason,
                    risk_dollars if risk_dollars else None,
                    r_multiple,
                    strategies_json,
                    opened_at,
                    closed_at,
                    edge_json,
                ),
            )
            self._conn.commit()

    def get_trades(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM trades ORDER BY id").fetchall()
        result = []
        for row in rows:
            trade = dict(row)
            trade["strategies"] = self._decode_json(trade.get("strategies"), [])
            trade["edge_snapshot"] = self._decode_json(trade.get("edge_snapshot"), {})
            result.append(trade)
        return result

    def set_state(self, key: str, value: Any):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                (key, self._encode_json(value)),
            )
            self._conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM bot_state WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return default
        return self._decode_json(row["value"], default)

    def set_pending_signal(self, symbol: str, prev_signal: bool):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pending_signals (symbol, prev_signal) VALUES (?, ?)",
                (symbol, int(bool(prev_signal))),
            )
            self._conn.commit()

    def get_pending_signal(self, symbol: str) -> bool:
        row = self._conn.execute(
            "SELECT prev_signal FROM pending_signals WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        return bool(row["prev_signal"]) if row else False

    def migrate_from_json(self):
        state_path = os.path.join(self._base_dir, "state.json")
        trades_path = os.path.join(self._base_dir, "trades.json")
        pending_path = os.path.join(self._base_dir, "watcher_pending.json")
        migrated = []

        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if "peak_equity" in state:
                    self.set_state("peak_equity", state["peak_equity"])
                for symbol, high in state.get("high_watermarks", {}).items():
                    self.upsert_watermark(symbol, high=float(high))
                for symbol, low in state.get("low_watermarks", {}).items():
                    self.upsert_watermark(symbol, low=float(low))
                for symbol, meta in state.get("position_meta", {}).items():
                    self.upsert_position(symbol, **meta)
                os.replace(state_path, state_path + ".bak")
                migrated.append("state.json")
            except Exception as e:
                log.error(f"Migration of state.json failed: {e}")

        if os.path.exists(trades_path):
            try:
                with open(trades_path, "r", encoding="utf-8") as f:
                    trades = json.load(f)
                for trade in trades:
                    self.record_trade(
                        symbol=trade.get("symbol", ""),
                        side=trade.get("side", "buy"),
                        qty=float(trade.get("qty", 0) or 0),
                        entry_price=float(trade.get("entry_price", 0) or 0),
                        exit_price=float(trade.get("exit_price", 0) or 0),
                        pnl=float(trade.get("pnl", 0) or 0),
                        pnl_pct=trade.get("pnl_pct"),
                        reason=trade.get("reason", ""),
                        risk_dollars=float(trade.get("risk_dollars", 0) or 0),
                        r_multiple=trade.get("r_multiple"),
                        strategies=trade.get("strategies", []),
                        opened_at=trade.get("opened_at", ""),
                        closed_at=trade.get("closed_at", ""),
                    )
                os.replace(trades_path, trades_path + ".bak")
                migrated.append("trades.json")
            except Exception as e:
                log.error(f"Migration of trades.json failed: {e}")

        if os.path.exists(pending_path):
            try:
                with open(pending_path, "r", encoding="utf-8") as f:
                    pending = json.load(f)
                for symbol, value in pending.items():
                    prev_signal = value if isinstance(value, bool) else bool(value.get("prev_signal", False))
                    self.set_pending_signal(symbol, prev_signal)
                os.replace(pending_path, pending_path + ".bak")
                migrated.append("watcher_pending.json")
            except Exception as e:
                log.error(f"Migration of watcher_pending.json failed: {e}")

        if migrated:
            log.info(f"SQLite migration complete: {', '.join(migrated)}")
