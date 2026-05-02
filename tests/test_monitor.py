"""Unit tests for runtime.monitor health metrics."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from runtime.monitor import (
    BASELINE,
    THRESHOLDS,
    _profit_factor,
    _win_rate,
    _expectancy,
    _loss_streaks,
    _drawdown,
    compute_health,
)


def _seed_db(path, trades, positions=None):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
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
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, side TEXT, qty REAL,
            entry_price REAL, exit_price REAL,
            pnl REAL, pnl_pct REAL, reason TEXT,
            risk_dollars REAL, r_multiple REAL,
            strategies TEXT, opened_at TEXT, closed_at TEXT
        );
        """
    )
    for t in trades:
        conn.execute(
            "INSERT INTO trades (symbol, side, qty, entry_price, exit_price, "
            "pnl, pnl_pct, reason, risk_dollars, r_multiple, strategies, "
            "opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                t.get("symbol", "AAA"),
                t.get("side", "long"),
                t.get("qty", 100),
                t.get("entry_price", 100.0),
                t.get("exit_price", 101.0),
                t.get("pnl", 0.0),
                t.get("pnl_pct", 0.0),
                t.get("reason", "tp"),
                t.get("risk_dollars", 100.0),
                t.get("r_multiple", 1.0),
                t.get("strategies", "[]"),
                t.get("opened_at", "2026-04-01T10:00:00"),
                t.get("closed_at", "2026-04-01T15:00:00"),
            ),
        )
    if positions:
        for p in positions:
            conn.execute(
                "INSERT INTO positions (symbol, entry_price, side, opened_at) "
                "VALUES (?,?,?,?)",
                (p["symbol"], p["entry_price"], p["side"], p["opened_at"]),
            )
    conn.commit()
    conn.close()


def test_profit_factor_basic():
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}]
    assert _profit_factor(trades) == 6.0


def test_profit_factor_zero_losses_inf():
    trades = [{"pnl": 100}, {"pnl": 50}]
    assert _profit_factor(trades) == float("inf")


def test_profit_factor_empty():
    assert _profit_factor([]) == 0.0


def test_win_rate_basic():
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}]
    assert _win_rate(trades) == round(2 / 3 * 100, 2)


def test_expectancy_basic():
    trades = [{"pnl": 100}, {"pnl": -50}]
    assert _expectancy(trades) == 25.0


def test_loss_streaks():
    trades = [{"pnl": x} for x in [10, -5, -5, -5, 10, -2, -2]]
    cur, longest = _loss_streaks(trades)
    assert cur == 2
    assert longest == 3


def test_drawdown():
    trades = [{"pnl": x} for x in [1000, -2000, -1000, 500]]
    peak, trough, mdd = _drawdown(trades, starting_equity=100_000.0)
    assert peak == 101_000.0
    assert mdd > 0


def test_compute_health_smoke(tmp_path):
    db = tmp_path / "test.db"
    now = datetime.now(timezone.utc)
    trades = [
        {"pnl": 500, "r_multiple": 1.0,
         "closed_at": (now - timedelta(days=2)).isoformat(timespec="seconds")},
        {"pnl": -100, "r_multiple": -1.0,
         "closed_at": (now - timedelta(days=1)).isoformat(timespec="seconds")},
        {"pnl": 800, "r_multiple": 1.5,
         "closed_at": now.isoformat(timespec="seconds")},
    ]
    _seed_db(str(db), trades)
    r = compute_health(str(db))
    assert r.total_trades == 3
    assert r.pf_all > 0
    assert r.severity == "ok"


def test_compute_health_alerts_on_loss_streak(tmp_path):
    db = tmp_path / "test.db"
    today = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trades = [{"pnl": -100, "r_multiple": -1.0, "closed_at": today} for _ in range(6)]
    _seed_db(str(db), trades)
    r = compute_health(str(db))
    assert r.severity in ("warn", "critical")
    assert any("Loss streak" in a for a in r.alerts)


def test_compute_health_critical_on_drawdown(tmp_path):
    db = tmp_path / "test.db"
    now = datetime.now(timezone.utc)
    trades = [
        {"pnl": -15_000, "r_multiple": -1.0,
         "closed_at": now.isoformat(timespec="seconds")},
    ]
    _seed_db(str(db), trades)
    r = compute_health(str(db), starting_equity=100_000.0)
    assert r.drawdown_pct > THRESHOLDS["dd_max"]
    assert r.severity == "critical"


def test_compute_health_idle_alert(tmp_path):
    db = tmp_path / "test.db"
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    trades = [{"pnl": 100, "r_multiple": 1.0, "closed_at": old}]
    _seed_db(str(db), trades)
    r = compute_health(str(db))
    assert any("No trade" in a for a in r.alerts)
