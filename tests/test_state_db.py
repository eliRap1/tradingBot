import json
import os

import pytest

os.environ["BOT_STATE_DB"] = ":memory:"

from state_db import StateDB


@pytest.fixture
def db():
    return StateDB(":memory:")


def test_upsert_and_get_position(db):
    db.upsert_position(
        "AAPL",
        entry_price=150.0,
        initial_risk=200.0,
        stop_loss=148.0,
        take_profit=156.0,
        strategies='["momentum"]',
        opened_at="2026-04-13T10:00:00",
        original_qty=10.0,
        side="buy",
        partial_done=0,
        second_partial_done=0,
        breakeven_armed=0,
    )
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
    db.upsert_watermark("AAPL", high=158.0)
    wm = db.get_watermarks()
    assert wm["AAPL"]["high_watermark"] == 158.0
    assert wm["AAPL"]["low_watermark"] == 148.0


def test_record_and_get_trades(db):
    db.record_trade(
        symbol="AAPL",
        side="buy",
        qty=10,
        entry_price=150.0,
        exit_price=155.0,
        pnl=50.0,
        reason="trailing_stop",
        risk_dollars=200.0,
        strategies='["momentum"]',
        opened_at="2026-04-13T10:00:00",
        closed_at="2026-04-13T15:00:00",
    )
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
    state_file = tmp_path / "state.json"
    trades_file = tmp_path / "trades.json"
    pending_file = tmp_path / "watcher_pending.json"
    state_file.write_text(json.dumps({"peak_equity": 102000.0, "high_watermarks": {"AAPL": 155.0}}))
    trades_file.write_text(json.dumps([
        {"symbol": "AAPL", "side": "buy", "qty": 10, "entry_price": 150.0,
         "exit_price": 155.0, "pnl": 50.0, "reason": "trailing_stop",
         "risk_dollars": 200.0, "strategies": ["momentum"],
         "closed_at": "2026-04-13T15:00:00"}
    ]))
    pending_file.write_text(json.dumps({"AAPL": True}))

    db = StateDB(":memory:", base_dir=str(tmp_path))
    db.migrate_from_json()

    assert db.get_state("peak_equity") == 102000.0
    wm = db.get_watermarks()
    assert wm.get("AAPL", {}).get("high_watermark") == 155.0
    trades = db.get_trades()
    assert len(trades) == 1
    assert trades[0]["pnl"] == 50.0
    assert db.get_pending_signal("AAPL") is True

    assert (tmp_path / "state.json.bak").exists()
    assert (tmp_path / "trades.json.bak").exists()
    assert (tmp_path / "watcher_pending.json.bak").exists()
    assert not state_file.exists()
    assert not trades_file.exists()
    assert not pending_file.exists()
