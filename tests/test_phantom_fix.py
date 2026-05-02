"""Regression tests for the phantom-record + short-pnl-sign bugs found
in live paper-trade on 2026-05-02.

Bug 1: tracker.record_trade used signed `qty` directly; passing qty=-14
       for a profitable short flipped the pnl sign to a fake loss.
Bug 2: portfolio.execute_exits did not delete_position from state DB or
       cooldown the symbol, so broker state lag caused the same closed
       position to be re-detected and re-recorded each cycle (15-20x
       phantom rows for a single real exit).
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tracker import TradeTracker
from portfolio import PortfolioManager


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_STATE_DB", str(tmp_path / "test.db"))
    trades_file = str(tmp_path / "trades.json")
    with patch("tracker.TRADES_FILE", trades_file):
        return TradeTracker()


# ---------------------------------------------------------------- bug 1


def test_short_pnl_sign_with_signed_qty(tracker):
    """Short with exit < entry must record positive pnl, even if caller
    passes a signed (negative) qty as IB does for shorts."""
    tracker.record_trade(
        symbol="CLM6", side="short", qty=-14,
        entry_price=104989.06, exit_price=102460.00,
        reason="trailing_stop", risk_dollars=1000.0,
    )
    pnl = tracker.trades[-1]["pnl"]
    assert pnl > 0, f"short profitable closure must be positive pnl, got {pnl}"
    expected = (104989.06 - 102460.00) * 14
    assert abs(pnl - expected) < 0.01


def test_short_pnl_sign_with_positive_qty(tracker):
    """Same short trade with positive qty (alt convention) — same answer."""
    tracker.record_trade(
        symbol="CLM6", side="short", qty=14,
        entry_price=104989.06, exit_price=102460.00,
        reason="trailing_stop",
    )
    pnl = tracker.trades[-1]["pnl"]
    expected = (104989.06 - 102460.00) * 14
    assert abs(pnl - expected) < 0.01


def test_short_loss_when_exit_above_entry(tracker):
    tracker.record_trade(
        symbol="CLM6", side="short", qty=-14,
        entry_price=100000.0, exit_price=102000.0,
        reason="stop_loss",
    )
    pnl = tracker.trades[-1]["pnl"]
    assert pnl < 0
    assert abs(pnl - (-2000.0 * 14)) < 0.01


def test_long_unchanged_with_positive_qty(tracker):
    tracker.record_trade(
        symbol="AAPL", side="buy", qty=10,
        entry_price=100.0, exit_price=110.0,
        reason="trailing_stop",
    )
    assert abs(tracker.trades[-1]["pnl"] - 100.0) < 0.01


# ---------------------------------------------------------------- bug 2


def _mock_broker_with_position(symbol, qty, side="short", avg=100.0):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.avg_price = avg
    pos.market_value = qty * avg
    pos.unrealized_pl = 0.0
    pos.side = side
    broker = MagicMock()
    broker.get_positions = MagicMock(return_value=[pos])
    broker.get_equity = MagicMock(return_value=100_000.0)
    return broker, pos


def test_recently_closed_cooldown_prevents_redetection(tmp_path, monkeypatch):
    """After execute_exits, broker may briefly still report the position;
    cooldown must skip it from get_current_positions."""
    monkeypatch.setenv("BOT_STATE_DB", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)

    broker, _ = _mock_broker_with_position("CLM6", -14)
    cfg = {"risk": {"trailing_stop_pct": 0.02, "chandelier_atr_mult": 3.0,
                    "partial_exit_enabled": False},
           "screener": {}, "execution": {"post_close_cooldown_sec": 60.0}}

    with patch("portfolio.load_state", return_value={}), \
         patch("portfolio.save_state"):
        pm = PortfolioManager(cfg, broker)
        pm._recently_closed["CLM6"] = time.time() + 60.0
        positions = pm.get_current_positions()
        assert "CLM6" not in positions, "cooldown must skip recently-closed sym"


def test_zero_qty_position_skipped(tmp_path, monkeypatch):
    """Broker may briefly return position with qty=0 after close;
    must not be added to positions dict."""
    monkeypatch.setenv("BOT_STATE_DB", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)

    broker, _ = _mock_broker_with_position("CLM6", 0)
    cfg = {"risk": {"trailing_stop_pct": 0.02, "chandelier_atr_mult": 3.0,
                    "partial_exit_enabled": False},
           "screener": {}, "execution": {}}

    with patch("portfolio.load_state", return_value={}), \
         patch("portfolio.save_state"):
        pm = PortfolioManager(cfg, broker)
        positions = pm.get_current_positions()
        assert "CLM6" not in positions


def test_cooldown_garbage_collected_after_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_STATE_DB", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)

    broker, _ = _mock_broker_with_position("CLM6", -14)
    cfg = {"risk": {"trailing_stop_pct": 0.02, "chandelier_atr_mult": 3.0,
                    "partial_exit_enabled": False},
           "screener": {}, "execution": {}}

    with patch("portfolio.load_state", return_value={}), \
         patch("portfolio.save_state"):
        pm = PortfolioManager(cfg, broker)
        pm._recently_closed["CLM6"] = time.time() - 1  # expired
        positions = pm.get_current_positions()
        # Expired entry should have been GC'd; position visible again
        assert "CLM6" in positions
        assert "CLM6" not in pm._recently_closed
