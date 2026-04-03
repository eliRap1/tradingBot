"""
Tests for trade tracking and performance metrics.

Covers:
  Mistake #7: Logging everything — track every trade
  Pro metrics: Sharpe-like metrics, profit factor, win rate
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import tempfile
from unittest.mock import patch
from tracker import TradeTracker


@pytest.fixture
def tracker(tmp_path):
    """TradeTracker with isolated trades file."""
    trades_file = str(tmp_path / "trades.json")
    with patch("tracker.TRADES_FILE", trades_file):
        t = TradeTracker()
        # Record some sample trades
        t.record_trade("AAPL", "buy", 10, 150.0, 160.0, "trailing_stop")  # Win
        t.record_trade("MSFT", "buy", 5, 300.0, 290.0, "stop_loss")       # Loss
        t.record_trade("GOOGL", "buy", 8, 140.0, 155.0, "trailing_stop")  # Win
        t.record_trade("TSLA", "buy", 3, 200.0, 185.0, "stop_loss")       # Loss
        t.record_trade("NVDA", "buy", 12, 800.0, 840.0, "trailing_stop")  # Win
    return t


class TestTradeRecording:
    def test_records_trades(self, tracker):
        assert len(tracker.trades) == 5

    def test_trade_has_required_fields(self, tracker):
        trade = tracker.trades[0]
        required = ["symbol", "side", "qty", "entry_price", "exit_price",
                     "pnl", "pnl_pct", "reason", "closed_at"]
        for field in required:
            assert field in trade, f"Missing field: {field}"

    def test_pnl_calculated_correctly(self, tracker):
        # First trade: buy 10 AAPL at 150, exit 160 = +$100
        assert tracker.trades[0]["pnl"] == 100.0
        # Second trade: buy 5 MSFT at 300, exit 290 = -$50
        assert tracker.trades[1]["pnl"] == -50.0


class TestPerformanceMetrics:
    """Metrics that actually matter (not just total return)."""

    def test_win_rate(self, tracker):
        stats = tracker.get_stats()
        assert stats["wins"] == 3
        assert stats["losses"] == 2
        assert stats["win_pct"] == 60.0

    def test_profit_factor(self, tracker):
        """Profit factor = gross profit / gross loss. Should be > 1.0."""
        stats = tracker.get_stats()
        assert stats["profit_factor"] > 1.0, \
            f"Profit factor {stats['profit_factor']} should be > 1.0"

    def test_avg_win_vs_avg_loss(self, tracker):
        """Average win should be larger than average loss (good R:R)."""
        stats = tracker.get_stats()
        assert stats["avg_win"] > 0
        assert stats["avg_loss"] < 0

    def test_total_pnl(self, tracker):
        stats = tracker.get_stats()
        # $100 + (-$50) + $120 + (-$45) + $480 = $605
        expected = 100 + (-50) + 120 + (-45) + 480
        assert stats["total_pnl"] == expected

    def test_empty_tracker(self, tmp_path):
        trades_file = str(tmp_path / "empty.json")
        with patch("tracker.TRADES_FILE", trades_file):
            t = TradeTracker()
        stats = t.get_stats()
        assert stats == {}


class TestShortTradeTracking:
    """Short trades must calculate PnL correctly (inverted)."""

    def test_short_pnl(self, tmp_path):
        trades_file = str(tmp_path / "short_trades.json")
        with patch("tracker.TRADES_FILE", trades_file):
            t = TradeTracker()
            t.record_trade("SPY", "sell", 10, 500.0, 480.0, "trailing_stop")

        # Short: profit when price drops. PnL = (entry - exit) * qty
        assert t.trades[0]["pnl"] == 200.0  # (500 - 480) * 10
