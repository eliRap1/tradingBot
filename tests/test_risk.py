"""
Tests for risk management.

Covers:
  Mistake #2: Transaction costs & slippage — R:R filter ensures enough room
  Mistake #3: No risk management — validates stop-losses, position sizing,
              drawdown limits, daily loss limits
  Mistake #8: Realistic backtesting — ATR-based stops, not arbitrary %
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import tempfile
import json
from unittest.mock import patch
from tests.helpers import make_config, make_uptrend_bars
from risk import RiskManager, SizedOrder
from signals import Opportunity


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def risk(config, tmp_path):
    """RiskManager with isolated state file."""
    state_file = str(tmp_path / "state.json")
    with patch("risk.load_state", return_value={"peak_equity": 100000.0}):
        rm = RiskManager(config)
    rm.starting_equity = 100000.0
    return rm


class TestPositionSizing:
    """Mistake #3: Position sizes must be bounded by risk limits."""

    def test_max_1pct_risk_per_trade(self, risk):
        """No single trade should risk more than 1% of equity."""
        bars = make_uptrend_bars(100)
        equity = 100000.0
        price = float(bars["close"].iloc[-1])

        opp = Opportunity("TEST", 0.5, "buy", {"momentum": 0.5}, 3)
        orders = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=equity, num_existing=0
        )

        if orders:
            order = orders[0]
            risk_per_share = abs(order.entry_price - order.stop_loss)
            total_risk = risk_per_share * order.qty
            max_risk = equity * 0.01  # 1% of equity
            assert total_risk <= max_risk * 1.01, \
                f"Trade risks ${total_risk:.2f} > 1% max (${max_risk:.2f})"

    def test_max_5pct_position_size(self, risk):
        """No single position should exceed 5% of portfolio."""
        bars = make_uptrend_bars(100)
        equity = 100000.0
        price = float(bars["close"].iloc[-1])

        opp = Opportunity("TEST", 0.5, "buy", {"momentum": 0.5}, 3)
        orders = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=equity, num_existing=0
        )

        if orders:
            order = orders[0]
            position_value = order.qty * order.entry_price
            max_position = equity * 0.05  # 5%
            assert position_value <= max_position * 1.01, \
                f"Position ${position_value:.2f} > 5% max (${max_position:.2f})"


class TestRiskRewardFilter:
    """Mistake #2: Must reject trades with bad R:R (costs eat profits)."""

    def test_minimum_2_to_1_rr(self, risk):
        """Orders must have at least 2:1 reward:risk ratio."""
        bars = make_uptrend_bars(100)
        equity = 100000.0
        price = float(bars["close"].iloc[-1])

        opp = Opportunity("TEST", 0.5, "buy", {"momentum": 0.5}, 3)
        orders = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=equity, num_existing=0
        )

        for order in orders:
            risk_amt = abs(order.entry_price - order.stop_loss)
            reward = abs(order.take_profit - order.entry_price)
            rr = reward / risk_amt if risk_amt > 0 else 0
            assert rr >= 2.0, f"R:R ratio {rr:.1f} < 2.0 minimum"

    def test_stop_loss_always_set(self, risk):
        """Every order must have a stop loss."""
        bars = make_uptrend_bars(100)
        equity = 100000.0
        price = float(bars["close"].iloc[-1])

        opp = Opportunity("TEST", 0.5, "buy", {"momentum": 0.5}, 3)
        orders = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=equity, num_existing=0
        )

        for order in orders:
            assert order.stop_loss > 0, "Stop loss not set"
            if order.side == "buy":
                assert order.stop_loss < order.entry_price, \
                    "Long stop should be below entry"
            else:
                assert order.stop_loss > order.entry_price, \
                    "Short stop should be above entry"


class TestShortSizing:
    """Short orders must have reversed stop/target."""

    def test_short_stop_above_entry(self, risk):
        bars = make_downtrend_bars(100)
        equity = 100000.0
        price = float(bars["close"].iloc[-1])

        opp = Opportunity("TEST", -0.5, "sell", {"momentum": -0.5}, 3)
        orders = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=equity, num_existing=0
        )

        for order in orders:
            assert order.side == "sell"
            assert order.stop_loss > order.entry_price, \
                f"Short stop {order.stop_loss} should be above entry {order.entry_price}"
            assert order.take_profit < order.entry_price, \
                f"Short target {order.take_profit} should be below entry {order.entry_price}"


class TestDrawdownProtection:
    """Mistake #3: Must halt trading when drawdown exceeds limit."""

    def test_drawdown_halt(self, risk):
        """Trading must stop at 10% drawdown."""
        risk.peak_equity = 100000.0
        # 10% drawdown
        assert risk.check_drawdown(90000.0) is True, \
            "Should halt at 10% drawdown"

    def test_no_halt_within_limits(self, risk):
        """Should not halt within acceptable drawdown."""
        risk.peak_equity = 100000.0
        assert risk.check_drawdown(95000.0) is False, \
            "Should not halt at 5% drawdown"

    def test_peak_updates(self, risk):
        """Peak equity should track new highs."""
        with patch("risk.load_state", return_value={}), \
             patch("risk.save_state"):
            risk.peak_equity = 100000.0
            risk.check_drawdown(105000.0)
            assert risk.peak_equity == 105000.0


class TestDailyLossLimit:
    """Mistake #3: Daily loss limit prevents blowup days."""

    def test_daily_loss_limit_blocks_trades(self, risk):
        """After 3% daily loss, no new trades."""
        risk.starting_equity = 100000.0
        bars = make_uptrend_bars(100)
        price = float(bars["close"].iloc[-1])
        equity = 97000.0  # 3% down

        opp = Opportunity("TEST", 0.5, "buy", {"momentum": 0.5}, 3)
        orders = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=equity, num_existing=0
        )

        assert len(orders) == 0, \
            "Should block new trades after daily loss limit hit"


class TestMaxPositions:
    """Mistake #3: Don't open too many positions."""

    def test_respects_max_positions(self, risk):
        """Should not open more positions than max."""
        bars = make_uptrend_bars(100)
        equity = 100000.0
        price = float(bars["close"].iloc[-1])

        opps = [
            Opportunity(f"TEST{i}", 0.5, "buy", {"momentum": 0.5}, 3)
            for i in range(10)
        ]
        orders = risk.size_orders(
            opps,
            {f"TEST{i}": bars for i in range(10)},
            {f"TEST{i}": price for i in range(10)},
            equity=equity,
            num_existing=7,  # already at 7, max is 8
        )

        assert len(orders) <= 1, \
            f"Opened {len(orders)} positions when only 1 slot available"


# Import for the downtrend helper
from tests.helpers import make_downtrend_bars
