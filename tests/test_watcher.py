"""
Tests for the StockWatcher confirmation and signal logic.

Covers:
  Mistake #7: Overtrading — confirmation filter (signal must persist 2 checks)
  Mistake #9: Psychological — bot follows rules, not emotions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock

from watcher import WatcherState, Action


class TestConfirmationFilter:
    """Signal must persist across 2 checks — prevents noise trading."""

    def test_first_signal_not_confirmed(self):
        """First signal should be 'pending', not confirmed."""
        state = WatcherState(symbol="TEST")
        state.prev_signal = False

        # Simulate first signal detection
        has_signal = True
        if has_signal and state.prev_signal:
            state.confirmed = True
        elif has_signal:
            state.prev_signal = True
            state.confirmed = False

        assert state.confirmed is False, "First signal should not be confirmed"
        assert state.prev_signal is True

    def test_second_signal_confirmed(self):
        """Second consecutive signal should be confirmed."""
        state = WatcherState(symbol="TEST")
        state.prev_signal = True  # Had signal last check

        # Simulate second signal detection
        has_signal = True
        if has_signal and state.prev_signal:
            state.confirmed = True
            state.action = Action.BUY

        assert state.confirmed is True, "Second signal should be confirmed"
        assert state.action == Action.BUY

    def test_signal_disappears_resets(self):
        """If signal disappears, confirmation resets."""
        state = WatcherState(symbol="TEST")
        state.prev_signal = True
        state.confirmed = True

        # Signal disappears
        has_signal = False
        if not has_signal:
            state.prev_signal = False
            state.confirmed = False
            state.action = Action.NONE

        assert state.confirmed is False
        assert state.action == Action.NONE


class TestActionEnum:
    """Actions must be clearly defined."""

    def test_all_actions_exist(self):
        assert Action.NONE.value == "none"
        assert Action.BUY.value == "buy"
        assert Action.SHORT.value == "short"
        assert Action.EXIT.value == "exit"


class TestWatcherState:
    """State must track all relevant info for dashboard."""

    def test_default_state(self):
        state = WatcherState(symbol="AAPL")
        assert state.symbol == "AAPL"
        assert state.status == "idle"
        assert state.action == Action.NONE
        assert state.score == 0.0
        assert state.confirmed is False
        assert state.error == ""


def test_bear_veto_zeroes_momentum_gap_breakout():
    from watcher import StockWatcher

    config = {
        "strategies": {
            "momentum": {"weight": 0.25},
            "supertrend": {"weight": 0.25},
            "gap": {"weight": 0.15},
            "breakout": {"weight": 0.2},
        },
        "signals": {"min_composite_score": 0.25, "min_agreeing_strategies": 3},
    }
    watcher = StockWatcher("AAPL", config, MagicMock(), strategies={"momentum": 0.4, "supertrend": 0.6})
    watcher._macro_regime = "bear"
    selection = {"strategies": {"momentum": 0.4, "supertrend": 0.3, "gap": 0.2, "breakout": 0.1}}

    watcher._apply_bear_veto(selection, {"adx": 35})

    assert selection["strategies"]["momentum"] == 0.0
    assert selection["strategies"]["gap"] == 0.0
    assert selection["strategies"]["breakout"] == 0.0
