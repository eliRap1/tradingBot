"""
Tests for signal aggregation and confluence filtering.

Covers:
  Mistake #7: Overtrading — confluence filter requires 3+ strategies to agree
  Mistake #1: Overfitting — single strategy can't force a trade
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from signals import aggregate_signals, Opportunity


class TestConfluenceFilter:
    """The single biggest win-rate improvement: require multiple strategies to agree."""

    def test_single_strategy_rejected(self):
        """One bullish strategy is NOT enough — needs 3+ confluence."""
        all_signals = {
            "momentum": {"AAPL": 0.8},  # Only one strategy fires
            "mean_reversion": {},
            "breakout": {},
            "supertrend": {},
            "stoch_rsi": {},
        }
        weights = {"momentum": 0.25, "mean_reversion": 0.15,
                   "breakout": 0.2, "supertrend": 0.25, "stoch_rsi": 0.15}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=0.25, max_positions=8,
            existing_positions=[], min_agreeing=3
        )

        assert len(opps) == 0, \
            "Single strategy signal should be rejected by confluence filter"

    def test_two_strategies_rejected(self):
        """Two strategies still not enough."""
        all_signals = {
            "momentum": {"AAPL": 0.6},
            "supertrend": {"AAPL": 0.5},
            "breakout": {},
            "mean_reversion": {},
            "stoch_rsi": {},
        }
        weights = {"momentum": 0.25, "mean_reversion": 0.15,
                   "breakout": 0.2, "supertrend": 0.25, "stoch_rsi": 0.15}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=0.25, max_positions=8,
            existing_positions=[], min_agreeing=3
        )

        assert len(opps) == 0, "2 strategies should not pass confluence filter"

    def test_three_strategies_accepted(self):
        """Three agreeing strategies should pass."""
        all_signals = {
            "momentum": {"AAPL": 0.5},
            "supertrend": {"AAPL": 0.4},
            "breakout": {"AAPL": 0.3},
            "mean_reversion": {},
            "stoch_rsi": {},
        }
        weights = {"momentum": 0.25, "mean_reversion": 0.15,
                   "breakout": 0.2, "supertrend": 0.25, "stoch_rsi": 0.15}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=0.25, max_positions=8,
            existing_positions=[], min_agreeing=3
        )

        assert len(opps) == 1, "3 agreeing strategies should generate a signal"
        assert opps[0].num_agreeing == 3

    def test_low_score_signals_dont_count(self):
        """Signals below 0.1 threshold shouldn't count as 'agreeing'."""
        all_signals = {
            "momentum": {"AAPL": 0.5},
            "supertrend": {"AAPL": 0.4},
            "breakout": {"AAPL": 0.05},  # Too weak to count
            "mean_reversion": {"AAPL": 0.03},  # Too weak
            "stoch_rsi": {"AAPL": 0.02},  # Too weak
        }
        weights = {"momentum": 0.25, "mean_reversion": 0.15,
                   "breakout": 0.2, "supertrend": 0.25, "stoch_rsi": 0.15}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=0.1, max_positions=8,
            existing_positions=[], min_agreeing=3
        )

        assert len(opps) == 0, "Weak signals shouldn't count as agreeing"


class TestExistingPositionFilter:
    """Don't open duplicate positions."""

    def test_skips_held_symbols(self):
        all_signals = {
            "momentum": {"AAPL": 0.5, "MSFT": 0.5},
            "supertrend": {"AAPL": 0.4, "MSFT": 0.4},
            "breakout": {"AAPL": 0.3, "MSFT": 0.3},
        }
        weights = {"momentum": 0.25, "supertrend": 0.25, "breakout": 0.2}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=0.25, max_positions=8,
            existing_positions=["AAPL"],  # Already holding
            min_agreeing=3
        )

        symbols = [o.symbol for o in opps]
        assert "AAPL" not in symbols, "Should skip already-held symbols"
        assert "MSFT" in symbols


class TestRanking:
    """Signals should be ranked by confluence then score."""

    def test_higher_confluence_ranks_first(self):
        all_signals = {
            "momentum": {"AAPL": 0.3, "MSFT": 0.5},
            "supertrend": {"AAPL": 0.3, "MSFT": 0.4},
            "breakout": {"AAPL": 0.3, "MSFT": 0.3},
            "mean_reversion": {"MSFT": 0.2},
            "stoch_rsi": {},
        }
        weights = {"momentum": 0.25, "supertrend": 0.25,
                   "breakout": 0.2, "mean_reversion": 0.15, "stoch_rsi": 0.15}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=0.2, max_positions=8,
            existing_positions=[], min_agreeing=3
        )

        if len(opps) >= 2:
            # MSFT has 4 agreeing, AAPL has 3 — MSFT should rank first
            assert opps[0].symbol == "MSFT"
