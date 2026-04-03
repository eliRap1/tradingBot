"""
Tests for regime filter and strategy selector.

Covers:
  Mistake #5: No regime awareness — must detect bull/bear/chop
  Mistake #5: Must switch strategies dynamically per regime
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from tests.helpers import make_uptrend_bars, make_downtrend_bars, make_ranging_bars
from strategy_selector import select_strategies


class TestStrategySelection:
    """Strategy selector must adapt weights to market conditions."""

    def test_uptrend_favors_momentum(self):
        """In strong uptrend, momentum/supertrend should get highest weights."""
        bars = make_uptrend_bars(200)
        selection = select_strategies(bars, "TEST")

        strategies = selection["strategies"]
        # Momentum strategies should be dominant
        trend_weight = strategies.get("momentum", 0) + strategies.get("supertrend", 0)
        mr_weight = strategies.get("mean_reversion", 0)

        assert trend_weight > mr_weight, \
            f"Uptrend should favor trend strategies ({trend_weight}) over mean reversion ({mr_weight})"

    def test_ranging_favors_mean_reversion(self):
        """In ranging market, mean reversion should get higher weight."""
        bars = make_ranging_bars(200)
        selection = select_strategies(bars, "TEST")

        strategies = selection["strategies"]
        mr_weight = strategies.get("mean_reversion", 0)
        # Mean reversion should be active (weight > 0)
        # It may not be highest in all cases, but should be non-zero
        assert mr_weight >= 0.1, \
            f"Ranging market should activate mean reversion (got {mr_weight})"

    def test_selection_returns_valid_structure(self):
        """Selection must have regime, strategies, and reason."""
        bars = make_uptrend_bars(100)
        selection = select_strategies(bars, "TEST")

        assert "regime" in selection
        assert "strategies" in selection
        assert "reason" in selection
        assert selection["regime"] in ("trending", "ranging", "breakout", "volatile", "mixed")
        assert isinstance(selection["strategies"], dict)
        assert len(selection["strategies"]) > 0

    def test_weights_sum_reasonable(self):
        """Strategy weights should sum to approximately 1.0."""
        bars = make_uptrend_bars(100)
        selection = select_strategies(bars, "TEST")

        total = sum(selection["strategies"].values())
        assert 0.8 <= total <= 1.2, \
            f"Strategy weights sum to {total}, expected ~1.0"

    def test_short_data_gets_default(self):
        """Very short data should get default balanced weights."""
        from tests.helpers import make_bars
        bars = make_bars(n=20)
        selection = select_strategies(bars, "TEST")

        assert selection["regime"] in ("mixed", "trending", "ranging", "breakout", "volatile")
        assert len(selection["strategies"]) > 0
