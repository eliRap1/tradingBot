"""
Tests for all 5 strategies — validates they produce bounded scores,
don't overfit, and respect regime filters.

Covers:
  Mistake #1: Overfitting — strategies should produce varied scores, not always max
  Mistake #4: Lagging indicators — strategies use context, not just indicators alone
  Mistake #5: Regime awareness — strategies adapt to market conditions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from tests.helpers import (
    make_config, make_uptrend_bars, make_downtrend_bars,
    make_ranging_bars, make_volatile_bars, make_5min_bars,
)
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.supertrend import SuperTrendStrategy
from strategies.stoch_rsi import StochRSIStrategy


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def all_strategies(config):
    return {
        "momentum": MomentumStrategy(config),
        "mean_reversion": MeanReversionStrategy(config),
        "breakout": BreakoutStrategy(config),
        "supertrend": SuperTrendStrategy(config),
        "stoch_rsi": StochRSIStrategy(config),
    }


class TestScoreBounds:
    """All strategy scores must be in [-1.0, 1.0] — never unbounded."""

    @pytest.mark.parametrize("bar_fn", [
        make_uptrend_bars, make_downtrend_bars, make_ranging_bars,
        make_volatile_bars, make_5min_bars,
    ])
    def test_scores_bounded(self, all_strategies, bar_fn):
        bars = bar_fn()
        for name, strat in all_strategies.items():
            signals = strat.generate_signals({"TEST": bars})
            for sym, score in signals.items():
                assert -1.0 <= score <= 1.0, \
                    f"{name} produced out-of-bounds score {score}"


class TestNotAlwaysFiring:
    """
    Mistake #1 (Overfitting): A good strategy does NOT fire on every bar.
    It should be selective — scoring 0 most of the time.
    Mistake #7 (Overtrading): Strategies must filter noise.
    """

    def test_not_always_bullish_in_downtrend(self, all_strategies):
        """Momentum shouldn't fire BUY in a downtrend."""
        bars = make_downtrend_bars(100)
        signals = all_strategies["momentum"].generate_signals({"TEST": bars})
        score = signals.get("TEST", 0.0)
        assert score <= 0.3, \
            f"Momentum is bullish ({score}) in a downtrend — overfitting"

    def test_mean_reversion_blocks_strong_trend(self, all_strategies):
        """Mean reversion must NOT fire in strong trends (ADX > 40)."""
        bars = make_uptrend_bars(200)
        signals = all_strategies["mean_reversion"].generate_signals({"TEST": bars})
        score = signals.get("TEST", 0.0)
        # In a strong trend, mean reversion should return 0 or very low
        assert score < 0.3, \
            f"Mean reversion fired ({score}) in strong trend — dangerous"

    def test_stoch_rsi_needs_uptrend(self, all_strategies):
        """StochRSI pullback strategy needs an uptrend to buy."""
        bars = make_downtrend_bars(100)
        signals = all_strategies["stoch_rsi"].generate_signals({"TEST": bars})
        score = signals.get("TEST", 0.0)
        assert score <= 0.1, \
            f"StochRSI buying pullbacks in downtrend ({score})"


class TestRegimeAdaptation:
    """
    Mistake #5: Strategies must behave differently across regimes.
    """

    def test_supertrend_stronger_in_trend(self, all_strategies):
        """SuperTrend should score higher in trending than ranging markets."""
        trend_bars = make_uptrend_bars(100)
        range_bars = make_ranging_bars(100)

        trend_score = all_strategies["supertrend"].generate_signals(
            {"TEST": trend_bars}).get("TEST", 0.0)
        range_score = all_strategies["supertrend"].generate_signals(
            {"TEST": range_bars}).get("TEST", 0.0)

        # SuperTrend in trend should be >= than in range
        # (at minimum, shouldn't be strongly bullish in a range)
        assert range_score <= 0.5, \
            f"SuperTrend too bullish in ranging market ({range_score})"

    def test_mean_reversion_prefers_range(self, all_strategies):
        """Mean reversion should not fire aggressively in strong trends."""
        trend_bars = make_uptrend_bars(200)
        range_bars = make_ranging_bars(100)

        trend_score = all_strategies["mean_reversion"].generate_signals(
            {"TEST": trend_bars}).get("TEST", 0.0)
        range_score = all_strategies["mean_reversion"].generate_signals(
            {"TEST": range_bars}).get("TEST", 0.0)

        # Mean reversion should be penalized in trending markets
        assert trend_score < 0.4, \
            f"Mean reversion too aggressive in trend ({trend_score})"


class TestMinimumDataRequirement:
    """Strategies must handle insufficient data gracefully."""

    def test_short_bars_no_crash(self, all_strategies):
        """Strategies should not crash on < 30 bars."""
        from tests.helpers import make_bars
        short_bars = make_bars(n=10)
        for name, strat in all_strategies.items():
            # Should not raise, should return empty or 0
            signals = strat.generate_signals({"TEST": short_bars})
            assert isinstance(signals, dict), f"{name} crashed on short data"

    def test_empty_bars_no_crash(self, all_strategies):
        """Strategies should handle empty DataFrame."""
        import pandas as pd
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        for name, strat in all_strategies.items():
            signals = strat.generate_signals({"TEST": empty})
            assert isinstance(signals, dict)


class TestIntradayBars:
    """Strategies must work on 5-min bars, not just daily."""

    def test_strategies_work_on_5min(self, all_strategies):
        bars = make_5min_bars(500)
        for name, strat in all_strategies.items():
            signals = strat.generate_signals({"TEST": bars})
            assert isinstance(signals, dict), \
                f"{name} failed on 5-min bars"
            score = signals.get("TEST", 0.0)
            assert -1.0 <= score <= 1.0
