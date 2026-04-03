"""
Tests for indicators and candle pattern detection.

Covers:
  Mistake #4: Lagging indicators — validate indicators produce valid output
  Mistake #6: Data quality — ensure indicators handle edge cases
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
from tests.helpers import make_uptrend_bars, make_bars
from indicators import supertrend, pivot_high, pivot_low, stochastic_rsi, crossover
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context, get_weekly_trend


class TestSuperTrend:
    def test_returns_valid_output(self):
        bars = make_uptrend_bars(100)
        st_line, direction = supertrend(bars)

        assert len(st_line) == len(bars)
        assert len(direction) == len(bars)
        # Direction should be -1 (bullish), 0 (init), or 1 (bearish)
        unique_dirs = set(direction.dropna().unique())
        assert unique_dirs.issubset({-1, 0, 1, -1.0, 0.0, 1.0}), \
            f"SuperTrend direction has unexpected values: {unique_dirs}"

    def test_st_line_non_negative(self):
        bars = make_uptrend_bars(100)
        st_line, _ = supertrend(bars)
        valid = st_line.dropna()
        assert (valid >= 0).all(), "SuperTrend line should be non-negative"


class TestPivots:
    def test_pivot_high_finds_peaks(self):
        bars = make_uptrend_bars(100)
        ph = pivot_high(bars["high"], left_bars=5, right_bars=5)
        valid = ph.dropna()
        assert len(valid) > 0, "Should find at least one pivot high"

    def test_pivot_low_finds_troughs(self):
        bars = make_uptrend_bars(100)
        pl = pivot_low(bars["low"], left_bars=5, right_bars=5)
        valid = pl.dropna()
        assert len(valid) > 0, "Should find at least one pivot low"


class TestStochRSI:
    def test_bounded_output(self):
        bars = make_uptrend_bars(100)
        k, d = stochastic_rsi(bars["close"])
        valid_k = k.dropna()
        valid_d = d.dropna()

        assert (valid_k >= 0).all() and (valid_k <= 100).all(), \
            f"StochRSI K out of bounds: min={valid_k.min()}, max={valid_k.max()}"
        assert (valid_d >= 0).all() and (valid_d <= 100).all(), \
            f"StochRSI D out of bounds: min={valid_d.min()}, max={valid_d.max()}"


class TestCrossover:
    def test_crossover_detection(self):
        a = pd.Series([1, 2, 3, 4, 5, 6, 7])
        b = pd.Series([3, 3, 3, 3, 3, 3, 3])
        crosses = crossover(a, b)
        # a crosses above b when a goes from <= b to > b
        assert crosses.any(), "Should detect crossover"


class TestCandlePatterns:
    def test_returns_dict(self):
        bars = make_uptrend_bars(100)
        patterns = detect_patterns(bars)
        assert isinstance(patterns, dict)

    def test_bullish_score_bounded(self):
        bars = make_uptrend_bars(100)
        patterns = detect_patterns(bars)
        score = bullish_score(patterns)
        assert 0.0 <= score <= 1.0, f"Bullish score {score} out of bounds"

    def test_bearish_score_bounded(self):
        bars = make_uptrend_bars(100)
        patterns = detect_patterns(bars)
        score = bearish_score(patterns)
        assert 0.0 <= score <= 1.0, f"Bearish score {score} out of bounds"


class TestTrendContext:
    def test_returns_all_fields(self):
        bars = make_uptrend_bars(100)
        ctx = get_trend_context(bars)

        required = ["adx", "trending", "strong_trend", "direction",
                     "di_plus", "di_minus", "above_vwap", "above_ema_200",
                     "higher_highs", "lower_lows"]
        for field in required:
            assert field in ctx, f"Missing field: {field}"

    def test_adx_bounded(self):
        bars = make_uptrend_bars(100)
        ctx = get_trend_context(bars)
        assert 0 <= ctx["adx"] <= 100, f"ADX {ctx['adx']} out of range"

    def test_direction_valid(self):
        bars = make_uptrend_bars(100)
        ctx = get_trend_context(bars)
        assert ctx["direction"] in ("up", "down", "neutral")

    def test_weekly_trend_structure(self):
        bars = make_uptrend_bars(200)
        wk = get_weekly_trend(bars)
        assert "weekly_trend_up" in wk
        assert isinstance(wk["weekly_trend_up"], (bool, np.bool_))

    def test_short_data_no_crash(self):
        bars = make_bars(n=15)
        ctx = get_trend_context(bars)
        assert isinstance(ctx, dict)
