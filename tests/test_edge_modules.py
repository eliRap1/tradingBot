"""Unit tests for the edge layer — cross-asset, microstructure, news, ML filter.

No live IB connection required — mocks broker/data everywhere.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from edge.cross_asset import CrossAssetEngine, CrossAssetSignals
from edge.microstructure import MicrostructureGate
from edge.ml_filter import MLSignalFilter
from edge.news_sentiment import NewsSentimentEngine


# ── Fixtures ─────────────────────────────────────────────────


def _bars(n: int = 80, start: float = 100.0, drift: float = 0.5, noise: float = 0.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    base = np.array([start + i * drift for i in range(n)], dtype=float)
    if noise:
        rng = np.random.default_rng(42)
        base = base + rng.normal(0, noise, size=n)
    close = pd.Series(base, index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )


def _volatile_bars(n: int = 80, start: float = 400.0):
    """High-volatility series — drives VIX proxy above 'panic' threshold."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 12.0, size=n)
    close = pd.Series(np.abs(start + noise.cumsum()), index=idx)
    return pd.DataFrame(
        {
            "open": close - 3.0,
            "high": close + 6.0,
            "low": close - 6.0,
            "close": close,
            "volume": 2_000_000,
        },
        index=idx,
    )


def _mock_data_basic() -> MagicMock:
    data = MagicMock()
    data.get_bars.return_value = {
        "SPY": _bars(),
        "TLT": _bars(start=120),
        "QQQ": _bars(start=200),
        "IWM": _bars(start=80),
        "RSP": _bars(start=140),
        "UUP": _bars(start=28),
        "XLK": _bars(start=180),
        "XLF": _bars(start=40),
    }
    data.get_intraday_bars.return_value = _bars(3)
    return data


# ── CrossAssetEngine ─────────────────────────────────────────


class TestCrossAssetEngine:
    def test_returns_bounded_multiplier(self):
        engine = CrossAssetEngine(_mock_data_basic())
        signals = engine.get_signals()
        assert 0.15 <= signals.size_multiplier <= 1.25
        assert signals.vix_regime in {"low", "normal", "elevated", "panic"}
        assert signals.breadth_signal in {"healthy", "neutral", "weak"}

    def test_cache_hit_skips_fetch(self):
        data = _mock_data_basic()
        engine = CrossAssetEngine(data, ttl_sec=600)
        engine.get_signals()
        fetch_count_after_first = data.get_bars.call_count
        engine.get_signals()
        assert data.get_bars.call_count == fetch_count_after_first, "Cache should suppress refetch"

    def test_panic_regime_reduces_size(self):
        data = MagicMock()
        data.get_bars.return_value = {"SPY": _volatile_bars()}
        data.get_intraday_bars.return_value = _bars(3)
        engine = CrossAssetEngine(data, ttl_sec=0)
        signals = engine.get_signals()
        assert signals.vix_regime in {"elevated", "panic"}, (
            f"Expected high vol regime, got {signals.vix_regime}"
        )
        assert signals.size_multiplier < 1.0, "Panic/elevated VIX must shrink size"

    def test_empty_data_falls_back_to_defaults(self):
        data = MagicMock()
        data.get_bars.return_value = {}
        data.get_intraday_bars.return_value = None
        engine = CrossAssetEngine(data, ttl_sec=0)
        signals = engine.get_signals()
        assert isinstance(signals, CrossAssetSignals)
        assert signals.size_multiplier == 1.0

    def test_size_multiplier_math_cascades(self):
        """Panic + backwardation + weak breadth + risk_off should compound."""
        data = MagicMock()
        # Build bars where breadth < 40% (most ETFs below their EMA).
        falling = _bars(n=80, start=200, drift=-0.5)
        data.get_bars.return_value = {
            "SPY": _volatile_bars(),
            "TLT": _bars(start=120, drift=0.8),  # TLT rising → risk_off
            "QQQ": falling,
            "IWM": falling,
            "RSP": falling,
            "UUP": _bars(start=28),
            "XLK": falling,
            "XLF": falling,
        }
        data.get_intraday_bars.return_value = _bars(3)
        engine = CrossAssetEngine(data, ttl_sec=0)
        signals = engine.get_signals()
        # Clamp floor is 0.15 — must hit or approach it under compound stress.
        assert signals.size_multiplier <= 0.50, (
            f"Expected compounded shrink under panic+risk_off+weak breadth, "
            f"got {signals.size_multiplier}"
        )


# ── MicrostructureGate ───────────────────────────────────────


class TestMicrostructureGate:
    def test_blocks_wide_spread(self):
        broker = MagicMock()
        broker.get_quote.return_value = MagicMock(bid=99.0, ask=101.0, mid=100.0)
        data = MagicMock()
        gate = MicrostructureGate(
            broker, data, {"edge": {"max_spread_pct": 0.005, "ofi_weight": 0.05}}
        )
        signal = gate.evaluate("AAPL", _bars(5))
        assert signal.blocked is True
        assert signal.spread_pct >= 0.005

    def test_passes_tight_spread(self):
        broker = MagicMock()
        broker.get_quote.return_value = MagicMock(bid=99.99, ask=100.01, mid=100.0)
        data = MagicMock()
        data.get_bars.return_value = {"AAPL": _bars(), "SPY": _bars()}
        gate = MicrostructureGate(
            broker, data, {"edge": {"max_spread_pct": 0.0015, "ofi_weight": 0.05}}
        )
        signal = gate.evaluate("AAPL", _bars(5))
        assert signal.blocked is False
        assert signal.spread_pct < 0.0015

    def test_missing_quote_does_not_crash(self):
        broker = MagicMock()
        broker.get_quote.return_value = None
        data = MagicMock()
        gate = MicrostructureGate(broker, data, {"edge": {"max_spread_pct": 0.0015}})
        signal = gate.evaluate("AAPL", None)
        assert signal.blocked is False
        assert signal.spread_pct == 0.0

    def test_ofi_score_bounded(self):
        broker = MagicMock()
        broker.get_quote.return_value = MagicMock(bid=100.0, ask=100.01, mid=100.005)
        data = MagicMock()
        gate = MicrostructureGate(
            broker, data, {"edge": {"max_spread_pct": 0.01, "ofi_weight": 0.05}}
        )
        signal = gate.evaluate("AAPL", _bars(10))
        assert -0.05 <= signal.ofi_score <= 0.05

    def test_spy_corr_defaults_on_missing_history(self):
        broker = MagicMock()
        broker.get_quote.return_value = MagicMock(bid=100.0, ask=100.02, mid=100.01)
        data = MagicMock()
        data.get_bars.return_value = {}
        gate = MicrostructureGate(broker, data, {"edge": {"max_spread_pct": 0.01}})
        signal = gate.evaluate("AAPL", _bars(5))
        assert signal.spy_corr == 0.5


# ── NewsSentimentEngine ──────────────────────────────────────


class TestNewsSentimentEngine:
    def test_earnings_avoidance_off_returns_empty(self):
        engine = NewsSentimentEngine({"edge": {"earnings_avoidance": False}})
        assert engine.get_blocked_symbols(["AAPL"]) == set()

    def test_earnings_avoidance_without_keys_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        engine = NewsSentimentEngine({"edge": {"earnings_avoidance": True}})
        assert engine.get_blocked_symbols(["AAPL", "MSFT"]) == set()

    def test_earnings_cache_populates_days(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        engine = NewsSentimentEngine({"edge": {"earnings_avoidance": True}})

        class _Resp:
            def json(self):
                return {
                    "announcements": [
                        {"symbol": "AAPL", "announcement_date": "2026-04-17"}
                    ]
                }

        with patch("edge.news_sentiment.requests.get", return_value=_Resp()):
            blocked = engine.get_blocked_symbols(["AAPL", "MSFT"])
        assert "AAPL" in blocked
        # days_to_earnings relative to today should be small int
        d = engine.get_days_to_earnings("AAPL")
        assert isinstance(d, int) and d < 30

    def test_days_to_earnings_default(self):
        engine = NewsSentimentEngine({"edge": {}})
        assert engine.get_days_to_earnings("UNKNOWN") == 99

    def test_news_score_disabled_returns_zero(self):
        engine = NewsSentimentEngine({"edge": {"news_sentiment": False}})
        assert engine.score_symbol_news("AAPL") == 0.0


# ── MLSignalFilter ───────────────────────────────────────────


class TestMLSignalFilter:
    def test_passthrough_without_model(self):
        filt = MLSignalFilter(min_trades=1000)
        assert filt.predict_quality({"strategy_scores": {}}) == 1.0

    def test_passthrough_below_min_trades(self):
        filt = MLSignalFilter(min_trades=1000)
        trained = filt.maybe_train(trades=[{"pnl": 1.0} for _ in range(50)])
        assert trained is False
        assert filt.predict_quality({"strategy_scores": {}}) == 1.0

    def test_feature_vector_shape(self):
        filt = MLSignalFilter(min_trades=1000)
        vec = filt._build_feature_vector({
            "strategy_scores": {"momentum": 0.5},
            "composite_score": 0.3,
            "num_agreeing": 3,
            "regime": "bull_trending",
            "high_vol": False,
            "size_multiplier": 1.0,
            "spread_pct": 0.001,
            "ofi_score": 0.01,
            "spy_corr": 0.7,
            "hour_of_day": 10,
            "day_of_week": 2,
            "session_bucket": "open",
            "days_to_earnings": 20,
            "nq_overnight_move": 0.002,
        })
        # 9 strategies (incl. DOL) + 13 scalar features
        assert len(vec) == 9 + 13

    def test_regime_encoding_stable(self):
        filt = MLSignalFilter(min_trades=1000)
        a = filt._build_feature_vector({"strategy_scores": {}, "regime": "bull_trending"})
        b = filt._build_feature_vector({"strategy_scores": {}, "regime": "bear_choppy"})
        regime_idx = 9 + 2  # strategies block (incl. DOL) + (composite, num_agreeing) then regime
        assert a[regime_idx] != b[regime_idx]
