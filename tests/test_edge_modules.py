from unittest.mock import MagicMock

import pandas as pd

from edge.cross_asset import CrossAssetEngine
from edge.microstructure import MicrostructureGate
from edge.ml_filter import MLSignalFilter


def _bars(n=80, start=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series([start + i * 0.5 for i in range(n)], index=idx)
    return pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 1_000_000,
    }, index=idx)


def test_cross_asset_engine_returns_multiplier():
    data = MagicMock()
    data.get_bars.return_value = {
        "SPY": _bars(),
        "TLT": _bars(start=120),
        "QQQ": _bars(start=200),
        "IWM": _bars(start=80),
        "RSP": _bars(start=140),
    }
    data.get_intraday_bars.return_value = _bars(3)
    engine = CrossAssetEngine(data)
    signals = engine.get_signals()
    assert 0.15 <= signals.size_multiplier <= 1.25


def test_microstructure_gate_blocks_wide_spread():
    broker = MagicMock()
    broker.get_quote.return_value = MagicMock(bid=99.0, ask=101.0, mid=100.0)
    data = MagicMock()
    gate = MicrostructureGate(broker, data, {"edge": {"max_spread_pct": 0.005, "ofi_weight": 0.05}})
    signal = gate.evaluate("AAPL", _bars(5))
    assert signal.blocked is True


def test_ml_filter_passthrough_without_model():
    filt = MLSignalFilter(min_trades=1000)
    assert filt.predict_quality({"strategy_scores": {}}) == 1.0
