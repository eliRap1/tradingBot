"""Shared fixtures for live IB integration tests."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nest_asyncio
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

nest_asyncio.apply()

from ib_broker import IBBroker
from ib_data import IBDataFetcher

SYMBOLS = {
    "stocks": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "TSLA", "AMD", "PLTR", "SNOW", "CRWD",
        "PEP", "LOW", "SO", "PM", "COF",
    ],
    "etfs": ["GLD", "SPY", "QQQ"],
    "sector_etfs": ["XLF", "XLE", "XLK"],
    "futures": ["NQ", "ES"],
    "crypto": ["BTC/USD", "ETH/USD"],
}
ALL_SYMBOLS = [symbol for group in SYMBOLS.values() for symbol in group]


def _load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _live_marker_requested(request: pytest.FixtureRequest) -> bool:
    markexpr = getattr(request.config.option, "markexpr", "") or ""
    return "live" in markexpr


@pytest.fixture(scope="module")
def ib_session(request: pytest.FixtureRequest):
    """Connect to IB paper, yield (broker, data, config), then clean up."""
    if not _live_marker_requested(request):
        pytest.skip("Live IB tests are skipped unless explicitly selected with `-m live`.")

    config = _load_config()
    broker = IBBroker(config)
    assert broker._ib.isConnected(), "Failed to connect to IB Gateway"

    data = IBDataFetcher(broker._ib, broker._contracts, config)

    try:
        yield broker, data, config
    finally:
        try:
            broker.cancel_all_orders()
            time.sleep(1)
        except Exception:
            pass
        try:
            broker.close_all_positions()
            time.sleep(2)
        except Exception:
            pass
        try:
            broker._ib.disconnect()
        except Exception:
            pass
