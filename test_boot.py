"""Smoke test — verifies coordinator builds with RoutingBroker/RoutingDataFetcher.

Run manually with IB Gateway active at 127.0.0.1:4002:
    python test_boot.py

Press Ctrl+C after seeing "Coordinator built OK".
"""
import pytest

from coordinator import Coordinator


pytestmark = pytest.mark.live


def _print_boot_summary(c: Coordinator) -> None:
    print("Coordinator built OK")
    print("Broker type:", type(c.broker).__name__)
    print("Data type:", type(c.data).__name__)
    print("Strategy router:", type(c._strategy_router).__name__)
    print("Classifier:", type(c._clf).__name__)
    print("NQ classified as:", c._clf.classify("NQ"))
    print("BTC/USD classified as:", c._clf.classify("BTC/USD"))
    print("AAPL classified as:", c._clf.classify("AAPL"))
    print("Futures strategies:", list(c._strategy_router.get_strategies("futures").keys()))


def test_coordinator_builds_with_gateway():
    c = Coordinator()
    _print_boot_summary(c)


if __name__ == "__main__":
    _print_boot_summary(Coordinator())
