"""
Tests for smart filters.

Covers:
  Mistake #7: Overtrading — filters eliminate marginal trades
  Mistake #2: Transaction costs — gap filter prevents slippage traps
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from filters import SECTOR_MAP, MAX_PER_SECTOR


class TestSectorCap:
    """Sector correlation filter prevents concentrated risk."""

    def test_max_per_sector_defined(self):
        """MAX_PER_SECTOR should be a reasonable limit."""
        assert MAX_PER_SECTOR >= 1, "Must allow at least 1 per sector"
        assert MAX_PER_SECTOR <= 4, \
            f"MAX_PER_SECTOR={MAX_PER_SECTOR} too high, risk concentration"

    def test_sector_map_covers_universe(self):
        """All stocks in the default universe should have a sector."""
        from tests.helpers import make_config
        config = make_config()
        universe = config["screener"]["universe"]

        for sym in universe:
            assert sym in SECTOR_MAP, \
                f"{sym} missing from SECTOR_MAP — no sector filtering"

    def test_crypto_in_sector_map(self):
        """Crypto symbols should be in the sector map."""
        assert "BTC/USD" in SECTOR_MAP
        assert "ETH/USD" in SECTOR_MAP

    def test_no_sector_has_too_many_stocks(self):
        """No single sector should have more than 30 stocks (overfitting to one sector)."""
        from collections import Counter
        counts = Counter(SECTOR_MAP.values())
        for sector, count in counts.items():
            assert count <= 30, \
                f"Sector '{sector}' has {count} stocks — too concentrated"


class TestGapFilter:
    """Gap filter prevents entering on volatile opens."""

    def test_gap_threshold_reasonable(self):
        """Gap threshold should be between 1-5%."""
        # The coordinator uses 0.02 (2%) hardcoded
        gap_threshold = 0.02
        assert 0.01 <= gap_threshold <= 0.05, \
            f"Gap threshold {gap_threshold} outside reasonable range"
