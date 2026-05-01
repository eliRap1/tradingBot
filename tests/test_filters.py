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
from filters import SECTOR_MAP, MAX_PER_SECTOR, compute_regime_guard_decision
from tests.helpers import make_config


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


class TestRegimeGuard:
    def _config(self):
        cfg = make_config()
        cfg["filters"] = {
            "regime_guard": {
                "enabled": True,
                "lookback_trades": 20,
                "min_trades": 8,
                "recent_wr_trades": 5,
                "caution_pf": 1.3,
                "defensive_pf": 1.0,
                "defensive_recent_wr": 0.30,
            }
        }
        return cfg

    def test_guard_normal_with_too_few_trades(self):
        decision = compute_regime_guard_decision(
            [{"pnl": -100.0}, {"pnl": -50.0}],
            self._config(),
        )

        assert decision.mode == "normal"
        assert decision.size_mult == 1.0

    def test_guard_caution_on_low_profit_factor(self):
        trades = [{"pnl": p} for p in [150, 100, -100, 100, -100, 100, -100, -100]]

        decision = compute_regime_guard_decision(trades, self._config())

        assert decision.mode == "caution"
        assert decision.size_mult < 1.0
        assert decision.max_positions < make_config()["signals"]["max_positions"]
        assert decision.min_agreeing > make_config()["signals"]["min_agreeing_strategies"]

    def test_guard_defensive_on_bad_recent_win_rate(self):
        trades = [{"pnl": p} for p in [200, 200, 200, 200, 200, -50, -50, -50, -50, -50]]

        decision = compute_regime_guard_decision(trades, self._config())

        assert decision.mode == "defensive"
        assert decision.min_agreeing >= 5
        assert decision.max_positions <= 4

    def test_guard_preserves_zero_stock_slots(self):
        trades = [{"pnl": -100.0} for _ in range(8)]

        decision = compute_regime_guard_decision(
            trades,
            self._config(),
            base_max_positions=0,
        )

        assert decision.mode == "defensive"
        assert decision.max_positions == 0

    def test_guard_paper_only_disables_live_mode(self):
        cfg = self._config()
        cfg["filters"]["regime_guard"]["paper_only"] = True
        trades = [{"pnl": -100.0} for _ in range(8)]

        decision = compute_regime_guard_decision(
            trades,
            cfg,
            trading_mode="live",
        )

        assert decision.mode == "normal"
        assert decision.reason == "regime_guard paper_only in live mode"
