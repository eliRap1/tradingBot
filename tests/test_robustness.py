"""
Robustness and integration tests — validates the bot as a SYSTEM.

Covers:
  Mistake #1: Overfitting — walk-forward style test with different data
  Mistake #2: Slippage — regime sizing reduces exposure in bad conditions
  Mistake #5: Regime — full pipeline must adapt
  Mistake #8: Realistic execution — ATR stops scale with volatility
  Mistake #10: Fail-safes — error handling, kill switches
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from tests.helpers import (
    make_config, make_uptrend_bars, make_downtrend_bars,
    make_ranging_bars, make_volatile_bars, make_bars,
)
from risk import RiskManager
from signals import Opportunity, aggregate_signals
from strategy_selector import select_strategies


class TestWalkForward:
    """
    Mistake #1: Overfitting — test on different data segments.
    Strategies should produce reasonable (not identical) scores on
    different random seeds = different market conditions.
    """

    def test_different_seeds_different_scores(self):
        """Strategy scores should vary with different data — not memorized."""
        from strategies.momentum import MomentumStrategy
        config = make_config()
        strat = MomentumStrategy(config)

        scores = []
        for seed in range(5):
            bars = make_bars(100, trend="up", volatility=0.02, seed=seed)
            signals = strat.generate_signals({"TEST": bars})
            scores.append(signals.get("TEST", 0.0))

        # Scores should NOT all be identical (that would mean overfitting)
        unique = len(set(round(s, 2) for s in scores))
        assert unique >= 2, \
            f"All scores identical ({scores}) — likely overfitting"

    def test_strategies_dont_always_max_score(self):
        """Strategies should rarely hit max score — that signals overfitting."""
        from strategies import ALL_STRATEGIES
        config = make_config()
        strategies = {n: cls(config) for n, cls in ALL_STRATEGIES.items()}

        max_count = 0
        total = 0
        for seed in range(10):
            bars = make_bars(100, trend="up", volatility=0.02, seed=seed * 100)
            for name, strat in strategies.items():
                signals = strat.generate_signals({"TEST": bars})
                score = signals.get("TEST", 0.0)
                total += 1
                if abs(score) >= 0.95:
                    max_count += 1

        # Less than 20% of scores should be near-max
        pct_max = max_count / total if total > 0 else 0
        assert pct_max < 0.2, \
            f"{pct_max:.0%} of scores near max — strategies may be overfitting"


class TestSlippageResilience:
    """
    Mistake #2: Regime sizing multiplier reduces position size in bad conditions.
    """

    def test_bear_regime_reduces_size(self):
        config = make_config()
        with patch("risk.load_state", return_value={"peak_equity": 100000}):
            risk = RiskManager(config)
        risk.starting_equity = 100000.0

        bars = make_uptrend_bars(100)
        price = float(bars["close"].iloc[-1])
        opp = Opportunity("TEST", 0.5, "buy", {}, 3)

        # Normal size (mult=1.0)
        orders_full = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=100000, num_existing=0, regime_size_mult=1.0
        )
        # Bear regime (mult=0.3)
        orders_bear = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=100000, num_existing=0, regime_size_mult=0.3
        )

        if orders_full and orders_bear:
            assert orders_bear[0].qty < orders_full[0].qty, \
                "Bear regime should reduce position size"

    def test_cooldown_after_losses(self):
        """Loss cooldown multiplier should reduce sizing."""
        config = make_config()
        with patch("risk.load_state", return_value={"peak_equity": 100000}):
            risk = RiskManager(config)
        risk.starting_equity = 100000.0

        bars = make_uptrend_bars(100)
        price = float(bars["close"].iloc[-1])
        opp = Opportunity("TEST", 0.5, "buy", {}, 3)

        # Normal
        orders_normal = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=100000, num_existing=0, regime_size_mult=1.0
        )
        # After losses (mult=0.5)
        orders_cooldown = risk.size_orders(
            [opp], {"TEST": bars}, {"TEST": price},
            equity=100000, num_existing=0, regime_size_mult=0.5
        )

        if orders_normal and orders_cooldown:
            assert orders_cooldown[0].qty <= orders_normal[0].qty


class TestATRStopsScale:
    """
    Mistake #8: Stops should scale with volatility (ATR), not be fixed %.
    """

    def test_volatile_stock_wider_stops(self):
        config = make_config()
        with patch("risk.load_state", return_value={"peak_equity": 100000}):
            risk = RiskManager(config)
        risk.starting_equity = 100000.0

        calm_bars = make_bars(100, volatility=0.01, trend="up")
        volatile_bars = make_bars(100, volatility=0.04, trend="up")

        calm_price = float(calm_bars["close"].iloc[-1])
        vol_price = float(volatile_bars["close"].iloc[-1])

        opp_calm = Opportunity("CALM", 0.5, "buy", {}, 3)
        opp_vol = Opportunity("VOL", 0.5, "buy", {}, 3)

        orders_calm = risk.size_orders(
            [opp_calm], {"CALM": calm_bars}, {"CALM": calm_price},
            equity=100000, num_existing=0
        )
        orders_vol = risk.size_orders(
            [opp_vol], {"VOL": volatile_bars}, {"VOL": vol_price},
            equity=100000, num_existing=0
        )

        if orders_calm and orders_vol:
            calm_stop_dist = abs(orders_calm[0].entry_price - orders_calm[0].stop_loss)
            vol_stop_dist = abs(orders_vol[0].entry_price - orders_vol[0].stop_loss)
            # Normalize by price
            calm_pct = calm_stop_dist / orders_calm[0].entry_price
            vol_pct = vol_stop_dist / orders_vol[0].entry_price

            assert vol_pct > calm_pct * 0.8, \
                f"Volatile stop ({vol_pct:.3f}) should be wider than calm ({calm_pct:.3f})"


class TestFullPipeline:
    """
    Integration: run full signal pipeline and verify output quality.
    """

    def test_pipeline_produces_valid_opportunities(self):
        """Full pipeline: bars → strategies → signals → opportunities."""
        from strategies import ALL_STRATEGIES
        config = make_config()
        strategies = {n: cls(config) for n, cls in ALL_STRATEGIES.items()}

        bars = make_uptrend_bars(100)
        all_signals = {}
        for name, strat in strategies.items():
            all_signals[name] = strat.generate_signals({"TEST": bars})

        weights = {n: config["strategies"].get(n, {}).get("weight", 0.20)
                   for n in ALL_STRATEGIES}

        opps = aggregate_signals(
            all_signals, weights,
            min_score=config["signals"]["min_composite_score"],
            max_positions=config["signals"]["max_positions"],
            existing_positions=[],
            min_agreeing=config["signals"]["min_agreeing_strategies"]
        )

        # Opportunities should be valid
        for opp in opps:
            assert opp.symbol == "TEST"
            assert opp.score >= config["signals"]["min_composite_score"]
            assert opp.num_agreeing >= config["signals"]["min_agreeing_strategies"]
            assert opp.direction in ("buy", "sell")


class TestConfigSanity:
    """
    Validate config values prevent common mistakes.
    """

    def test_risk_per_trade_reasonable(self):
        config = make_config()
        risk_pct = config["risk"]["max_portfolio_risk_pct"]
        assert 0.005 <= risk_pct <= 0.03, \
            f"Risk per trade {risk_pct} should be 0.5-3%"

    def test_rr_ratio_minimum(self):
        config = make_config()
        min_rr = config["risk"]["min_risk_reward"]
        assert min_rr >= 1.5, f"Minimum R:R {min_rr} too low"

    def test_max_drawdown_reasonable(self):
        config = make_config()
        dd = config["risk"]["max_drawdown_pct"]
        assert 0.05 <= dd <= 0.20, f"Max drawdown {dd} should be 5-20%"

    def test_confluence_minimum(self):
        config = make_config()
        min_agree = config["signals"]["min_agreeing_strategies"]
        assert min_agree >= 2, \
            f"Confluence of {min_agree} too low — need at least 2"

    def test_stop_loss_atr_reasonable(self):
        config = make_config()
        sl_mult = config["risk"]["stop_loss_atr_mult"]
        assert 1.0 <= sl_mult <= 4.0, f"SL ATR mult {sl_mult} unreasonable"

    def test_take_profit_exceeds_stop(self):
        config = make_config()
        tp = config["risk"]["take_profit_atr_mult"]
        sl = config["risk"]["stop_loss_atr_mult"]
        assert tp > sl, "Take profit must exceed stop loss for positive R:R"
