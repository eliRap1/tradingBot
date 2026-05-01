import math

import pandas as pd

from backtester import Backtester
from performance import apr_pct, profit_usd
from tests.helpers import make_bars, make_config


def _flat_bars(n=70, price=100.0, high_extra=1.0, low_extra=1.0):
    dates = pd.date_range("2025-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price + high_extra] * n,
            "low": [price - low_extra] * n,
            "close": [price] * n,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


class OneShotLong:
    def generate_signals(self, bars):
        sym, df = next(iter(bars.items()))
        return {sym: 1.0} if len(df) == 51 else {}


def test_apr_and_profit_calculation():
    curve = [("start", 100000.0), ("end", 110000.0)]

    assert profit_usd(curve, 100000.0) == 10000.0
    assert apr_pct(curve, 100000.0, bars=252) == 10.0


def test_backtester_reports_profit_and_apr_fields():
    cfg = make_config()
    bt = Backtester(cfg, initial_equity=100000)

    result = bt._calculate_result([], [("2025-01-01", 100000), ("2025-12-31", 101000)], [])

    assert result.profit_usd == 0.0  # empty-result path has no closed trades
    assert result.apr_pct == 0.0


def test_futures_multiplier_affects_pnl_and_sizing():
    cfg = make_config()
    cfg["risk"]["asset_overrides"]["futures"]["volatility_target"] = {"enabled": False}
    cfg["risk"]["asset_overrides"]["futures"]["stop_loss_atr_mult"] = 1.0
    cfg["risk"]["asset_overrides"]["futures"]["take_profit_atr_mult"] = 2.0
    cfg["risk"]["asset_overrides"]["futures"]["min_risk_reward"] = 1.0
    cfg["risk"]["asset_overrides"]["futures"]["min_agreeing"] = 1
    cfg["risk"]["asset_overrides"]["futures"]["min_score"] = 0.1
    cfg["backtest"]["slippage_pct"] = 0.0
    cfg["backtest"]["spread_pct"] = 0.0
    cfg["backtest"]["futures_slippage_pct"] = 0.0
    cfg["backtest"]["futures_spread_pct"] = 0.0

    bars = _flat_bars()
    bars.iloc[51, bars.columns.get_loc("high")] = 105.0
    bt = Backtester(cfg, initial_equity=100000)
    bt.strategies = {"fake": OneShotLong()}
    bt._router_weights = lambda sym, hist: {"fake": 1.0}

    result = bt.run({"NQ": bars}, min_bars=50)

    assert result.total_trades >= 1
    first = result.trades[0]
    assert first["qty"] >= 1
    assert first["pnl"] >= 400.0  # requires NQ multiplier=20; no-multiplier pnl would be tiny


def test_crypto_rounding_allows_fractional_quantity():
    cfg = make_config()
    bt = Backtester(cfg)

    qty = bt._round_qty(0.123456789, "crypto")

    assert math.isclose(qty, 0.12345679)
    assert bt._round_qty(0.9, "futures") == 0.0


def test_crypto_regime_filter_requires_btc_uptrend():
    cfg = make_config()
    bt = Backtester(cfg)
    eth = make_bars(80, start_price=3000, trend="up", volatility=0.005, seed=10)
    btc_down = make_bars(80, start_price=100000, trend="down", volatility=0.005, seed=11)
    btc_up = make_bars(80, start_price=100000, trend="up", volatility=0.005, seed=12)
    date = eth.index[-1]

    blocked = bt._crypto_regime_allows_long(
        "ETH/USD", eth, date, {"ETH/USD": eth, "BTC/USD": btc_down}
    )
    allowed = bt._crypto_regime_allows_long(
        "ETH/USD", eth, date, {"ETH/USD": eth, "BTC/USD": btc_up}
    )

    assert blocked is False
    assert allowed is True


def test_strategy_router_asset_filters_disable_legacy_by_asset():
    from strategy_router import StrategyRouter

    cfg = make_config()
    router = StrategyRouter(cfg)

    futures = router.get_strategies("futures")
    crypto = router.get_strategies("crypto")
    stock = router.get_strategies("stock")

    assert "mean_reversion" not in futures
    assert "gap" not in crypto
    assert "futures_trend" not in stock
    assert "time_series_momentum" in futures


def test_new_strategies_return_bounded_scores():
    from strategies.time_series_momentum import TimeSeriesMomentumStrategy
    from strategies.donchian_breakout import DonchianBreakoutStrategy
    from strategies.relative_strength_rotation import RelativeStrengthRotationStrategy

    cfg = make_config()
    bars = {
        "AAA": make_bars(160, trend="up", seed=1),
        "BBB": make_bars(160, trend="down", seed=2),
        "CCC": make_bars(160, trend="flat", seed=3),
    }

    for cls in [TimeSeriesMomentumStrategy, DonchianBreakoutStrategy, RelativeStrengthRotationStrategy]:
        signals = cls(cfg).generate_signals(bars)
        assert isinstance(signals, dict)
        assert all(-1.0 <= score <= 1.0 for score in signals.values())


def test_optimizer_walk_forward_slices_do_not_overlap():
    from research.apr_optimizer import make_walk_forward_slices

    bars = {"AAA": make_bars(90), "BBB": make_bars(90, seed=2)}
    folds = make_walk_forward_slices(bars, folds=3)

    assert len(folds) == 3
    ranges = [(fold["AAA"].index[0], fold["AAA"].index[-1]) for fold in folds]
    assert ranges[0][1] < ranges[1][0] < ranges[1][1] < ranges[2][0]


def test_all_asset_validation_splits_are_date_aligned():
    from oos_harness import make_date_aligned_folds, split_bars

    bars = {
        "AAA": make_bars(100, start_date="2025-01-01", seed=1),
        "BTC/USD": make_bars(120, start_date="2024-12-15", seed=2),
    }

    is_bars, oos_bars = split_bars(bars, 0.30)
    assert min(df.index.min() for df in is_bars.values()) >= pd.Timestamp("2025-01-01")
    assert max(df.index.max() for df in is_bars.values()) < min(df.index.min() for df in oos_bars.values())

    folds = make_date_aligned_folds(bars, folds=2, min_bars=30)
    assert len(folds) == 2
    assert max(df.index.max() for df in folds[0].values()) < min(df.index.min() for df in folds[1].values())
