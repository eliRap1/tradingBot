"""Test helpers — generate realistic OHLCV data for strategy testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_config():
    """Return a minimal config dict for testing."""
    return {
        "strategies": {
            "momentum": {
                "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
                "roc_period": 10, "ema_fast": 8, "ema_slow": 21, "weight": 0.25,
            },
            "mean_reversion": {
                "bb_period": 20, "bb_std": 2.0, "zscore_threshold": 1.5,
                "rsi_period": 14, "weight": 0.15,
            },
            "breakout": {
                "volume_multiplier": 1.5, "atr_period": 14,
                "lookback_days": 20, "weight": 0.2,
            },
            "supertrend": {
                "atr_period": 10, "multiplier": 3.0, "weight": 0.25,
            },
            "stoch_rsi": {
                "rsi_period": 14, "stoch_period": 14, "k_smooth": 3,
                "d_smooth": 3, "ema_period": 50, "oversold": 20,
                "overbought": 80, "weight": 0.15,
            },
            "vwap_reclaim": {
                "weight": 0.15,
            },
            "gap": {
                "min_gap_pct": 1.5,
                "strong_gap_pct": 3.0,
                "weight": 0.15,
            },
            "liquidity_sweep": {
                "pivot_lookback": 5,
                "sweep_tolerance_pct": 0.005,
                "min_wick_body_ratio": 2.0,
                "min_volume_ratio": 1.5,
                "min_confluence": 3,
                "weight": 0.20,
            },
            "dol": {
                "weight": 0.15,
                "pivot_lookback": 5,
                "fvg_min_atr_mult": 0.1,
                "ob_displacement_mult": 1.0,
                "freshness_decay_bars": 30,
                "top_n_levels": 3,
                "min_verdict": 0.20,
                "require_htf_align": True,
                "scan_lookback_bars": 60,
            },
            "futures_trend": {"weight": 0.30},
            "time_series_momentum": {
                "weight": 0.30,
                "lookbacks": [20, 60, 120],
                "lookback_weights": [0.50, 0.30, 0.20],
                "ema_fast": 20,
                "ema_slow": 100,
                "min_abs_score": 0.15,
                "max_realized_vol": 1.20,
                "target_annual_vol": 0.35,
            },
            "donchian_breakout": {
                "weight": 0.25,
                "fast_lookback": 20,
                "slow_lookback": 55,
                "atr_period": 14,
                "min_volume_ratio": 1.05,
                "max_atr_extension": 3.0,
            },
            "relative_strength_rotation": {
                "weight": 0.20,
                "lookbacks": [20, 60, 126],
                "lookback_weights": [0.50, 0.30, 0.20],
                "top_pct": 0.25,
                "bottom_pct": 0.20,
                "min_abs_momentum": 0.02,
                "allow_shorts": True,
            },
        },
        "signals": {
            "min_composite_score": 0.25,
            "max_positions": 8,
            "min_agreeing_strategies": 3,
            "allow_shorts_in_bull": False,
            "entry_timeframe": "5Min",
            "trend_timeframe": "1Day",
            "intraday_lookback_days": 5,
        },
        "risk": {
            "max_position_pct": 0.05,
            "max_portfolio_risk_pct": 0.01,
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 4.0,
            "min_risk_reward": 2.0,
            "max_drawdown_pct": 0.10,
            "trailing_stop_pct": 0.03,
            "daily_loss_limit_pct": 0.03,
            "max_positions": 8,
            "leverage": 1.0,
            "volatility_target": {
                "enabled": True,
                "target_annual_vol": 0.25,
                "lookback_bars": 20,
                "min_mult": 0.35,
                "max_mult": 1.50,
            },
            "asset_overrides": {
                "stock": {
                    "max_positions": 0,
                    "max_position_pct": 0.05,
                    "max_portfolio_risk_pct": 0.01,
                    "min_risk_reward": 2.0,
                    "contract_multiplier": 1,
                    "allow_fractional_qty": False,
                    "allow_shorts": False,
                    "min_qty": 1,
                },
                "futures": {
                    "max_positions": 2,
                    "max_position_pct": 0.12,
                    "max_portfolio_risk_pct": 0.01,
                    "min_risk_reward": 1.8,
                    "stop_loss_atr_mult": 2.0,
                    "take_profit_atr_mult": 4.0,
                    "min_score": 0.18,
                    "min_agreeing": 2,
                    "allow_fractional_qty": False,
                    "allow_shorts": True,
                    "min_qty": 1,
                    "contract_multipliers": {
                        "NQ": 20, "ES": 50, "CL": 1000, "GC": 100,
                        "MNQ": 2, "MES": 5, "MCL": 100, "MGC": 10,
                    },
                },
                "crypto": {
                    "max_positions": 1,
                    "max_position_pct": 0.12,
                    "max_portfolio_risk_pct": 0.012,
                    "min_risk_reward": 1.8,
                    "stop_loss_atr_mult": 2.5,
                    "take_profit_atr_mult": 5.0,
                    "min_score": 0.20,
                    "min_agreeing": 2,
                    "contract_multiplier": 1,
                    "allow_fractional_qty": True,
                    "allow_shorts": False,
                    "qty_decimals": 8,
                    "min_qty": 0.000001,
                    "regime_filter": {
                        "enabled": True,
                        "benchmark_symbol": "BTC/USD",
                        "ema_period": 20,
                        "return_lookback_bars": 20,
                        "min_benchmark_return": 0.0,
                        "require_symbol_above_ema": True,
                        "require_benchmark_above_ema": True,
                    },
                },
            },
        },
        "backtest": {
            "slippage_pct": 0.001,
            "spread_pct": 0.0005,
            "futures_slippage_pct": 0.0005,
            "futures_spread_pct": 0.0002,
            "crypto_slippage_pct": 0.002,
            "crypto_spread_pct": 0.001,
            "volume_impact": True,
            "commission_per_share": 0.0,
        },
        "schedule": {
            "market_open_delay_min": 15,
            "cycle_interval_sec": 300,
            "market_close_buffer_min": 15,
        },
        "screener": {
            "min_price": 10.0,
            "max_price": 1000.0,
            "min_avg_volume": 500000,
            "universe": ["AAPL", "MSFT", "GOOGL"],
            "crypto": ["BTC/USD", "ETH/USD"],
        },
        "futures": {
            "contracts": [
                {"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"},
                {"root": "MNQ"}, {"root": "MES"}, {"root": "MCL"}, {"root": "MGC"},
            ],
            "symbols": ["MNQ", "MES", "MCL", "MGC"],
        },
        "optimization": {
            "apr_target": 0.30,
            "max_drawdown_pct": 0.15,
            "assets": ["stocks", "futures", "crypto"],
            "paper_only_on_partial_pass": True,
            "strategy_filters": {
                "stock": {
                    "enabled": ["relative_strength_rotation"],
                    "disabled": [
                        "time_series_momentum",
                        "donchian_breakout",
                        "momentum",
                        "supertrend",
                        "futures_trend",
                        "mean_reversion",
                        "gap",
                    ],
                },
                "futures": {
                    "enabled": ["time_series_momentum", "donchian_breakout", "futures_trend", "supertrend", "momentum"]
                },
                "crypto": {
                    "enabled": ["time_series_momentum", "donchian_breakout", "relative_strength_rotation", "momentum", "supertrend"]
                },
            },
        },
    }


def make_bars(n: int = 100, start_price: float = 100.0,
              trend: str = "up", volatility: float = 0.02,
              volume_base: int = 1_000_000,
              start_date: str = "2025-01-01",
              freq: str = "1D",
              seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic OHLCV bars.

    trend: "up", "down", "flat", "volatile"
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n, freq=freq)

    drift = {
        "up": 0.001,
        "down": -0.001,
        "flat": 0.0,
        "volatile": 0.0,
    }.get(trend, 0.0)

    vol = volatility if trend != "volatile" else volatility * 2.5

    prices = [start_price]
    for _ in range(n - 1):
        ret = drift + np.random.normal(0, vol)
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    # Realistic OHLC relationships
    highs = closes * (1 + np.abs(np.random.normal(0, vol * 0.5, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, vol * 0.5, n)))
    opens = np.roll(closes, 1) * (1 + np.random.normal(0, vol * 0.3, n))
    opens[0] = start_price

    # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    volumes = (volume_base * (1 + np.random.normal(0, 0.3, n))).astype(int)
    volumes = np.maximum(volumes, 100)

    # VWAP approximation
    typical = (highs + lows + closes) / 3
    cum_tp_vol = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    vwap = cum_tp_vol / cum_vol

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "vwap": vwap,
    }, index=dates)

    return df


def make_uptrend_bars(n=100):
    """Strong uptrend with clear higher highs."""
    return make_bars(n, start_price=50, trend="up", volatility=0.015)


def make_downtrend_bars(n=100):
    """Strong downtrend with clear lower lows."""
    return make_bars(n, start_price=150, trend="down", volatility=0.015)


def make_ranging_bars(n=100):
    """Flat/ranging market."""
    return make_bars(n, start_price=100, trend="flat", volatility=0.01)


def make_volatile_bars(n=100):
    """High volatility bars."""
    return make_bars(n, start_price=100, trend="volatile", volatility=0.03)


def make_5min_bars(n=500):
    """5-minute intraday bars."""
    return make_bars(n, start_price=100, trend="up", volatility=0.005,
                     volume_base=50_000, freq="5min",
                     start_date="2025-03-31 09:30:00")
