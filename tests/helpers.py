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
