"""
StockWatcher — one dedicated thread per stock.

Each watcher:
  1. Fetches its own candle data
  2. Reads the chart (candlestick patterns, indicators)
  3. Picks the right strategies for this stock's behavior
  4. Generates a trading decision (buy / hold / exit)
  5. Reports back to the Coordinator for execution

This mimics how a real trader watches a chart — focused on one instrument
at a time, reading price action, and applying the right setup.
"""

import json
import os
import threading
import time
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from strategies import ALL_STRATEGIES
from strategy_selector import select_strategies
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context, get_weekly_trend
from indicators import supertrend, pivot_high, pivot_low, stochastic_rsi, last_pivot_value
from utils import setup_logger

PENDING_STATE_FILE = os.path.join(os.path.dirname(__file__), "watcher_pending.json")


class Action(Enum):
    NONE = "none"
    BUY = "buy"
    SHORT = "short"
    EXIT = "exit"


@dataclass
class WatcherState:
    """Current state of a stock watcher — visible to the dashboard."""
    symbol: str
    status: str = "idle"            # idle, watching, analyzing, signal, cooldown
    action: Action = Action.NONE
    score: float = 0.0
    num_agreeing: int = 0
    strategy_scores: dict = field(default_factory=dict)
    strategy_weights: dict = field(default_factory=dict)
    regime: str = ""                # trending, ranging, breakout, volatile
    regime_reason: str = ""
    candle_patterns: list = field(default_factory=list)
    adx: float = 0.0
    trend_direction: str = ""
    above_200ema: bool = False
    above_vwap: bool = False
    weekly_trend_up: bool = False
    last_price: float = 0.0
    last_update: str = ""
    error: str = ""
    confirmed: bool = False         # signal persisted across 2 checks
    prev_signal: bool = False       # had signal last check


class StockWatcher:
    """
    Dedicated thread that watches a single stock.
    """

    def __init__(self, symbol: str, config: dict, data_fetcher,
                 interval: int = 60):
        self.symbol = symbol
        self.config = config
        self.data = data_fetcher
        self.interval = interval
        self.state = WatcherState(symbol=symbol)
        self.log = setup_logger(f"watcher.{symbol}")

        # Initialize strategy instances (one per watcher)
        self.strategies = {
            name: cls(config) for name, cls in ALL_STRATEGIES.items()
        }

        self._thread = None
        self._stop_event = threading.Event()
        self._bars = None  # cached bars

        # Restore pending signal state from disk (survives restarts)
        self.state.prev_signal = _load_pending_state(symbol)

    def start(self):
        """Start watching in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"watcher-{self.symbol}",
            daemon=True
        )
        self._thread.start()
        self.log.info(f"Started watching {self.symbol}")

    def stop(self):
        """Stop the watcher thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.state.status = "stopped"
        self.log.info(f"Stopped watching {self.symbol}")

    def _run_loop(self):
        """Main loop — fetch data, analyze, decide."""
        while not self._stop_event.is_set():
            try:
                self.state.status = "watching"
                self._analyze()
                self.state.last_update = datetime.now().strftime("%H:%M:%S")
            except Exception as e:
                self.state.error = str(e)
                self.state.status = "error"
                self.log.error(f"Error: {e}")

            self._stop_event.wait(timeout=self.interval)

    def _analyze(self):
        """Full analysis cycle for this stock.

        Multi-timeframe approach:
          - Daily bars → trend context (200 EMA, ADX, weekly trend)
          - 5-min bars → entry signals (strategies, candle patterns)
        """
        self.state.status = "analyzing"

        # 1a. Fetch DAILY bars for trend context
        daily_bars = self.data.get_bars([self.symbol], timeframe="1Day", days=250)
        if self.symbol not in daily_bars or len(daily_bars[self.symbol]) < 30:
            self.state.status = "no_data"
            return

        daily_df = daily_bars[self.symbol]

        # 1b. Fetch 5-MINUTE bars for entry signals
        intraday_df = self.data.get_intraday_bars(self.symbol, timeframe="5Min", days=5)
        if intraday_df is None or len(intraday_df) < 30:
            # Fall back to daily if intraday not available (market closed, etc.)
            intraday_df = daily_df

        self._bars = intraday_df  # used for order sizing (ATR on entry timeframe)
        self.state.last_price = float(intraday_df["close"].iloc[-1])

        # 2. Read the chart — DAILY timeframe for trend context
        ctx = get_trend_context(daily_df)
        self.state.adx = float(round(ctx["adx"], 1))
        self.state.trend_direction = str(ctx["direction"])
        self.state.above_200ema = bool(ctx["above_ema_200"])
        self.state.above_vwap = bool(ctx["above_vwap"])

        wk = get_weekly_trend(daily_df)
        self.state.weekly_trend_up = bool(wk["weekly_trend_up"])

        # 3. Detect candlestick patterns on 5-MIN bars
        patterns = detect_patterns(intraday_df)
        self.state.candle_patterns = [k for k, v in patterns.items() if v]

        # 4. Pick strategies based on DAILY regime
        selection = select_strategies(daily_df, self.symbol)
        self.state.regime = selection["regime"]
        self.state.regime_reason = selection["reason"]
        self.state.strategy_weights = selection["strategies"]

        # 5. Run selected strategies on 5-MIN bars (entry timeframe)
        strategy_scores = {}
        num_bullish = 0
        num_bearish = 0
        weighted_sum = 0.0
        total_weight = 0.0

        for strat_name, weight in selection["strategies"].items():
            if weight <= 0:
                continue

            strat = self.strategies.get(strat_name)
            if not strat:
                continue

            signals = strat.generate_signals({self.symbol: intraday_df})
            score = signals.get(self.symbol, 0.0)
            strategy_scores[strat_name] = round(float(score), 3)

            weighted_sum += score * weight
            total_weight += weight

            if score > 0.1:
                num_bullish += 1
            elif score < -0.1:
                num_bearish += 1

        self.state.strategy_scores = strategy_scores
        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.state.score = round(float(composite), 3)

        # 6. Decision: LONG, SHORT, or NONE
        from data import CRYPTO_SYMBOLS
        is_crypto = self.symbol in CRYPTO_SYMBOLS
        # Crypto needs only 2/5 strategies (fewer patterns in 24/7 markets)
        min_agreeing = self.config["signals"].get("min_crypto_agreeing", 2) if is_crypto \
            else self.config["signals"].get("min_agreeing_strategies", 3)
        min_score = self.config["signals"].get("min_crypto_score", 0.15) if is_crypto \
            else self.config["signals"]["min_composite_score"]

        # Check for LONG signal
        has_long = (num_bullish >= min_agreeing and composite >= min_score)
        # Check for SHORT signal (3+ strategies bearish, negative composite)
        has_short = (num_bearish >= min_agreeing and composite <= -min_score)

        self.state.num_agreeing = num_bullish if has_long else num_bearish if has_short else max(num_bullish, num_bearish)

        has_signal = has_long or has_short
        signal_type = Action.BUY if has_long else Action.SHORT if has_short else Action.NONE
        direction_str = "LONG" if has_long else "SHORT" if has_short else ""

        # Confirmation: signal must persist across 2 checks (persisted to disk)
        if has_signal and self.state.prev_signal:
            self.state.confirmed = True
            self.state.action = signal_type
            self.state.status = "signal"
            _save_pending_state(self.symbol, True)
            self.log.info(
                f"CONFIRMED {direction_str} SIGNAL: score={composite:.3f} "
                f"confluence={self.state.num_agreeing}/{len(selection['strategies'])} "
                f"regime={selection['regime']}"
            )
        elif has_signal:
            self.state.prev_signal = True
            self.state.confirmed = False
            self.state.action = Action.NONE
            self.state.status = "pending"
            _save_pending_state(self.symbol, True)
            self.log.info(
                f"Pending {direction_str}: score={composite:.3f} "
                f"confluence={self.state.num_agreeing} — waiting for confirmation"
            )
        else:
            self.state.prev_signal = False
            self.state.confirmed = False
            self.state.action = Action.NONE
            self.state.status = "watching"
            _save_pending_state(self.symbol, False)

    def get_bars(self) -> pd.DataFrame | None:
        """Return cached bars for order sizing."""
        return self._bars


# ── Pending signal persistence (survives restarts) ────────

_pending_lock = threading.Lock()


def _load_pending_state(symbol: str) -> bool:
    """Load whether a symbol had a pending signal from disk."""
    try:
        if not os.path.exists(PENDING_STATE_FILE):
            return False
        with open(PENDING_STATE_FILE, "r") as f:
            data = json.load(f)
        return data.get(symbol, False)
    except Exception:
        return False


def _save_pending_state(symbol: str, has_signal: bool):
    """Save pending signal state to disk (thread-safe)."""
    with _pending_lock:
        try:
            data = {}
            if os.path.exists(PENDING_STATE_FILE):
                with open(PENDING_STATE_FILE, "r") as f:
                    data = json.load(f)
            data[symbol] = has_signal
            with open(PENDING_STATE_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
