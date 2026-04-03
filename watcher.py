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


class Action(Enum):
    NONE = "none"
    BUY = "buy"
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
        """Full analysis cycle for this stock."""
        self.state.status = "analyzing"

        # 1. Fetch candle data
        bars = self.data.get_bars([self.symbol], timeframe="1Day", days=250)
        if self.symbol not in bars or len(bars[self.symbol]) < 30:
            self.state.status = "no_data"
            return

        df = bars[self.symbol]
        self._bars = df
        self.state.last_price = float(df["close"].iloc[-1])

        # 2. Read the chart — what kind of market is this stock in?
        ctx = get_trend_context(df)
        self.state.adx = float(round(ctx["adx"], 1))
        self.state.trend_direction = str(ctx["direction"])
        self.state.above_200ema = bool(ctx["above_ema_200"])
        self.state.above_vwap = bool(ctx["above_vwap"])

        wk = get_weekly_trend(df)
        self.state.weekly_trend_up = bool(wk["weekly_trend_up"])

        # 3. Detect candlestick patterns
        patterns = detect_patterns(df)
        self.state.candle_patterns = [k for k, v in patterns.items() if v]

        # 4. Pick the right strategies for THIS stock
        selection = select_strategies(df, self.symbol)
        self.state.regime = selection["regime"]
        self.state.regime_reason = selection["reason"]
        self.state.strategy_weights = selection["strategies"]

        # 5. Run selected strategies with their weights
        strategy_scores = {}
        num_agreeing = 0
        weighted_sum = 0.0
        total_weight = 0.0

        for strat_name, weight in selection["strategies"].items():
            if weight <= 0:
                continue

            strat = self.strategies.get(strat_name)
            if not strat:
                continue

            signals = strat.generate_signals({self.symbol: df})
            score = signals.get(self.symbol, 0.0)
            strategy_scores[strat_name] = round(float(score), 3)

            weighted_sum += score * weight
            total_weight += weight

            if score > 0.1:
                num_agreeing += 1

        self.state.strategy_scores = strategy_scores
        self.state.num_agreeing = num_agreeing

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.state.score = round(float(composite), 3)

        # 6. Decision: does this stock pass all filters?
        min_agreeing = self.config["signals"].get("min_agreeing_strategies", 3)
        min_score = self.config["signals"]["min_composite_score"]

        has_signal = (num_agreeing >= min_agreeing and composite >= min_score)

        # Confirmation: signal must persist across 2 checks
        if has_signal and self.state.prev_signal:
            self.state.confirmed = True
            self.state.action = Action.BUY
            self.state.status = "signal"
            self.log.info(
                f"CONFIRMED BUY SIGNAL: score={composite:.3f} "
                f"confluence={num_agreeing}/{len(selection['strategies'])} "
                f"regime={selection['regime']}"
            )
        elif has_signal:
            self.state.prev_signal = True
            self.state.confirmed = False
            self.state.action = Action.NONE
            self.state.status = "pending"
            self.log.info(
                f"Pending signal: score={composite:.3f} "
                f"confluence={num_agreeing} — waiting for confirmation"
            )
        else:
            self.state.prev_signal = False
            self.state.confirmed = False
            self.state.action = Action.NONE
            self.state.status = "watching"

    def get_bars(self) -> pd.DataFrame | None:
        """Return cached bars for order sizing."""
        return self._bars
