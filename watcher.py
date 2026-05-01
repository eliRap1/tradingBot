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
from trend import get_trend_context, get_weekly_trend, get_hourly_bias
from indicators import supertrend, pivot_high, pivot_low, stochastic_rsi, last_pivot_value, order_flow_imbalance
from crypto_sentiment import crypto_sentiment
from ib_data import IB_CRYPTO_SYMBOLS
from instrument_classifier import InstrumentClassifier
from state_db import StateDB
from utils import setup_logger


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
    macro_regime: str = ""
    sector: str = ""


class StockWatcher:
    """
    Dedicated thread that watches a single stock.
    """

    def __init__(self, symbol: str, config: dict, data_fetcher,
                 interval: int = 60, sector_regime_getter=None,
                 strategies: dict | None = None, sector: str = "other",
                 current_regime: str = "bull_choppy"):
        self.symbol = symbol
        self.config = config
        self.data = data_fetcher
        self.interval = interval
        self.state = WatcherState(symbol=symbol)
        self.log = setup_logger(f"watcher.{symbol}")
        self._sector_regime_getter = sector_regime_getter
        self._sector = sector
        self._current_regime = current_regime
        self.state.sector = sector

        # Initialize strategy instances (one per watcher).
        # If strategies dict provided (from StrategyRouter), instantiate only those.
        # Otherwise fall back to all strategies for backward compat.
        if strategies is not None:
            self.strategies = {
                name: cls(config) for name, cls in ALL_STRATEGIES.items()
                if name in strategies
            }
        else:
            self.strategies = {
                name: cls(config) for name, cls in ALL_STRATEGIES.items()
            }
        # Weight overrides from StrategyRouter (None = use select_strategies())
        self._strategy_weights_override = strategies

        self._thread = None
        self._stop_event = threading.Event()
        self._bars = None  # cached bars
        self._bars_cache_time = None  # track cache age
        self._max_cache_age_seconds = 300  # refresh bars every 5 min max
        self._alpha_decay = {}  # {strategy_name: float} set by coordinator
        self._macro_regime: str = "bull"
        self._threshold_overrides: dict | None = None
        self._clf = InstrumentClassifier(config)

        # Restore pending signal state from disk (survives restarts)
        self.state.prev_signal = _load_pending_state(symbol)

    def set_strategy_weights(self, weights: dict[str, float], regime: str | None = None):
        self._strategy_weights_override = weights
        if regime:
            self._current_regime = regime
        for name in weights:
            if name not in self.strategies and name in ALL_STRATEGIES:
                self.strategies[name] = ALL_STRATEGIES[name](self.config)

    def set_threshold_overrides(
        self,
        min_score: float | None = None,
        min_agreeing: int | None = None,
        mode: str = "normal",
        reason: str = "",
    ):
        if min_score is None and min_agreeing is None:
            self._threshold_overrides = None
            return
        self._threshold_overrides = {
            "min_score": min_score,
            "min_agreeing": min_agreeing,
            "mode": mode,
            "reason": reason,
        }

    def clear_threshold_overrides(self):
        self._threshold_overrides = None

    def _entry_thresholds(
        self,
        asset_type: str | None = None,
        is_crypto: bool | None = None,
    ) -> tuple[int, float]:
        if asset_type is None:
            asset_type = "crypto" if is_crypto else "stock"
        asset_cfg = self.config.get("risk", {}).get("asset_overrides", {}).get(asset_type, {})
        if asset_type == "crypto":
            min_agreeing = self.config["signals"].get("min_crypto_agreeing", 2)
            min_score = self.config["signals"].get("min_crypto_score", 0.15)
        else:
            min_agreeing = self.config["signals"].get("min_agreeing_strategies", 3)
            min_score = self.config["signals"]["min_composite_score"]
        min_agreeing = int(asset_cfg.get("min_agreeing", min_agreeing))
        min_score = float(asset_cfg.get("min_score", min_score))
        if asset_type == "stock" and self._threshold_overrides:
            override_score = self._threshold_overrides.get("min_score")
            override_agreeing = self._threshold_overrides.get("min_agreeing")
            if override_score is not None:
                min_score = max(min_score, float(override_score))
            if override_agreeing is not None:
                min_agreeing = max(min_agreeing, int(override_agreeing))
        return min_agreeing, min_score

    def _crypto_regime_allows_long(self, daily_df: pd.DataFrame) -> bool:
        cfg = (
            self.config.get("risk", {})
            .get("asset_overrides", {})
            .get("crypto", {})
            .get("regime_filter", {})
        )
        if not cfg.get("enabled", False):
            return True

        ema_period = int(cfg.get("ema_period", 20))
        lookback = int(cfg.get("return_lookback_bars", 20))
        min_len = max(ema_period, lookback) + 1
        if daily_df is None or len(daily_df) < min_len:
            return False

        def above_ema(df: pd.DataFrame) -> bool:
            close = df["close"]
            ema = close.ewm(span=ema_period, min_periods=max(2, ema_period // 2)).mean().iloc[-1]
            return float(close.iloc[-1]) > float(ema)

        def ret_ok(df: pd.DataFrame) -> bool:
            if len(df) <= lookback:
                return False
            close = df["close"]
            base = float(close.iloc[-lookback - 1])
            if base <= 0:
                return False
            ret = float(close.iloc[-1]) / base - 1.0
            return ret >= float(cfg.get("min_benchmark_return", 0.0))

        if cfg.get("require_symbol_above_ema", True) and not above_ema(daily_df):
            return False

        benchmark = str(cfg.get("benchmark_symbol", "BTC/USD"))
        if benchmark == self.symbol:
            benchmark_df = daily_df
        else:
            benchmark_df = self.data.get_intraday_bars(
                benchmark, timeframe="1Day", days=250, cache_only=True
            )
        if benchmark_df is None or len(benchmark_df) < min_len:
            return False
        if cfg.get("require_benchmark_above_ema", True) and not above_ema(benchmark_df):
            return False
        return ret_ok(benchmark_df)

    def start(self):
        """No-op — watchers no longer spawn threads. Coordinator calls
        analyze_once() directly on its main loop after priming the bar cache.
        Kept as a shim so existing call sites stay valid.
        """
        self.state.status = "watching"
        self.log.info(f"Started watching {self.symbol}")

    def stop(self):
        """No-op — no thread to stop."""
        self._stop_event.set()
        self.state.status = "stopped"
        self.log.info(f"Stopped watching {self.symbol}")

    def analyze_once(self):
        """Run a single analyze cycle on the caller's thread.
        Must be called after ib_data._cache is primed — never blocks on IB.
        """
        t0 = time.time()
        try:
            self.state.status = "watching"
            self._analyze()
            self.state.last_update = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            self.state.error = str(e)
            self.state.status = "error"
            self.log.error(f"Error: {e}")
        dur = time.time() - t0
        self.log.debug(
            f"analyze status={self.state.status} score={self.state.score:+.3f} "
            f"action={self.state.action.name} dur={dur:.2f}s"
        )

    def _analyze(self):
        """Full analysis cycle for this stock.

        Multi-timeframe approach:
          - Daily bars → trend context (200 EMA, ADX, weekly trend)
          - 5-min bars → entry signals (strategies, candle patterns)
          
        Memory management: Clear old bars to prevent memory leaks in long-running bot.
        """
        self.state.status = "analyzing"
        
        # Clear old cached bars periodically to prevent memory buildup
        now = time.time()
        if self._bars is not None and self._bars_cache_time:
            if now - self._bars_cache_time > self._max_cache_age_seconds:
                self._bars = None  # Force refresh

        # 1a. Fetch DAILY bars for trend context (cache-only — coordinator primes)
        daily_df = self.data.get_intraday_bars(self.symbol, timeframe="1Day", days=250, cache_only=True)
        if daily_df is None or len(daily_df) < 30:
            self.state.status = "no_data"
            return

        # 1b. Fetch 5-MINUTE bars for entry signals (cache-only)
        intraday_df = self.data.get_intraday_bars(self.symbol, timeframe="5Min", days=5, cache_only=True)
        if intraday_df is None or len(intraday_df) < 30:
            # Fall back to daily if intraday not available (market closed, etc.)
            intraday_df = daily_df

        # 1c. Fetch 1-HOUR bars for intermediate confirmation (cache-only)
        hourly_df = self.data.get_intraday_bars(self.symbol, timeframe="1Hour", days=10, cache_only=True)
        if hourly_df is not None and len(hourly_df) >= 25:
            self._hourly_bias = get_hourly_bias(hourly_df)
        else:
            self._hourly_bias = {"bias": "neutral"}  # neutral = don't block

        self._bars = intraday_df  # used for order sizing (ATR on entry timeframe)
        self._bars_cache_time = time.time()  # Track cache age
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

        # 4. Pick strategies based on DAILY regime (+ sector context if available)
        _sector_reg = None
        if self._sector_regime_getter is not None:
            try:
                from filters import SECTOR_MAP
                _sector_label = SECTOR_MAP.get(self.symbol, "other")
                if _sector_label != "other":
                    _sector_reg = self._sector_regime_getter(_sector_label)
            except Exception:
                _sector_reg = None
        if self._strategy_weights_override is not None:
            # StrategyRouter provided weights — bypass select_strategies
            selection = {
                "regime": "router_assigned",
                "reason": "StrategyRouter per-instrument weights",
                "strategies": self._strategy_weights_override,
            }
        else:
            selection = select_strategies(daily_df, self.symbol, sector_regime=_sector_reg)
        self._apply_bear_veto(selection, ctx)
        self.state.regime = selection["regime"]
        self.state.regime_reason = selection["reason"]
        self.state.strategy_weights = selection["strategies"]
        self.state.macro_regime = self._current_regime

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

            # Apply alpha decay factor (reduces weight of decaying strategies)
            decay = self._alpha_decay.get(strat_name, 1.0)
            adj_weight = weight * decay

            signals = strat.generate_signals({self.symbol: intraday_df})
            score = signals.get(self.symbol, 0.0)
            strategy_scores[strat_name] = round(float(score), 3)

            weighted_sum += score * adj_weight
            total_weight += adj_weight

            if score > 0.1:
                num_bullish += 1
            elif score < -0.1:
                num_bearish += 1

        self.state.strategy_scores = strategy_scores
        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.state.score = round(float(composite), 3)

        # 5b. Order flow imbalance filter
        flow = order_flow_imbalance(intraday_df, lookback=20)

        # 6. Decision: LONG, SHORT, or NONE
        asset_type = self._clf.classify(self.symbol)
        is_crypto = asset_type == "crypto" or self.symbol in IB_CRYPTO_SYMBOLS
        # Crypto needs only 2/5 strategies (fewer patterns in 24/7 markets)
        min_agreeing, min_score = self._entry_thresholds(asset_type)

        # Crypto sentiment filter (funding rate proxy)
        crypto_penalize_long = False
        crypto_penalize_short = False
        if is_crypto:
            sent = crypto_sentiment(intraday_df)
            crypto_penalize_long = sent["penalize_longs"]
            crypto_penalize_short = sent["penalize_shorts"]
            if sent["extreme_greed"] or sent["extreme_fear"]:
                self.log.info(f"Crypto sentiment: {sent['sentiment']:+.2f} "
                              f"({'EXTREME GREED' if sent['extreme_greed'] else 'EXTREME FEAR'})")

        # Check for LONG signal (blocked if strong bearish order flow or crypto extreme greed)
        crypto_regime_allows_long = True if not is_crypto else self._crypto_regime_allows_long(daily_df)
        has_long = (num_bullish >= min_agreeing and composite >= min_score
                    and not flow["is_bearish_flow"]
                    and not crypto_penalize_long
                    and crypto_regime_allows_long)
        asset_cfg = self.config.get("risk", {}).get("asset_overrides", {}).get(asset_type, {})
        allow_shorts = bool(asset_cfg.get("allow_shorts", asset_type != "stock"))
        # Check for SHORT signal — only block if shorts are already overcrowded
        # (extreme bearish flow = squeeze risk). Bullish flow on a bearish signal
        # means trapped longs — that IS the short setup, do NOT block it.
        has_short = (allow_shorts
                     and num_bearish >= min_agreeing and composite <= -min_score
                     and not (flow["is_bearish_flow"] and flow.get("flow_strength", 0) > 0.40)
                     and not crypto_penalize_short)

        self.state.num_agreeing = num_bullish if has_long else num_bearish if has_short else max(num_bullish, num_bearish)

        has_signal = has_long or has_short
        signal_type = Action.BUY if has_long else Action.SHORT if has_short else Action.NONE
        direction_str = "LONG" if has_long else "SHORT" if has_short else ""

        # Hourly bias gate: 1H must agree or be neutral
        hourly_bias = getattr(self, "_hourly_bias", {}).get("bias", "neutral")
        hourly_agrees = (
            hourly_bias == "neutral"
            or (has_long and hourly_bias == "bullish")
            or (has_short and hourly_bias == "bearish")
        )

        # Confirmation: signal must persist across 2 checks + hourly agreement (persisted to disk)
        if has_signal and self.state.prev_signal and hourly_agrees:
            self.state.confirmed = True
            self.state.action = signal_type
            self.state.status = "signal"
            _save_pending_state(self.symbol, True)
            self.log.info(
                f"CONFIRMED {direction_str} SIGNAL: score={composite:.3f} "
                f"confluence={self.state.num_agreeing}/{len(selection['strategies'])} "
                f"regime={selection['regime']} 1H_bias={hourly_bias}"
            )
        elif has_signal and self.state.prev_signal and not hourly_agrees:
            # Signal confirmed on 5min but 1H disagrees — block it
            self.state.prev_signal = True
            self.state.confirmed = False
            self.state.action = Action.NONE
            self.state.status = "pending"
            _save_pending_state(self.symbol, True)
            self.log.info(
                f"BLOCKED {direction_str}: 1H bias={hourly_bias} conflicts — waiting for alignment"
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

    def _apply_bear_veto(self, selection: dict, ctx: dict):
        macro = getattr(self, "_macro_regime", "bull")
        if macro != "bear" or ctx.get("adx", 0) <= 30:
            return

        for strat in ("momentum", "gap", "breakout"):
            if strat in selection["strategies"]:
                selection["strategies"][strat] = 0.0

        total = sum(selection["strategies"].values())
        if total > 0:
            selection["strategies"] = {
                name: round(weight / total, 4)
                for name, weight in selection["strategies"].items()
            }
        self.log.debug(
            f"Bear veto applied (ADX={ctx.get('adx', 0):.0f}): momentum/gap/breakout weights zeroed"
        )


# ── Pending signal persistence (survives restarts) ────────

_pending_lock = threading.Lock()
_pending_db: StateDB | None = None
_pending_db_init_lock = threading.Lock()


def _get_pending_db() -> StateDB:
    """Return module-level StateDB singleton (lazy init, thread-safe)."""
    global _pending_db
    if _pending_db is None:
        with _pending_db_init_lock:
            if _pending_db is None:
                db = StateDB()
                db.migrate_from_json()
                _pending_db = db
    return _pending_db


def _load_pending_state(symbol: str) -> bool:
    """Load whether a symbol had a pending signal from SQLite."""
    try:
        return _get_pending_db().get_pending_signal(symbol)
    except Exception:
        return False


def _save_pending_state(symbol: str, has_signal: bool):
    """Save pending signal state to SQLite (thread-safe)."""
    with _pending_lock:
        try:
            _get_pending_db().set_pending_signal(symbol, has_signal)
        except Exception:
            pass
