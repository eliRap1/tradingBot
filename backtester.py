"""
Backtesting engine with realistic slippage modeling.

Simulates the full trading pipeline bar-by-bar:
  bars → strategies → confluence → sizing → fill simulation → stops/targets

Key rules:
  - NO look-ahead bias: only uses bars[:i] at step i
  - Realistic fills: slippage + volume impact
  - Stops checked on bar high/low (not just close)
  - Gap handling: stops fill at open when price gaps through
"""

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from strategies import ALL_STRATEGIES
from strategy_selector import select_strategies
from strategy_router import StrategyRouter
from instrument_classifier import InstrumentClassifier
from filters import SECTOR_MAP, compute_regime_guard_decision
from edge.regime_gate import is_chop_or_panic
from candles import detect_patterns, bullish_score
from trend import get_trend_context
from risk import RiskManager
from signals import Opportunity
from ib_data import IB_CRYPTO_SYMBOLS as CRYPTO_SYMBOLS
from utils import setup_logger
from performance import apr_pct, profit_usd
import ta as _ta_root

log = setup_logger("backtester")

CROSS_SECTIONAL_STRATEGIES = {"relative_strength_rotation"}


# ── Slippage Model ──────────────────────────────────────────

class SlippageModel:
    """Realistic fill price simulation with slippage + spread costs.

    Stocks: ~15 bps round-trip (10 slippage + 5 spread per side)
    Crypto: ~60 bps round-trip (20 slippage + 10 spread per side, wider Alpaca spreads)
    """

    def __init__(self, config: dict):
        bt_cfg = config.get("backtest", {})
        self._config = bt_cfg
        self.base_slippage_pct = bt_cfg.get("slippage_pct", 0.001)
        self.spread_pct = bt_cfg.get("spread_pct", 0.0005)
        self.crypto_slippage_pct = bt_cfg.get("crypto_slippage_pct", 0.002)
        self.crypto_spread_pct = bt_cfg.get("crypto_spread_pct", 0.001)
        self.volume_impact = bt_cfg.get("volume_impact", True)
        self.commission_per_share = bt_cfg.get("commission_per_share", 0.0)

        # Smart orders simulate limit @ midpoint fills (entry only).
        # Round-trip cost drops because we earn half the spread on entry,
        # but stop/TP fills (exits) still cross the spread.
        exec_cfg = config.get("execution", {})
        self.smart_orders = bool(exec_cfg.get("smart_orders", False))
        self.smart_entry_discount = float(exec_cfg.get("smart_entry_discount", 0.6))

    def get_fill_price(self, price: float, side: str, volume: int,
                       qty: float, is_crypto: bool = False,
                       asset_type: str | None = None,
                       is_entry: bool = False) -> float:
        """
        Calculate realistic fill price including slippage and spread.
        Uses higher costs for crypto due to wider Alpaca spreads.
        When smart_orders + is_entry, scale impact down by smart_entry_discount
        to model limit-at-midpoint entries (saves half-spread).
        """
        asset_type = asset_type or ("crypto" if is_crypto else "stock")
        participation = float(qty) / max(float(volume), 1.0)
        if asset_type == "crypto":
            impact = self.crypto_slippage_pct + self.crypto_spread_pct
        elif asset_type == "futures":
            bt_cfg = getattr(self, "_config", {})
            impact = (
                bt_cfg.get("futures_slippage_pct", self.base_slippage_pct)
                + bt_cfg.get("futures_spread_pct", self.spread_pct)
            )
        else:
            impact = self.base_slippage_pct + self.spread_pct
        if self.volume_impact and participation > 0.01:
            impact += participation * 0.01

        if self.smart_orders and is_entry and asset_type != "crypto":
            impact *= self.smart_entry_discount

        if side == "buy":
            return price * (1 + impact)
        else:
            return price * (1 - impact)

    def get_commission(self, qty: float) -> float:
        return self.commission_per_share * float(qty)


# ── Data Classes ────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    side: str       # "buy" or "sell"
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    entry_date: str = ""


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str  # "stop_loss", "take_profit", "end_of_data"
    commission: float = 0.0


@dataclass
class BacktestResult:
    trades: list
    equity_curve: list         # [(date, equity), ...]
    total_return_pct: float
    profit_usd: float
    apr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    avg_bars_held: float
    commission_total: float


# ── Backtester Engine ───────────────────────────────────────

class Backtester:
    """
    Bar-by-bar backtesting engine.

    Usage:
        bt = Backtester(config)
        result = bt.run(bars_dict, min_bars=50)
    """

    def __init__(self, config: dict, initial_equity: float = 100000.0):
        self.config = config
        self.initial_equity = initial_equity
        self.slippage = SlippageModel(config)

        # Initialize strategies
        self.strategies = {
            name: cls(config) for name, cls in ALL_STRATEGIES.items()
        }

        self.min_score = config["signals"]["min_composite_score"]
        self.min_agreeing = config["signals"]["min_agreeing_strategies"]
        self.max_positions = config["signals"]["max_positions"]

        # StrategyRouter: per-sector/regime weights from research/sector_weights.json
        # Falls back to select_strategies when sector missing or no router weights.
        self.router = StrategyRouter(config)
        self.clf = InstrumentClassifier(config)
        self._use_router = bool(self.router._sector_weights)

    def _regime_label(self, hist: pd.DataFrame) -> str:
        """Lightweight regime label for router lookup: bull/bear + trending/choppy."""
        if len(hist) < 50:
            return "bull_choppy"
        close = hist["close"]
        ema50 = close.ewm(span=50, min_periods=20).mean().iloc[-1]
        bull = close.iloc[-1] > ema50
        try:
            adx = _ta_root.trend.ADXIndicator(
                high=hist["high"], low=hist["low"], close=close, window=14
            ).adx().iloc[-1]
            trending = adx is not None and adx >= 20.0
        except Exception:
            trending = False
        tail = "trending" if trending else "choppy"
        head = "bull" if bull else "bear"
        return f"{head}_{tail}"

    def _router_weights(self, sym: str, hist: pd.DataFrame) -> dict | None:
        """Return router weights for symbol, or None to fall back."""
        instr = self.clf.classify(sym)
        if instr != "stock":
            weights = self.router.get_strategies(instr, None, self._regime_label(hist))
            return weights if weights else None
        if not self._use_router:
            return None
        sector = SECTOR_MAP.get(sym, "other")
        regime = self._regime_label(hist)
        weights = self.router.get_strategies(instr, sector, regime)
        # Empty dict = no data for that (sector, regime) cell -> fall back
        return weights if weights else None

    def _regime_guard(self, trades: list[BacktestTrade]):
        stock_cfg = self._asset_risk_cfg("stock")
        guard_trades = [{"pnl": t.pnl} for t in trades]
        return compute_regime_guard_decision(
            guard_trades,
            self.config,
            base_max_positions=int(stock_cfg.get("max_positions", self.max_positions)),
            base_min_score=float(stock_cfg.get("min_score", self.min_score)),
            base_min_agreeing=int(stock_cfg.get("min_agreeing", self.min_agreeing)),
        )

    def _futures_root(self, symbol: str) -> str:
        sym = symbol.upper()
        for contract in self.config.get("futures", {}).get("contracts", []):
            root = str(contract.get("root", "")).upper()
            if root and sym.startswith(root):
                return root
        return sym

    def _asset_risk_cfg(self, asset_type: str) -> dict:
        return self.config.get("risk", {}).get("asset_overrides", {}).get(asset_type, {})

    def _risk_value(self, asset_type: str, key: str, default_key: str | None = None):
        asset_cfg = self._asset_risk_cfg(asset_type)
        if key in asset_cfg:
            return asset_cfg[key]
        return self.config.get("risk", {}).get(default_key or key)

    def _contract_multiplier(self, symbol: str, asset_type: str | None = None) -> float:
        asset_type = asset_type or self.clf.classify(symbol)
        if asset_type != "futures":
            return float(self._asset_risk_cfg(asset_type).get("contract_multiplier", 1.0))
        cfg = self._asset_risk_cfg("futures")
        multipliers = cfg.get("contract_multipliers", {})
        return float(multipliers.get(self._futures_root(symbol), cfg.get("contract_multiplier", 1.0)))

    def _allow_fractional_qty(self, asset_type: str) -> bool:
        cfg = self._asset_risk_cfg(asset_type)
        return bool(cfg.get("allow_fractional_qty", asset_type == "crypto"))

    def _min_qty(self, asset_type: str) -> float:
        return float(self._asset_risk_cfg(asset_type).get("min_qty", 0.000001 if asset_type == "crypto" else 1.0))

    def _asset_position_count(self, positions: dict[str, Position], asset_type: str) -> int:
        return sum(1 for open_sym in positions if self.clf.classify(open_sym) == asset_type)

    def _asset_slots_available(self, positions: dict[str, Position], asset_type: str, regime_guard) -> int:
        if asset_type == "stock":
            max_positions = regime_guard.max_positions
        else:
            max_positions = int(
                self._asset_risk_cfg(asset_type).get(
                    "max_positions",
                    self.config.get("signals", {}).get("max_crypto_positions", 3)
                    if asset_type == "crypto" else self.max_positions,
                )
            )
        return max(0, max_positions - self._asset_position_count(positions, asset_type))

    def _volatility_size_mult(self, hist: pd.DataFrame, asset_type: str) -> float:
        cfg = self._asset_risk_cfg(asset_type)
        vol_cfg = cfg.get("volatility_target", self.config.get("risk", {}).get("volatility_target", {}))
        if not vol_cfg or not vol_cfg.get("enabled", True):
            return 1.0
        lookback = int(vol_cfg.get("lookback_bars", 20))
        if len(hist) < lookback + 2:
            return 1.0
        returns = hist["close"].pct_change().dropna().tail(lookback)
        realized = float(returns.std() * math.sqrt(252)) if len(returns) > 1 else 0.0
        if not np.isfinite(realized) or realized <= 0:
            return 1.0
        target = float(vol_cfg.get("target_annual_vol", 0.25))
        mult = target / realized
        return max(float(vol_cfg.get("min_mult", 0.35)), min(float(vol_cfg.get("max_mult", 1.50)), mult))

    def _round_qty(self, qty: float, asset_type: str) -> float:
        if self._allow_fractional_qty(asset_type):
            decimals = int(self._asset_risk_cfg(asset_type).get("qty_decimals", 8))
            return round(max(0.0, qty), decimals)
        return float(int(qty))

    def _score_strategy(self, strat_name: str, strat, sym: str, hist: pd.DataFrame,
                        date, bars_dict: dict[str, pd.DataFrame],
                        positions: dict[str, Position], cache: dict) -> float:
        if strat_name in CROSS_SECTIONAL_STRATEGIES:
            if strat_name not in cache:
                histories = {}
                for other_sym, other_df in bars_dict.items():
                    if other_sym in positions:
                        continue
                    other_hist = other_df.loc[other_df.index <= date]
                    if len(other_hist) >= 30:
                        histories[other_sym] = other_hist
                cache[strat_name] = strat.generate_signals(histories)
            return float(cache[strat_name].get(sym, 0.0) or 0.0)
        result = strat.generate_signals({sym: hist})
        return float(result.get(sym, 0.0) or 0.0)

    def _crypto_regime_allows_long(
        self,
        sym: str,
        hist: pd.DataFrame,
        date,
        bars_dict: dict[str, pd.DataFrame],
    ) -> bool:
        cfg = self._asset_risk_cfg("crypto").get("regime_filter", {})
        if not cfg.get("enabled", False):
            return True

        ema_period = int(cfg.get("ema_period", 20))
        lookback = int(cfg.get("return_lookback_bars", 20))
        min_len = max(ema_period, lookback) + 1
        if len(hist) < min_len:
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

        if cfg.get("require_symbol_above_ema", True) and not above_ema(hist):
            return False

        benchmark = str(cfg.get("benchmark_symbol", "BTC/USD"))
        bench_df = bars_dict.get(benchmark)
        if bench_df is None and benchmark != sym:
            return False
        bench_hist = hist if benchmark == sym else bench_df.loc[bench_df.index <= date]
        if len(bench_hist) < min_len:
            return False
        if cfg.get("require_benchmark_above_ema", True) and not above_ema(bench_hist):
            return False
        return ret_ok(bench_hist)

    def run(self, bars_dict: dict[str, pd.DataFrame],
            min_bars: int = 50,
            benchmark_bars: pd.DataFrame | None = None) -> BacktestResult:
        """
        Run backtest on provided bars.

        bars_dict: {symbol: DataFrame with OHLCV}
        min_bars: minimum bars before starting to trade
        benchmark_bars: optional SPY (or other index) DataFrame for the
            leading regime gate (chop/panic kill-switch). Falls back to
            bars_dict.get("SPY") if not provided.
        """
        # Find the common date range
        all_dates = set()
        for df in bars_dict.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        if len(all_dates) < min_bars:
            log.warning("Not enough bars for backtesting")
            return self._empty_result()

        equity = self.initial_equity
        positions: dict[str, Position] = {}
        trades: list[BacktestTrade] = []
        equity_curve = [(str(all_dates[0]), equity)]
        daily_returns = []

        log.info(f"Backtesting {len(bars_dict)} symbols over {len(all_dates)} bars "
                 f"(starting equity=${equity:,.2f})")

        total_signals_found = 0
        progress_interval = max(1, len(all_dates) // 10)
        last_guard_mode = ""

        # Bar-by-bar simulation
        for i in range(min_bars, len(all_dates)):
            if (i - min_bars) % progress_interval == 0:
                pct = (i - min_bars) / (len(all_dates) - min_bars) * 100
                log.info(f"  Progress: {pct:.0f}% | Trades: {len(trades)} | "
                         f"Positions: {len(positions)} | Signals found: {total_signals_found}")
            date = all_dates[i]
            prev_equity = equity

            # ── 1. Check exits on current bar ───────────────
            symbols_to_close = []
            for sym, pos in list(positions.items()):
                if sym not in bars_dict:
                    continue
                df = bars_dict[sym]
                # Get the bar at this date
                if date not in df.index:
                    continue

                bar_high = float(df.loc[date, "high"])
                bar_low = float(df.loc[date, "low"])
                bar_open = float(df.loc[date, "open"])
                bar_close = float(df.loc[date, "close"])
                bar_vol = int(df.loc[date, "volume"])

                exit_price = None
                exit_reason = None
                asset_type = self.clf.classify(sym)
                multiplier = self._contract_multiplier(sym, asset_type)

                if pos.side == "buy":
                    # Stop loss hit? (low goes below stop)
                    if bar_low <= pos.stop_loss:
                        if bar_open <= pos.stop_loss:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "sell", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.stop_loss, "sell", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        exit_reason = "stop_loss"

                    elif bar_high >= pos.take_profit:
                        if bar_open >= pos.take_profit:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "sell", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.take_profit, "sell", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        exit_reason = "take_profit"

                else:  # short
                    if bar_high >= pos.stop_loss:
                        if bar_open >= pos.stop_loss:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "buy", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.stop_loss, "buy", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        exit_reason = "stop_loss"

                    elif bar_low <= pos.take_profit:
                        if bar_open <= pos.take_profit:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "buy", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.take_profit, "buy", bar_vol, pos.qty,
                                is_crypto=asset_type == "crypto",
                                asset_type=asset_type)
                        exit_reason = "take_profit"

                if exit_price is not None:
                    if pos.side == "buy":
                        pnl = (exit_price - pos.entry_price) * pos.qty * multiplier
                    else:
                        pnl = (pos.entry_price - exit_price) * pos.qty * multiplier

                    commission = self.slippage.get_commission(pos.qty) * 2
                    pnl -= commission
                    notional = pos.entry_price * pos.qty * multiplier
                    pnl_pct = pnl / notional if notional > 0 else 0

                    trades.append(BacktestTrade(
                        symbol=sym, side=pos.side, qty=pos.qty,
                        entry_price=pos.entry_price,
                        exit_price=round(exit_price, 2),
                        entry_date=pos.entry_date,
                        exit_date=str(date),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 4),
                        bars_held=i - pos.entry_bar,
                        exit_reason=exit_reason,
                        commission=round(commission, 2),
                    ))
                    equity += pnl
                    symbols_to_close.append(sym)

            for sym in symbols_to_close:
                del positions[sym]

            regime_guard = self._regime_guard(trades)
            if regime_guard.mode != last_guard_mode:
                log.info(
                    f"Backtest regime guard: {last_guard_mode or 'initial'} -> "
                    f"{regime_guard.mode} ({regime_guard.reason})"
                )
                last_guard_mode = regime_guard.mode

            # Leading regime gate (SPY ADX + vol). Skip new entries on chop/panic.
            bench_df = benchmark_bars if benchmark_bars is not None else bars_dict.get("SPY")
            if bench_df is not None:
                bench_hist = bench_df.loc[bench_df.index <= date]
                blocked, gate_reason = is_chop_or_panic(bench_hist, self.config)
                if blocked:
                    self._gate_block_count = getattr(self, "_gate_block_count", 0) + 1
                    equity_curve.append((str(date), equity))
                    daily_returns.append(
                        (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                    )
                    continue

            # ── 2. Generate signals (NO look-ahead) ─────────
            if len(positions) >= self.max_positions:
                equity_curve.append((str(date), equity))
                daily_returns.append((equity - prev_equity) / prev_equity if prev_equity > 0 else 0)
                continue

            # Run strategies on historical bars only (bars up to and including current)
            signals_by_sym = {}
            strategy_result_cache = {}
            for sym, df in bars_dict.items():
                if sym in positions:
                    continue

                # Only use bars up to current date (NO future data)
                hist = df.loc[df.index <= date]
                if len(hist) < 30:
                    continue

                # Strategy selection: prefer per-sector router weights, fall
                # back to regime-based select_strategies when no router data.
                router_w = self._router_weights(sym, hist)
                if router_w:
                    selection = {"strategies": router_w, "regime": "router"}
                else:
                    selection = select_strategies(hist, sym)

                # Run strategies — track both bullish and bearish agreement
                weighted_sum = 0.0
                total_weight = 0.0
                num_bullish = 0
                num_bearish = 0
                strat_scores = {}

                for strat_name, weight in selection["strategies"].items():
                    if weight <= 0:
                        continue
                    strat = self.strategies.get(strat_name)
                    if not strat:
                        continue

                    score = self._score_strategy(
                        strat_name, strat, sym, hist, date,
                        bars_dict, positions, strategy_result_cache,
                    )
                    strat_scores[strat_name] = score
                    weighted_sum += score * weight
                    total_weight += weight

                    if score > 0.1:
                        num_bullish += 1
                    elif score < -0.1:
                        num_bearish += 1

                if total_weight == 0:
                    continue

                composite = weighted_sum / total_weight
                asset_type = self.clf.classify(sym)
                asset_cfg = self._asset_risk_cfg(asset_type)
                if asset_type == "stock":
                    min_score = regime_guard.min_score
                    min_agreeing = regime_guard.min_agreeing
                elif asset_type == "crypto":
                    min_score = float(asset_cfg.get("min_score", self.config["signals"].get("min_crypto_score", self.min_score)))
                    min_agreeing = int(asset_cfg.get("min_agreeing", self.config["signals"].get("min_crypto_agreeing", 2)))
                else:
                    min_score = float(asset_cfg.get("min_score", self.min_score))
                    min_agreeing = int(asset_cfg.get("min_agreeing", self.min_agreeing))

                # Cap min_agreeing by strategies available in the bucket.
                # Sector-routed buckets sometimes hold only 1-2 strategies for
                # a regime cell, which would make min_agreeing=3 unreachable.
                bucket_size = sum(1 for w in selection["strategies"].values() if w > 0)
                min_agreeing = min(min_agreeing, max(2, bucket_size))

                # Check long signal
                long_allowed = (
                    asset_type != "crypto"
                    or self._crypto_regime_allows_long(sym, hist, date, bars_dict)
                )
                if num_bullish >= min_agreeing and composite >= min_score and long_allowed:
                    signals_by_sym[sym] = {
                        "score": composite,
                        "direction": "buy",
                        "num_agreeing": num_bullish,
                        "strat_scores": strat_scores,
                        "hist": hist,
                    }
                    total_signals_found += 1
                # Check short signal
                elif (
                    bool(asset_cfg.get("allow_shorts", asset_type != "stock"))
                    and num_bearish >= min_agreeing
                    and composite <= -min_score
                ):
                    signals_by_sym[sym] = {
                        "score": composite,
                        "direction": "sell",
                        "num_agreeing": num_bearish,
                        "strat_scores": strat_scores,
                        "hist": hist,
                    }
                    total_signals_found += 1

            # ── 3. Size and execute (long + short) ─────────
            # Sort by absolute score
            sorted_signals = sorted(signals_by_sym.items(),
                                    key=lambda x: abs(x[1]["score"]), reverse=True)

            global_slots = self.max_positions - len(positions)
            for sym, sig in sorted_signals:
                if global_slots <= 0:
                    break
                asset_type_for_slot = self.clf.classify(sym)
                if self._asset_slots_available(positions, asset_type_for_slot, regime_guard) <= 0:
                    continue

                hist = sig["hist"]
                bar = bars_dict[sym].loc[date] if date in bars_dict[sym].index else None
                if bar is None:
                    continue

                bar_close = float(bar["close"])
                bar_vol = int(bar["volume"])
                direction = sig["direction"]

                # ATR-based stops
                import ta
                atr_val = ta.volatility.AverageTrueRange(
                    high=hist["high"], low=hist["low"], close=hist["close"],
                    window=14
                ).average_true_range().iloc[-1]

                if atr_val <= 0:
                    continue

                asset_cfg = self._asset_risk_cfg(asset_type_for_slot)
                multiplier = self._contract_multiplier(sym, asset_type_for_slot)
                sl_mult = float(asset_cfg.get("stop_loss_atr_mult", self.config["risk"]["stop_loss_atr_mult"]))
                tp_mult = float(asset_cfg.get("take_profit_atr_mult", self.config["risk"]["take_profit_atr_mult"]))

                if direction == "sell":
                    # SHORT: stop above, target below
                    stop_loss = bar_close + (atr_val * sl_mult)
                    take_profit = bar_close - (atr_val * tp_mult)
                    risk_per_share = (stop_loss - bar_close) * multiplier
                    reward = bar_close - take_profit
                else:
                    # LONG: stop below, target above
                    stop_loss = bar_close - (atr_val * sl_mult)
                    take_profit = bar_close + (atr_val * tp_mult)
                    risk_per_share = (bar_close - stop_loss) * multiplier
                    reward = take_profit - bar_close

                if risk_per_share <= 0:
                    continue

                # R:R filter
                rr = (reward * multiplier) / risk_per_share
                min_rr = float(asset_cfg.get("min_risk_reward", self.config["risk"]["min_risk_reward"]))
                if rr < min_rr:
                    continue

                # Position sizing — base risk_pct, optionally Kelly-scaled.
                base_risk_pct = float(asset_cfg.get("max_portfolio_risk_pct", self.config["risk"]["max_portfolio_risk_pct"]))
                sizing_method = str(
                    asset_cfg.get("sizing_method", self.config["risk"].get("sizing_method", "volatility_adjusted"))
                ).lower()
                risk_pct = base_risk_pct
                if sizing_method in ("kelly", "kelly_fractional") and asset_type_for_slot == "stock":
                    kelly_min = int(self.config["risk"].get("kelly_min_trades", 30))
                    closed_pnls = [float(t.pnl) for t in trades if t.pnl != 0]
                    if len(closed_pnls) >= kelly_min:
                        wins = [p for p in closed_pnls if p > 0]
                        losses = [abs(p) for p in closed_pnls if p < 0]
                        if wins and losses:
                            p_win = len(wins) / len(closed_pnls)
                            avg_win = sum(wins) / len(wins)
                            avg_loss = sum(losses) / len(losses)
                            b = min(avg_win / avg_loss, 5.0) if avg_loss > 0 else 1.0
                            kelly_f = p_win - ((1.0 - p_win) / b) if b > 0 else 0.0
                            kelly_f *= float(self.config["risk"].get("kelly_fraction", 0.5))
                            if kelly_f > 0:
                                risk_pct = min(kelly_f, base_risk_pct * 2.0)
                            else:
                                # Negative-edge: skip trade entirely
                                continue
                max_risk = equity * risk_pct
                qty_by_risk = max_risk / risk_per_share
                max_pos_pct = float(asset_cfg.get("max_position_pct", self.config["risk"]["max_position_pct"]))
                leverage = float(asset_cfg.get("leverage", self.config["risk"].get("leverage", 1.0)))
                max_pos_value = equity * max_pos_pct * leverage
                qty_by_notional = max_pos_value / max(bar_close * multiplier, 1e-9)
                qty = min(qty_by_risk, qty_by_notional)
                qty *= self._volatility_size_mult(hist, asset_type_for_slot)
                if asset_type_for_slot == "stock":
                    qty *= regime_guard.size_mult
                qty = self._round_qty(qty, asset_type_for_slot)

                if qty < self._min_qty(asset_type_for_slot):
                    continue

                # Simulate fill with slippage (crypto gets wider spreads).
                # Entry: smart_orders model midpoint fill (cheaper).
                fill_price = self.slippage.get_fill_price(
                    bar_close, direction, bar_vol, qty,
                    is_crypto=asset_type_for_slot == "crypto",
                    asset_type=asset_type_for_slot,
                    is_entry=True)
                entry_price = round(fill_price, 2)
                if direction == "sell":
                    stop_loss = entry_price + (atr_val * sl_mult)
                    take_profit = entry_price - (atr_val * tp_mult)
                else:
                    stop_loss = entry_price - (atr_val * sl_mult)
                    take_profit = entry_price + (atr_val * tp_mult)

                positions[sym] = Position(
                    symbol=sym, side=direction, qty=qty,
                    entry_price=entry_price,
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    entry_bar=i,
                    entry_date=str(date),
                )
                global_slots -= 1

            # ── 4. Update equity curve ──────────────────────
            # Mark-to-market open positions
            mtm = equity
            for sym, pos in positions.items():
                if sym in bars_dict and date in bars_dict[sym].index:
                    current = float(bars_dict[sym].loc[date, "close"])
                    asset_type = self.clf.classify(sym)
                    multiplier = self._contract_multiplier(sym, asset_type)
                    if pos.side == "buy":
                        mtm += (current - pos.entry_price) * pos.qty * multiplier
                    else:
                        mtm += (pos.entry_price - current) * pos.qty * multiplier

            equity_curve.append((str(date), round(mtm, 2)))
            daily_returns.append(
                (mtm - prev_equity) / prev_equity if prev_equity > 0 else 0
            )

        # ── Close remaining positions at end ────────────────
        last_date = all_dates[-1]
        for sym, pos in list(positions.items()):
            if sym in bars_dict and last_date in bars_dict[sym].index:
                exit_price = float(bars_dict[sym].loc[last_date, "close"])
                asset_type = self.clf.classify(sym)
                multiplier = self._contract_multiplier(sym, asset_type)
                if pos.side == "buy":
                    pnl = (exit_price - pos.entry_price) * pos.qty * multiplier
                else:
                    pnl = (pos.entry_price - exit_price) * pos.qty * multiplier
                notional = pos.entry_price * pos.qty * multiplier

                trades.append(BacktestTrade(
                    symbol=sym, side=pos.side, qty=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=round(exit_price, 2),
                    entry_date=pos.entry_date,
                    exit_date=str(last_date),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl / notional, 4) if notional > 0 else 0.0,
                    bars_held=len(all_dates) - pos.entry_bar,
                    exit_reason="end_of_data",
                ))
                equity += pnl

        if equity_curve:
            equity_curve[-1] = (str(last_date), round(equity, 2))

        # ── Calculate metrics ───────────────────────────────
        return self._calculate_result(trades, equity_curve, daily_returns)

    def _calculate_result(self, trades: list[BacktestTrade],
                          equity_curve: list,
                          daily_returns: list) -> BacktestResult:
        """Calculate all performance metrics from completed trades."""
        if not trades:
            return self._empty_result(equity_curve)

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]
        n = len(pnls)

        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        ending_equity = float(equity_curve[-1][1]) if equity_curve else self.initial_equity + total_pnl
        total_profit = profit_usd(equity_curve, self.initial_equity) if equity_curve else total_pnl
        total_return = (ending_equity - self.initial_equity) / self.initial_equity
        annual_return_pct = apr_pct(equity_curve, self.initial_equity) if equity_curve else 0.0

        # Win rate
        win_rate = len(wins) / n if n > 0 else 0.0

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe ratio
        if len(daily_returns) > 1:
            mean_r = sum(daily_returns) / len(daily_returns)
            var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            std_r = math.sqrt(var_r)
            sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown from equity curve
        peak = self.initial_equity
        max_dd = 0.0
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Average bars held
        avg_bars = sum(t.bars_held for t in trades) / n if n > 0 else 0

        # Total commission
        total_comm = sum(t.commission for t in trades)

        result = BacktestResult(
            trades=[{
                "symbol": t.symbol, "side": t.side, "qty": t.qty,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                "bars_held": t.bars_held, "exit_reason": t.exit_reason,
            } for t in trades],
            equity_curve=equity_curve,
            total_return_pct=round(total_return * 100, 2),
            profit_usd=round(total_profit, 2),
            apr_pct=annual_return_pct,
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            win_rate=round(win_rate * 100, 1),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            total_trades=n,
            avg_bars_held=round(avg_bars, 1),
            commission_total=round(total_comm, 2),
        )

        gate_blocks = getattr(self, "_gate_block_count", 0)
        if gate_blocks:
            log.info(f"Regime gate blocked entries on {gate_blocks} bars")
        self._log_result(result)
        return result

    def _log_result(self, r: BacktestResult):
        log.info("=" * 50)
        log.info("BACKTEST RESULTS")
        log.info("=" * 50)
        log.info(f"Total Trades: {r.total_trades}")
        log.info(f"Win Rate: {r.win_rate}%")
        log.info(f"Total Return: {r.total_return_pct}%")
        log.info(f"Profit: ${r.profit_usd:+,.2f}")
        log.info(f"APR: {r.apr_pct}%")
        log.info(f"Sharpe Ratio: {r.sharpe_ratio}")
        log.info(f"Max Drawdown: {r.max_drawdown_pct}%")
        log.info(f"Profit Factor: {r.profit_factor}")
        log.info(f"Expectancy: ${r.expectancy:+,.2f}/trade")
        log.info(f"Avg Bars Held: {r.avg_bars_held}")
        log.info(f"Commission: ${r.commission_total:,.2f}")

        # Trade breakdown by exit reason
        reasons = {}
        for t in r.trades:
            reason = t["exit_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        log.info(f"Exit Reasons: {reasons}")

        # Survivorship bias warning
        log.warning(
            "SURVIVORSHIP BIAS NOTE: This backtest uses only currently-listed "
            "stocks. Delisted stocks (bankruptcies, acquisitions) are excluded, "
            "which inflates results by ~1-3% annually. Adjust expectations "
            "accordingly. Real OOS performance is the true benchmark."
        )

    def _empty_result(self, equity_curve: list = None) -> BacktestResult:
        return BacktestResult(
            trades=[], equity_curve=equity_curve or [], total_return_pct=0.0,
            profit_usd=0.0, apr_pct=0.0,
            sharpe_ratio=0.0, max_drawdown_pct=0.0, win_rate=0.0,
            profit_factor=0.0, expectancy=0.0, total_trades=0,
            avg_bars_held=0.0, commission_total=0.0,
        )
