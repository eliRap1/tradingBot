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
from candles import detect_patterns, bullish_score
from trend import get_trend_context
from risk import RiskManager
from signals import Opportunity
from utils import setup_logger

log = setup_logger("backtester")


# ── Slippage Model ──────────────────────────────────────────

class SlippageModel:
    """Realistic fill price simulation."""

    def __init__(self, config: dict):
        bt_cfg = config.get("backtest", {})
        self.base_slippage_pct = bt_cfg.get("slippage_pct", 0.001)  # 10 bps
        self.volume_impact = bt_cfg.get("volume_impact", True)
        self.commission_per_share = bt_cfg.get("commission_per_share", 0.0)

    def get_fill_price(self, price: float, side: str, volume: int,
                       qty: int) -> float:
        """
        Calculate realistic fill price.
        - Base slippage (10 bps default)
        - Volume impact: extra slippage if qty > 1% of bar volume
        - Direction: buys fill higher, sells fill lower
        """
        participation = qty / max(volume, 1)
        impact = self.base_slippage_pct
        if self.volume_impact and participation > 0.01:
            impact += participation * 0.01

        if side == "buy":
            return price * (1 + impact)
        else:
            return price * (1 - impact)

    def get_commission(self, qty: int) -> float:
        return self.commission_per_share * qty


# ── Data Classes ────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    side: str       # "buy" or "sell"
    qty: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    entry_date: str = ""


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    qty: int
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

    def run(self, bars_dict: dict[str, pd.DataFrame],
            min_bars: int = 50) -> BacktestResult:
        """
        Run backtest on provided bars.

        bars_dict: {symbol: DataFrame with OHLCV}
        min_bars: minimum bars before starting to trade
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

                if pos.side == "buy":
                    # Stop loss hit? (low goes below stop)
                    if bar_low <= pos.stop_loss:
                        # Gap through stop → fill at open
                        if bar_open <= pos.stop_loss:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "sell", bar_vol, pos.qty)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.stop_loss, "sell", bar_vol, pos.qty)
                        exit_reason = "stop_loss"

                    # Take profit hit? (high goes above TP)
                    elif bar_high >= pos.take_profit:
                        if bar_open >= pos.take_profit:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "sell", bar_vol, pos.qty)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.take_profit, "sell", bar_vol, pos.qty)
                        exit_reason = "take_profit"

                else:  # short
                    # Stop loss hit? (high goes above stop)
                    if bar_high >= pos.stop_loss:
                        if bar_open >= pos.stop_loss:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "buy", bar_vol, pos.qty)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.stop_loss, "buy", bar_vol, pos.qty)
                        exit_reason = "stop_loss"

                    elif bar_low <= pos.take_profit:
                        if bar_open <= pos.take_profit:
                            exit_price = self.slippage.get_fill_price(
                                bar_open, "buy", bar_vol, pos.qty)
                        else:
                            exit_price = self.slippage.get_fill_price(
                                pos.take_profit, "buy", bar_vol, pos.qty)
                        exit_reason = "take_profit"

                if exit_price is not None:
                    if pos.side == "buy":
                        pnl = (exit_price - pos.entry_price) * pos.qty
                    else:
                        pnl = (pos.entry_price - exit_price) * pos.qty

                    commission = self.slippage.get_commission(pos.qty) * 2
                    pnl -= commission
                    pnl_pct = pnl / (pos.entry_price * pos.qty) if pos.entry_price > 0 else 0

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

            # ── 2. Generate signals (NO look-ahead) ─────────
            if len(positions) >= self.max_positions:
                equity_curve.append((str(date), equity))
                daily_returns.append((equity - prev_equity) / prev_equity if prev_equity > 0 else 0)
                continue

            # Run strategies on historical bars only (bars up to and including current)
            signals_by_sym = {}
            for sym, df in bars_dict.items():
                if sym in positions:
                    continue

                # Only use bars up to current date (NO future data)
                hist = df.loc[df.index <= date]
                if len(hist) < 30:
                    continue

                # Strategy selection based on historical data
                selection = select_strategies(hist, sym)

                # Run strategies
                weighted_sum = 0.0
                total_weight = 0.0
                num_agreeing = 0
                strat_scores = {}

                for strat_name, weight in selection["strategies"].items():
                    if weight <= 0:
                        continue
                    strat = self.strategies.get(strat_name)
                    if not strat:
                        continue

                    result = strat.generate_signals({sym: hist})
                    score = result.get(sym, 0.0)
                    strat_scores[strat_name] = score
                    weighted_sum += score * weight
                    total_weight += weight

                    if score > 0.1:
                        num_agreeing += 1

                if total_weight == 0:
                    continue

                composite = weighted_sum / total_weight

                if num_agreeing >= self.min_agreeing and composite >= self.min_score:
                    signals_by_sym[sym] = {
                        "score": composite,
                        "num_agreeing": num_agreeing,
                        "strat_scores": strat_scores,
                        "hist": hist,
                    }
                    total_signals_found += 1

            # ── 3. Size and execute ─────────────────────────
            # Sort by score
            sorted_signals = sorted(signals_by_sym.items(),
                                    key=lambda x: x[1]["score"], reverse=True)

            slots = self.max_positions - len(positions)
            for sym, sig in sorted_signals[:slots]:
                hist = sig["hist"]
                bar = bars_dict[sym].loc[date] if date in bars_dict[sym].index else None
                if bar is None:
                    continue

                bar_close = float(bar["close"])
                bar_vol = int(bar["volume"])

                # ATR-based stops
                import ta
                atr_val = ta.volatility.AverageTrueRange(
                    high=hist["high"], low=hist["low"], close=hist["close"],
                    window=14
                ).average_true_range().iloc[-1]

                if atr_val <= 0:
                    continue

                sl_mult = self.config["risk"]["stop_loss_atr_mult"]
                tp_mult = self.config["risk"]["take_profit_atr_mult"]
                stop_loss = bar_close - (atr_val * sl_mult)
                take_profit = bar_close + (atr_val * tp_mult)

                risk_per_share = bar_close - stop_loss
                if risk_per_share <= 0:
                    continue

                # R:R filter
                reward = take_profit - bar_close
                rr = reward / risk_per_share
                if rr < self.config["risk"]["min_risk_reward"]:
                    continue

                # Position sizing (1% risk)
                max_risk = equity * self.config["risk"]["max_portfolio_risk_pct"]
                qty = int(max_risk / risk_per_share)
                max_pos_value = equity * self.config["risk"]["max_position_pct"]
                qty = min(qty, int(max_pos_value / bar_close))

                if qty <= 0:
                    continue

                # Simulate fill with slippage
                fill_price = self.slippage.get_fill_price(
                    bar_close, "buy", bar_vol, qty)

                positions[sym] = Position(
                    symbol=sym, side="buy", qty=qty,
                    entry_price=round(fill_price, 2),
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    entry_bar=i,
                    entry_date=str(date),
                )

            # ── 4. Update equity curve ──────────────────────
            # Mark-to-market open positions
            mtm = equity
            for sym, pos in positions.items():
                if sym in bars_dict and date in bars_dict[sym].index:
                    current = float(bars_dict[sym].loc[date, "close"])
                    if pos.side == "buy":
                        mtm += (current - pos.entry_price) * pos.qty
                    else:
                        mtm += (pos.entry_price - current) * pos.qty

            equity_curve.append((str(date), round(mtm, 2)))
            daily_returns.append(
                (mtm - prev_equity) / prev_equity if prev_equity > 0 else 0
            )

        # ── Close remaining positions at end ────────────────
        last_date = all_dates[-1]
        for sym, pos in list(positions.items()):
            if sym in bars_dict and last_date in bars_dict[sym].index:
                exit_price = float(bars_dict[sym].loc[last_date, "close"])
                if pos.side == "buy":
                    pnl = (exit_price - pos.entry_price) * pos.qty
                else:
                    pnl = (pos.entry_price - exit_price) * pos.qty

                trades.append(BacktestTrade(
                    symbol=sym, side=pos.side, qty=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=round(exit_price, 2),
                    entry_date=pos.entry_date,
                    exit_date=str(last_date),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl / (pos.entry_price * pos.qty), 4),
                    bars_held=len(all_dates) - pos.entry_bar,
                    exit_reason="end_of_data",
                ))
                equity += pnl

        # ── Calculate metrics ───────────────────────────────
        return self._calculate_result(trades, equity_curve, daily_returns)

    def _calculate_result(self, trades: list[BacktestTrade],
                          equity_curve: list,
                          daily_returns: list) -> BacktestResult:
        """Calculate all performance metrics from completed trades."""
        if not trades:
            return self._empty_result()

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]
        n = len(pnls)

        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        total_return = total_pnl / self.initial_equity

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
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            win_rate=round(win_rate * 100, 1),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            total_trades=n,
            avg_bars_held=round(avg_bars, 1),
            commission_total=round(total_comm, 2),
        )

        self._log_result(result)
        return result

    def _log_result(self, r: BacktestResult):
        log.info("=" * 50)
        log.info("BACKTEST RESULTS")
        log.info("=" * 50)
        log.info(f"Total Trades: {r.total_trades}")
        log.info(f"Win Rate: {r.win_rate}%")
        log.info(f"Total Return: {r.total_return_pct}%")
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

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            trades=[], equity_curve=[], total_return_pct=0.0,
            sharpe_ratio=0.0, max_drawdown_pct=0.0, win_rate=0.0,
            profit_factor=0.0, expectancy=0.0, total_trades=0,
            avg_bars_held=0.0, commission_total=0.0,
        )
