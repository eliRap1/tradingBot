"""
Multi-timeframe IB backtest — mirrors live watcher architecture.

Live bot decides on BOTH 1Day (trend context + strategy selection) AND
5Min (entry signals). A single-timeframe backtest massively undercounts
signals because strategy selection fires on 5min noise instead of daily
regime. This runner replays the real multi-TF pipeline:

  for each 5Min bar i:
    daily_hist  = daily bars with date <= bar_i.date()       (trend context)
    intraday    = 5Min bars [:i+1]                           (no look-ahead)
    selection   = select_strategies(daily_hist)              (regime-aware)
    scores      = strategies.generate_signals(intraday)      (5min entries)
    composite   = weighted_sum / total_weight
    if composite >= min_score AND num_agreeing >= min_agreeing:
        open position with ATR stop + target
    for each open position:
        if next bar high >= TP or low <= SL → close, record WIN/LOSS

Usage:
  python ib_multitf_backtest.py --days 10
  python ib_multitf_backtest.py --days 20 --symbols-limit 50
  python ib_multitf_backtest.py --days 5 --symbols AAPL,NVDA,TSLA,AMD
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time

import pandas as pd

from utils import load_config, setup_logger
from backtester import SlippageModel, Position, BacktestTrade
from strategies import ALL_STRATEGIES
from strategy_selector import select_strategies
from trend import get_trend_context

log = setup_logger("multitf_bt")


# Futures contract metadata (multiplier = $/pt, margin = approx init margin/contract USD)
# Source: CME specs + typical IB margin requirements (Apr 2026).
FUTURES_META: dict[str, dict] = {
    "ES": {"multiplier": 50.0,   "margin": 13000.0, "tick_size": 0.25},
    "NQ": {"multiplier": 20.0,   "margin": 17000.0, "tick_size": 0.25},
    "CL": {"multiplier": 1000.0, "margin": 5500.0,  "tick_size": 0.01},
    "GC": {"multiplier": 100.0,  "margin": 11000.0, "tick_size": 0.10},
}


def _is_session_valid(ts, min_minute: int, skip_lunch: bool) -> bool:
    """Check if 5-min bar ts is within tradable window (ET)."""
    try:
        t = pd.Timestamp(ts)
        if t.tz is None:
            t = t.tz_localize("UTC")
        et = t.tz_convert("America/New_York")
    except Exception:
        return True
    hh, mm = et.hour, et.minute

    # Market hours 9:30-16:00 ET
    if hh < 9 or (hh == 9 and mm < 30):
        return False
    if hh >= 16:
        return False

    # Skip first N minutes after open
    open_minutes = (hh - 9) * 60 + (mm - 30)
    if open_minutes < min_minute:
        return False

    # Skip last 15 min (thin / closing auction noise)
    close_minutes = (16 - hh) * 60 - mm
    if close_minutes < 15:
        return False

    # Skip lunch lull 11:30-13:30 ET
    if skip_lunch:
        lunch_start = 11 * 60 + 30
        lunch_end = 13 * 60 + 30
        now_min = hh * 60 + mm
        if lunch_start <= now_min < lunch_end:
            return False

    return True


def _connect_ib(config: dict, client_id: int = 99):
    from ib_broker import IBBroker
    from ib_data import IBDataFetcher
    cfg = dict(config)
    cfg["ib"] = dict(cfg.get("ib", {}))
    cfg["ib"]["client_id"] = client_id
    broker = IBBroker(cfg)
    data = IBDataFetcher(broker._ib, broker._contracts, cfg)
    return broker, data


def _fetch_tf(data, symbols: list[str], timeframe: str,
              days: int) -> dict[str, pd.DataFrame]:
    out, failed = {}, []
    for sym in symbols:
        try:
            df = data._fetch_with_retry(sym, timeframe, days)
        except Exception as e:
            log.error(f"{sym} {timeframe}: {e}")
            failed.append(sym)
            continue
        if df is not None and not df.empty:
            out[sym] = df
        else:
            failed.append(sym)
        time.sleep(0.3)
    log.info(f"{timeframe}: {len(out)}/{len(symbols)} OK, failed={len(failed)}")
    return out


class MultiTFBacktester:
    def __init__(self, config: dict, initial_equity: float = 100_000.0):
        self.config = config
        self.initial_equity = initial_equity
        self.slippage = SlippageModel(config)
        self.strategies = {name: cls(config) for name, cls in ALL_STRATEGIES.items()}

        sig_cfg = config["signals"]
        self.min_score = float(sig_cfg["min_composite_score"])
        self.min_agreeing = int(sig_cfg["min_agreeing_strategies"])
        self.max_positions = int(sig_cfg["max_positions"])
        self.require_trend_align = bool(sig_cfg.get("require_trend_alignment", True))
        self.min_bar_minute = int(sig_cfg.get("min_bar_minute", 45))
        self.skip_lunch = bool(sig_cfg.get("skip_lunch", True))
        self.min_vol_ratio = float(sig_cfg.get("min_volume_ratio", 1.2))
        self.loss_cooldown_bars = int(sig_cfg.get("loss_cooldown_bars", 24))
        self.min_adx = float(sig_cfg.get("min_adx_for_entry", 20.0))

        risk_cfg = config["risk"]
        self.sl_mult = float(risk_cfg["stop_loss_atr_mult"])
        self.tp_mult = float(risk_cfg["take_profit_atr_mult"])
        self.min_rr = float(risk_cfg["min_risk_reward"])
        self.max_pos_pct = float(risk_cfg["max_position_pct"])
        self.max_risk_pct = float(risk_cfg["max_portfolio_risk_pct"])
        self.leverage = float(risk_cfg.get("leverage", 1.0))
        # Partial exits
        self.partial1_enabled = bool(risk_cfg.get("partial_exit_enabled", True))
        self.partial1_r = float(risk_cfg.get("partial_exit_r", 1.2))
        self.partial1_pct = float(risk_cfg.get("partial_exit_pct", 0.40))
        self.partial2_enabled = bool(risk_cfg.get("second_partial_enabled", True))
        self.partial2_r = float(risk_cfg.get("second_partial_r", 2.5))
        self.partial2_pct = float(risk_cfg.get("second_partial_pct", 0.30))
        # Trailing stop
        self.trail_activate_r = float(risk_cfg.get("trail_activate_r", 1.0))
        self.chandelier_mult = float(risk_cfg.get("chandelier_atr_mult", 3.0))

    def run(self, intraday_bars: dict[str, pd.DataFrame],
            daily_bars: dict[str, pd.DataFrame],
            min_intraday_bars: int = 60):
        # Align on union of 5-min timestamps across symbols
        all_ts = sorted({ts for df in intraday_bars.values() for ts in df.index.tolist()})
        if len(all_ts) < min_intraday_bars:
            log.warning("Not enough intraday bars")
            return [], []

        equity = self.initial_equity
        positions: dict[str, Position] = {}
        trades: list[BacktestTrade] = []
        equity_curve = [(str(all_ts[0]), equity)]
        signals_fired = 0
        # Track previous-bar signal direction per symbol (for 2-bar confirmation)
        prev_sig: dict[str, str] = {}
        # Per-symbol cooldown: bar index when symbol becomes tradable again
        cooldown_until: dict[str, int] = {}
        # Filter counters
        filtered = {"session": 0, "trend": 0, "volume": 0, "confirm": 0,
                    "cooldown": 0, "adx": 0}
        progress_every = max(1, len(all_ts) // 10)

        log.info(f"Multi-TF replay: {len(intraday_bars)} symbols × "
                 f"{len(all_ts)} 5-min bars, equity=${equity:,.0f}")

        for i, ts in enumerate(all_ts):
            if i < min_intraday_bars:
                continue
            if (i - min_intraday_bars) % progress_every == 0:
                pct = (i - min_intraday_bars) / max(1, len(all_ts) - min_intraday_bars) * 100
                log.info(f"  {pct:.0f}% | trades={len(trades)} "
                         f"open={len(positions)} signals={signals_fired}")

            # 1. Check exits on current bar — partials + trailing + SL/TP
            to_close = []
            for sym, pos in list(positions.items()):
                df = intraday_bars.get(sym)
                if df is None or ts not in df.index:
                    continue
                row = df.loc[ts]
                hi, lo, op = float(row["high"]), float(row["low"]), float(row["open"])
                vol = int(row["volume"])

                # Track high/low water for trailing stop
                pos.high_water = max(pos.high_water, hi)
                pos.low_water = min(pos.low_water, lo)

                # Compute current R multiple (best of bar)
                if pos.side == "buy":
                    best_profit = pos.high_water - pos.entry_price
                else:
                    best_profit = pos.entry_price - pos.low_water
                r_mult = best_profit / pos.initial_risk if pos.initial_risk > 0 else 0

                m = pos.multiplier
                # Partial TP 1 — at partial1_r, close partial1_pct of original qty
                if (self.partial1_enabled and not pos.partial1_taken
                        and r_mult >= self.partial1_r):
                    close_qty = max(1, int(pos.original_qty * self.partial1_pct))
                    close_qty = min(close_qty, pos.qty)
                    # Fill at partial1_r target price
                    target_px = pos.entry_price + pos.initial_risk * self.partial1_r \
                        if pos.side == "buy" else pos.entry_price - pos.initial_risk * self.partial1_r
                    fill_px = self.slippage.get_fill_price(
                        target_px, "sell" if pos.side == "buy" else "buy", vol, close_qty)
                    pnl_pts = (fill_px - pos.entry_price) if pos.side == "buy" \
                        else (pos.entry_price - fill_px)
                    pnl = pnl_pts * close_qty * m
                    trades.append(BacktestTrade(
                        symbol=sym, side=pos.side, qty=close_qty,
                        entry_price=pos.entry_price, exit_price=round(fill_px, 2),
                        entry_date=pos.entry_date, exit_date=str(ts),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl / (pos.entry_price * close_qty * m), 4),
                        bars_held=i - pos.entry_bar,
                        exit_reason="partial_tp_1",
                    ))
                    equity += pnl
                    pos.qty -= close_qty
                    pos.partial1_taken = True

                # Partial TP 2 — at partial2_r, close partial2_pct of original qty
                if (self.partial2_enabled and pos.partial1_taken and not pos.partial2_taken
                        and r_mult >= self.partial2_r and pos.qty > 0):
                    close_qty = max(1, int(pos.original_qty * self.partial2_pct))
                    close_qty = min(close_qty, pos.qty)
                    target_px = pos.entry_price + pos.initial_risk * self.partial2_r \
                        if pos.side == "buy" else pos.entry_price - pos.initial_risk * self.partial2_r
                    fill_px = self.slippage.get_fill_price(
                        target_px, "sell" if pos.side == "buy" else "buy", vol, close_qty)
                    pnl_pts = (fill_px - pos.entry_price) if pos.side == "buy" \
                        else (pos.entry_price - fill_px)
                    pnl = pnl_pts * close_qty * m
                    trades.append(BacktestTrade(
                        symbol=sym, side=pos.side, qty=close_qty,
                        entry_price=pos.entry_price, exit_price=round(fill_px, 2),
                        entry_date=pos.entry_date, exit_date=str(ts),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl / (pos.entry_price * close_qty * m), 4),
                        bars_held=i - pos.entry_bar,
                        exit_reason="partial_tp_2",
                    ))
                    equity += pnl
                    pos.qty -= close_qty
                    pos.partial2_taken = True

                # Trailing stop — activate at trail_activate_r, use chandelier (N*ATR from extreme)
                if r_mult >= self.trail_activate_r:
                    if pos.side == "buy":
                        new_stop = pos.high_water - pos.atr * self.chandelier_mult
                        if new_stop > pos.stop_loss:
                            pos.stop_loss = new_stop
                    else:
                        new_stop = pos.low_water + pos.atr * self.chandelier_mult
                        if new_stop < pos.stop_loss:
                            pos.stop_loss = new_stop

                # Skip rest if fully closed by partials
                if pos.qty <= 0:
                    to_close.append(sym)
                    continue

                # Regular SL / final TP check on remaining qty
                exit_price, exit_reason = None, None
                if pos.side == "buy":
                    if lo <= pos.stop_loss:
                        px = op if op <= pos.stop_loss else pos.stop_loss
                        exit_price = self.slippage.get_fill_price(px, "sell", vol, pos.qty)
                        exit_reason = "stop_loss" if not pos.partial1_taken else "trailing_stop"
                    elif hi >= pos.take_profit:
                        px = op if op >= pos.take_profit else pos.take_profit
                        exit_price = self.slippage.get_fill_price(px, "sell", vol, pos.qty)
                        exit_reason = "take_profit"
                else:
                    if hi >= pos.stop_loss:
                        px = op if op >= pos.stop_loss else pos.stop_loss
                        exit_price = self.slippage.get_fill_price(px, "buy", vol, pos.qty)
                        exit_reason = "stop_loss" if not pos.partial1_taken else "trailing_stop"
                    elif lo <= pos.take_profit:
                        px = op if op <= pos.take_profit else pos.take_profit
                        exit_price = self.slippage.get_fill_price(px, "buy", vol, pos.qty)
                        exit_reason = "take_profit"

                if exit_price is not None:
                    pnl_pts = (exit_price - pos.entry_price) if pos.side == "buy" \
                        else (pos.entry_price - exit_price)
                    pnl = pnl_pts * pos.qty * m
                    # Commission: stocks use slippage model, futures ~$5 round-trip per contract
                    if sym in FUTURES_META:
                        comm = 5.0 * pos.qty
                    else:
                        comm = self.slippage.get_commission(pos.qty) * 2
                    pnl -= comm
                    trades.append(BacktestTrade(
                        symbol=sym, side=pos.side, qty=pos.qty,
                        entry_price=pos.entry_price, exit_price=round(exit_price, 2),
                        entry_date=pos.entry_date, exit_date=str(ts),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl / (pos.entry_price * pos.qty * m), 4),
                        bars_held=i - pos.entry_bar,
                        exit_reason=exit_reason,
                        commission=round(comm, 2),
                    ))
                    equity += pnl
                    to_close.append(sym)
                    if exit_reason == "stop_loss":
                        cooldown_until[sym] = i + self.loss_cooldown_bars
            for s in to_close:
                del positions[s]

            if len(positions) >= self.max_positions:
                equity_curve.append((str(ts), round(equity, 2)))
                continue

            # 2. Session-time gate — skip open/close/lunch
            if not _is_session_valid(ts, self.min_bar_minute, self.skip_lunch):
                filtered["session"] += len(intraday_bars)
                equity_curve.append((str(ts), round(equity, 2)))
                continue

            # 3. Generate signals — daily regime + 5-min entries
            candidates = []
            current_sig_dir: dict[str, str] = {}
            bar_date = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
            for sym, intraday in intraday_bars.items():
                if sym in positions:
                    continue
                # Cooldown gate — skip symbols that recently got stopped out
                if i < cooldown_until.get(sym, 0):
                    filtered["cooldown"] += 1
                    continue
                if ts not in intraday.index:
                    continue
                hist_intraday = intraday.loc[:ts]
                if len(hist_intraday) < 30:
                    continue

                daily = daily_bars.get(sym)
                if daily is None:
                    continue
                daily_mask = daily.index.date <= bar_date if hasattr(daily.index, "date") \
                    else [pd.Timestamp(d).date() <= bar_date for d in daily.index]
                daily_hist = daily[daily_mask]
                if len(daily_hist) < 30:
                    continue

                try:
                    selection = select_strategies(daily_hist, sym)
                    trend_ctx = get_trend_context(daily_hist)
                except Exception:
                    continue

                weighted, total_w = 0.0, 0.0
                bull, bear = 0, 0
                for strat_name, w in selection["strategies"].items():
                    if w <= 0:
                        continue
                    strat = self.strategies.get(strat_name)
                    if not strat:
                        continue
                    try:
                        res = strat.generate_signals({sym: hist_intraday})
                        score = float(res.get(sym, 0.0))
                    except Exception:
                        continue
                    weighted += score * w
                    total_w += w
                    if score > 0.1:
                        bull += 1
                    elif score < -0.1:
                        bear += 1
                if total_w == 0:
                    continue
                composite = weighted / total_w

                direction = None
                if bull >= self.min_agreeing and composite >= self.min_score:
                    direction = "buy"
                elif bear >= self.min_agreeing and composite <= -self.min_score:
                    direction = "sell"
                if not direction:
                    continue

                # ADX gate — only enter on strong trend
                try:
                    adx_val = float(trend_ctx.get("adx", 0.0))
                    if adx_val < self.min_adx:
                        filtered["adx"] += 1
                        continue
                except Exception:
                    pass

                # Trend alignment gate — daily must agree
                if self.require_trend_align:
                    trend_dir = str(trend_ctx.get("direction", "neutral"))
                    if direction == "buy" and trend_dir == "down":
                        filtered["trend"] += 1
                        continue
                    if direction == "sell" and trend_dir == "up":
                        filtered["trend"] += 1
                        continue

                # Volume gate — current bar vol vs 20-bar trailing avg
                try:
                    cur_vol = float(hist_intraday["volume"].iloc[-1])
                    avg_vol = float(hist_intraday["volume"].iloc[-21:-1].mean())
                    if avg_vol > 0 and cur_vol / avg_vol < self.min_vol_ratio:
                        filtered["volume"] += 1
                        continue
                except Exception:
                    pass

                candidates.append((sym, composite, direction, bull if direction == "buy" else bear, hist_intraday))

            # 3. Size + enter (top N by |score|)
            candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            slots = self.max_positions - len(positions)
            for sym, comp, direction, n_agree, hist in candidates[:slots]:
                bar = intraday_bars[sym].loc[ts]
                close_px = float(bar["close"])
                bar_vol = int(bar["volume"])

                import ta
                atr = ta.volatility.AverageTrueRange(
                    high=hist["high"], low=hist["low"], close=hist["close"], window=14
                ).average_true_range().iloc[-1]
                if atr <= 0 or math.isnan(atr):
                    continue

                if direction == "sell":
                    stop = close_px + atr * self.sl_mult
                    tp = close_px - atr * self.tp_mult
                    risk = stop - close_px
                    reward = close_px - tp
                else:
                    stop = close_px - atr * self.sl_mult
                    tp = close_px + atr * self.tp_mult
                    risk = close_px - stop
                    reward = tp - close_px

                if risk <= 0 or reward / risk < self.min_rr:
                    continue

                max_risk_dollars = equity * self.max_risk_pct
                if sym in FUTURES_META:
                    meta = FUTURES_META[sym]
                    mult_pt = meta["multiplier"]
                    # Risk per contract = risk_pts × multiplier
                    qty_by_risk = int(max_risk_dollars / (risk * mult_pt))
                    # Margin-based cap (leverage-aware)
                    margin_budget = equity * self.leverage * self.max_pos_pct
                    qty_by_size = int(margin_budget / meta["margin"])
                    qty = min(qty_by_risk, qty_by_size)
                    qty = min(qty, 5)  # hard cap: 5 contracts / root
                else:
                    qty = int(max_risk_dollars / risk)
                    # Notional cap scaled by leverage (risk budget stays on cash equity)
                    qty = min(qty, int(equity * self.leverage * self.max_pos_pct / close_px))
                if qty <= 0:
                    continue

                fill = self.slippage.get_fill_price(close_px, direction, bar_vol, qty)
                entry_px = round(fill, 2)
                pos = Position(
                    symbol=sym, side=direction, qty=qty,
                    entry_price=entry_px,
                    stop_loss=round(stop, 2),
                    take_profit=round(tp, 2),
                    entry_bar=i,
                    entry_date=str(ts),
                )
                # Dynamic attrs for partial TP + trailing
                pos.original_qty = qty
                pos.atr = float(atr)
                pos.initial_risk = float(risk)
                pos.high_water = entry_px
                pos.low_water = entry_px
                pos.partial1_taken = False
                pos.partial2_taken = False
                # Contract multiplier: futures use $/pt, stocks = 1.0
                pos.multiplier = FUTURES_META.get(sym, {}).get("multiplier", 1.0)
                positions[sym] = pos
                signals_fired += 1

            # 4. Mark-to-market equity
            mtm = equity
            for sym, pos in positions.items():
                df = intraday_bars.get(sym)
                if df is not None and ts in df.index:
                    px = float(df.loc[ts, "close"])
                    pnl_pts = (px - pos.entry_price) if pos.side == "buy" \
                        else (pos.entry_price - px)
                    mtm += pnl_pts * pos.qty * pos.multiplier
            equity_curve.append((str(ts), round(mtm, 2)))

        # Close remaining at last bar
        last_ts = all_ts[-1]
        for sym, pos in list(positions.items()):
            df = intraday_bars.get(sym)
            if df is None or last_ts not in df.index:
                continue
            px = float(df.loc[last_ts, "close"])
            m = pos.multiplier
            pnl_pts = (px - pos.entry_price) if pos.side == "buy" \
                else (pos.entry_price - px)
            pnl = pnl_pts * pos.qty * m
            trades.append(BacktestTrade(
                symbol=sym, side=pos.side, qty=pos.qty,
                entry_price=pos.entry_price, exit_price=round(px, 2),
                entry_date=pos.entry_date, exit_date=str(last_ts),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / (pos.entry_price * pos.qty * m), 4),
                bars_held=len(all_ts) - pos.entry_bar,
                exit_reason="end_of_data",
            ))
            equity += pnl

        log.info(f"Filter counters: {filtered}")
        return trades, equity_curve


def _summary(trades, equity_curve, initial_equity):
    if not trades:
        return
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    total_pnl = sum(t.pnl for t in trades)
    gross_p = sum(t.pnl for t in trades if t.pnl > 0)
    gross_l = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    peak, mdd = initial_equity, 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak if peak else 0)

    print()
    print("=" * 50)
    print("MULTI-TF BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Trades:        {len(trades)}")
    print(f"Wins:          {wins}")
    print(f"Losses:        {losses}")
    print(f"Hit rate:      {wins/len(trades)*100:.1f}%")
    print(f"Total P&L:     ${total_pnl:+,.2f}")
    print(f"Return:        {total_pnl/initial_equity*100:+.2f}%")
    print(f"Profit factor: {pf:.2f}")
    print(f"Max drawdown:  {mdd*100:.2f}%")
    print()

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    print(f"Exit reasons:  {exit_reasons}")
    print()

    print("=" * 80)
    print("PER-TRADE OUTCOMES")
    print("=" * 80)
    print(f"{'#':<4}{'SYM':<8}{'SIDE':<6}{'ENTRY':<10}{'EXIT':<10}"
          f"{'PNL$':<12}{'PNL%':<9}{'BARS':<6}{'REASON':<12}VERDICT")
    print("-" * 80)
    for i, t in enumerate(trades, 1):
        v = "WIN" if t.pnl > 0 else "LOSS" if t.pnl < 0 else "FLAT"
        print(f"{i:<4}{t.symbol:<8}{t.side:<6}"
              f"{t.entry_price:<10.2f}{t.exit_price:<10.2f}"
              f"{t.pnl:<+12.2f}{t.pnl_pct*100:<+9.2f}"
              f"{t.bars_held:<6}{t.exit_reason:<12}{v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--daily-days", type=int, default=250,
                   help="Daily history for trend context (default 250)")
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--symbols-limit", type=int, default=40,
                   help="When --symbols not given, use first N of universe")
    p.add_argument("--include-futures", action="store_true",
                   help="Include futures (NQ/ES/CL/GC) from config.futures.symbols")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--client-id", type=int, default=99)
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    config = load_config()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = config["screener"]["universe"][:args.symbols_limit]

    fut_syms: list[str] = []
    if args.include_futures:
        fut_syms = list(config.get("futures", {}).get("symbols", []))
        fut_syms = [s for s in fut_syms if s in FUTURES_META]

    print(f"Multi-TF backtest: {len(symbols)} stocks + {len(fut_syms)} futures | "
          f"{args.days}d intraday × {args.daily_days}d trend | "
          f"equity=${args.equity:,.0f} | leverage={config['risk'].get('leverage', 1.0)}x")
    print()

    try:
        broker, data = _connect_ib(config, client_id=args.client_id)
    except Exception as e:
        print(f"ERROR: IB connect failed: {e}")
        sys.exit(1)

    try:
        all_syms = symbols + fut_syms
        print("Fetching 1Day trend bars...")
        t0 = time.time()
        daily = _fetch_tf(data, all_syms, "1Day", args.daily_days)
        print(f"  {len(daily)} OK ({time.time()-t0:.1f}s)")

        print("Fetching 5Min entry bars...")
        t0 = time.time()
        intraday = _fetch_tf(data, all_syms, "5Min", args.days)
        print(f"  {len(intraday)} OK ({time.time()-t0:.1f}s)")

        # Keep only symbols that have both
        common = set(daily.keys()) & set(intraday.keys())
        daily = {s: daily[s] for s in common}
        intraday = {s: intraday[s] for s in common}
        fut_in = [s for s in fut_syms if s in common]
        print(f"Common symbols: {len(common)} ({len(fut_in)} futures: {fut_in})")
        if not common:
            print("No symbols with both timeframes. Abort.")
            sys.exit(1)

        bt = MultiTFBacktester(config, initial_equity=args.equity)
        trades, eqc = bt.run(intraday, daily)
        _summary(trades, eqc, args.equity)

        if args.output and eqc:
            with open(args.output, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "equity"])
                for ts, eq in eqc:
                    w.writerow([ts, eq])
            print(f"\nEquity curve -> {args.output}")

    finally:
        try:
            broker._ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
