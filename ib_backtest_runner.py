"""
IB historical backtest — replay bot logic on real IB bars.

Usage:
  python ib_backtest_runner.py --days 30
  python ib_backtest_runner.py --symbols AAPL,NVDA,TSLA --days 60 --timeframe 5Min
  python ib_backtest_runner.py --days 30 --output equity.csv

Fetches real historical bars from IB Gateway (same data source as live bot),
then replays bot strategy + confluence + sizing bar-by-bar. For each trade:
  - Stop or target hit on bar high/low -> WIN or LOSS recorded.
  - Timeout at end of window -> exit at last close.

No look-ahead: strategies only see bars <= current step.
No edge gates wired (news/econ/gap/insider/RS/calendar/volume) — those
require point-in-time API replay which isn't available historically.
Core strategy + confluence + stops/targets + slippage are real.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from utils import load_config, setup_logger
from backtester import Backtester
from asset_universe import symbols_for_assets

log = setup_logger("ib_backtest")


def _connect_ib(config: dict, client_id: int = 99):
    """Connect to IB Gateway and return (broker, data_fetcher).

    Uses a distinct client_id from the live bot (default 1) so both can
    coexist on the same Gateway.
    """
    from ib_broker import IBBroker
    from ib_data import IBDataFetcher

    # Override client_id to avoid collision with live bot
    cfg = dict(config)
    cfg["ib"] = dict(cfg.get("ib", {}))
    cfg["ib"]["client_id"] = client_id

    broker = IBBroker(cfg)
    data = IBDataFetcher(broker._ib, broker._contracts, cfg)
    return broker, data


def _fetch_window_bars(
    data,
    symbols: list[str],
    timeframe: str,
    days: int,
    warmup: int,
) -> dict[str, pd.DataFrame]:
    """Fetch `days + warmup` days of bars per symbol from IB.

    The warmup prefix gives strategies enough history before the first
    replay bar (indicators need ~30-50 bars to settle).
    """
    total_days = days + warmup
    bars: dict[str, pd.DataFrame] = {}
    fetched, failed = 0, []

    for i, sym in enumerate(symbols, 1):
        try:
            df = data._fetch_with_retry(sym, timeframe, total_days)
        except Exception as e:
            log.error(f"fetch failed {sym}: {e}")
            failed.append(sym)
            continue
        if df is None or df.empty:
            failed.append(sym)
            continue
        bars[sym] = df
        fetched += 1
        if i % 10 == 0:
            log.info(f"  fetched {i}/{len(symbols)}...")
        # IB pacing safety
        time.sleep(0.3)

    log.info(f"IB fetch: {fetched}/{len(symbols)} OK | failed={len(failed)}")
    if failed:
        log.info(f"  skipped: {', '.join(failed[:20])}"
                 f"{' ...' if len(failed) > 20 else ''}")
    return bars


def _trim_to_window(
    bars: dict[str, pd.DataFrame],
    days: int,
) -> dict[str, pd.DataFrame]:
    """Keep entire history; backtester uses first N as warmup via min_bars.

    We don't slice — backtester expects full df and uses `min_bars` to skip
    the warmup. But for a 30-day "a month ago" replay on daily bars, we
    want: bars from (today-30-warmup) to (today). All bars from IB are
    already up to now; just return as-is.
    """
    return bars


def _print_per_trade(trades: list[dict]) -> None:
    """Print every trade with WIN/LOSS verdict."""
    if not trades:
        print("No trades generated in window.")
        return

    print()
    print("=" * 80)
    print("PER-TRADE OUTCOMES")
    print("=" * 80)
    print(f"{'#':<4}{'SYM':<8}{'SIDE':<6}{'ENTRY':<10}{'EXIT':<10}"
          f"{'PNL$':<12}{'PNL%':<8}{'BARS':<6}{'REASON':<12}{'VERDICT'}")
    print("-" * 80)
    wins = losses = breakeven = 0
    for i, t in enumerate(trades, 1):
        pnl = t["pnl"]
        if pnl > 0:
            verdict = "WIN"
            wins += 1
        elif pnl < 0:
            verdict = "LOSS"
            losses += 1
        else:
            verdict = "FLAT"
            breakeven += 1
        print(f"{i:<4}{t['symbol']:<8}{t['side']:<6}"
              f"{t['entry_price']:<10.2f}{t['exit_price']:<10.2f}"
              f"{pnl:<+12.2f}{t['pnl_pct']*100:<+8.2f}"
              f"{t['bars_held']:<6}{t['exit_reason']:<12}{verdict}")
    print("-" * 80)
    total = len(trades)
    print(f"  WINS={wins}  LOSSES={losses}  FLAT={breakeven}  "
          f"hit_rate={wins/total*100:.1f}%")


def _write_report(result, symbols: list[str], args) -> None:
    out_dir = os.path.join("research", "results")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "assets": args.assets,
        "symbols": symbols,
        "days": args.days,
        "timeframe": args.timeframe,
        "initial_equity": args.equity,
        "total_trades": result.total_trades,
        "return_pct": result.total_return_pct,
        "profit_usd": result.profit_usd,
        "apr_pct": result.apr_pct,
        "profit_factor": result.profit_factor,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "win_rate": result.win_rate,
    }
    with open(os.path.join(out_dir, "backtest_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "apr_report.md"), "w", encoding="utf-8") as f:
        f.write("# APR Backtest Report\n\n")
        f.write(f"- Assets: `{args.assets}`\n")
        f.write(f"- Symbols: `{len(symbols)}`\n")
        f.write(f"- Return: `{result.total_return_pct:+.2f}%`\n")
        f.write(f"- APR: `{result.apr_pct:+.2f}%`\n")
        f.write(f"- Profit: `${result.profit_usd:+,.2f}`\n")
        f.write(f"- PF: `{result.profit_factor}`\n")
        f.write(f"- Max DD: `{result.max_drawdown_pct}%`\n")
        f.write(f"- Trades: `{result.total_trades}`\n")
    print(f"\nReport -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="IB historical backtest")
    parser.add_argument("--days", type=int, default=30,
                        help="Replay window in bars (default: 30)")
    parser.add_argument("--warmup", type=int, default=60,
                        help="Extra bars before window for strategy warmup (default: 60)")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (default: first 20 from config universe)")
    parser.add_argument("--assets", type=str, default="stocks",
                        help="stocks,futures,crypto or all")
    parser.add_argument("--universe", type=str, default="small",
                        choices=["small", "large"])
    parser.add_argument("--n", type=int, default=20,
                        help="Number of stock symbols when --symbols is omitted")
    parser.add_argument("--timeframe", type=str, default="1Day",
                        choices=["5Min", "15Min", "1Hour", "1Day"],
                        help="Bar size (default: 1Day)")
    parser.add_argument("--equity", type=float, default=100000.0,
                        help="Starting equity (default: 100000)")
    parser.add_argument("--output", type=str, default="",
                        help="Save equity curve to CSV")
    parser.add_argument("--report", action="store_true",
                        help="Write research/results APR report files")
    parser.add_argument("--client-id", type=int, default=99,
                        help="IB API client_id (default 99; avoid collision with live bot)")
    args = parser.parse_args()

    config = load_config()

    # Symbol list — default to first 20 from universe to stay in IB pacing budget
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = symbols_for_assets(config, assets=args.assets, universe=args.universe, n=args.n)

    print(f"IB backtest: {len(symbols)} symbols | {args.timeframe} bars | "
          f"window={args.days}d warmup={args.warmup}d")
    print(f"Starting equity: ${args.equity:,.2f}")
    print()

    # 1. Connect to IB Gateway
    try:
        broker, data = _connect_ib(config, client_id=args.client_id)
    except Exception as e:
        print(f"ERROR: could not connect to IB Gateway: {e}")
        print("Make sure IB Gateway is running on the configured port.")
        sys.exit(1)

    try:
        # 2. Fetch historical bars
        t0 = time.time()
        bars = _fetch_window_bars(
            data, symbols, args.timeframe,
            days=args.days, warmup=args.warmup,
        )
        if not bars:
            print("No bars fetched. Aborting.")
            sys.exit(1)
        print(f"Fetched {len(bars)} symbols in {time.time()-t0:.1f}s")
        print()

        # 3. Run backtester
        bt = Backtester(config, initial_equity=args.equity)
        result = bt.run(bars, min_bars=args.warmup)

        # 4. Summary
        print()
        print("=" * 50)
        print("BACKTEST SUMMARY")
        print("=" * 50)
        print(f"Total Trades:   {result.total_trades}")
        print(f"Win Rate:       {result.win_rate}%")
        print(f"Total Return:   {result.total_return_pct}%")
        print(f"Profit:         ${result.profit_usd:+,.2f}")
        print(f"APR:            {result.apr_pct}%")
        print(f"Sharpe Ratio:   {result.sharpe_ratio}")
        print(f"Max Drawdown:   {result.max_drawdown_pct}%")
        print(f"Profit Factor:  {result.profit_factor}")
        print(f"Expectancy:     ${result.expectancy:+,.2f}/trade")
        print(f"Avg Bars Held:  {result.avg_bars_held}")

        # 5. Per-trade WIN/LOSS table
        _print_per_trade(result.trades)

        # 6. CSV equity curve
        if args.output and result.equity_curve:
            with open(args.output, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "equity"])
                for d, eq in result.equity_curve:
                    w.writerow([d, eq])
            print(f"\nEquity curve -> {args.output}")

        if args.report:
            _write_report(result, symbols, args)

    finally:
        try:
            broker._ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
