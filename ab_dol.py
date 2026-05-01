"""
A/B test: backtester with DOL disabled vs enabled.

Identical synthetic bars on both runs (seeded per symbol). Baseline pops
DOL from the strategy registry AND zeroes its selector/router weights,
so it makes zero contribution.
"""
import argparse
import copy
import sys

from utils import load_config
from backtester import Backtester
from backtest_runner import generate_test_data

import strategies as strat_mod
import strategy_selector
import strategy_router


def _connect_ib_broker():
    from ib_broker import IBBroker
    from ib_data import IBDataFetcher
    config = load_config()
    cfg = dict(config)
    cfg["ib"] = dict(cfg.get("ib", {}))
    cfg["ib"]["client_id"] = 99
    broker = IBBroker(cfg)
    data = IBDataFetcher(broker._ib, broker._contracts, cfg)
    return broker, data, cfg


def fetch_live_data(symbols, days, timeframe="1Day", warmup=60):
    """Fetch real IB historical bars with auto-reconnect on drop."""
    import time

    broker, data, cfg = _connect_ib_broker()
    bars = {}
    total = days + warmup

    for i, sym in enumerate(symbols, 1):
        for attempt in range(3):
            try:
                if not broker._ib.isConnected():
                    print(f"  reconnecting IB (attempt {attempt + 1})...")
                    try:
                        broker._ib.disconnect()
                    except Exception:
                        pass
                    time.sleep(2)
                    broker, data, cfg = _connect_ib_broker()

                df = data._fetch_with_retry(sym, timeframe, total)
                if df is not None and not df.empty:
                    bars[sym] = df
                break
            except Exception as e:
                print(f"  {sym} attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        if i % 10 == 0:
            print(f"  fetched {i}/{len(symbols)} (got {len(bars)} OK)")
        time.sleep(0.4)

    try:
        broker._ib.disconnect()
    except Exception:
        pass
    return bars


def _strip_dol():
    """Remove DOL everywhere so baseline run has no DOL influence."""
    strat_mod.ALL_STRATEGIES.pop("dol", None)
    strategy_router._STOCK_WEIGHTS.pop("dol", None)
    strategy_router._FUTURES_WEIGHTS.pop("dol", None)


def _restore_dol():
    from strategies.dol import DOLStrategy
    strat_mod.ALL_STRATEGIES["dol"] = DOLStrategy
    strategy_router._STOCK_WEIGHTS["dol"] = 0.15
    strategy_router._FUTURES_WEIGHTS["dol"] = 0.15


def run(label: str, bars, config):
    bt = Backtester(config, initial_equity=100000)
    result = bt.run(bars)

    # April-only slice (backtester uses `exit_date`, not `closed_at`)
    april_trades = [t for t in (result.trades or [])
                    if "2026-04" in str(t.get("exit_date", ""))]
    april_wins = [t for t in april_trades if t.get("pnl", 0) > 0]
    april_pnl = sum(t.get("pnl", 0) for t in april_trades)
    april_win_rate = (len(april_wins) / len(april_trades) * 100) if april_trades else 0.0

    return {
        "label": label,
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "return_pct": result.total_return_pct,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown_pct,
        "profit_factor": result.profit_factor,
        "expectancy": result.expectancy,
        "april_trades": len(april_trades),
        "april_win_rate": april_win_rate,
        "april_pnl": april_pnl,
    }


def print_row(r):
    print(f"{r['label']:<12} trades={r['trades']:>4}  win%={r['win_rate']:>5.1f}  "
          f"ret%={r['return_pct']:>6.2f}  sharpe={r['sharpe']:>5.2f}  "
          f"mdd%={r['max_dd']:>5.2f}  pf={r['profit_factor']:>5.2f}  "
          f"exp=${r['expectancy']:>+8.2f}")
    print(f"             [April] trades={r['april_trades']}  "
          f"win%={r['april_win_rate']:.1f}  pnl=${r['april_pnl']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--live-data", action="store_true")
    ap.add_argument("--timeframe", type=str, default="1Day",
                    choices=["5Min", "15Min", "1Hour", "1Day"])
    ap.add_argument("--max-symbols", type=int, default=50)
    args = ap.parse_args()

    config = load_config()
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = config["screener"]["universe"][:args.max_symbols]

    mode = "LIVE IB" if args.live_data else "synthetic"
    print(f"A/B DOL: {len(symbols)} symbols, {args.days} days {args.timeframe} ({mode})")
    if args.live_data:
        bars = fetch_live_data(symbols, args.days, timeframe=args.timeframe)
    else:
        bars = generate_test_data(symbols, args.days)
    print(f"Got bars for {len(bars)} symbols")
    if bars:
        first_sym = next(iter(bars))
        df0 = bars[first_sym]
        if len(df0) > 0:
            print(f"Date range ({first_sym}): {df0.index[0]} -> {df0.index[-1]} ({len(df0)} bars)")

    # Baseline
    _strip_dol()
    base = run("baseline", copy.deepcopy(bars), config)

    # DOL-on
    _restore_dol()
    dol = run("dol_on", copy.deepcopy(bars), config)

    print()
    print_row(base)
    print_row(dol)
    print()

    # Deltas
    d_win = dol["win_rate"] - base["win_rate"]
    d_pf  = dol["profit_factor"] - base["profit_factor"]
    d_dd  = dol["max_dd"] - base["max_dd"]
    d_tr  = dol["trades"] - base["trades"]
    d_ret = dol["return_pct"] - base["return_pct"]

    print(f"delta win%        = {d_win:+.2f}")
    print(f"delta profit_fac  = {d_pf:+.2f}")
    print(f"delta max_dd%     = {d_dd:+.2f}  (negative = better)")
    print(f"delta trades      = {d_tr:+d}")
    print(f"delta return%     = {d_ret:+.2f}")

    # Verdict
    keep = (
        (d_win >= 2.0 or d_pf >= 0.15)
        and d_dd <= 0.5
        and abs(d_tr) <= 0.15 * max(1, base["trades"])
    )
    print()
    print("VERDICT:", "KEEP @ 0.15" if keep else "inconclusive - consider diversifier@0.08 or shelve")


if __name__ == "__main__":
    main()
