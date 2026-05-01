"""
A/B test: small (20 tech) universe vs large (100+) universe.
Same window, same config, real IB data. Tests user's claim that
larger universe produces LESS return than smaller.
"""
import argparse
import copy
import time

from utils import load_config
from backtester import Backtester


def _connect():
    from ib_broker import IBBroker
    from ib_data import IBDataFetcher
    config = load_config()
    cfg = dict(config)
    cfg["ib"] = dict(cfg.get("ib", {}))
    cfg["ib"]["client_id"] = 98
    broker = IBBroker(cfg)
    data = IBDataFetcher(broker._ib, broker._contracts, cfg)
    return broker, data, cfg


def fetch_bars(symbols, days, timeframe="1Day", warmup=60):
    broker, data, cfg = _connect()
    bars = {}
    total = days + warmup
    for i, sym in enumerate(symbols, 1):
        for attempt in range(3):
            try:
                if not broker._ib.isConnected():
                    print(f"  reconnect attempt {attempt + 1}...")
                    try: broker._ib.disconnect()
                    except Exception: pass
                    time.sleep(2)
                    broker, data, cfg = _connect()
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
    try: broker._ib.disconnect()
    except Exception: pass
    return bars


def run(label, bars, config):
    bt = Backtester(config, initial_equity=100000)
    result = bt.run(bars)
    return {
        "label": label,
        "universe": len(bars),
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "return_pct": result.total_return_pct,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown_pct,
        "profit_factor": result.profit_factor,
        "expectancy": result.expectancy,
    }


def row(r):
    print(f"{r['label']:<14} n={r['universe']:>3}  trades={r['trades']:>4}  "
          f"win%={r['win_rate']:>5.1f}  ret%={r['return_pct']:>7.2f}  "
          f"sharpe={r['sharpe']:>5.2f}  mdd%={r['max_dd']:>5.2f}  "
          f"pf={r['profit_factor']:>5.2f}  exp=${r['expectancy']:>+8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--small", type=int, default=20)
    ap.add_argument("--large", type=int, default=100)
    ap.add_argument("--timeframe", type=str, default="1Day",
                    choices=["5Min", "15Min", "1Hour", "1Day"])
    args = ap.parse_args()

    config = load_config()
    small_syms = config["screener"]["universe"][:args.small]
    large_syms = config["screener"]["universe_full"][:args.large]

    all_syms = list(dict.fromkeys(large_syms + small_syms))
    print(f"A/B UNIVERSE: small={len(small_syms)} vs large={len(large_syms)} "
          f"(fetching union of {len(all_syms)}) {args.days}d {args.timeframe}")

    bars = fetch_bars(all_syms, args.days, timeframe=args.timeframe)
    print(f"Got bars for {len(bars)} symbols")

    small_bars = {s: bars[s] for s in small_syms if s in bars}
    large_bars = {s: bars[s] for s in large_syms if s in bars}
    print(f"small_bars={len(small_bars)}  large_bars={len(large_bars)}")

    small_res = run("small_universe", copy.deepcopy(small_bars), config)
    large_res = run("large_universe", copy.deepcopy(large_bars), config)

    print()
    row(small_res)
    row(large_res)
    print()

    d_ret = large_res["return_pct"] - small_res["return_pct"]
    d_pf  = large_res["profit_factor"] - small_res["profit_factor"]
    d_win = large_res["win_rate"] - small_res["win_rate"]
    d_dd  = large_res["max_dd"] - small_res["max_dd"]
    d_tr  = large_res["trades"] - small_res["trades"]

    print(f"delta return%    = {d_ret:+.2f}  (large - small)")
    print(f"delta profit_fac = {d_pf:+.2f}")
    print(f"delta win%       = {d_win:+.2f}")
    print(f"delta max_dd%    = {d_dd:+.2f}  (negative = better)")
    print(f"delta trades     = {d_tr:+d}")
    print()
    if d_ret < -1.0:
        print("CONFIRMED: larger universe produces LESS return (dilution)")
    elif d_ret > 1.0:
        print("REJECTED: larger universe produces MORE return")
    else:
        print("NEUTRAL: within 1pp")


if __name__ == "__main__":
    main()
