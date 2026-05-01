"""
OOS harness: train/test split on real IB bars. Runs backtester twice —
first 70% (IS) and last 30% (OOS). Accepts if OOS metrics within 20% of IS.
"""
import argparse
import copy
import time

from utils import load_config
from backtester import Backtester
from asset_universe import symbols_for_assets
from performance import result_row


def _connect(client_id=97):
    from ib_broker import IBBroker
    from ib_data import IBDataFetcher
    config = load_config()
    cfg = dict(config)
    cfg["ib"] = dict(cfg.get("ib", {}))
    cfg["ib"]["client_id"] = client_id
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


def split_bars(bars, pct):
    if not bars:
        return {}, {}

    starts = [df.index.min() for df in bars.values() if df is not None and len(df) >= 30]
    ends = [df.index.max() for df in bars.values() if df is not None and len(df) >= 30]
    if not starts or not ends:
        return {}, {}

    common_start = max(starts)
    common_end = min(ends)
    dates = sorted({
        idx
        for df in bars.values()
        for idx in df.index
        if common_start <= idx <= common_end
    })
    if len(dates) < 30:
        return {}, {}

    cut = int(len(dates) * (1.0 - pct))
    cut = max(1, min(cut, len(dates) - 1))
    train_start, train_end = dates[0], dates[cut - 1]
    test_start, test_end = dates[cut], dates[-1]

    is_bars, oos_bars = {}, {}
    for sym, df in bars.items():
        if df is None or len(df) < 30:
            continue
        train = df.loc[(df.index >= train_start) & (df.index <= train_end)]
        test = df.loc[(df.index >= test_start) & (df.index <= test_end)]
        if len(train) >= 30:
            is_bars[sym] = train
        if len(test) >= 30:
            oos_bars[sym] = test
    return is_bars, oos_bars


def make_date_aligned_folds(bars, folds, min_bars=30):
    """Split a mixed-asset bar set into calendar-aligned folds."""
    starts = [df.index.min() for df in bars.values() if df is not None and len(df) >= min_bars]
    ends = [df.index.max() for df in bars.values() if df is not None and len(df) >= min_bars]
    if not starts or not ends:
        return []

    common_start = max(starts)
    common_end = min(ends)
    dates = sorted({
        idx
        for df in bars.values()
        for idx in df.index
        if common_start <= idx <= common_end
    })
    fold_size = len(dates) // folds if folds else 0
    if fold_size < min_bars:
        return []

    result = []
    for i in range(folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < folds - 1 else len(dates)
        fold_dates = dates[start_idx:end_idx]
        if not fold_dates:
            continue
        start_date, end_date = fold_dates[0], fold_dates[-1]
        fold_bars = {}
        for sym, df in bars.items():
            sliced = df.loc[(df.index >= start_date) & (df.index <= end_date)]
            if len(sliced) >= min_bars:
                fold_bars[sym] = sliced
        if fold_bars:
            result.append(fold_bars)
    return result


def run(label, bars, config):
    bt = Backtester(config, initial_equity=100000)
    r = bt.run(bars)
    return result_row(r, label)


def _row(r):
    print(f"{r['label']:<8}  trades={r['trades']:>3}  win%={r['win_rate']:>5.1f}  "
          f"ret%={r['return_pct']:>+7.2f}  APR%={r['apr_pct']:>+7.2f}  "
          f"profit=${r['profit_usd']:>+9,.0f}  PF={r['profit_factor']:>5.2f}  "
          f"Sharpe={r['sharpe']:>5.2f}  MDD%={r['max_dd']:>5.2f}")


def _within(a, b, tol):
    if abs(a) < 1e-6:
        return abs(b) <= tol
    return abs(b - a) / abs(a) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--oos-pct", type=float, default=0.3)
    ap.add_argument("--universe", type=str, default="large",
                    choices=["small", "large"])
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--assets", type=str, default="stocks",
                    help="stocks,futures,crypto or all")
    ap.add_argument("--timeframe", type=str, default="1Day")
    ap.add_argument("--tol", type=float, default=0.20)
    args = ap.parse_args()

    config = load_config()
    syms = symbols_for_assets(config, assets=args.assets, universe=args.universe, n=args.n)
    print(f"OOS split: assets={args.assets} symbols={len(syms)} days={args.days} "
          f"oos_pct={args.oos_pct} tol={args.tol:.0%}")

    bars = fetch_bars(syms, args.days, timeframe=args.timeframe)
    print(f"Got bars for {len(bars)} symbols\n")

    is_bars, oos_bars = split_bars(bars, args.oos_pct)
    lens_is = [len(b) for b in is_bars.values()]
    lens_oos = [len(b) for b in oos_bars.values()]
    print(f"IS bars per sym avg={sum(lens_is)/max(1,len(lens_is)):.0f}  "
          f"OOS bars per sym avg={sum(lens_oos)/max(1,len(lens_oos)):.0f}\n")

    is_res = run("IS", copy.deepcopy(is_bars), config)
    oos_res = run("OOS", copy.deepcopy(oos_bars), config)

    print()
    _row(is_res)
    _row(oos_res)
    print()

    ok_ret = _within(is_res["return_pct"], oos_res["return_pct"], args.tol)
    ok_pf = _within(is_res["profit_factor"], oos_res["profit_factor"], args.tol)
    ok_wr = _within(is_res["win_rate"], oos_res["win_rate"], args.tol)
    print(f"OOS within {args.tol:.0%} of IS: "
          f"return={ok_ret}  PF={ok_pf}  WR={ok_wr}")
    verdict = "PASS" if (ok_ret and ok_pf and ok_wr) else "FAIL"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
