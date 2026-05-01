"""
Walk-forward 3-fold rolling validation on IB data.
Splits history into non-overlapping folds and reports per-fold metrics.
Accept if all folds PF > 1.3.
"""
import argparse
import copy

from utils import load_config
from backtester import Backtester
from oos_harness import fetch_bars, make_date_aligned_folds
from asset_universe import symbols_for_assets
from performance import result_row


def run_fold(label, bars, config):
    bt = Backtester(config, initial_equity=100000)
    r = bt.run(bars)
    return result_row(r, label)


def _row(r):
    print(f"{r['label']:<10}  trades={r['trades']:>3}  win%={r['win_rate']:>5.1f}  "
          f"ret%={r['return_pct']:>+7.2f}  APR%={r['apr_pct']:>+7.2f}  "
          f"profit=${r['profit_usd']:>+9,.0f}  PF={r['profit_factor']:>5.2f}  "
          f"Sharpe={r['sharpe']:>5.2f}  MDD%={r['max_dd']:>5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--universe", type=str, default="large",
                    choices=["small", "large"])
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--assets", type=str, default="stocks",
                    help="stocks,futures,crypto or all")
    ap.add_argument("--min-pf", type=float, default=1.3)
    ap.add_argument("--timeframe", type=str, default="1Day")
    args = ap.parse_args()

    config = load_config()
    syms = symbols_for_assets(config, assets=args.assets, universe=args.universe, n=args.n)
    print(f"Walk-forward {args.folds}-fold: assets={args.assets} symbols={len(syms)} "
          f"days={args.days} min_PF={args.min_pf}")

    bars = fetch_bars(syms, args.days, timeframe=args.timeframe)
    print(f"Got bars for {len(bars)} symbols\n")

    fold_sets = make_date_aligned_folds(bars, args.folds, min_bars=40)
    if not fold_sets:
        print("Date-aligned fold size too small. Abort.")
        return

    print(f"Date-aligned folds={len(fold_sets)}\n")

    results = []
    for i, fold_bars in enumerate(fold_sets):
        res = run_fold(f"fold{i+1}", copy.deepcopy(fold_bars), config)
        results.append(res)

    print()
    print("=" * 80)
    print("WALK-FORWARD RESULTS")
    print("=" * 80)
    for r in results:
        _row(r)
    print()

    all_pass = all(r["profit_factor"] >= args.min_pf for r in results)
    passed = sum(1 for r in results if r["profit_factor"] >= args.min_pf)
    median_apr = sorted(r["apr_pct"] for r in results)[len(results) // 2] if results else 0.0
    worst_apr = min((r["apr_pct"] for r in results), default=0.0)
    print(f"Median APR%: {median_apr:.2f}  Worst APR%: {worst_apr:.2f}")
    print(f"Folds PF >= {args.min_pf}: {passed}/{len(results)}")
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
