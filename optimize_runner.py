"""
Walk-forward optimization runner.

Usage:
  python optimize_runner.py
  python optimize_runner.py --days 500 --train 180 --test 60
  python optimize_runner.py --live-data --symbols AAPL,NVDA,TSLA,MSFT,AMD
"""

import argparse
import json
from utils import load_config
from optimizer import WalkForwardOptimizer
from backtest_runner import generate_test_data


def main():
    parser = argparse.ArgumentParser(description="Walk-forward parameter optimization")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=500,
                        help="Total days of data (default: 500)")
    parser.add_argument("--train", type=int, default=180,
                        help="Training window in days (default: 180)")
    parser.add_argument("--test", type=int, default=60,
                        help="Testing window in days (default: 60)")
    parser.add_argument("--step", type=int, default=30,
                        help="Step size in days (default: 30)")
    parser.add_argument("--equity", type=float, default=100000,
                        help="Starting equity (default: 100000)")
    parser.add_argument("--live-data", action="store_true",
                        help="Use real Alpaca data")
    parser.add_argument("--output", type=str, default="",
                        help="Save results to JSON file")
    args = parser.parse_args()

    config = load_config()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = config["screener"]["universe"][:10]  # Top 10 for speed

    print(f"Walk-Forward Optimization")
    print(f"Symbols: {symbols}")
    print(f"Data: {args.days} days | Train: {args.train}d | Test: {args.test}d | Step: {args.step}d")
    print()

    if args.live_data:
        from broker import Broker
        from data import DataFetcher
        broker = Broker(config)
        data = DataFetcher(broker)
        bars = data.get_bars(symbols, timeframe="1Day", days=args.days)
    else:
        print("Using synthetic data (add --live-data for real data)")
        bars = generate_test_data(symbols, args.days)

    optimizer = WalkForwardOptimizer(config, initial_equity=args.equity)
    result = optimizer.optimize(
        bars, train_days=args.train, test_days=args.test, step_days=args.step
    )

    print()
    print("=" * 60)
    print(result.summary)
    print("=" * 60)

    if result.is_overfit:
        print("\nWARNING: Strategy appears OVERFIT to historical data.")
        print("Consider: fewer parameters, simpler strategies, more data.")
    else:
        print(f"\nStrategy appears ROBUST.")
        print(f"Recommended params: {json.dumps(result.best_stable_params, indent=2)}")

    if result.windows:
        print(f"\nPer-window breakdown:")
        for w in result.windows:
            print(f"  Window {w['window']}: IS Sharpe={w['is_sharpe']} -> "
                  f"OOS Sharpe={w['oos_sharpe']} "
                  f"(degradation={w['degradation']}%, trades={w['oos_trades']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "avg_oos_sharpe": result.avg_oos_sharpe,
                "avg_oos_win_rate": result.avg_oos_win_rate,
                "avg_degradation_pct": result.avg_degradation_pct,
                "total_oos_trades": result.total_oos_trades,
                "best_params": result.best_stable_params,
                "param_stability": result.param_stability,
                "is_overfit": result.is_overfit,
                "windows": result.windows,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
