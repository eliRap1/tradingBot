"""
Backtest runner — CLI tool to test strategies on historical data.

Usage:
  python backtest_runner.py
  python backtest_runner.py --days 365
  python backtest_runner.py --symbols AAPL,NVDA,TSLA --days 180
"""

import argparse
import json
import csv
from datetime import datetime

from utils import load_config
from backtester import Backtester


def generate_test_data(symbols: list[str], days: int) -> dict:
    """
    Generate realistic synthetic test data with regime changes.
    Each stock gets a mix of trending, ranging, and pullback phases
    to ensure strategies actually have setups to detect.
    """
    import pandas as pd
    import numpy as np

    bars = {}
    for idx, sym in enumerate(symbols):
        np.random.seed((hash(sym) % 2**31 + idx) % 2**31)
        n = days
        dates = pd.date_range(end=datetime.now(), periods=n, freq="1D")

        start_price = np.random.uniform(50, 500)
        vol = np.random.uniform(0.012, 0.025)

        # Generate price with regime changes (trending + pullbacks + ranges)
        prices = [start_price]
        regime_len = np.random.randint(20, 50)
        drift = np.random.choice([0.002, -0.001, 0.0, 0.003])

        for j in range(n - 1):
            # Switch regime periodically
            if j % regime_len == 0:
                drift = np.random.choice([
                    0.003,   # strong uptrend
                    0.001,   # mild uptrend
                    -0.002,  # downtrend
                    0.0,     # ranging
                    -0.001,  # mild downtrend
                    0.004,   # strong bull run
                ])
                regime_len = np.random.randint(15, 60)
                vol = np.random.uniform(0.008, 0.03)

            ret = drift + np.random.normal(0, vol)
            prices.append(prices[-1] * (1 + ret))

        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, vol * 0.6, n)))
        lows = closes * (1 - np.abs(np.random.normal(0, vol * 0.6, n)))
        opens = np.roll(closes, 1) * (1 + np.random.normal(0, vol * 0.3, n))
        opens[0] = start_price
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Volume with occasional spikes (breakout volume)
        base_vol = np.random.uniform(500_000, 2_000_000)
        volumes = (base_vol * (1 + np.abs(np.random.normal(0, 0.3, n)))).astype(int)
        # Add volume spikes on big moves
        for j in range(1, n):
            price_change = abs(closes[j] - closes[j-1]) / closes[j-1]
            if price_change > 0.02:
                volumes[j] = int(volumes[j] * np.random.uniform(1.5, 3.0))
        volumes = np.maximum(volumes, 100)

        typical = (highs + lows + closes) / 3
        cum_tp_vol = np.cumsum(typical * volumes)
        cum_vol = np.cumsum(volumes)
        vwap = cum_tp_vol / cum_vol

        bars[sym] = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes, "vwap": vwap,
        }, index=dates)

    return bars


def main():
    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (default: config universe)")
    parser.add_argument("--days", type=int, default=250,
                        help="Days of historical data (default: 250)")
    parser.add_argument("--equity", type=float, default=100000,
                        help="Starting equity (default: 100000)")
    parser.add_argument("--live-data", action="store_true",
                        help="Use real Alpaca data (requires API keys)")
    parser.add_argument("--output", type=str, default="",
                        help="Save equity curve to CSV file")
    args = parser.parse_args()

    config = load_config()

    # Get symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = config["screener"]["universe"]

    print(f"Backtesting {len(symbols)} symbols over {args.days} days")
    print(f"Starting equity: ${args.equity:,.2f}")
    print()

    # Get data
    if args.live_data:
        from broker import Broker
        from data import DataFetcher
        broker = Broker(config)
        data = DataFetcher(broker)
        bars = data.get_bars(symbols, timeframe="1Day", days=args.days)
    else:
        print("Using synthetic data (add --live-data for real Alpaca data)")
        bars = generate_test_data(symbols, args.days)

    # Run backtest
    bt = Backtester(config, initial_equity=args.equity)
    result = bt.run(bars)

    # Print summary
    print()
    print("=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Total Trades:   {result.total_trades}")
    print(f"Win Rate:       {result.win_rate}%")
    print(f"Total Return:   {result.total_return_pct}%")
    print(f"Sharpe Ratio:   {result.sharpe_ratio}")
    print(f"Max Drawdown:   {result.max_drawdown_pct}%")
    print(f"Profit Factor:  {result.profit_factor}")
    print(f"Expectancy:     ${result.expectancy:+,.2f}/trade")
    print(f"Avg Bars Held:  {result.avg_bars_held}")
    print(f"Commissions:    ${result.commission_total:,.2f}")

    if result.total_trades == 0:
        print()
        print("No trades were generated. This means the confluence filter")
        print("(3/5 strategies agree) blocked all entries. This is CORRECT")
        print("behavior on random/synthetic data -- the bot is selective.")
        print()
        print("For meaningful results, use real data:")
        print("  python backtest_runner.py --live-data")

    if result.trades:
        print()
        print("Recent Trades:")
        for t in result.trades[-10:]:
            tag = "WIN " if t["pnl"] >= 0 else "LOSS"
            print(f"  [{tag}] {t['symbol']}: ${t['pnl']:+,.2f} "
                  f"({t['exit_reason']}) held {t['bars_held']} bars")

    # Save equity curve
    if args.output and result.equity_curve:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "equity"])
            for date, eq in result.equity_curve:
                writer.writerow([date, eq])
        print(f"\nEquity curve saved to {args.output}")


if __name__ == "__main__":
    main()
