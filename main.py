"""
Trading Bot — Threaded Architecture.

Each stock gets its own watcher thread that:
  1. Monitors its candles and indicators continuously
  2. Picks the right strategy for that stock's behavior
  3. Reports signals to the Coordinator for execution

Usage:
    1. Set your Alpaca API keys in .env
    2. Set TRADING_MODE=paper in .env
    3. Tune parameters in config.yaml
    4. Run: python main.py
"""

from coordinator import Coordinator


def main():
    bot = Coordinator()
    bot.run()


if __name__ == "__main__":
    main()
