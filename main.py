"""
Trading Bot — LIVE TRADING with Real-Time Data.

Each stock gets its own watcher thread that:
  1. Monitors its candles and indicators continuously
  2. Picks the right strategy for that stock's behavior
  3. Reports signals to the Coordinator for execution

LIVE TRADING FEATURES:
  - Real-time price validation (rejects stale data)
  - Market hours awareness (stocks vs crypto)
  - Walk-forward optimized parameters
  - Profit maximization enhancements

Usage:
    1. Set your Alpaca API keys in .env:
       ALPACA_API_KEY=your_key
       ALPACA_SECRET_KEY=your_secret
    2. Set TRADING_MODE=paper in .env (or 'live' for real money)
    3. Tune parameters in config.yaml
    4. Run: python main.py

Optional: Run walk-forward optimization first:
    python walk_forward_optimizer.py --start 2024-01-01 --end 2025-12-31
"""

import os
import sys
from datetime import datetime

from coordinator import Coordinator
from live_trading import ensure_live_trading_mode


def main():
    print("=" * 60)
    print("TRADING BOT - LIVE TRADING MODE")
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    
    # Verify trading mode
    if not ensure_live_trading_mode():
        print("ERROR: Invalid trading mode configuration")
        sys.exit(1)
    
    # Check for API keys
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        print("\nERROR: Alpaca API keys not set!")
        print("Please set the following environment variables:")
        print("  export ALPACA_API_KEY=your_key")
        print("  export ALPACA_SECRET_KEY=your_secret")
        print("\nOr add them to your .env file")
        sys.exit(1)
    
    # Start the bot
    bot = Coordinator()
    bot.run()


if __name__ == "__main__":
    main()
