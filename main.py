"""
Trading Bot - LIVE TRADING with Interactive Brokers.

All execution and market data runs through IB Gateway / TWS.
No Alpaca dependency.

Broker:  Interactive Brokers (IB Gateway on port 4002)
Assets:  Stocks, Futures (NQ/ES/CL/GC), Crypto (BTC/USD, ETH/USD via PAXOS)

Usage:
    1. Start IB Gateway (or TWS) and enable the API:
         Configure -> API -> Settings -> Enable ActiveX and Socket Clients = ON
         Socket port = 4002   (Gateway paper default)
    2. Set TRADING_MODE in .env:
         TRADING_MODE=paper   (safe default - uses IB paper account)
         TRADING_MODE=live    (real money - double-check everything first)
    3. Tune parameters in config.yaml.
    4. python main.py

Optional - run walk-forward optimisation before live trading:
    python walk_forward_optimizer.py --start 2024-01-01 --end 2025-12-31
"""

import os
import sys
import socket
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from utils import load_config, setup_logger
from live_trading import ensure_live_trading_mode

log = setup_logger("main")


def _check_ib_gateway(host: str, port: int, timeout: float = 4.0) -> bool:
    """Return True if IB Gateway / TWS is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    print("=" * 60)
    print("  TRADING BOT - Interactive Brokers")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # 1. Verify trading mode (.env TRADING_MODE=paper|live)
    if not ensure_live_trading_mode():
        print("ERROR: Set TRADING_MODE=paper or TRADING_MODE=live in your .env file")
        sys.exit(1)

    mode = os.getenv("TRADING_MODE", "paper").lower()
    print(f"  Mode: {mode.upper()}")
    if mode == "live":
        print("  *** LIVE TRADING - real money at risk ***")

    # 2. Check IB Gateway / TWS is reachable
    cfg = load_config()
    ib_cfg = cfg.get("ib", {})
    ib_host = ib_cfg.get("host", "127.0.0.1")
    ib_port = ib_cfg.get("port", 4002)

    if not _check_ib_gateway(ib_host, ib_port):
        print(f"\nERROR: Cannot reach IB Gateway at {ib_host}:{ib_port}")
        print("Steps to fix:")
        print("  1. Open IB Gateway (or TWS)")
        print("  2. Log in to your account (paper or live)")
        print(f"  3. Configure -> API -> Settings -> Socket port = {ib_port}")
        print("  4. Enable 'Enable ActiveX and Socket Clients'")
        print("  5. Re-run this script")
        sys.exit(1)

    print(f"  IB Gateway: reachable at {ib_host}:{ib_port}")
    print("")

    # 3. Start the coordinator
    from coordinator import Coordinator
    bot = Coordinator()
    bot.run()


if __name__ == "__main__":
    main()
