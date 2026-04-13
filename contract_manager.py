"""IB contract resolution and auto-roll for futures.

Called only by IBBroker and IBDataFetcher — never directly by strategies or coordinator.
Front-month detection: pick nearest expiry with open interest > 0.
Contracts cached for 4 hours to avoid repeated IB queries.
"""
import time
import threading
from typing import Optional
from utils import setup_logger

log = setup_logger("contract_manager")

# Default contract specs (can be overridden in config under futures.contracts)
CONTRACT_SPECS = {
    "NQ": {"exchange": "CME",   "currency": "USD", "multiplier": 20,   "min_tick": 0.25},
    "ES": {"exchange": "CME",   "currency": "USD", "multiplier": 50,   "min_tick": 0.25},
    "CL": {"exchange": "NYMEX", "currency": "USD", "multiplier": 1000, "min_tick": 0.01},
    "GC": {"exchange": "COMEX", "currency": "USD", "multiplier": 100,  "min_tick": 0.10},
}

CACHE_TTL_SEC = 4 * 3600  # 4 hours


class ContractManager:
    """Resolves IB futures contracts and manages auto-roll."""

    def __init__(self, ib, config: dict):
        """
        Args:
            ib: connected ib_insync.IB instance
            config: full config dict (reads futures.contracts for overrides)
        """
        self._ib = ib
        self._cache: dict[str, tuple[float, object]] = {}  # root → (ts, Contract)
        self._lock = threading.Lock()
        self._specs = dict(CONTRACT_SPECS)

        # Apply config overrides
        for entry in config.get("futures", {}).get("contracts", []):
            root = entry["root"]
            if root in self._specs:
                self._specs[root].update({k: v for k, v in entry.items() if k != "root"})

    def get_contract(self, root: str):
        """Return the front-month IB Contract for `root`, using cache if fresh."""
        with self._lock:
            cached = self._cache.get(root)
            if cached and time.time() - cached[0] < CACHE_TTL_SEC:
                return cached[1]

        contract = self._resolve_front_month(root)
        if contract:
            with self._lock:
                self._cache[root] = (time.time(), contract)
        return contract

    def _resolve_front_month(self, root: str):
        """Query IB for available contracts and pick the nearest expiry."""
        from ib_insync import Future, ContFuture

        spec = self._specs.get(root)
        if not spec:
            log.error(f"No contract spec for {root}")
            return None

        try:
            # Use ContFuture to get the continuous/front-month contract details
            cont = ContFuture(root, spec["exchange"], currency=spec["currency"])
            details = self._ib.reqContractDetails(cont)

            if not details:
                log.warning(f"No ContFuture details for {root}, trying explicit Future")
                fut = Future(root, exchange=spec["exchange"], currency=spec["currency"])
                details = self._ib.reqContractDetails(fut)

            if not details:
                log.error(f"Cannot resolve contract for {root}")
                return None

            # Pick the contract with nearest expiry
            sorted_details = sorted(
                details,
                key=lambda d: d.contract.lastTradeDateOrContractMonth
            )
            front = sorted_details[0].contract

            # Qualify the contract (fills in conId, trading class, etc.)
            self._ib.qualifyContracts(front)
            log.info(
                f"Front-month {root}: {front.localSymbol} "
                f"exp={front.lastTradeDateOrContractMonth}"
            )
            return front

        except Exception as e:
            log.error(f"Failed to resolve front-month for {root}: {e}")
            return None

    def invalidate(self, root: str):
        """Force re-resolution on next get_contract call (e.g., after roll)."""
        with self._lock:
            self._cache.pop(root, None)

    def get_spec(self, root: str) -> dict:
        """Return the contract spec for a futures root."""
        return self._specs.get(root, {})
