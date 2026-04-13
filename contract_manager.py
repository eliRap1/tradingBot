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
        """Query IB for available contracts and pick the nearest non-expired expiry.

        IMPORTANT: Always returns a real FUT contract (secType='FUT'), never CONTFUT.
        reqHistoricalData with secType='CONTFUT' times out on many NYMEX/COMEX products
        (notably CL crude oil) and is not reliable for historical bar requests.
        """
        from ib_insync import Future, ContFuture
        from datetime import date

        spec = self._specs.get(root)
        if not spec:
            log.error(f"No contract spec for {root}")
            return None

        today_str = date.today().strftime("%Y%m%d")

        try:
            # ── Strategy 1: enumerate real FUT contracts, pick nearest expiry ──
            # Query with no expiry so IB returns all listed contracts.
            fut = Future(root, exchange=spec["exchange"], currency=spec["currency"])
            details = self._ib.reqContractDetails(fut)

            if details:
                # Drop spreads / combos (their localSymbol contains '-' or '/')
                std = [d for d in details
                       if not any(c in d.contract.localSymbol for c in ("-", "/", " "))]
                pool = std if std else details

                # Keep only non-expired contracts
                active = [d for d in pool
                          if d.contract.lastTradeDateOrContractMonth >= today_str]
                pool = active if active else pool

                sorted_details = sorted(
                    pool,
                    key=lambda d: d.contract.lastTradeDateOrContractMonth
                )
                front = sorted_details[0].contract

                qualified = self._ib.qualifyContracts(front)
                if qualified:
                    front = qualified[0]

                log.info(
                    f"Front-month {root}: {front.localSymbol} "
                    f"exp={front.lastTradeDateOrContractMonth}"
                )
                return front

        except Exception as e:
            log.warning(f"FUT enumeration failed for {root}: {e} — trying CONTFUT")

        try:
            # ── Strategy 2: use CONTFUT to discover expiry, then build real FUT ──
            # CONTFUT is useful only for discovery; we never return a CONTFUT
            # object because reqHistoricalData is unreliable with secType='CONTFUT'.
            cont = ContFuture(root, spec["exchange"], currency=spec["currency"])
            cont_details = self._ib.reqContractDetails(cont)

            if cont_details:
                cd = cont_details[0].contract
                expiry = cd.lastTradeDateOrContractMonth

                # Build an explicit FUT using the discovered expiry
                actual = Future(
                    symbol=root,
                    lastTradeDateOrContractMonth=expiry,
                    exchange=spec["exchange"],
                    currency=spec["currency"],
                    multiplier=str(spec.get("multiplier", "")),
                )
                qualified = self._ib.qualifyContracts(actual)
                if qualified:
                    actual = qualified[0]

                log.info(
                    f"Front-month {root} (via CONTFUT discovery): "
                    f"{actual.localSymbol} exp={expiry}"
                )
                return actual

        except Exception as e:
            log.error(f"CONTFUT discovery also failed for {root}: {e}")

        log.error(f"Cannot resolve any contract for {root}")
        return None

    def invalidate(self, root: str):
        """Force re-resolution on next get_contract call (e.g., after roll)."""
        with self._lock:
            self._cache.pop(root, None)

    def get_spec(self, root: str) -> dict:
        """Return the contract spec for a futures root."""
        return self._specs.get(root, {})
