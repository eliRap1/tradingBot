"""IB contract resolution and auto-roll for futures.

Called only by IBBroker and IBDataFetcher — never directly by strategies or coordinator.
Front-month detection: pick nearest expiry with open interest > 0.
Contracts cached for 4 hours to avoid repeated IB queries.
"""
import time
import threading
import re
from datetime import date
from typing import Optional
from utils import setup_logger

log = setup_logger("contract_manager")

# Default contract specs (can be overridden in config under futures.contracts)
CONTRACT_SPECS = {
    "NQ": {"exchange": "CME",   "currency": "USD", "multiplier": 20,   "min_tick": 0.25},
    "ES": {"exchange": "CME",   "currency": "USD", "multiplier": 50,   "min_tick": 0.25},
    "CL": {"exchange": "NYMEX", "currency": "USD", "multiplier": 1000, "min_tick": 0.01},
    "GC": {"exchange": "COMEX", "currency": "USD", "multiplier": 100,  "min_tick": 0.10},
    "MNQ": {"exchange": "CME",   "currency": "USD", "multiplier": 2,    "min_tick": 0.25},
    "MES": {"exchange": "CME",   "currency": "USD", "multiplier": 5,    "min_tick": 0.25},
    "MCL": {"exchange": "NYMEX", "currency": "USD", "multiplier": 100,  "min_tick": 0.01},
    "MGC": {"exchange": "COMEX", "currency": "USD", "multiplier": 10,   "min_tick": 0.10},
}

CACHE_TTL_SEC = 4 * 3600  # 4 hours
_MONTH_CODE_TO_NUM = {
    "F": "01", "G": "02", "H": "03", "J": "04",
    "K": "05", "M": "06", "N": "07", "Q": "08",
    "U": "09", "V": "10", "X": "11", "Z": "12",
}
_FUTURES_MONTH_SUFFIX_RE = re.compile(r"^([FGHJKMNQUVXZ])(\d{1,2})$")


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
        self._sorted_roots = sorted(self._specs, key=len, reverse=True)

    def get_contract(self, root: str):
        """Return an IB futures contract for a root or month-coded symbol."""
        parsed_root, explicit_expiry = self._parse_symbol(root)
        if not parsed_root:
            log.error(f"No contract spec for {root}")
            return None
        cache_key = root.upper()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and time.time() - cached[0] < CACHE_TTL_SEC:
                return cached[1]

        if explicit_expiry:
            contract = self._resolve_explicit_contract(parsed_root, explicit_expiry)
        else:
            contract = self._resolve_front_month(parsed_root)
        if contract:
            with self._lock:
                self._cache[cache_key] = (time.time(), contract)
        return contract

    def _parse_symbol(self, symbol: str) -> tuple[Optional[str], Optional[str]]:
        sym = symbol.upper()
        if sym in self._specs:
            return sym, None

        for root in self._sorted_roots:
            if not sym.startswith(root):
                continue
            suffix = sym[len(root):]
            match = _FUTURES_MONTH_SUFFIX_RE.fullmatch(suffix)
            if match:
                month_code, year_digits = match.groups()
                return root, self._expiry_from_month_code(month_code, year_digits)
        return None, None

    def _expiry_from_month_code(self, month_code: str, year_digits: str) -> str:
        month = _MONTH_CODE_TO_NUM[month_code]
        if len(year_digits) == 2:
            year = 2000 + int(year_digits)
        else:
            current_year = date.today().year
            current_decade = (current_year // 10) * 10
            year = current_decade + int(year_digits)
            if year < current_year:
                year += 10
        return f"{year}{month}"

    def _resolve_explicit_contract(self, root: str, expiry: str):
        from ib_insync import Future

        spec = self._specs.get(root)
        if not spec:
            log.error(f"No contract spec for {root}")
            return None

        try:
            fut = Future(
                symbol=root,
                lastTradeDateOrContractMonth=expiry,
                exchange=spec["exchange"],
                currency=spec["currency"],
                multiplier=str(spec.get("multiplier", "")),
            )
            qualified = self._ib.qualifyContracts(fut)
            if qualified:
                contract = qualified[0]
                log.info(
                    f"Explicit contract {root}{expiry[4:]}: "
                    f"{getattr(contract, 'localSymbol', root)} exp={expiry}"
                )
                return contract
        except Exception as e:
            log.warning(f"Explicit contract resolve failed for {root} {expiry}: {e}")
        return self._resolve_front_month(root)

    def _resolve_front_month(self, root: str):
        """Query IB for available contracts and pick the nearest non-expired expiry.

        IMPORTANT: Always returns a real FUT contract (secType='FUT'), never CONTFUT.
        reqHistoricalData with secType='CONTFUT' times out on many NYMEX/COMEX products
        (notably CL crude oil) and is not reliable for historical bar requests.
        """
        from ib_insync import Future, ContFuture

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
