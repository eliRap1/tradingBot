"""Classifies any symbol as 'stock', 'crypto', or 'futures'.

Priority order (first match wins):
  1. Symbol in FUTURES_ROOTS set  → 'futures'
  2. Symbol in CRYPTO_SYMBOLS set → 'crypto'
  3. Everything else              → 'stock'
"""

import re


_FUTURES_MONTH_SUFFIX_RE = re.compile(r"^[FGHJKMNQUVXZ]\d{1,2}$")


class InstrumentClassifier:
    def __init__(self, config: dict):
        futures_contracts = config.get("futures", {}).get("contracts", [])
        self._futures_roots = {c["root"].upper() for c in futures_contracts}
        self._sorted_futures_roots = sorted(self._futures_roots, key=len, reverse=True)

        crypto_list = config.get("screener", {}).get("crypto", [])
        # Store both slash and no-slash formats
        self._crypto_symbols: set[str] = set()
        for sym in crypto_list:
            self._crypto_symbols.add(sym)
            self._crypto_symbols.add(sym.replace("/", ""))

    def classify(self, symbol: str) -> str:
        """Return 'futures', 'crypto', or 'stock'."""
        if self._futures_root(symbol) is not None:
            return "futures"
        if symbol in self._crypto_symbols:
            return "crypto"
        return "stock"

    def _futures_root(self, symbol: str) -> str | None:
        sym = symbol.upper()
        if sym in self._futures_roots:
            return sym
        for root in self._sorted_futures_roots:
            if not sym.startswith(root):
                continue
            suffix = sym[len(root):]
            if _FUTURES_MONTH_SUFFIX_RE.fullmatch(suffix):
                return root
        return None

    def is_futures(self, symbol: str) -> bool:
        return self.classify(symbol) == "futures"

    def is_crypto(self, symbol: str) -> bool:
        return self.classify(symbol) == "crypto"

    def is_stock(self, symbol: str) -> bool:
        return self.classify(symbol) == "stock"
