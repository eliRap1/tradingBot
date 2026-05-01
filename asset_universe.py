"""Helpers for selecting validation symbols by asset class."""

from __future__ import annotations


def parse_assets(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return ["stocks"]
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in value if str(p).strip()]
    if not parts or "all" in parts:
        return ["stocks", "futures", "crypto"]
    aliases = {"stock": "stocks", "future": "futures"}
    normalized = []
    for part in parts:
        item = aliases.get(part, part)
        if item not in {"stocks", "futures", "crypto"}:
            raise ValueError(f"Unknown asset class: {part}")
        if item not in normalized:
            normalized.append(item)
    return normalized


def symbols_for_assets(
    config: dict,
    assets: str | list[str] | tuple[str, ...] | None = None,
    universe: str = "small",
    n: int | None = None,
) -> list[str]:
    """Return a de-duplicated symbol list for validation."""
    selected_assets = parse_assets(assets)
    symbols: list[str] = []

    if "stocks" in selected_assets:
        key = "universe" if universe == "small" else "universe_full"
        stock_symbols = list(config.get("screener", {}).get(key, []))
        if n is not None:
            stock_symbols = stock_symbols[: int(n)]
        symbols.extend(stock_symbols)

    if "futures" in selected_assets:
        symbols.extend(config.get("futures", {}).get("symbols", []))

    if "crypto" in selected_assets:
        crypto_symbols = config.get("screener", {}).get("crypto", [])
        # IB scope is BTC/ETH only for this optimization pass.
        symbols.extend([s for s in crypto_symbols if s in {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}])

    deduped: list[str] = []
    seen = set()
    for sym in symbols:
        if sym not in seen:
            deduped.append(sym)
            seen.add(sym)
    return deduped
