"""Per-instrument strategy assignment with optional sector x regime overrides."""

from __future__ import annotations

import json
import os

_STOCK_WEIGHTS = {
    "momentum": 0.20,
    "mean_reversion": 0.15,
    "breakout": 0.20,
    "supertrend": 0.20,
    "stoch_rsi": 0.15,
    "vwap_reclaim": 0.10,
    "gap": 0.10,
    "liquidity_sweep": 0.20,
}

_CRYPTO_WEIGHTS = {
    "momentum": 0.25,
    "breakout": 0.25,
    "supertrend": 0.25,
    "stoch_rsi": 0.25,
    "liquidity_sweep": 0.25,
}

_FUTURES_WEIGHTS = {
    "momentum": 0.20,
    "breakout": 0.20,
    "supertrend": 0.25,
    "stoch_rsi": 0.15,
    "vwap_reclaim": 0.10,
    "liquidity_sweep": 0.25,
    "futures_trend": 0.30,
}


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)
    return {k: round(v / total, 4) for k, v in weights.items() if v > 0}


class StrategyRouter:
    """Returns normalized strategy weights for instrument, sector, and regime."""

    def __init__(self, config: dict):
        self._config = config
        self._stock_weights = _normalize(_STOCK_WEIGHTS)
        self._crypto_weights = _normalize(_CRYPTO_WEIGHTS)
        self._futures_weights = _normalize(_FUTURES_WEIGHTS)
        self._sector_weights = self._load_sector_weights()

    def _load_sector_weights(self) -> dict:
        path = os.path.join(os.path.dirname(__file__), "research", "sector_weights.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {
                sector: {key: _normalize(value) for key, value in mapping.items() if isinstance(value, dict)}
                for sector, mapping in raw.items()
                if isinstance(mapping, dict)
            }
        except Exception:
            return {}

    def get_strategies(
        self,
        instrument_type: str,
        sector: str | None = None,
        regime: str | None = None,
    ) -> dict[str, float]:
        if instrument_type == "crypto":
            return dict(self._crypto_weights)
        if instrument_type == "futures":
            return dict(self._futures_weights)

        if sector and sector in self._sector_weights:
            sector_map = self._sector_weights[sector]
            if regime and regime in sector_map:
                return dict(sector_map[regime])
            if "_fallback" in sector_map:
                return dict(sector_map["_fallback"])

        return dict(self._stock_weights)
