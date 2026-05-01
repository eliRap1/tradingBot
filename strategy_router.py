"""Per-instrument strategy assignment with optional sector x regime overrides."""

from __future__ import annotations

import json
import os

_STOCK_WEIGHTS = {
    "time_series_momentum": 0.24,
    "relative_strength_rotation": 0.20,
    "donchian_breakout": 0.18,
    "momentum": 0.12,
    "supertrend": 0.10,
    "breakout": 0.08,
    "liquidity_sweep": 0.05,
    "dol": 0.03,
    "mean_reversion": 0.02,
    "gap": 0.02,
}

_CRYPTO_WEIGHTS = {
    "time_series_momentum": 0.35,
    "donchian_breakout": 0.25,
    "relative_strength_rotation": 0.15,
    "momentum": 0.15,
    "supertrend": 0.10,
}

_FUTURES_WEIGHTS = {
    "time_series_momentum": 0.35,
    "donchian_breakout": 0.25,
    "futures_trend": 0.20,
    "supertrend": 0.12,
    "momentum": 0.08,
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

    def _apply_strategy_filters(
        self,
        weights: dict[str, float],
        instrument_type: str,
        regime: str | None = None,
    ) -> dict[str, float]:
        filters = self._config.get("optimization", {}).get("strategy_filters", {})
        cfg = filters.get(instrument_type, {})
        if regime:
            regime_cfg = cfg.get("regimes", {}).get(regime, {})
        else:
            regime_cfg = {}
        enabled = regime_cfg.get("enabled", cfg.get("enabled"))
        disabled = set(regime_cfg.get("disabled", cfg.get("disabled", [])))
        result = dict(weights)
        if enabled:
            allowed = set(enabled)
            result = {k: v for k, v in result.items() if k in allowed}
        if disabled:
            result = {k: v for k, v in result.items() if k not in disabled}
        if not result:
            return {}
        return _normalize(result)

    def get_strategies(
        self,
        instrument_type: str,
        sector: str | None = None,
        regime: str | None = None,
    ) -> dict[str, float]:
        if instrument_type == "crypto":
            return self._apply_strategy_filters(self._crypto_weights, "crypto", regime)
        if instrument_type == "futures":
            return self._apply_strategy_filters(self._futures_weights, "futures", regime)

        if sector and sector in self._sector_weights:
            sector_map = self._sector_weights[sector]
            if regime and regime in sector_map:
                filtered = self._apply_strategy_filters(sector_map[regime], "stock", regime)
                if filtered:
                    return filtered
            if "_fallback" in sector_map:
                filtered = self._apply_strategy_filters(sector_map["_fallback"], "stock", regime)
                if filtered:
                    return filtered

        return self._apply_strategy_filters(self._stock_weights, "stock", regime)
