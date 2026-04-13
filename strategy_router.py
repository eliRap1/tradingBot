"""Per-instrument strategy assignment with normalized weights.

Strategy matrix (weights before normalization):
  Strategy         Stocks   Crypto   Futures
  momentum         0.20     0.25     0.20
  mean_reversion   0.15     ❌       ❌
  breakout         0.20     0.25     0.20
  supertrend       0.20     0.25     0.25
  stoch_rsi        0.15     0.25     0.15
  vwap_reclaim     0.10     ❌       0.10
  gap              0.10     ❌       ❌
  liquidity_sweep  0.20     0.25     0.25
  futures_trend    ❌       ❌       0.30

Weights are normalized to sum to 1.0 per instrument type.
"""

# Raw weights before normalization — edit here to tune per-instrument emphasis
_STOCK_WEIGHTS = {
    "momentum":        0.20,
    "mean_reversion":  0.15,
    "breakout":        0.20,
    "supertrend":      0.20,
    "stoch_rsi":       0.15,
    "vwap_reclaim":    0.10,
    "gap":             0.10,
    "liquidity_sweep": 0.20,
}

_CRYPTO_WEIGHTS = {
    "momentum":        0.25,
    "breakout":        0.25,
    "supertrend":      0.25,
    "stoch_rsi":       0.25,
    "liquidity_sweep": 0.25,
}

_FUTURES_WEIGHTS = {
    "momentum":        0.20,
    "breakout":        0.20,
    "supertrend":      0.25,
    "stoch_rsi":       0.15,
    "vwap_reclaim":    0.10,
    "liquidity_sweep": 0.25,
    "futures_trend":   0.30,
}


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: round(v / total, 4) for k, v in weights.items()}


class StrategyRouter:
    """Returns the normalized strategy weight dict for a given instrument type."""

    def __init__(self, config: dict):
        self._config = config
        self._stock_weights = _normalize(_STOCK_WEIGHTS)
        self._crypto_weights = _normalize(_CRYPTO_WEIGHTS)
        self._futures_weights = _normalize(_FUTURES_WEIGHTS)

    def get_strategies(self, instrument_type: str) -> dict[str, float]:
        """Return {strategy_name: normalized_weight} for the instrument type.

        Args:
            instrument_type: 'stock', 'crypto', or 'futures'

        Returns:
            Dict of strategy name → normalized weight (sums to 1.0)
        """
        if instrument_type == "crypto":
            return dict(self._crypto_weights)
        elif instrument_type == "futures":
            return dict(self._futures_weights)
        else:  # stock (default for unknown types too)
            return dict(self._stock_weights)
