from dataclasses import dataclass, field


@dataclass
class EdgeContext:
    current_regime: str = "bull_choppy"
    size_multiplier: float = 1.0
    blocked_symbols: set[str] = field(default_factory=set)
    ml_confidence: float = 1.0
