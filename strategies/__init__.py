from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .supertrend import SuperTrendStrategy
from .stoch_rsi import StochRSIStrategy

ALL_STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "supertrend": SuperTrendStrategy,
    "stoch_rsi": StochRSIStrategy,
}
