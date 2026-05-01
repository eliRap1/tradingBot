from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .supertrend import SuperTrendStrategy
from .stoch_rsi import StochRSIStrategy
from .vwap_reclaim import VWAPReclaimStrategy
from .gap import GapStrategy
from .liquidity_sweep import LiquiditySweepStrategy
from .futures_trend import FuturesTrendStrategy
from .dol import DOLStrategy
from .time_series_momentum import TimeSeriesMomentumStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .relative_strength_rotation import RelativeStrengthRotationStrategy

ALL_STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "supertrend": SuperTrendStrategy,
    "stoch_rsi": StochRSIStrategy,
    "vwap_reclaim": VWAPReclaimStrategy,
    "gap": GapStrategy,
    "liquidity_sweep": LiquiditySweepStrategy,
    "futures_trend": FuturesTrendStrategy,
    "dol": DOLStrategy,
    "time_series_momentum": TimeSeriesMomentumStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "relative_strength_rotation": RelativeStrengthRotationStrategy,
}
