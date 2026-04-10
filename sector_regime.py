"""
Sector Regime Filter — Layer 2 of the multi-layer regime system.

Checks per-sector ETFs (XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLRE, XLU)
plus BTC/USD as a crypto proxy.

For each sector:
  - 50 EMA: price above/below
  - 20 EMA: price above/below (fast trend)
  - RSI(14)
  - ADX(14) + DI+/DI-

Size multiplier matrix (macro x sector):
  bull x bull  = 1.00   chop x bull = 0.80   bear x bull = 0.15
  bull x chop  = 0.70   chop x chop = 0.60   bear x chop = 0.10
  bull x bear  = 0.50   chop x bear = 0.40   bear x bear = 0.00 (long veto)
"""

import time
import ta
from utils import setup_logger

log = setup_logger("sector_regime")

SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLP", "XLI", "XLRE", "XLU"]

# Internal sector label -> ETF key used in get_sector_regimes() result dict
SECTOR_LABEL_TO_ETF = {
    "tech":          "XLK",
    "semi":          "XLK",
    "cloud":         "XLK",
    "cyber":         "XLK",
    "energy":        "XLE",
    "finance":       "XLF",
    "health":        "XLV",
    "consumer_disc": "XLY",
    "consumer_stap": "XLP",
    "industrial":    "XLI",
    "reit":          "XLRE",
    "utility":       "XLU",
    "crypto":        "crypto",
}

_MULT_MATRIX = {
    ("bull", "bull"):  1.00,
    ("bull", "chop"):  0.70,
    ("bull", "bear"):  0.50,
    ("chop", "bull"):  0.80,
    ("chop", "chop"):  0.60,
    ("chop", "bear"):  0.40,
    ("bear", "bull"):  0.15,
    ("bear", "chop"):  0.10,
    ("bear", "bear"):  0.00,
}


class SectorRegimeFilter:
    def __init__(self, data_fetcher, config: dict = None):
        self.data = data_fetcher
        self.config = config or {}
        self._last_regimes: dict = {}
        self._cached_result: dict = {}
        self._cache_time: float = 0.0
        self._cache_ttl = self.config.get("sector_regime", {}).get("cache_ttl_sec", 900)

    def get_sector_regimes(self) -> dict:
        """
        Classify all sector ETFs. Cached for cache_ttl_sec (default 15 min).

        Returns:
            {
                "XLK": {"regime": "bull"|"bear"|"chop", "size_mult": float,
                        "rsi": float, "adx": float, "above_50ema": bool,
                        "description": str, "symbol": str},
                ...
                "crypto": {...},
            }
        """
        now = time.time()
        if self._cached_result and now - self._cache_time < self._cache_ttl:
            return self._cached_result

        result = {}
        for etf in SECTOR_ETFS:
            result[etf] = self._classify(etf)
        result["crypto"] = self._classify("BTC/USD", label="crypto")

        # Log regime changes
        for key, data in result.items():
            prev = self._last_regimes.get(key)
            curr = data["regime"]
            if prev is not None and prev != curr:
                log.info(
                    f"SECTOR REGIME CHANGE: {key} {prev.upper()} -> {curr.upper()} "
                    f"({data['description']})"
                )
            self._last_regimes[key] = curr

        self._cached_result = result
        self._cache_time = now
        return result

    def get_regime_for_sector(self, sector_label: str) -> dict:
        """Look up sector regime by internal label (e.g. 'semi', 'tech', 'energy')."""
        etf_key = SECTOR_LABEL_TO_ETF.get(sector_label)
        if etf_key is None:
            return self._neutral()
        return self.get_sector_regimes().get(etf_key, self._neutral())

    def compute_size_mult(self, sector_label: str, macro_regime: str) -> float:
        """
        Combined macro x sector size multiplier.
        Returns 0.0 when macro=bear AND sector=bear (long veto).
        """
        sector_data = self.get_regime_for_sector(sector_label)
        sector_reg = sector_data["regime"]
        return _MULT_MATRIX.get((macro_regime, sector_reg), sector_data["size_mult"])

    # -- Internal ------------------------------------------------------

    def _classify(self, symbol: str, label: str = None) -> dict:
        label = label or symbol
        df = self.data.get_intraday_bars(symbol, timeframe="1Day", days=120)

        if df is None or len(df) < 30:
            log.warning(f"Sector regime: no data for {symbol} -- neutral fallback")
            return self._neutral(label)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        price = float(close.iloc[-1])

        ema50 = ta.trend.EMAIndicator(close, window=min(50, len(close) - 1)).ema_indicator()
        ema20 = ta.trend.EMAIndicator(close, window=min(20, len(close) - 1)).ema_indicator()
        above_50ema = price > float(ema50.iloc[-1])
        above_20ema = price > float(ema20.iloc[-1])

        rsi_val = 50.0
        try:
            rsi_val = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        except Exception:
            pass

        adx_val, di_plus, di_minus = 20.0, 10.0, 10.0
        try:
            adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
            adx_val  = float(adx_ind.adx().iloc[-1])
            di_plus  = float(adx_ind.adx_pos().iloc[-1])
            di_minus = float(adx_ind.adx_neg().iloc[-1])
        except Exception:
            pass

        if above_50ema and above_20ema and di_plus > di_minus:
            regime, size_mult = "bull", 1.0
            desc = f"{label} BULL -- above 50/20 EMA, RSI={rsi_val:.0f}, ADX={adx_val:.0f}"
        elif not above_50ema and not above_20ema and di_minus > di_plus:
            regime, size_mult = "bear", 0.30
            desc = f"{label} BEAR -- below 50/20 EMA, RSI={rsi_val:.0f}, ADX={adx_val:.0f}"
        else:
            regime, size_mult = "chop", 0.65
            desc = f"{label} CHOP -- mixed signals, RSI={rsi_val:.0f}, ADX={adx_val:.0f}"

        if rsi_val > 75 and regime == "bull":
            size_mult *= 0.80
            desc += " (overbought)"
        elif rsi_val < 30 and regime == "bear":
            size_mult = max(size_mult, 0.40)
            desc += " (oversold -- bounce watch)"

        if adx_val > 30 and regime == "bear":
            size_mult *= 0.75
            desc += " (strong downtrend)"

        return {
            "regime": regime,
            "size_mult": round(size_mult, 2),
            "rsi": round(rsi_val, 1),
            "adx": round(adx_val, 1),
            "above_50ema": above_50ema,
            "above_20ema": above_20ema,
            "description": desc,
            "symbol": symbol,
        }

    def _neutral(self, label: str = "unknown") -> dict:
        return {
            "regime": "chop", "size_mult": 0.65,
            "rsi": 50.0, "adx": 20.0,
            "above_50ema": True, "above_20ema": True,
            "description": f"{label} UNKNOWN -- data unavailable",
            "symbol": label,
        }
