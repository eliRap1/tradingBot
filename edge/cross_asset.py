from __future__ import annotations

from dataclasses import dataclass, field
import time

import ta
import numpy as np


# Sector ETFs → sector label for momentum map
_SECTOR_ETFS: dict[str, str] = {
    "XLK": "tech",
    "XLF": "finance",
    "XLV": "health",
    "XLE": "energy",
    "XLY": "consumer_disc",
    "XLP": "consumer_stap",
    "XLI": "industrial",
    "XLRE": "reit",
    "XLU": "utility",
}

_FETCH_TICKERS = ["SPY", "TLT", "QQQ", "IWM", "RSP", "UUP"] + list(_SECTOR_ETFS)


@dataclass
class CrossAssetSignals:
    vix_regime: str = "normal"           # low / normal / elevated / panic
    vix_term_structure: str = "contango" # contango / backwardation
    bond_trend: str = "risk_on"          # risk_on / risk_off
    dxy_trend: str = "neutral"           # strong / neutral / weak
    market_breadth: float = 50.0
    breadth_signal: str = "neutral"      # healthy / neutral / weak
    sector_momentum: dict = field(default_factory=dict)  # {sector: leading/neutral/lagging}
    nq_overnight_move: float = 0.0
    size_multiplier: float = 1.0


class CrossAssetEngine:
    def __init__(self, data_fetcher, ttl_sec: int = 300):
        self.data = data_fetcher
        self.ttl_sec = ttl_sec
        self._cached = CrossAssetSignals()
        self._cached_at = 0.0

    def get_signals(self) -> CrossAssetSignals:
        if time.time() - self._cached_at < self.ttl_sec:
            return self._cached

        bars = self.data.get_bars(_FETCH_TICKERS, timeframe="1Day", days=80)
        spy_df = bars.get("SPY")

        breadth = self._compute_breadth(bars)
        bond_trend = self._bond_trend(bars.get("TLT"))
        nq_move = self._overnight_move(
            self.data.get_intraday_bars("NQ", timeframe="1Day", days=3)
        )
        vix_regime, vix_term, vix_mult = self._vix_proxy(spy_df)
        dxy_trend = self._dxy_trend(bars.get("UUP"))
        sector_mom = self._sector_momentum(bars, spy_df)

        # --- Size multiplier ---
        mult = 1.0

        # Breadth
        if breadth < 40:
            mult *= 0.70
            breadth_signal = "weak"
        elif breadth > 60:
            breadth_signal = "healthy"
        else:
            breadth_signal = "neutral"

        # Bond trend
        if bond_trend == "risk_off":
            mult *= 0.85

        # VIX regime (highest impact — dominates in tail events)
        mult *= vix_mult

        # VIX term structure: backwardation = near-term stress, reduce further
        if vix_term == "backwardation":
            mult *= 0.80

        # DXY: strong USD hurts EM/commodities but is usually neutral for US equities
        # No size change — used as ML feature only

        mult = max(0.15, min(1.25, mult))

        self._cached = CrossAssetSignals(
            vix_regime=vix_regime,
            vix_term_structure=vix_term,
            bond_trend=bond_trend,
            dxy_trend=dxy_trend,
            market_breadth=round(breadth, 1),
            breadth_signal=breadth_signal,
            sector_momentum=sector_mom,
            nq_overnight_move=round(nq_move, 4),
            size_multiplier=round(mult, 3),
        )
        self._cached_at = time.time()
        return self._cached

    # ── signal helpers ──────────────────────────────────────────────

    def _compute_breadth(self, bars: dict) -> float:
        total = above = 0
        for ticker, df in bars.items():
            if df is None or len(df) < 50:
                continue
            ema50 = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
            total += 1
            if df["close"].iloc[-1] > ema50.iloc[-1]:
                above += 1
        return (above / total * 100) if total else 50.0

    def _bond_trend(self, df) -> str:
        if df is None or len(df) < 20:
            return "risk_on"
        ema20 = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        # TLT rising = flight to safety = risk-off
        return "risk_off" if df["close"].iloc[-1] > ema20.iloc[-1] else "risk_on"

    def _overnight_move(self, df) -> float:
        if df is None or len(df) < 2:
            return 0.0
        prev_close = float(df["close"].iloc[-2])
        today_open = float(df["open"].iloc[-1])
        return ((today_open - prev_close) / prev_close) if prev_close else 0.0

    def _vix_proxy(self, spy_df) -> tuple[str, str, float]:
        """Approximate VIX from SPY annualised rolling volatility.

        Returns (regime, term_structure, size_multiplier_factor).
        """
        if spy_df is None or len(spy_df) < 30:
            return "normal", "contango", 1.0

        returns = spy_df["close"].pct_change().dropna()
        vol_20d = float(returns.tail(20).std()) * (252 ** 0.5) * 100  # annualised %
        vol_5d  = float(returns.tail(5).std())  * (252 ** 0.5) * 100

        if vol_20d < 15:
            regime = "low"
            mult = 1.0
        elif vol_20d < 25:
            regime = "normal"
            mult = 1.0
        elif vol_20d < 35:
            regime = "elevated"
            mult = 0.60
        else:
            regime = "panic"
            mult = 0.25

        # Term structure: spike in short-term vol vs trailing = backwardation
        term = "backwardation" if vol_5d > vol_20d * 1.15 else "contango"

        return regime, term, mult

    def _dxy_trend(self, uup_df) -> str:
        """UUP ETF tracks DXY Dollar Index — use as USD strength proxy."""
        if uup_df is None or len(uup_df) < 20:
            return "neutral"
        ema20 = ta.trend.EMAIndicator(uup_df["close"], window=20).ema_indicator()
        price = float(uup_df["close"].iloc[-1])
        ema_val = float(ema20.iloc[-1])
        if price > ema_val * 1.005:
            return "strong"
        if price < ema_val * 0.995:
            return "weak"
        return "neutral"

    def _sector_momentum(self, bars: dict, spy_df) -> dict:
        """Compare each sector ETF 20-day return vs SPY to classify as
        leading / neutral / lagging.
        """
        result: dict[str, str] = {}
        if spy_df is None or len(spy_df) < 21:
            return result

        spy_ret = float(spy_df["close"].iloc[-1] / spy_df["close"].iloc[-20] - 1)

        for etf, sector in _SECTOR_ETFS.items():
            df = bars.get(etf)
            if df is None or len(df) < 21:
                continue
            etf_ret = float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1)
            diff = etf_ret - spy_ret
            if diff > 0.02:
                result[sector] = "leading"
            elif diff < -0.02:
                result[sector] = "lagging"
            else:
                result[sector] = "neutral"

        return result
