from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MicrostructureSignal:
    spread_pct: float = 0.0
    ofi_score: float = 0.0
    spy_corr: float = 0.5
    blocked: bool = False


class MicrostructureGate:
    def __init__(self, broker, data_fetcher, config: dict):
        self.broker = broker
        self.data = data_fetcher
        edge_cfg = config.get("edge", {})
        self.max_spread_pct = edge_cfg.get("max_spread_pct", 0.0015)
        self.ofi_weight = edge_cfg.get("ofi_weight", 0.05)

    def evaluate(self, symbol: str, intraday_bars: pd.DataFrame | None = None) -> MicrostructureSignal:
        spread_pct = self._spread_pct(symbol)
        ofi_score = self._ofi_score(intraday_bars)
        spy_corr = self._spy_corr(symbol)
        blocked = spread_pct > self.max_spread_pct if spread_pct > 0 else False
        return MicrostructureSignal(
            spread_pct=round(spread_pct, 6),
            ofi_score=round(ofi_score, 4),
            spy_corr=round(spy_corr, 4),
            blocked=blocked,
        )

    def _spread_pct(self, symbol: str) -> float:
        try:
            quote = self.broker.get_quote(symbol)
            if not quote or not quote.mid:
                return 0.0
            return max(0.0, (quote.ask - quote.bid) / quote.mid)
        except Exception:
            return 0.0

    def _ofi_score(self, bars: pd.DataFrame | None) -> float:
        if bars is None or len(bars) < 3:
            return 0.0
        recent = bars.tail(3)
        score = 0.0
        for _, row in recent.iterrows():
            rng = float(row["high"] - row["low"])
            if rng <= 0:
                continue
            score += (float(row["close"]) - float(row["open"])) / rng
        score = score / 3.0
        return max(-self.ofi_weight, min(self.ofi_weight, score * self.ofi_weight))

    def _spy_corr(self, symbol: str) -> float:
        try:
            bars = self.data.get_bars([symbol, "SPY"], timeframe="1Day", days=30)
            sym_df = bars.get(symbol)
            spy_df = bars.get("SPY")
            if sym_df is None or spy_df is None or len(sym_df) < 21 or len(spy_df) < 21:
                return 0.5
            sym_ret = sym_df["close"].pct_change().dropna().tail(20)
            spy_ret = spy_df["close"].pct_change().dropna().tail(20)
            if len(sym_ret) != len(spy_ret) or len(sym_ret) < 5:
                return 0.5
            return float(sym_ret.corr(spy_ret))
        except Exception:
            return 0.5
