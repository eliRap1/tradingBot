"""Pairs / stat-arb edge — cointegration mean reversion.

Selects same-sector candidate pairs, runs Engle-Granger cointegration
test, computes hedge ratio + z-score of the spread, and emits long-laggard
/ short-leader signals when |z| > entry threshold.

Lit. Sharpe: 1.0-2.0 standalone, low correlation to momentum.

Usage (live coordinator):
    pe = PairsEdge(data_fetcher, config, sector_map)
    pe.refresh_pairs()                    # daily / hourly
    sig = pe.evaluate(symbol)             # per-symbol entry hint

Usage (research):
    PairsEdge(...).discover_pairs(symbols, lookback=180)

Notes:
- Cointegration test requires statsmodels; if not installed, falls
  back to correlation-only fallback (looser, less rigorous).
- Bars are daily close. Half-life filter rejects pairs that revert
  too slowly (>15 days) or too fast (<2 days, noise).
- Refreshes pair list on a TTL — default 1 day (Engle-Granger is slow).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint, adfuller
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


@dataclass
class PairSignal:
    in_pair: bool = False
    partner: str = ""
    role: str = ""              # "laggard" → long, "leader" → short
    z_score: float = 0.0
    hedge_ratio: float = 0.0
    half_life_days: float = 0.0
    p_value: float = 1.0
    reason: str = ""
    size_mult: float = 1.0


@dataclass
class PairDef:
    a: str
    b: str
    hedge_ratio: float
    z_mean: float
    z_std: float
    half_life: float
    p_value: float


class PairsEdge:
    def __init__(self, data_fetcher, config: dict, sector_map: dict[str, str] | None = None):
        self.data = data_fetcher
        self.config = config.get("edge", {}).get("pairs", {})
        self.enabled = bool(self.config.get("enabled", True))
        self.lookback_days = int(self.config.get("lookback_days", 180))
        self.refresh_ttl_sec = int(self.config.get("refresh_ttl_sec", 86_400))
        self.entry_z = float(self.config.get("entry_z", 2.0))
        self.exit_z = float(self.config.get("exit_z", 0.5))
        self.max_p_value = float(self.config.get("max_p_value", 0.05))
        self.min_half_life = float(self.config.get("min_half_life_days", 2.0))
        self.max_half_life = float(self.config.get("max_half_life_days", 15.0))
        self.size_mult = float(self.config.get("size_mult", 1.10))
        self.sector_map = sector_map or {}

        self._pairs: list[PairDef] = []
        self._pair_by_symbol: dict[str, list[PairDef]] = {}
        self._refreshed_at: float = 0.0

    @staticmethod
    def half_life_ou(spread: pd.Series) -> float:
        """Estimate Ornstein-Uhlenbeck half-life of spread reversion."""
        spread = spread.dropna()
        if len(spread) < 20:
            return float("inf")
        spread_lag = spread.shift(1).dropna()
        delta = (spread - spread.shift(1)).dropna()
        spread_lag = spread_lag.loc[delta.index]
        if len(spread_lag) < 10:
            return float("inf")
        # ΔS_t = κ(μ - S_{t-1}) + ε  → κ from regression
        x = spread_lag.values
        y = delta.values
        denom = ((x - x.mean()) ** 2).sum()
        if denom == 0:
            return float("inf")
        kappa = -((x - x.mean()) * (y - y.mean())).sum() / denom
        if kappa <= 0:
            return float("inf")
        return math.log(2) / kappa

    @staticmethod
    def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
        """OLS slope: y = beta * x + alpha."""
        y_v = y.values
        x_v = x.values
        x_mean = x_v.mean()
        denom = ((x_v - x_mean) ** 2).sum()
        if denom == 0:
            return 1.0
        return float(((x_v - x_mean) * (y_v - y_v.mean())).sum() / denom)

    def discover_pairs(self, symbols: list[str], bars_by_symbol: dict[str, pd.DataFrame] | None = None) -> list[PairDef]:
        """Find cointegrated pairs among same-sector symbols."""
        out: list[PairDef] = []
        bars = bars_by_symbol or {}
        # Group by sector
        groups: dict[str, list[str]] = {}
        for s in symbols:
            sec = self.sector_map.get(s, "other")
            groups.setdefault(sec, []).append(s)

        for sector, syms in groups.items():
            if len(syms) < 2:
                continue
            # Closes per symbol
            closes: dict[str, pd.Series] = {}
            for s in syms:
                df = bars.get(s)
                if df is None or "close" not in df.columns or len(df) < self.lookback_days:
                    continue
                closes[s] = df["close"].astype(float).iloc[-self.lookback_days:]
            keys = sorted(closes.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    sa, sb = closes[a].align(closes[b], join="inner")
                    if len(sa) < 60:
                        continue
                    p_value = 1.0
                    if _HAS_STATSMODELS:
                        try:
                            _, p_value, _ = coint(sa.values, sb.values)
                        except Exception:
                            continue
                        if p_value > self.max_p_value:
                            continue
                    else:
                        # Correlation-only fallback
                        if sa.corr(sb) < 0.85:
                            continue
                    beta = self.hedge_ratio_ols(sa, sb)
                    if beta <= 0 or not math.isfinite(beta):
                        continue
                    spread = sa - beta * sb
                    hl = self.half_life_ou(spread)
                    if hl < self.min_half_life or hl > self.max_half_life:
                        continue
                    out.append(PairDef(
                        a=a, b=b,
                        hedge_ratio=beta,
                        z_mean=float(spread.mean()),
                        z_std=float(spread.std()) if spread.std() > 0 else 1.0,
                        half_life=hl,
                        p_value=p_value,
                    ))
        return out

    def refresh_pairs(self, symbols: list[str], bars_by_symbol: dict[str, pd.DataFrame] | None = None) -> int:
        """Re-run discovery if TTL elapsed. Returns count of valid pairs."""
        if not self.enabled:
            return 0
        if time.time() - self._refreshed_at < self.refresh_ttl_sec and self._pairs:
            return len(self._pairs)
        pairs = self.discover_pairs(symbols, bars_by_symbol)
        self._pairs = pairs
        self._refreshed_at = time.time()
        idx: dict[str, list[PairDef]] = {}
        for p in pairs:
            idx.setdefault(p.a, []).append(p)
            idx.setdefault(p.b, []).append(p)
        self._pair_by_symbol = idx
        return len(pairs)

    def evaluate(self, symbol: str, bars_by_symbol: dict[str, pd.DataFrame] | None = None) -> PairSignal:
        """Score a symbol's role in any active pair."""
        if not self.enabled:
            return PairSignal(reason="pairs disabled")
        pairs = self._pair_by_symbol.get(symbol, [])
        if not pairs:
            return PairSignal(reason="no active pair")
        bars = bars_by_symbol or {}

        best: PairSignal | None = None
        for pair in pairs:
            df_a = bars.get(pair.a)
            df_b = bars.get(pair.b)
            if df_a is None or df_b is None or len(df_a) < 5 or len(df_b) < 5:
                continue
            ca = float(df_a["close"].iloc[-1])
            cb = float(df_b["close"].iloc[-1])
            spread_now = ca - pair.hedge_ratio * cb
            if pair.z_std == 0:
                continue
            z = (spread_now - pair.z_mean) / pair.z_std
            partner = pair.b if symbol == pair.a else pair.a
            # If z > entry → spread elevated → A overpriced relative to B → short A, long B
            # If z < -entry → spread depressed → A underpriced relative to B → long A, short B
            if abs(z) < self.entry_z:
                continue
            if symbol == pair.a:
                role = "leader" if z > 0 else "laggard"
            else:
                role = "laggard" if z > 0 else "leader"
            sig = PairSignal(
                in_pair=True,
                partner=partner,
                role=role,
                z_score=float(z),
                hedge_ratio=pair.hedge_ratio,
                half_life_days=pair.half_life,
                p_value=pair.p_value,
                reason=f"|z|={abs(z):.2f} > {self.entry_z}, half-life={pair.half_life:.1f}d",
                size_mult=self.size_mult,
            )
            if best is None or abs(z) > abs(best.z_score):
                best = sig
        return best or PairSignal(reason="no |z|>entry on active pairs")
