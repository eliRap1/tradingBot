"""Offline strategy audit harness with sector x regime scorecard output."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import ta

from backtester import Backtester
from strategy_router import StrategyRouter
from utils import load_config

SECTOR_GROUPS = {
    "tech_mega": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "HPQ", "INTC"],
    "semis": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "ON", "SWKS"],
    "software": ["CRM", "NOW", "PANW", "CRWD", "DDOG", "OKTA", "ZM"],
    "financials": ["JPM", "GS", "MS", "V", "MA", "COF", "AFRM"],
    "healthcare": ["UNH", "LLY", "ISRG", "VRTX", "AMGN", "BIIB", "IDXX"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "HAL", "OXY"],
    "industrials": ["CAT", "HON", "GE", "RTX", "DE", "MMM", "CSX"],
    "consumer": ["COST", "HD", "NKE", "MCD", "SBUX", "TGT", "LYFT"],
    "crypto": ["BTC/USD", "ETH/USD"],
    "futures": ["NQ", "ES", "CL", "GC"],
}

OUT_DIR = os.path.dirname(__file__)
SCORECARD_FILE = os.path.join(OUT_DIR, "strategy_scorecard.csv")
WEIGHTS_FILE = os.path.join(OUT_DIR, "sector_weights.json")


@dataclass
class AuditRow:
    strategy: str
    sector: str
    signals: int
    win_rate: float
    avg_r: float
    alpha: float
    best_session: str
    best_regime: str
    flag: str
    quality_score: float


def classify_regime(daily_bars: pd.DataFrame) -> str:
    if daily_bars is None or len(daily_bars) < 60:
        return "bull_choppy"
    adx = ta.trend.ADXIndicator(daily_bars["high"], daily_bars["low"], daily_bars["close"], window=14).adx().iloc[-1]
    trending = adx > 25
    ema50 = daily_bars["close"].ewm(span=50).mean()
    bull = daily_bars["close"].iloc[-1] > ema50.iloc[-1]
    if bull and trending:
        return "bull_trending"
    if bull and not trending:
        return "bull_choppy"
    if not bull and trending:
        return "bear_trending"
    return "bear_choppy"


class StrategyAudit:
    def __init__(self, data_fetcher, config: dict | None = None):
        self.config = config or load_config()
        self.data = data_fetcher
        self.backtester = Backtester(self.config)
        self.router = StrategyRouter(self.config)

    def run(self, months: int = 6) -> pd.DataFrame:
        rows: list[AuditRow] = []
        sector_weights: dict[str, dict] = {}

        for sector, symbols in SECTOR_GROUPS.items():
            sector_rows = []
            for strategy in self.config.get("strategies", {}):
                stats = self._evaluate_strategy(strategy, sector, symbols, months)
                sector_rows.append(stats)
                rows.append(stats)
            sector_weights[sector] = self._build_sector_weights(sector_rows)

        df = pd.DataFrame([asdict(r) for r in rows])
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_csv(SCORECARD_FILE, index=False)
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(sector_weights, f, indent=2)
        return df

    def _evaluate_strategy(self, strategy: str, sector: str, symbols: list[str], months: int) -> AuditRow:
        bars = {}
        days = max(120, months * 21)
        for sym in symbols:
            df = self.data.get_intraday_bars(sym, timeframe="1Day", days=days)
            if df is not None and len(df) >= 80:
                bars[sym] = df

        trades = []
        if bars:
            test_config = json.loads(json.dumps(self.config))
            for name in list(test_config["strategies"].keys()):
                if name != strategy and "weight" in test_config["strategies"][name]:
                    test_config["strategies"][name]["weight"] = 0.0
            result = Backtester(test_config).run(bars, min_bars=60)
            trades = result.trades
            strategy_return = result.total_return_pct / 100.0
        else:
            strategy_return = 0.0

        signals = len(trades)
        wins = [t for t in trades if t["pnl"] >= 0]
        win_rate = len(wins) / signals if signals else 0.0
        avg_r = self._avg_r(trades)
        baseline_hold = self._baseline_return(bars)
        alpha = strategy_return - baseline_hold
        quality = max(0.0, (win_rate * avg_r) * math.log(max(signals, 1))) if win_rate >= 0.40 and avg_r >= 0.5 else 0.0
        flag = self._flag(signals, win_rate, avg_r, alpha)

        regimes = {}
        for sym, df in bars.items():
            regimes[classify_regime(df)] = regimes.get(classify_regime(df), 0) + 1
        best_regime = max(regimes, key=regimes.get) if regimes else "bull_choppy"

        return AuditRow(
            strategy=strategy,
            sector=sector,
            signals=signals,
            win_rate=round(win_rate * 100, 1),
            avg_r=round(avg_r, 2),
            alpha=round(alpha * 100, 2),
            best_session="mid",
            best_regime=best_regime,
            flag=flag,
            quality_score=round(quality, 3),
        )

    def _avg_r(self, trades: list[dict]) -> float:
        vals = []
        for t in trades:
            risk = abs(float(t["entry_price"]) - float(t["exit_price"])) or 1.0
            vals.append(float(t["pnl"]) / risk)
        return sum(vals) / len(vals) if vals else 0.0

    def _baseline_return(self, bars: dict[str, pd.DataFrame]) -> float:
        rets = []
        for df in bars.values():
            first_close = float(df["close"].iloc[0])
            last_close = float(df["close"].iloc[-1])
            if first_close > 0:
                rets.append((last_close / first_close) - 1)
        return sum(rets) / len(rets) if rets else 0.0

    def _flag(self, signals: int, win_rate: float, avg_r: float, alpha: float) -> str:
        if signals < 30:
            return "LOW_SAMPLE"
        if win_rate < 0.40 and avg_r < 0.5:
            return "REMOVE"
        if alpha < 0:
            return "NO_ALPHA"
        if win_rate > 0.55 and avg_r > 1.2 and alpha > 0:
            return "STRONG"
        return "OK"

    def _build_sector_weights(self, rows: list[AuditRow]) -> dict:
        ranked = sorted(rows, key=lambda r: r.quality_score, reverse=True)
        top = ranked[:3] if ranked else []
        fallback = {r.strategy: r.quality_score for r in top if r.quality_score > 0}
        if not fallback:
            fallback = self.router.get_strategies("stock")
        else:
            total = sum(fallback.values()) or 1.0
            fallback = {k: round(v / total, 4) for k, v in fallback.items()}
        return {
            "bull_trending": fallback,
            "bull_choppy": fallback,
            "bear_trending": fallback,
            "bear_choppy": fallback,
            "_fallback": fallback,
            "_generated_at": datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    from coordinator import Coordinator

    coord = Coordinator()
    audit = StrategyAudit(coord.data, coord.config)
    audit.run()
