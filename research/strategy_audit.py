"""Offline strategy audit harness with sector x regime scorecard output.

Per-regime bucketing: each trade is tagged with the 4-state regime active at
its entry bar (computed from that symbol's daily bars up to entry_date).
Weights are derived per (sector x regime) cell with fallback chain per spec.
"""

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

# Sector labels MUST match filters.SECTOR_MAP — router looks up by same key.
SECTOR_GROUPS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "ADBE", "CRM", "NOW"],
    "semi": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "TXN", "AMAT"],
    "cloud": ["SNOW", "PLTR", "DDOG", "OKTA", "NET", "ZS", "MDB"],
    "cyber": ["PANW", "CRWD"],
    "finance": ["JPM", "GS", "MS", "V", "MA", "COF", "BAC"],
    "health": ["UNH", "LLY", "ISRG", "VRTX", "AMGN", "DHR", "PFE"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "HAL", "OXY"],
    "industrial": ["CAT", "HON", "GE", "RTX", "DE", "MMM", "CSX"],
    "consumer_disc": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT"],
    "consumer_stap": ["COST", "WMT", "PEP", "KO", "PG", "PM", "MDLZ"],
    "reit": ["AMT", "EQIX", "PLD"],
    "utility": ["NEE", "DUK", "SO"],
    "crypto": ["BTC/USD", "ETH/USD"],
    "futures": ["NQ", "ES", "CL", "GC"],
}

REGIMES = ["bull_trending", "bull_choppy", "bear_trending", "bear_choppy"]

# Minimum sample thresholds per spec
CELL_MIN_SAMPLES = 20
SECTOR_MIN_SAMPLES = 30

OUT_DIR = os.path.dirname(__file__)
SCORECARD_FILE = os.path.join(OUT_DIR, "strategy_scorecard.csv")
WEIGHTS_FILE = os.path.join(OUT_DIR, "sector_weights.json")


@dataclass
class AuditRow:
    strategy: str
    sector: str
    regime: str
    signals: int
    win_rate: float
    avg_r: float
    alpha: float
    best_session: str
    flag: str
    quality_score: float


def classify_regime(daily_bars: pd.DataFrame) -> str:
    if daily_bars is None or len(daily_bars) < 60:
        return "bull_choppy"
    adx = ta.trend.ADXIndicator(
        daily_bars["high"], daily_bars["low"], daily_bars["close"], window=14
    ).adx().iloc[-1]
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


def tag_trade_regime(trade: dict, bars: pd.DataFrame) -> str:
    """Classify regime at the trade's entry bar using history up to entry_date."""
    entry_date = trade.get("entry_date")
    if not entry_date or bars is None or bars.empty:
        return "bull_choppy"
    try:
        mask = bars.index <= pd.Timestamp(entry_date)
        window = bars.loc[mask]
    except Exception:
        return "bull_choppy"
    if len(window) < 60:
        return "bull_choppy"
    return classify_regime(window.tail(80))


class StrategyAudit:
    def __init__(self, data_fetcher, config: dict | None = None):
        self.config = config or load_config()
        self.data = data_fetcher
        self.router = StrategyRouter(self.config)

    def run(self, months: int = 6) -> pd.DataFrame:
        rows: list[AuditRow] = []
        sector_weights: dict[str, dict] = {}

        for sector, symbols in SECTOR_GROUPS.items():
            print(f"[audit] sector={sector} symbols={symbols}")
            sector_rows: list[AuditRow] = []
            strategy_trades: dict[str, list[dict]] = {}
            bars_per_symbol: dict[str, pd.DataFrame] = {}

            for strategy in self.config.get("strategies", {}):
                result = self._evaluate_strategy(strategy, symbols, months)
                trades = result["trades"]
                bars_per_symbol = result["bars"]  # same each loop (cached)
                strategy_trades[strategy] = trades

                # Bucket trades by regime
                regime_buckets: dict[str, list[dict]] = {r: [] for r in REGIMES}
                for trade in trades:
                    sym = trade.get("symbol")
                    regime = tag_trade_regime(trade, bars_per_symbol.get(sym))
                    regime_buckets.setdefault(regime, []).append(trade)

                baseline = self._baseline_return(bars_per_symbol)

                for regime in REGIMES:
                    bucket = regime_buckets.get(regime, [])
                    row = self._row_from_bucket(
                        strategy=strategy,
                        sector=sector,
                        regime=regime,
                        trades=bucket,
                        baseline=baseline,
                    )
                    sector_rows.append(row)
                    rows.append(row)

            sector_weights[sector] = self._build_sector_weights(
                sector_rows, strategy_trades
            )

        df = pd.DataFrame([asdict(r) for r in rows])
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_csv(SCORECARD_FILE, index=False)
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(sector_weights, f, indent=2)
        print(f"[audit] wrote {SCORECARD_FILE}")
        print(f"[audit] wrote {WEIGHTS_FILE}")
        return df

    def _evaluate_strategy(self, strategy: str, symbols: list[str], months: int) -> dict:
        bars = {}
        days = max(180, months * 22)
        for sym in symbols:
            try:
                df = self.data.get_intraday_bars(sym, timeframe="1Day", days=days)
            except Exception:
                df = None
            if df is None or len(df) < 80:
                try:
                    bars_dict = self.data.get_bars([sym], timeframe="1Day", days=days)
                    df = bars_dict.get(sym)
                except Exception:
                    df = None
            if df is not None and len(df) >= 80:
                bars[sym] = df

        trades: list[dict] = []
        if bars:
            test_config = json.loads(json.dumps(self.config))
            # Single-strategy isolation: Backtester uses select_strategies() which
            # ignores config weights — monkey-patch it to force just our target.
            test_config.setdefault("signals", {})
            test_config["signals"]["min_agreeing_strategies"] = 1
            test_config["signals"]["min_composite_score"] = 0.1
            import strategy_selector as _sel
            orig_select = _sel.select_strategies

            def _forced(df, symbol, sector_regime=None, _strat=strategy):
                return {"regime": "forced", "strategies": {_strat: 1.0}, "reason": "audit"}

            _sel.select_strategies = _forced
            try:
                # Patch the backtester's imported reference too
                import backtester as _bt
                orig_bt_select = _bt.select_strategies
                _bt.select_strategies = _forced
                result = Backtester(test_config).run(bars, min_bars=60)
                # BacktestResult.trades is already a list[dict] (backtester.py:509)
                trades = list(result.trades)
            except Exception as exc:
                print(f"[audit] backtest failed {strategy}: {exc}")
                trades = []
            finally:
                _sel.select_strategies = orig_select
                try:
                    _bt.select_strategies = orig_bt_select
                except NameError:
                    pass

        return {"trades": trades, "bars": bars}

    def _row_from_bucket(
        self, strategy: str, sector: str, regime: str, trades: list[dict], baseline: float
    ) -> AuditRow:
        signals = len(trades)
        wins = [t for t in trades if t["pnl"] >= 0]
        win_rate = len(wins) / signals if signals else 0.0
        avg_r = self._avg_r(trades)
        trade_return = sum(t["pnl_pct"] for t in trades) if trades else 0.0
        alpha = trade_return - baseline
        quality = (
            max(0.0, (win_rate * avg_r) * math.log(max(signals, 1)))
            if win_rate >= 0.40 and avg_r >= 0.5
            else 0.0
        )
        flag = self._flag(signals, win_rate, avg_r, alpha)
        return AuditRow(
            strategy=strategy,
            sector=sector,
            regime=regime,
            signals=signals,
            win_rate=round(win_rate * 100, 1),
            avg_r=round(avg_r, 2),
            alpha=round(alpha * 100, 2),
            best_session="mid",
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
            if df is None or df.empty:
                continue
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

    def _build_sector_weights(
        self, rows: list[AuditRow], strategy_trades: dict[str, list[dict]]
    ) -> dict:
        """Produce sector_weights[sector][regime] with fallback chain.

        Per spec:
          - cell with >= CELL_MIN_SAMPLES signals uses its own top-3 weighted
            by quality_score.
          - cell below threshold uses sector _fallback (weighted avg across
            regimes where quality > 0).
          - sector with < SECTOR_MIN_SAMPLES total uses global stock weights.
        """
        total_signals = sum(sum(len(t) for t in strategy_trades.values()) for _ in [0])
        # total trades across all strategies in this sector
        total_sector_signals = sum(len(t) for t in strategy_trades.values())

        # Group rows by regime
        by_regime: dict[str, list[AuditRow]] = {r: [] for r in REGIMES}
        for row in rows:
            by_regime.setdefault(row.regime, []).append(row)

        result: dict[str, dict] = {}

        # Per-regime cells
        for regime in REGIMES:
            regime_rows = by_regime.get(regime, [])
            regime_signals = sum(r.signals for r in regime_rows)
            if regime_signals >= CELL_MIN_SAMPLES:
                result[regime] = self._top_weights(regime_rows)
            # else: skip — will be filled from fallback below

        # Sector-level fallback: weighted avg across ALL rows (any regime) with quality > 0
        fallback = self._top_weights(rows)
        if not fallback:
            if total_sector_signals < SECTOR_MIN_SAMPLES:
                # Global fallback — use current stock_weights from router
                fallback = self.router.get_strategies("stock")
                total = sum(fallback.values()) or 1.0
                fallback = {k: round(v / total, 4) for k, v in fallback.items()}
            else:
                # Sector has samples but no quality strategies — degrade gracefully
                fallback = self.router.get_strategies("stock")
                total = sum(fallback.values()) or 1.0
                fallback = {k: round(v / total, 4) for k, v in fallback.items()}

        result["_fallback"] = fallback

        # Fill any missing regime cells with _fallback
        for regime in REGIMES:
            if regime not in result:
                result[regime] = dict(fallback)

        result["_generated_at"] = datetime.utcnow().isoformat()
        result["_total_signals"] = total_sector_signals
        return result

    def _top_weights(self, rows: list[AuditRow], top_n: int = 3) -> dict:
        """Return normalized top-N weights by quality_score from a row set."""
        scored = [r for r in rows if r.quality_score > 0]
        if not scored:
            return {}
        # If multiple rows for same strategy (different regimes), keep max quality
        best: dict[str, float] = {}
        for row in scored:
            if row.quality_score > best.get(row.strategy, 0.0):
                best[row.strategy] = row.quality_score
        # Top-N
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        total = sum(q for _, q in ranked) or 1.0
        return {name: round(q / total, 4) for name, q in ranked}


if __name__ == "__main__":
    # Light harness — skip Discord/watchers, just broker + data.
    import nest_asyncio
    nest_asyncio.apply()

    from ib_broker import IBBroker
    from ib_data import IBDataFetcher

    config = load_config()
    broker = IBBroker(config)
    data = IBDataFetcher(broker._ib, broker._contracts, config)
    try:
        audit = StrategyAudit(data, config)
        audit.run(months=12)
    finally:
        try:
            broker._ib.disconnect()
        except Exception:
            pass
