"""Live smoke test for the full edge pipeline.

Connects to IB paper, instantiates every edge module, asks each one for
live signals, and prints sanity-checked values. Fails fast if any module
crashes or returns obviously-broken output.

Usage:
    python tools/edge_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

import nest_asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

nest_asyncio.apply()

from edge.cross_asset import CrossAssetEngine
from edge.microstructure import MicrostructureGate
from edge.ml_filter import MLSignalFilter
from edge.news_sentiment import NewsSentimentEngine
from ib_broker import IBBroker
from ib_data import IBDataFetcher
from regime import RegimeFilter
from strategy_router import StrategyRouter
from utils import load_config


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def main() -> int:
    failures: list[str] = []
    config = load_config()
    broker = IBBroker(config)
    data = IBDataFetcher(broker._ib, broker._contracts, config)

    try:
        _section("Cross-Asset Engine")
        t0 = time.time()
        cross = CrossAssetEngine(data, ttl_sec=0)
        signals = cross.get_signals()
        dt = time.time() - t0
        print(f"VIX regime:        {signals.vix_regime}")
        print(f"VIX term:          {signals.vix_term_structure}")
        print(f"Bond trend:        {signals.bond_trend}")
        print(f"DXY trend:         {signals.dxy_trend}")
        print(f"Breadth:           {signals.market_breadth:.1f}% ({signals.breadth_signal})")
        print(f"Sector momentum:   {signals.sector_momentum}")
        print(f"NQ overnight:      {signals.nq_overnight_move:+.4f}")
        print(f"Size multiplier:   {signals.size_multiplier}")
        print(f"Elapsed:           {dt:.1f}s")
        if not (0.15 <= signals.size_multiplier <= 1.25):
            failures.append(f"size_multiplier out of bounds: {signals.size_multiplier}")
        if signals.vix_regime not in {"low", "normal", "elevated", "panic"}:
            failures.append(f"vix_regime invalid: {signals.vix_regime}")
        if signals.breadth_signal not in {"healthy", "neutral", "weak"}:
            failures.append(f"breadth_signal invalid: {signals.breadth_signal}")

        _section("Microstructure Gate")
        micro = MicrostructureGate(broker, data, config)
        test_symbols = ["AAPL", "SPY", "QQQ", "NVDA"]
        for sym in test_symbols:
            try:
                sig = micro.evaluate(sym, data.get_intraday_bars(sym, timeframe="5Min", days=1))
                print(f"{sym}: spread={sig.spread_pct*100:.3f}% ofi={sig.ofi_score:+.3f} "
                      f"spy_corr={sig.spy_corr:+.2f} blocked={sig.blocked}")
                if sig.spread_pct < 0 or sig.spread_pct > 0.10:
                    failures.append(f"{sym} spread_pct insane: {sig.spread_pct}")
                if abs(sig.spy_corr) > 1.0:
                    failures.append(f"{sym} spy_corr out of [-1,1]: {sig.spy_corr}")
            except Exception as exc:
                failures.append(f"microstructure {sym}: {exc}")
                print(f"{sym}: ERROR {exc}")

        _section("News / Earnings Engine")
        news = NewsSentimentEngine(config)
        blocked = news.get_blocked_symbols(["AAPL", "MSFT", "NVDA", "TSLA", "META"])
        print(f"Blocked this week: {sorted(blocked) if blocked else '(none)'}")
        sample = news.get_days_to_earnings("AAPL")
        print(f"AAPL days to earnings: {sample}")
        if not isinstance(sample, int):
            failures.append(f"days_to_earnings not int: {sample}")

        _section("ML Signal Filter")
        ml = MLSignalFilter(min_trades=config.get("edge", {}).get("ml_min_trades", 100))
        print(f"Model loaded:      {ml.model is not None}")
        print(f"Meta:              n={ml.meta.n_samples} auc={ml.meta.auc}")
        conf = ml.predict_quality({
            "strategy_scores": {"momentum": 0.6, "breakout": 0.4},
            "composite_score": 0.5,
            "num_agreeing": 2,
            "regime": "bull_trending",
            "size_multiplier": 1.0,
            "spread_pct": 0.0008,
            "ofi_score": 0.02,
            "spy_corr": 0.65,
            "hour_of_day": 11,
            "day_of_week": 2,
            "session_bucket": "mid",
            "days_to_earnings": 25,
            "nq_overnight_move": 0.001,
        })
        print(f"Passthrough conf:  {conf}")
        if not (0.1 <= conf <= 1.0):
            failures.append(f"ml confidence out of bounds: {conf}")

        _section("Regime + Strategy Router")
        regime = RegimeFilter(data, universe=["SPY", "QQQ", "IWM"])
        regime_str = regime.classify_4state()
        router = StrategyRouter(config)
        print(f"Current regime:    {regime_str}")
        for sector in ["tech", "semi", "finance", "energy"]:
            weights = router.get_strategies("stock", sector=sector, regime=regime_str)
            total = sum(weights.values())
            print(f"{sector:12} ({regime_str}): total={total:.3f}  {weights}")
            if not (0.99 <= total <= 1.01):
                failures.append(f"{sector}/{regime_str} weights sum={total}")

        _section("Summary")
        if failures:
            print(f"FAILURES ({len(failures)}):")
            for msg in failures:
                print(f"  - {msg}")
            return 1
        print("ALL EDGE MODULES LIVE-OK")
        return 0
    finally:
        try:
            broker._ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
