import time
import pandas as pd
from utils import setup_logger

log = setup_logger("screener")


class Screener:
    def __init__(self, config: dict, data_fetcher):
        self.config = config["screener"]
        self.data = data_fetcher
        self._cached_universe: list[str] = []
        self._cached_at: float = 0.0
        self._cache_ttl_sec = int(self.config.get("rs_cache_ttl_sec", 3600))

    def get_universe(self) -> list[str]:
        """Filter + rank. Prefers universe_full ranked by RS top-quartile.

        Also applies activity prefilter (ATR%, $vol) to eliminate
        illiquid/dead sectors that dilute edge.
        """
        if time.time() - self._cached_at < self._cache_ttl_sec and self._cached_universe:
            return list(self._cached_universe)

        full = self.config.get("universe_full") or self.config["universe"]
        static = self.config["universe"]
        log.info(f"Screening {len(full)} candidates (full universe)")

        bars = self.data.get_bars(full, timeframe="1Day", days=32)

        # Stage 1: basic liquidity + price gates
        passed_basic = []
        for sym, df in bars.items():
            if df is None or len(df) < 22:
                continue
            latest_close = float(df["close"].iloc[-1])
            avg_volume = float(df["volume"].tail(20).mean())
            if latest_close < self.config["min_price"]:
                continue
            if latest_close > self.config["max_price"]:
                continue
            if avg_volume < self.config["min_avg_volume"]:
                continue
            passed_basic.append(sym)

        # Stage 2: activity prefilter — ATR%>=1%, 20d avg$vol>=$50M
        min_atr_pct = float(self.config.get("min_atr_pct", 0.010))
        min_dollar_vol = float(self.config.get("min_dollar_volume", 50_000_000))
        passed_active = []
        for sym in passed_basic:
            df = bars[sym]
            try:
                tr = (df["high"] - df["low"]).tail(14).mean()
                atr_pct = tr / float(df["close"].iloc[-1])
                dollar_vol = float(df["close"].tail(20).mean() * df["volume"].tail(20).mean())
                if atr_pct < min_atr_pct:
                    continue
                if dollar_vol < min_dollar_vol:
                    continue
            except Exception:
                continue
            passed_active.append(sym)

        # Stage 3: rank by 20d return, keep top quartile
        use_rs = bool(self.config.get("rs_universe", True))
        if use_rs and len(passed_active) >= 20:
            returns = {}
            lookback = int(self.config.get("rs_lookback_days", 20))
            for sym in passed_active:
                df = bars[sym]
                try:
                    start = float(df["close"].iloc[-(lookback + 1)])
                    end = float(df["close"].iloc[-1])
                    if start > 0:
                        returns[sym] = (end / start) - 1.0
                except Exception:
                    continue
            ranked = sorted(returns.items(), key=lambda kv: kv[1], reverse=True)
            quartile_pct = float(self.config.get("rs_top_pct", 0.25))
            keep_n = max(20, int(len(ranked) * quartile_pct))
            qualified = [s for s, _ in ranked[:keep_n]]
        else:
            qualified = passed_active

        # Always include static universe (mega-cap anchors) so core coverage stays
        for sym in static:
            if sym not in qualified and sym in passed_basic:
                qualified.append(sym)

        log.info(
            f"Screener passed: {len(qualified)}/{len(full)} "
            f"(basic={len(passed_basic)} active={len(passed_active)})"
        )
        self._cached_universe = qualified
        self._cached_at = time.time()
        return qualified
