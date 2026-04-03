import pandas as pd
from utils import setup_logger

log = setup_logger("screener")


class Screener:
    def __init__(self, config: dict, data_fetcher):
        self.config = config["screener"]
        self.data = data_fetcher

    def get_universe(self) -> list[str]:
        """Get filtered list of tradeable symbols."""
        candidates = self.config["universe"]
        log.info(f"Screening {len(candidates)} candidates")

        bars = self.data.get_bars(candidates, timeframe="1Day", days=30)

        qualified = []
        for sym, df in bars.items():
            if len(df) < 10:
                continue

            latest_close = df["close"].iloc[-1]
            avg_volume = df["volume"].tail(20).mean()

            # Price filter
            if latest_close < self.config["min_price"]:
                continue
            if latest_close > self.config["max_price"]:
                continue

            # Volume filter
            if avg_volume < self.config["min_avg_volume"]:
                continue

            qualified.append(sym)

        log.info(f"Screener passed: {len(qualified)} symbols: {qualified}")
        return qualified
