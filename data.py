import pandas as pd
from datetime import datetime, timedelta
from utils import setup_logger

log = setup_logger("data")


class DataFetcher:
    def __init__(self, broker):
        self.api = broker.api

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        """Fetch historical bars for multiple symbols."""
        end = datetime.now()
        start = end - timedelta(days=days)

        bars = {}
        # Fetch in batches to avoid API limits
        batch_size = 20
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                raw = self.api.get_bars(
                    batch,
                    timeframe,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    adjustment="split",
                    feed="iex"
                )
                for bar in raw:
                    sym = bar.S
                    if sym not in bars:
                        bars[sym] = []
                    bars[sym].append({
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": int(bar.v),
                        "vwap": float(bar.vw) if hasattr(bar, "vw") else None,
                    })
            except Exception as e:
                log.error(f"Failed to fetch bars for {batch}: {e}")

        # Convert to DataFrames
        result = {}
        for sym, data in bars.items():
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                result[sym] = df

        log.info(f"Fetched bars for {len(result)}/{len(symbols)} symbols")
        return result

    def get_latest_price(self, symbol: str) -> float | None:
        try:
            trade = self.api.get_latest_trade(symbol, feed="iex")
            return float(trade.price)
        except Exception as e:
            log.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        prices = {}
        for sym in symbols:
            price = self.get_latest_price(sym)
            if price:
                prices[sym] = price
        return prices

    def get_snapshot(self, symbol: str):
        try:
            return self.api.get_snapshot(symbol, feed="iex")
        except Exception as e:
            log.error(f"Failed to get snapshot for {symbol}: {e}")
            return None
