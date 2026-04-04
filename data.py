import pandas as pd
from datetime import datetime, timedelta
from utils import setup_logger

log = setup_logger("data")

CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}


class DataFetcher:
    def __init__(self, broker):
        self.api = broker.api

    def get_bars(self, symbols: list[str], timeframe: str = "1Day",
                 days: int = 60) -> dict[str, pd.DataFrame]:
        """Fetch historical bars for multiple symbols."""
        end = datetime.now()
        start = end - timedelta(days=days)

        # Separate crypto and stock symbols
        stock_symbols = [s for s in symbols if s not in CRYPTO_SYMBOLS]
        crypto_symbols = [s for s in symbols if s in CRYPTO_SYMBOLS]

        bars = {}
        time_fmt = "%Y-%m-%dT%H:%M:%SZ" if "Min" in timeframe else "%Y-%m-%d"

        # Fetch stock bars
        batch_size = 20
        for i in range(0, len(stock_symbols), batch_size):
            batch = stock_symbols[i:i + batch_size]
            try:
                raw = self.api.get_bars(
                    batch,
                    timeframe,
                    start=start.strftime(time_fmt),
                    end=end.strftime(time_fmt),
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

        # Fetch crypto bars (use "us_equity" feed doesn't apply — Alpaca crypto uses different endpoint)
        for sym in crypto_symbols:
            try:
                raw = self.api.get_crypto_bars(
                    sym,
                    timeframe,
                    start=start.strftime(time_fmt),
                    end=end.strftime(time_fmt),
                )
                bars[sym] = []
                for bar in raw:
                    bars[sym].append({
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": float(bar.v),
                        "vwap": float(bar.vw) if hasattr(bar, "vw") else None,
                    })
            except Exception as e:
                log.error(f"Failed to fetch crypto bars for {sym}: {e}")

        # Convert to DataFrames
        result = {}
        for sym, data in bars.items():
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                result[sym] = df

        log.info(f"Fetched {timeframe} bars for {len(result)}/{len(symbols)} symbols")
        return result

    def get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                          days: int = 5) -> pd.DataFrame | None:
        """Fetch intraday bars for a single symbol."""
        result = self.get_bars([symbol], timeframe=timeframe, days=days)
        return result.get(symbol)

    def get_latest_price(self, symbol: str) -> float | None:
        try:
            if symbol in CRYPTO_SYMBOLS:
                # Alpaca v2: use quote (bid/ask midpoint) for crypto
                quote = self.api.get_latest_crypto_quote(symbol)
                bid = float(quote.bp) if hasattr(quote, 'bp') else 0
                ask = float(quote.ap) if hasattr(quote, 'ap') else 0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return ask or bid or None
            else:
                trade = self.api.get_latest_trade(symbol, feed="iex")
                return float(trade.price)
        except Exception as e:
            log.error(f"Failed to get price for {symbol}: {e}")
            # Fallback: try getting price from latest bar
            try:
                if symbol in CRYPTO_SYMBOLS:
                    bars = self.api.get_crypto_bars(symbol, "1Min", limit=1)
                    for bar in bars:
                        return float(bar.c)
            except Exception:
                pass
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
