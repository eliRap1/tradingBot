"""
LIVE TRADING MODULE - Real-time market data and execution.

This module ensures the bot:
1. Only trades during LIVE market hours (per asset class)
2. Uses REAL-TIME prices (not stale data)
3. Validates data freshness before trading
4. Handles market open/close transitions
5. Supports stocks (NYSE), futures (CME 23h), and crypto (24/7)

Asset market hours:
  Stocks  : NYSE  09:30 - 16:00 ET Mon-Fri
  Futures : CME   18:00 ET (Sun/Mon-Thu) to 17:00 ET next day, 1h break daily
  Crypto  : 24/7 (IB PAXOS near-continuous)

CRITICAL: This prevents trading on old/stale data!
"""

import os
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import setup_logger

log = setup_logger("live_trading")

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# NYSE observed holidays (2025-2027) — same set used in ib_broker.py
_NYSE_HOLIDAYS = frozenset({
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25",
    "2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
    "2027-01-01","2027-01-18","2027-02-15","2027-03-26","2027-05-31",
    "2027-06-18","2027-07-05","2027-09-06","2027-11-25","2027-12-24",
})


def _is_nyse_trading_day(d: date) -> bool:
    """Return True if d is a regular NYSE trading day."""
    return d.weekday() < 5 and d.isoformat() not in _NYSE_HOLIDAYS


def _is_cme_futures_open() -> bool:
    """
    Return True if CME Globex (ES/NQ/CL/GC) is currently open.

    CME Globex schedule (US Central → ET offset is +1h in standard, +1h in daylight):
      Opens : Sunday 18:00 ET (i.e. Sunday 17:00 CT)
      Closes: Friday 17:00 ET
      Daily maintenance break: 17:00 – 18:00 ET every weekday
    """
    now = datetime.now(ET)
    wd = now.weekday()   # 0=Mon … 6=Sun
    h, m = now.hour, now.minute

    # Saturday: always closed
    if wd == 5:
        return False
    # Sunday: only open after 18:00 ET
    if wd == 6:
        return h >= 18
    # Friday: closes at 17:00 ET
    if wd == 4:
        return not (h >= 17)
    # Monday–Thursday: open except during 17:00–18:00 ET maintenance break
    return not (h == 17)


def _is_crypto_open() -> bool:
    """IB PAXOS crypto is essentially 24/7 — always return True."""
    return True


def _classify_symbol(symbol: str) -> str:
    """
    Light-weight asset classifier without loading the full config.
    Returns 'futures', 'crypto', or 'stock'.
    Uses simple heuristics — for authoritative classification use InstrumentClassifier.
    """
    _FUTURES_ROOTS = {"NQ", "ES", "CL", "GC", "RTY", "YM", "NKD"}
    _CRYPTO_SUFFIXES = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}
    if symbol in _FUTURES_ROOTS:
        return "futures"
    if symbol in _CRYPTO_SUFFIXES:
        return "crypto"
    return "stock"


@dataclass
class MarketStatus:
    """Current market status."""
    is_open: bool
    next_open: Optional[datetime]
    next_close: Optional[datetime]
    session: str  # "pre", "regular", "post", "closed"
    crypto_open: bool = True  # Crypto trades 24/7


class LiveTradingManager:
    """
    Manages live trading with real-time data validation.
    
    Key responsibilities:
    1. Check market hours before trading
    2. Validate data freshness (reject stale prices)
    3. Handle pre-market and after-hours sessions
    4. Ensure crypto trades 24/7
    """
    
    # Maximum age of price data before considered stale
    MAX_STOCK_PRICE_AGE_SECONDS = 60  # 1 minute for stocks
    MAX_CRYPTO_PRICE_AGE_SECONDS = 30  # 30 seconds for crypto (faster moving)
    
    # Market hours (Eastern Time)
    PREMARKET_OPEN = 4, 0   # 4:00 AM ET
    MARKET_OPEN = 9, 30     # 9:30 AM ET
    MARKET_CLOSE = 16, 0    # 4:00 PM ET
    AFTERHOURS_CLOSE = 20, 0  # 8:00 PM ET
    
    def __init__(self, broker):
        self.broker = broker
        self._last_market_check = None
        self._cached_status = None
        
    def get_market_status(self) -> MarketStatus:
        """Get current market status from Alpaca."""
        # Cache for 1 minute to reduce API calls
        now = datetime.now()
        if (self._last_market_check and 
            (now - self._last_market_check).total_seconds() < 60 and
            self._cached_status):
            return self._cached_status
        
        try:
            clock = self.broker.get_clock()
            
            status = MarketStatus(
                is_open=clock.is_open,
                next_open=clock.next_open,
                next_close=clock.next_close,
                session=self._determine_session(clock),
                crypto_open=True,  # Crypto always open on Alpaca
            )
            
            self._cached_status = status
            self._last_market_check = now
            
            return status
            
        except Exception as e:
            log.error(f"Failed to get market status: {e}")
            # Default to checking manually
            return self._manual_market_check()
    
    def _determine_session(self, clock) -> str:
        """Determine which trading session we're in."""
        if clock.is_open:
            return "regular"
        
        now_et = datetime.now(ET)
        
        # Check if premarket
        premarket_start = now_et.replace(
            hour=self.PREMARKET_OPEN[0], 
            minute=self.PREMARKET_OPEN[1],
            second=0
        )
        market_open = now_et.replace(
            hour=self.MARKET_OPEN[0],
            minute=self.MARKET_OPEN[1],
            second=0
        )
        
        if premarket_start <= now_et < market_open:
            return "pre"
        
        # Check if after-hours
        market_close = now_et.replace(
            hour=self.MARKET_CLOSE[0],
            minute=self.MARKET_CLOSE[1],
            second=0
        )
        afterhours_close = now_et.replace(
            hour=self.AFTERHOURS_CLOSE[0],
            minute=self.AFTERHOURS_CLOSE[1],
            second=0
        )
        
        if market_close <= now_et < afterhours_close:
            return "post"
        
        return "closed"
    
    def _manual_market_check(self) -> MarketStatus:
        """Manual market hours check if API fails."""
        now_et = datetime.now(ET)
        weekday = now_et.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend = closed
        if weekday >= 5:
            return MarketStatus(
                is_open=False,
                next_open=None,
                next_close=None,
                session="closed",
                crypto_open=True,
            )
        
        # Check time
        current_time = (now_et.hour, now_et.minute)
        
        if self.MARKET_OPEN <= current_time < self.MARKET_CLOSE:
            return MarketStatus(
                is_open=True,
                next_open=None,
                next_close=now_et.replace(
                    hour=self.MARKET_CLOSE[0],
                    minute=self.MARKET_CLOSE[1]
                ),
                session="regular",
            )
        elif self.PREMARKET_OPEN <= current_time < self.MARKET_OPEN:
            return MarketStatus(
                is_open=False,
                next_open=now_et.replace(
                    hour=self.MARKET_OPEN[0],
                    minute=self.MARKET_OPEN[1]
                ),
                next_close=None,
                session="pre",
            )
        elif self.MARKET_CLOSE <= current_time < self.AFTERHOURS_CLOSE:
            return MarketStatus(
                is_open=False,
                next_open=None,
                next_close=None,
                session="post",
            )
        else:
            return MarketStatus(
                is_open=False,
                next_open=None,
                next_close=None,
                session="closed",
            )
    
    def can_trade_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if we can trade a symbol right now (per asset class).

        Returns:
            (can_trade: bool, reason: str)
        """
        asset = _classify_symbol(symbol)

        if asset == "crypto":
            return _is_crypto_open(), "crypto_24_7" if _is_crypto_open() else "crypto_closed"

        if asset == "futures":
            open_ = _is_cme_futures_open()
            return open_, "cme_open" if open_ else "cme_maintenance_break"

        # Stocks: need NYSE regular session
        status = self.get_market_status()
        if status.is_open:
            return True, "market_open"
        if status.session == "pre":
            return False, "premarket_only"
        if status.session == "post":
            return False, "afterhours_only"
        return False, f"market_closed_next_open_{status.next_open}"
    
    def validate_price_freshness(self, symbol: str, price: float, 
                                  timestamp: datetime = None) -> Tuple[bool, str]:
        """
        Validate that a price is fresh enough to trade on.
        
        This is CRITICAL for live trading - prevents trading on stale data!
        
        Args:
            symbol: The symbol
            price: The price to validate
            timestamp: When the price was fetched (default: now)
            
        Returns:
            (is_fresh: bool, reason: str)
        """
        if price is None or price <= 0:
            return False, "invalid_price"

        is_crypto = _classify_symbol(symbol) == "crypto"
        max_age = (self.MAX_CRYPTO_PRICE_AGE_SECONDS if is_crypto
                   else self.MAX_STOCK_PRICE_AGE_SECONDS)
        
        # If we have a timestamp, check age
        if timestamp:
            ts = timestamp.astimezone(UTC) if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - ts).total_seconds()
            if age > max_age:
                return False, f"stale_price_age_{int(age)}s"
        
        return True, "price_fresh"
    
    def get_live_price(self, symbol: str, data_fetcher) -> Tuple[Optional[float], str]:
        """
        Get a verified live price for a symbol.
        
        This ensures we're trading on current market prices, not stale data.
        
        Returns:
            (price: float or None, status: str)
        """
        is_crypto = _classify_symbol(symbol) == "crypto"

        try:
            price = data_fetcher.get_latest_price(symbol)
            
            if price is None:
                return None, "no_price_available"
            
            # For stocks, verify market is open
            if not is_crypto:
                can_trade, reason = self.can_trade_symbol(symbol)
                if not can_trade:
                    log.warning(f"Market closed for {symbol}: {reason}")
                    # Still return price for informational purposes
                    return price, f"market_closed_{reason}"
            
            return price, "live_price_ok"
            
        except Exception as e:
            log.error(f"Failed to get live price for {symbol}: {e}")
            return None, f"error_{str(e)}"
    
    def should_skip_cycle(self) -> Tuple[bool, str]:
        """
        Check if we should skip this trading cycle entirely.
        
        Skip when:
        - Market closed and no crypto positions
        - Within first 15 min of market open (too volatile)
        - Within last 15 min of market close (spreads widen)
        
        Returns:
            (should_skip: bool, reason: str)
        """
        status = self.get_market_status()
        now_et = datetime.now(ET)
        
        # Market completely closed (not even pre/post)
        if status.session == "closed":
            return False, "crypto_only_mode"  # Still run for crypto

        # Note: Opening delay is handled by coordinator's market_open_delay_min
        # to avoid double-skipping the first 30 minutes.

        # Skip last 15 minutes (closing spreads)
        if status.is_open and status.next_close:
            try:
                close_time = status.next_close
                if hasattr(close_time, 'astimezone'):
                    close_time = close_time.astimezone(ET)
                minutes_to_close = (close_time - now_et).total_seconds() / 60
                
                if 0 < minutes_to_close < 15:
                    return True, f"closing_spreads_{int(minutes_to_close)}min_left"
            except Exception:
                pass
        
        return False, "ok_to_trade"
    
    def wait_for_market_open(self, max_wait_hours: int = 16) -> bool:
        """
        Wait for market to open (with progress logging).
        
        Useful for bots that start before market hours.
        
        Returns:
            True if market opened, False if timeout
        """
        status = self.get_market_status()
        
        if status.is_open:
            return True
        
        if not status.next_open:
            log.info("Market closed, no next open time available")
            return False
        
        wait_seconds = (status.next_open - datetime.now(UTC)).total_seconds()
        
        if wait_seconds > max_wait_hours * 3600:
            log.info(f"Market opens in {wait_seconds/3600:.1f} hours - too long to wait")
            return False
        
        if wait_seconds > 0:
            log.info(f"Waiting for market open in {wait_seconds/60:.0f} minutes...")
            
            while wait_seconds > 0:
                sleep_time = min(300, wait_seconds)  # Check every 5 min
                time.sleep(sleep_time)
                
                status = self.get_market_status()
                if status.is_open:
                    log.info("Market is now OPEN!")
                    return True
                
                wait_seconds = (status.next_open - datetime.now(UTC)).total_seconds()
                if wait_seconds > 0:
                    log.info(f"  ... {wait_seconds/60:.0f} minutes remaining")
        
        return self.get_market_status().is_open


class DataFreshnessValidator:
    """Validates IB bar data freshness per asset type."""

    MAX_BAR_AGE_MINUTES = {
        "1Min": 3, "5Min": 10, "15Min": 30, "1Hour": 120, "1Day": 1440,
    }
    MAX_CRYPTO_BAR_AGE_MINUTES = {
        "1Min": 5, "5Min": 25, "15Min": 50, "1Hour": 200, "1Day": 1440,
    }
    MAX_FUTURES_BAR_AGE_MINUTES = {
        "1Min": 5, "5Min": 20, "15Min": 40, "1Hour": 150, "1Day": 1440,
    }

    def validate_bars(self, df, timeframe: str = "5Min",
                      symbol: str = None) -> Tuple[bool, str]:
        """Return (is_valid, reason). Checks bar age only."""
        if df is None or df.empty:
            return False, "no_data"
        asset = _classify_symbol(symbol) if symbol else "stock"
        if asset == "crypto":
            age_table = self.MAX_CRYPTO_BAR_AGE_MINUTES
        elif asset == "futures":
            age_table = self.MAX_FUTURES_BAR_AGE_MINUTES
        else:
            age_table = self.MAX_BAR_AGE_MINUTES
        max_age = age_table.get(timeframe, 60)
        try:
            last_bar_time = df.index[-1]
            if hasattr(last_bar_time, "tz_localize") and last_bar_time.tzinfo is None:
                last_bar_time = last_bar_time.tz_localize(UTC)
            age_minutes = (datetime.now(UTC) - last_bar_time).total_seconds() / 60
            if age_minutes > max_age:
                return False, f"bars_stale_{int(age_minutes)}min_old"
            return True, f"bars_fresh_{int(age_minutes)}min"
        except Exception as exc:
            return False, f"validation_error_{exc}"


def ensure_live_trading_mode() -> bool:
    """Return True when TRADING_MODE is paper or live (not backtest)."""
    mode = os.getenv("TRADING_MODE", "paper").lower()
    if mode == "backtest":
        log.error("TRADING_MODE=backtest -- not suitable for live trading\!")
        return False
    if mode not in ("paper", "live"):
        log.warning(f"Unrecognised TRADING_MODE='{mode}' -- defaulting to paper")
    return True
