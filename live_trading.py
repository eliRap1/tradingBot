"""
LIVE TRADING MODULE - Real-time market data and execution.

This module ensures the bot:
1. Only trades during LIVE market hours
2. Uses REAL-TIME prices (not stale data)
3. Validates data freshness before trading
4. Handles market open/close transitions
5. Supports both US stocks and 24/7 crypto

CRITICAL: This prevents trading on old/stale data!
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import setup_logger

log = setup_logger("live_trading")

# Timezone for US markets
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


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
        Check if we can trade a symbol right now.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        from broker import CRYPTO_SYMBOLS
        
        is_crypto = symbol in CRYPTO_SYMBOLS
        status = self.get_market_status()
        
        # Crypto trades 24/7
        if is_crypto:
            return True, "crypto_24_7"
        
        # Stocks need market to be open
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
        from broker import CRYPTO_SYMBOLS
        
        if price is None or price <= 0:
            return False, "invalid_price"
        
        is_crypto = symbol in CRYPTO_SYMBOLS
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
        from broker import CRYPTO_SYMBOLS
        
        is_crypto = symbol in CRYPTO_SYMBOLS
        
        try:
            # Get latest price
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
    """
    Validates that all data used for trading decisions is fresh.

    This prevents the bot from trading on old/stale market data.
    """

    MAX_BAR_AGE_MINUTES = {
        "1Min": 2,
        "5Min": 10,
        "15Min": 30,
        "1Hour": 120,
        "1Day": 1440,  # 24 hours
    }

    # Crypto gets more lenient staleness thresholds because:
    # 1. REST polling for 170+ symbols means crypto bars can lag
    # 2. Crypto trades 24/7 but volume drops off-peak, widening bar gaps
    MAX_CRYPTO_BAR_AGE_MINUTES = {
        "1Min": 5,
        "5Min": 20,     # Was 10 — too tight, caused repeated SOL/USD skips
        "15Min": 45,
        "1Hour": 180,
        "1Day": 1440,
    }
    
    def validate_bars(self, df, timeframe: str = "5Min", symbol: str = None) -> Tuple[bool, str]:
        """
        Validate that bar data is fresh enough for trading.

        Args:
            df: DataFrame with timestamp index
            timeframe: The timeframe of the bars
            symbol: Optional symbol — crypto gets more lenient thresholds

        Returns:
            (is_valid: bool, reason: str)
        """
        if df is None or df.empty:
            return False, "no_data"

        # Crypto gets more lenient thresholds (REST polling lag for 170+ symbols)
        from broker import CRYPTO_SYMBOLS
        is_crypto = symbol and (symbol in CRYPTO_SYMBOLS or symbol.replace("/", "") in CRYPTO_SYMBOLS)
        age_table = self.MAX_CRYPTO_BAR_AGE_MINUTES if is_crypto else self.MAX_BAR_AGE_MINUTES
        max_age = age_table.get(timeframe, 60)
        
        # Get latest bar timestamp
        try:
            last_bar_time = df.index[-1]
            
            # Convert to UTC if needed
            if hasattr(last_bar_time, 'tz_localize'):
                if last_bar_time.tzinfo is None:
                    last_bar_time = last_bar_time.tz_localize(UTC)
            
            now = datetime.now(UTC)
            age_minutes = (now - last_bar_time).total_seconds() / 60
            
            if age_minutes > max_age:
                return False, f"bars_stale_{int(age_minutes)}min_old"
            
            return True, f"bars_fresh_{int(age_minutes)}min"
            
        except Exception as e:
            return False, f"validation_error_{str(e)}"


def ensure_live_trading_mode():
    """
    Verify that the bot is configured for live/paper trading (not backtest).
    
    Call this at startup to prevent accidental backtesting when live trading intended.
    """
    mode = os.getenv("TRADING_MODE", "paper")
    
    if mode == "backtest":
        log.error("TRADING_MODE is set to 'backtest' - not suitable for live trading!")
        log.error("Set TRADING_MODE=paper or TRADING_MODE=live")
        return False
    
    if mode == "live":
        log.warning("=" * 60)
        log.warning("*** LIVE TRADING MODE - REAL MONEY AT RISK ***")
        log.warning("=" * 60)
    else:
        log.info("Paper trading mode - no real money at risk")
    
    return True
