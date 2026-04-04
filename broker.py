import os
import time
import threading
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from utils import setup_logger

load_dotenv()
log = setup_logger("broker")

CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}


class Broker:
    def __init__(self, config: dict):
        mode = os.getenv("TRADING_MODE", "paper")
        if mode == "live":
            base_url = config["alpaca"]["live_url"]
            log.warning("*** LIVE TRADING MODE ***")
        else:
            base_url = config["alpaca"]["paper_url"]
            log.info("Paper trading mode")

        self.api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=base_url,
            api_version="v2"
        )

    def get_account(self):
        return self.api.get_account()

    def get_equity(self) -> float:
        return float(self.api.get_account().equity)

    def get_cash(self) -> float:
        return float(self.api.get_account().cash)

    def get_buying_power(self) -> float:
        return float(self.api.get_account().buying_power)

    def get_positions(self) -> list:
        return self.api.list_positions()

    def get_position(self, symbol: str):
        try:
            return self.api.get_position(symbol)
        except Exception:
            return None

    def get_open_orders(self) -> list:
        return self.api.list_orders(status="open")

    def submit_bracket_order(self, symbol: str, qty: int, side: str,
                              take_profit: float, stop_loss: float):
        """Submit order with take-profit and stop-loss. Works for both long and short."""
        log.info(f"BRACKET ORDER: {side} {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": round(take_profit, 2)},
            stop_loss={"stop_price": round(stop_loss, 2)}
        )

    def submit_short_bracket(self, symbol: str, qty: int,
                              take_profit: float, stop_loss: float):
        """Short sell with bracket order. TP is below entry, SL is above."""
        log.info(f"SHORT BRACKET: sell {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": round(take_profit, 2)},
            stop_loss={"stop_price": round(stop_loss, 2)}
        )

    def submit_crypto_order(self, symbol: str, qty: float, side: str,
                             take_profit: float, stop_loss: float):
        """Submit crypto order with separate TP and SL orders.
        Crypto doesn't support bracket orders on Alpaca, so we place
        entry + TP limit + SL stop as 3 separate orders."""
        log.info(f"CRYPTO ORDER: {side} {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")

        # Round crypto prices (BTC to 2 decimals, ETH to 2)
        tp_price = round(take_profit, 2)
        sl_price = round(stop_loss, 2)

        # Entry order
        entry = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc"
        )

        # Exit side (opposite of entry)
        exit_side = "sell" if side == "buy" else "buy"

        # Take-profit limit order
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                type="limit",
                limit_price=tp_price,
                time_in_force="gtc"
            )
            log.info(f"CRYPTO TP order placed: {exit_side} {qty} {symbol} @ ${tp_price}")
        except Exception as e:
            log.error(f"Failed to place crypto TP order: {e}")

        # Stop-loss order
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                type="stop",
                stop_price=sl_price,
                time_in_force="gtc"
            )
            log.info(f"CRYPTO SL order placed: {exit_side} {qty} {symbol} @ ${sl_price}")
        except Exception as e:
            log.error(f"Failed to place crypto SL order: {e}")

        return entry

    def get_quote(self, symbol: str):
        """Get current bid/ask quote."""
        try:
            if symbol in CRYPTO_SYMBOLS:
                return self.api.get_latest_crypto_quote(symbol)
            return self.api.get_latest_quote(symbol, feed="iex")
        except Exception as e:
            log.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def submit_smart_order(self, symbol: str, qty: int, side: str,
                           take_profit: float, stop_loss: float,
                           limit_offset_pct: float = 0.0002,
                           timeout_sec: int = 30) -> dict:
        """
        Smart execution: limit order with timeout fallback to market.

        1. Gets current price and places limit order slightly better
        2. Waits up to timeout_sec for fill
        3. If not filled, cancels and sends market bracket order
        """
        quote = self.get_quote(symbol)
        if not quote:
            # Fallback to immediate bracket order
            log.info(f"No quote for {symbol}, using market order")
            if side == "sell":
                self.submit_short_bracket(symbol, qty, take_profit, stop_loss)
            else:
                self.submit_bracket_order(symbol, qty, side, take_profit, stop_loss)
            return {"method": "market", "symbol": symbol}

        # Calculate limit price (slightly better than market)
        if side == "buy":
            ask = float(quote.ap) if hasattr(quote, 'ap') else float(quote.ask_price)
            limit_price = round(ask * (1 - limit_offset_pct), 2)
        else:
            bid = float(quote.bp) if hasattr(quote, 'bp') else float(quote.bid_price)
            limit_price = round(bid * (1 + limit_offset_pct), 2)

        log.info(f"SMART ORDER: {side} {qty} {symbol} limit=${limit_price:.2f} "
                 f"(timeout={timeout_sec}s)")

        try:
            # Place limit order (no bracket yet)
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="limit", limit_price=limit_price,
                time_in_force="day"
            )
            order_id = order.id

            # Poll for fill
            filled = False
            fill_price = limit_price
            elapsed = 0
            while elapsed < timeout_sec:
                time.sleep(2)
                elapsed += 2
                status = self.api.get_order(order_id)
                if status.status == "filled":
                    filled = True
                    fill_price = float(status.filled_avg_price)
                    break
                elif status.status in ("canceled", "expired", "rejected"):
                    break

            if filled:
                log.info(f"LIMIT FILLED: {symbol} @ ${fill_price:.2f}")
                # Now place bracket stop/TP as OCO
                tp_side = "sell" if side == "buy" else "buy"
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=qty, side=tp_side,
                        type="limit", limit_price=round(take_profit, 2),
                        time_in_force="gtc", order_class="oco",
                        stop_loss={"stop_price": round(stop_loss, 2)}
                    )
                except Exception:
                    # OCO not available, submit separate orders
                    self.api.submit_order(
                        symbol=symbol, qty=qty, side=tp_side,
                        type="limit", limit_price=round(take_profit, 2),
                        time_in_force="gtc"
                    )
                return {"method": "limit", "fill_price": fill_price, "symbol": symbol}
            else:
                # Cancel unfilled limit order
                try:
                    self.api.cancel_order(order_id)
                except Exception:
                    # May already be canceled/filled
                    check = self.api.get_order(order_id)
                    if check.status == "filled":
                        return {"method": "limit", "fill_price": float(check.filled_avg_price), "symbol": symbol}

                # Fallback to market bracket
                log.info(f"LIMIT TIMEOUT: {symbol} — falling back to market bracket")
                if side == "sell":
                    self.submit_short_bracket(symbol, qty, take_profit, stop_loss)
                else:
                    self.submit_bracket_order(symbol, qty, side, take_profit, stop_loss)
                return {"method": "market_fallback", "symbol": symbol}

        except Exception as e:
            log.error(f"Smart order failed for {symbol}: {e} — using market")
            if side == "sell":
                self.submit_short_bracket(symbol, qty, take_profit, stop_loss)
            else:
                self.submit_bracket_order(symbol, qty, side, take_profit, stop_loss)
            return {"method": "market_error", "symbol": symbol}

    def submit_market_order(self, symbol: str, qty: int, side: str):
        log.info(f"MARKET ORDER: {side} {qty} {symbol}")
        tif = "gtc" if symbol in CRYPTO_SYMBOLS else "day"
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force=tif
        )

    def submit_trailing_stop(self, symbol: str, qty: int, trail_percent: float):
        log.info(f"TRAILING STOP: sell {qty} {symbol} trail={trail_percent}%")
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="trailing_stop",
            trail_percent=str(round(trail_percent, 2)),
            time_in_force="gtc"
        )

    def cancel_all_orders(self):
        log.warning("Cancelling all open orders")
        self.api.cancel_all_orders()

    def close_position(self, symbol: str):
        log.info(f"Closing position: {symbol}")
        self.api.close_position(symbol)

    def close_all_positions(self):
        log.warning("Closing all positions")
        self.api.close_all_positions()

    def is_market_open(self) -> bool:
        clock = self.api.get_clock()
        return clock.is_open

    def get_clock(self):
        return self.api.get_clock()
