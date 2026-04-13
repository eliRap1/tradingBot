import os
import time
import threading
import uuid
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from utils import setup_logger

load_dotenv()
log = setup_logger("broker")

CRYPTO_SYMBOLS = {
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD",
    "BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD", "DOGEUSD",
}


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
        
        # Track crypto exit orders for proper OCO behavior
        # {symbol: {"tp_order_id": str, "sl_order_id": str, "qty": float}}
        self._crypto_exit_orders = {}
        self._crypto_lock = threading.Lock()

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
        """Submit crypto order with linked TP and SL orders.

        Crypto doesn't support bracket orders on Alpaca, so we:
        1. Place entry order and wait for fill confirmation
        2. Place TP limit and SL stop orders with the actual filled qty
        3. Track both exit orders so we can cancel the other when one fills

        CRITICAL: The exit orders are tracked in _crypto_exit_orders so that
        when one fills, we can cancel the other (manual OCO behavior).
        """
        # Hard-reject crypto shorts — Alpaca does not support crypto short selling
        if side == "sell":
            log.error(f"CRYPTO SHORT REJECTED: {symbol} — Alpaca does not support crypto short selling")
            return None

        log.info(f"CRYPTO ORDER: {side} {qty} {symbol} | TP={take_profit:.2f} SL={stop_loss:.2f}")

        # Round crypto prices (BTC to 2 decimals, ETH to 2)
        tp_price = round(take_profit, 2)
        sl_price = round(stop_loss, 2)
        
        # Use client_order_id for idempotency
        client_id = f"crypto_{symbol.replace('/', '')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Entry order
        entry = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
            client_order_id=f"{client_id}_entry"
        )
        
        # Wait for entry fill (up to 30 seconds) to get actual filled qty
        filled_qty = qty
        for _ in range(15):
            time.sleep(2)
            try:
                order_status = self.api.get_order(entry.id)
                if order_status.status == "filled":
                    filled_qty = float(order_status.filled_qty)
                    log.info(f"CRYPTO ENTRY FILLED: {symbol} qty={filled_qty}")
                    break
                elif order_status.status in ("canceled", "expired", "rejected"):
                    log.error(f"CRYPTO ENTRY FAILED: {symbol} status={order_status.status}")
                    return entry
            except Exception as e:
                log.warning(f"Error checking entry status: {e}")
        
        # Reduce exit qty by 0.1% to account for Alpaca fee/rounding edge cases
        # This prevents "insufficient balance" errors when placing exit orders
        exit_qty = round(filled_qty * 0.999, 8)
        log.info(f"CRYPTO EXIT QTY: {symbol} filled={filled_qty} exit_qty={exit_qty} (0.1% buffer)")

        # Exit side (opposite of entry)
        exit_side = "sell" if side == "buy" else "buy"

        tp_order_id = None
        sl_order_id = None

        # Take-profit limit order
        try:
            tp_order = self.api.submit_order(
                symbol=symbol,
                qty=exit_qty,
                side=exit_side,
                type="limit",
                limit_price=tp_price,
                time_in_force="gtc",
                client_order_id=f"{client_id}_tp"
            )
            tp_order_id = tp_order.id
            log.info(f"CRYPTO TP order placed: {exit_side} {exit_qty} {symbol} @ ${tp_price}")
        except Exception as e:
            log.error(f"Failed to place crypto TP order: {e}")

        # Stop-loss order — Alpaca crypto requires stop_limit (plain stop not supported)
        # Limit price is set 0.3% beyond the stop to ensure fill after trigger
        try:
            sl_limit = round(sl_price * (0.997 if exit_side == "sell" else 1.003), 2)
            sl_order = self.api.submit_order(
                symbol=symbol,
                qty=exit_qty,
                side=exit_side,
                type="stop_limit",
                stop_price=sl_price,
                limit_price=sl_limit,
                time_in_force="gtc",
                client_order_id=f"{client_id}_sl"
            )
            sl_order_id = sl_order.id
            log.info(f"CRYPTO SL order placed: {exit_side} {exit_qty} {symbol} @ ${sl_price}")
        except Exception as e:
            log.error(f"Failed to place crypto SL order: {e}")

        # Track exit orders for OCO behavior
        if tp_order_id or sl_order_id:
            with self._crypto_lock:
                self._crypto_exit_orders[symbol] = {
                    "tp_order_id": tp_order_id,
                    "sl_order_id": sl_order_id,
                    "qty": exit_qty,
                    "entry_side": side
                }

        return entry
    
    def check_crypto_exit_fills(self):
        """Check if any crypto TP/SL orders have filled and cancel the other.
        
        This should be called periodically (e.g., in coordinator cycle) to
        implement manual OCO behavior for crypto positions.
        """
        filled_exits = {}  # {symbol: reason} — returned to coordinator for trade recording
        with self._crypto_lock:
            symbols_to_remove = []

            for symbol, orders in self._crypto_exit_orders.items():
                tp_id = orders.get("tp_order_id")
                sl_id = orders.get("sl_order_id")
                
                tp_filled = False
                sl_filled = False
                
                # Check TP order status
                if tp_id:
                    try:
                        tp_status = self.api.get_order(tp_id)
                        if tp_status.status == "filled":
                            tp_filled = True
                            log.info(f"CRYPTO TP FILLED: {symbol}")
                        elif tp_status.status in ("canceled", "expired", "rejected"):
                            orders["tp_order_id"] = None
                    except Exception:
                        pass
                
                # Check SL order status
                if sl_id:
                    try:
                        sl_status = self.api.get_order(sl_id)
                        if sl_status.status == "filled":
                            sl_filled = True
                            log.info(f"CRYPTO SL FILLED: {symbol}")
                        elif sl_status.status in ("canceled", "expired", "rejected"):
                            orders["sl_order_id"] = None
                    except Exception:
                        pass
                
                # If one filled, cancel the other (OCO behavior)
                if tp_filled and sl_id:
                    try:
                        self.api.cancel_order(sl_id)
                        log.info(f"CRYPTO OCO: Cancelled SL for {symbol} (TP filled)")
                    except Exception as e:
                        log.warning(f"Failed to cancel SL after TP fill: {e}")
                    filled_exits[symbol] = "take_profit"
                    symbols_to_remove.append(symbol)

                elif sl_filled and tp_id:
                    try:
                        self.api.cancel_order(tp_id)
                        log.info(f"CRYPTO OCO: Cancelled TP for {symbol} (SL filled)")
                    except Exception as e:
                        log.warning(f"Failed to cancel TP after SL fill: {e}")
                    filled_exits[symbol] = "stop_loss"
                    symbols_to_remove.append(symbol)

                # Clean up if both orders are gone
                elif not orders.get("tp_order_id") and not orders.get("sl_order_id"):
                    symbols_to_remove.append(symbol)

            for symbol in symbols_to_remove:
                self._crypto_exit_orders.pop(symbol, None)

        return filled_exits  # {symbol: "take_profit"|"stop_loss"}
    
    def cancel_crypto_exit_orders(self, symbol: str):
        """Cancel any pending TP/SL orders for a crypto position."""
        with self._crypto_lock:
            orders = self._crypto_exit_orders.pop(symbol, {})
        
        if orders.get("tp_order_id"):
            try:
                self.api.cancel_order(orders["tp_order_id"])
                log.info(f"Cancelled crypto TP order for {symbol}")
            except Exception:
                pass
        
        if orders.get("sl_order_id"):
            try:
                self.api.cancel_order(orders["sl_order_id"])
                log.info(f"Cancelled crypto SL order for {symbol}")
            except Exception:
                pass

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
        
        Uses client_order_id for idempotency on retries.
        """
        # Generate idempotency key
        client_id = f"smart_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
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
            # Place limit order with idempotency key (no bracket yet)
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="limit", limit_price=limit_price,
                time_in_force="day",
                client_order_id=f"{client_id}_entry"
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
