import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from utils import setup_logger

load_dotenv()
log = setup_logger("broker")


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
        """Submit order with take-profit and stop-loss."""
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

    def submit_market_order(self, symbol: str, qty: int, side: str):
        log.info(f"MARKET ORDER: {side} {qty} {symbol}")
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
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
