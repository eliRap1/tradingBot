"""RoutingBroker — dispatches broker calls by asset type.

Routing table (config-driven):
  futures → IBBroker
  stocks  → IBBroker
  crypto  → AlpacaBroker  (IB paper only supports BTC/ETH; Alpaca covers full universe)

The coordinator holds exactly one broker reference: RoutingBroker.
No broker-specific code outside broker files.
"""
from typing import Optional
from utils import setup_logger
from base_broker import (
    BaseBroker, Position, Order, OrderRequest, Quote, Clock
)

log = setup_logger("routing_broker")


class RoutingBroker(BaseBroker):
    def __init__(self, ib_broker: BaseBroker, alpaca_broker: BaseBroker,
                 classifier):
        """
        Args:
            ib_broker: IBBroker instance (handles stocks + futures)
            alpaca_broker: AlpacaBroker instance (handles crypto)
            classifier: InstrumentClassifier instance
        """
        self._ib = ib_broker
        self._alpaca = alpaca_broker
        self._clf = classifier

    def _broker_for(self, symbol: str) -> BaseBroker:
        asset = self._clf.classify(symbol)
        if asset == "crypto":
            return self._alpaca
        return self._ib  # stocks and futures go to IB

    # ── Account / portfolio queries go to IB (primary account) ───

    def get_account(self) -> dict:
        return self._ib.get_account()

    def get_equity(self) -> float:
        return self._ib.get_equity()

    def get_cash(self) -> float:
        return self._ib.get_cash()

    def get_buying_power(self) -> float:
        return self._ib.get_buying_power()

    def get_positions(self) -> list[Position]:
        """Aggregate positions from both brokers."""
        positions = self._ib.get_positions()
        try:
            positions += self._alpaca.get_positions()
        except Exception as e:
            log.warning(f"Could not fetch Alpaca positions: {e}")
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._broker_for(symbol).get_position(symbol)

    def get_open_orders(self) -> list[Order]:
        orders = self._ib.get_open_orders()
        try:
            orders += self._alpaca.get_open_orders()
        except Exception as e:
            log.warning(f"Could not fetch Alpaca orders: {e}")
        return orders

    def submit_order(self, req: OrderRequest) -> Order:
        broker = self._broker_for(req.symbol)
        log.info(
            f"Routing {req.symbol} ({self._clf.classify(req.symbol)}) "
            f"→ {broker.__class__.__name__}"
        )
        return broker.submit_order(req)

    def cancel_order(self, order_id: str):
        # Try IB first, then Alpaca
        try:
            self._ib.cancel_order(order_id)
        except Exception:
            self._alpaca.cancel_order(order_id)

    def cancel_all_orders(self):
        self._ib.cancel_all_orders()
        self._alpaca.cancel_all_orders()

    def close_position(self, symbol: str):
        self._broker_for(symbol).close_position(symbol)

    def close_all_positions(self):
        self._ib.close_all_positions()
        self._alpaca.close_all_positions()

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._broker_for(symbol).get_quote(symbol)

    def is_market_open(self) -> bool:
        return self._ib.is_market_open()

    def get_clock(self) -> Clock:
        return self._ib.get_clock()

    def asset_type(self, symbol: str) -> str:
        return self._clf.classify(symbol)

    # ── Alpaca-specific pass-throughs (for coordinator's crypto OCO logic) ─

    def check_crypto_exit_fills(self):
        return self._alpaca.check_crypto_exit_fills()

    def cancel_crypto_exit_orders(self, symbol: str):
        self._alpaca.cancel_crypto_exit_orders(symbol)

    def submit_bracket_order(self, symbol: str, qty: int, side: str,
                              take_profit: float, stop_loss: float):
        return self._broker_for(symbol).submit_bracket_order(
            symbol, qty, side, take_profit, stop_loss
        )

    def submit_short_bracket(self, symbol: str, qty: int,
                              take_profit: float, stop_loss: float):
        return self._broker_for(symbol).submit_short_bracket(
            symbol, qty, take_profit, stop_loss
        )

    def submit_crypto_order(self, symbol: str, qty: float, side: str,
                             take_profit: float, stop_loss: float):
        return self._alpaca.submit_crypto_order(symbol, qty, side, take_profit, stop_loss)

    def submit_smart_order(self, symbol: str, qty: int, side: str,
                           take_profit: float, stop_loss: float,
                           limit_offset_pct: float = 0.0002,
                           timeout_sec: int = 30) -> dict:
        return self._broker_for(symbol).submit_smart_order(
            symbol, qty, side, take_profit, stop_loss, limit_offset_pct, timeout_sec
        )

    def submit_market_order(self, symbol: str, qty: int, side: str):
        return self._broker_for(symbol).submit_market_order(symbol, qty, side)

    def submit_trailing_stop(self, symbol: str, qty: int, trail_percent: float):
        return self._broker_for(symbol).submit_trailing_stop(symbol, qty, trail_percent)
