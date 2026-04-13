"""Interactive Brokers broker implementation of BaseBroker.

Connects to IB Gateway at 127.0.0.1:4002 (paper account).
Uses ib_insync for all IB communication.
"""
import time
import threading
from typing import Optional
from utils import setup_logger
from base_broker import (
    BaseBroker, BrokerConnectionError,
    Position, Order, OrderRequest, Quote, Clock
)

log = setup_logger("ib_broker")

RECONNECT_DELAYS = [30, 60, 120]  # exponential backoff on connection loss


class IBBroker(BaseBroker):
    def __init__(self, config: dict):
        ib_cfg = config.get("ib", {})
        self._host = ib_cfg.get("host", "127.0.0.1")
        self._port = ib_cfg.get("port", 4002)
        self._client_id = ib_cfg.get("client_id", 1)
        self._timeout = ib_cfg.get("timeout_sec", 10)
        self._config = config
        self._lock = threading.Lock()

        from ib_insync import IB
        self._ib = IB()
        self._connect()

        from contract_manager import ContractManager
        self._contracts = ContractManager(self._ib, config)

    def _connect(self):
        """Connect to IB Gateway with exponential backoff retry."""
        for attempt, delay in enumerate([0] + RECONNECT_DELAYS):
            if delay:
                log.warning(f"IB reconnect attempt {attempt}, waiting {delay}s...")
                time.sleep(delay)
            try:
                if not self._ib.isConnected():
                    self._ib.connect(
                        self._host, self._port,
                        clientId=self._client_id,
                        timeout=self._timeout
                    )
                log.info(f"Connected to IB Gateway at {self._host}:{self._port}")
                return
            except Exception as e:
                log.error(f"IB connection failed (attempt {attempt + 1}): {e}")
        raise BrokerConnectionError(
            f"Cannot connect to IB Gateway at {self._host}:{self._port} "
            f"after {len(RECONNECT_DELAYS) + 1} attempts"
        )

    def _ensure_connected(self):
        """Reconnect if IB Gateway connection was lost."""
        if not self._ib.isConnected():
            log.warning("IB disconnected — attempting reconnect")
            self._connect()

    # ── BaseBroker interface ──────────────────────────────────────

    def get_account(self) -> dict:
        self._ensure_connected()
        summary = self._ib.accountSummary()
        return {item.tag: item.value for item in summary}

    def get_equity(self) -> float:
        acct = self.get_account()
        return float(acct.get("NetLiquidation", 0))

    def get_cash(self) -> float:
        acct = self.get_account()
        return float(acct.get("CashBalance", 0))

    def get_buying_power(self) -> float:
        acct = self.get_account()
        return float(acct.get("BuyingPower", 0))

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        result = []
        for item in self._ib.portfolio():
            qty = float(item.position)
            if qty == 0:
                continue
            result.append(Position(
                symbol=item.contract.localSymbol or item.contract.symbol,
                qty=qty,
                avg_price=float(item.averageCost),
                market_value=float(item.marketValue),
                unrealized_pl=float(item.unrealizedPNL),
                side="long" if qty > 0 else "short",
            ))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def get_open_orders(self) -> list[Order]:
        self._ensure_connected()
        trades = self._ib.openTrades()
        result = []
        for trade in trades:
            result.append(Order(
                id=str(trade.order.orderId),
                symbol=trade.contract.localSymbol or trade.contract.symbol,
                qty=float(trade.order.totalQuantity),
                side=trade.order.action.lower(),
                order_type=trade.order.orderType.lower(),
                status=trade.orderStatus.status.lower(),
                filled_qty=float(trade.orderStatus.filled),
                filled_avg_price=(
                    float(trade.orderStatus.avgFillPrice)
                    if trade.orderStatus.avgFillPrice else None
                ),
            ))
        return result

    def submit_order(self, req: OrderRequest) -> Order:
        """Submit bracket order to IB. Resolves futures vs stock/crypto contracts."""
        self._ensure_connected()
        from ib_insync import MarketOrder, LimitOrder, StopOrder

        asset = self.asset_type(req.symbol)
        contract = self._resolve_contract(req.symbol, asset)
        if contract is None:
            raise ValueError(f"Cannot resolve IB contract for {req.symbol}")

        ib_side = "BUY" if req.side == "buy" else "SELL"

        if req.take_profit and req.stop_loss:
            # IB native bracket: parent market order + TP limit + SL stop
            parent = MarketOrder(ib_side, req.qty)
            parent.transmit = False

            tp_side = "SELL" if ib_side == "BUY" else "BUY"
            tp_order = LimitOrder(tp_side, req.qty, req.take_profit)
            tp_order.parentId = parent.orderId
            tp_order.transmit = False

            sl_order = StopOrder(tp_side, req.qty, req.stop_loss)
            sl_order.parentId = parent.orderId
            sl_order.transmit = True  # transmits the whole bracket at once

            parent_trade = self._ib.placeOrder(contract, parent)
            self._ib.placeOrder(contract, tp_order)
            self._ib.placeOrder(contract, sl_order)

            log.info(
                f"IB BRACKET: {ib_side} {req.qty} {req.symbol} "
                f"TP={req.take_profit} SL={req.stop_loss}"
            )
            return Order(
                id=str(parent_trade.order.orderId),
                symbol=req.symbol,
                qty=req.qty,
                side=req.side,
                order_type="bracket",
                status="submitted",
            )
        else:
            order = MarketOrder(ib_side, req.qty)
            trade = self._ib.placeOrder(contract, order)
            log.info(f"IB MARKET: {ib_side} {req.qty} {req.symbol}")
            return Order(
                id=str(trade.order.orderId),
                symbol=req.symbol,
                qty=req.qty,
                side=req.side,
                order_type="market",
                status="submitted",
            )

    def cancel_order(self, order_id: str):
        self._ensure_connected()
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == str(order_id):
                self._ib.cancelOrder(trade.order)
                return

    def cancel_all_orders(self):
        self._ensure_connected()
        self._ib.reqGlobalCancel()

    def close_position(self, symbol: str):
        pos = self.get_position(symbol)
        if not pos:
            log.warning(f"No IB position to close: {symbol}")
            return
        side = "sell" if pos.qty > 0 else "buy"
        req = OrderRequest(symbol=symbol, qty=abs(pos.qty), side=side)
        self.submit_order(req)

    def close_all_positions(self):
        for pos in self.get_positions():
            self.close_position(pos.symbol)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        self._ensure_connected()
        asset = self.asset_type(symbol)
        contract = self._resolve_contract(symbol, asset)
        if not contract:
            return None
        try:
            ticker = self._ib.reqMktData(contract, "", True, False)
            self._ib.sleep(1)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
            self._ib.cancelMktData(contract)
            if bid > 0 and ask > 0:
                return Quote(symbol=symbol, bid=bid, ask=ask)
            # Fallback: use last price with estimated spread
            last = float(ticker.last) if ticker.last else 0.0
            if last > 0:
                spread = last * 0.0001  # 1 bps estimated spread
                return Quote(symbol=symbol, bid=last - spread, ask=last + spread)
        except Exception as e:
            log.error(f"IB quote failed for {symbol}: {e}")
        return None

    def is_market_open(self) -> bool:
        return self.get_clock().is_open

    def get_clock(self) -> Clock:
        """IB doesn't have a direct clock API — derive from wall time."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        now = datetime.now(ET)
        weekday = now.weekday()
        hour, minute = now.hour, now.minute
        is_open = (
            weekday < 5
            and (hour > 9 or (hour == 9 and minute >= 30))
            and hour < 16
        )
        return Clock(is_open=is_open, next_open=None, next_close=None)

    def asset_type(self, symbol: str) -> str:
        from instrument_classifier import InstrumentClassifier
        return InstrumentClassifier(self._config).classify(symbol)

    # ── Alpaca-compatible pass-throughs (used by coordinator's bracket logic) ─

    def submit_bracket_order(self, symbol: str, qty: int, side: str,
                              take_profit: float, stop_loss: float):
        req = OrderRequest(
            symbol=symbol, qty=qty, side=side,
            take_profit=take_profit, stop_loss=stop_loss
        )
        return self.submit_order(req)

    def submit_short_bracket(self, symbol: str, qty: int,
                              take_profit: float, stop_loss: float):
        req = OrderRequest(
            symbol=symbol, qty=qty, side="sell",
            take_profit=take_profit, stop_loss=stop_loss
        )
        return self.submit_order(req)

    def submit_market_order(self, symbol: str, qty: int, side: str):
        req = OrderRequest(symbol=symbol, qty=qty, side=side)
        return self.submit_order(req)

    def submit_smart_order(self, symbol: str, qty: int, side: str,
                           take_profit: float, stop_loss: float,
                           limit_offset_pct: float = 0.0002,
                           timeout_sec: int = 30) -> dict:
        """IB uses native bracket orders — smart order just submits bracket."""
        req = OrderRequest(
            symbol=symbol, qty=qty, side=side,
            take_profit=take_profit, stop_loss=stop_loss
        )
        self.submit_order(req)
        return {"method": "ib_bracket", "symbol": symbol}

    def submit_trailing_stop(self, symbol: str, qty: int, trail_percent: float):
        """Submit IB trailing stop order."""
        self._ensure_connected()
        from ib_insync import Order as IBOrder
        asset = self.asset_type(symbol)
        contract = self._resolve_contract(symbol, asset)
        if not contract:
            log.error(f"Cannot resolve contract for trailing stop: {symbol}")
            return None
        order = IBOrder()
        order.action = "SELL"
        order.orderType = "TRAIL"
        order.totalQuantity = qty
        order.trailingPercent = trail_percent
        trade = self._ib.placeOrder(contract, order)
        log.info(f"IB TRAILING STOP: sell {qty} {symbol} trail={trail_percent}%")
        return trade

    # ── Internal helpers ──────────────────────────────────────────

    def _resolve_contract(self, symbol: str, asset: str):
        """Resolve IB contract object for a symbol."""
        if asset == "futures":
            return self._contracts.get_contract(symbol)
        elif asset == "stock":
            from ib_insync import Stock
            contract = Stock(symbol, "SMART", "USD")
            try:
                self._ib.qualifyContracts(contract)
            except Exception:
                pass
            return contract
        elif asset == "crypto":
            from ib_insync import Crypto
            # IB paper only supports BTC and ETH
            base = symbol.split("/")[0] if "/" in symbol else symbol[:-3]
            return Crypto(base, "PAXOS", "USD")
        return None

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IB Gateway")
