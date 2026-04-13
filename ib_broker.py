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
    # Exchanges to try (in order) when SMART qualification fails
    _STOCK_EXCHANGE_HINTS = ("NYSE", "NASDAQ", "ARCA", "BATS", "IEX")

    # Hard-coded primary exchange for symbols that IB won't qualify via SMART.
    # Add any symbol that keeps logging "cannot qualify" here.
    _SYMBOL_EXCHANGE: dict[str, str] = {
        "HOOD": "NASDAQ",
        "RIVN": "NASDAQ",
        "LCID": "NASDAQ",
        "SOFI": "NASDAQ",
        "UWMC": "NYSE",
        "DKNG": "NASDAQ",
        "OPEN": "NASDAQ",
        "COIN": "NASDAQ",
        "AFRM": "NASDAQ",
        "BILL": "NYSE",
        "PATH": "NYSE",
        "GTLB": "NASDAQ",
        "MNST": "NASDAQ",
        "ON":   "NASDAQ",   # ON Semiconductor (reserved word risk)
        # ETFs used for regime/breadth checks — ARCA not in SMART hints by default
        "SPY":  "ARCA",
        "QQQ":  "NASDAQ",
        "IWM":  "ARCA",
        "DIA":  "ARCA",
        "TLT":  "NASDAQ",
        "GLD":  "ARCA",
        "SLV":  "ARCA",
    }

    def __init__(self, config: dict):
        ib_cfg = config.get("ib", {})
        self._host = ib_cfg.get("host", "127.0.0.1")
        self._port = ib_cfg.get("port", 4002)
        self._client_id = ib_cfg.get("client_id", 1)
        self._timeout = ib_cfg.get("timeout_sec", 10)
        self._config = config
        self._lock = threading.Lock()
        # Symbols that failed all contract qualification attempts.
        # TTL dict: symbol → retry-after timestamp (30-min cooldown for transient failures).
        self._bad_contracts: dict[str, float] = {}

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
        # accountValues() reads ib_insync's internal subscription cache —
        # no async request, safe to call from any thread (e.g. Discord bot).
        values = self._ib.accountValues()
        return {item.tag: item.value for item in values}

    def get_equity(self) -> float:
        acct = self.get_account()
        return float(acct.get("NetLiquidation", 0))

    def get_cash(self) -> float:
        acct = self.get_account()
        # IB emits "TotalCashValue" (not "CashBalance") for the account-level USD cash total
        return float(acct.get("TotalCashValue", acct.get("CashBalance", 0)))

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
            # Use ib.bracketOrder() so IB assigns real orderIds before we set parentId.
            # Manual construction (parent.orderId before placeOrder) gives orderId=0,
            # which orphans the TP/SL legs — this is the correct approach.
            tp_side = "SELL" if ib_side == "BUY" else "BUY"
            bracket = self._ib.bracketOrder(
                ib_side,
                req.qty,
                limitPrice=req.take_profit,   # TP limit leg
                stopLossPrice=req.stop_loss,  # SL stop leg
            )
            # bracketOrder returns (parent, takeProfit, stopLoss) tuple
            parent_order, tp_order, sl_order = bracket

            parent_trade = self._ib.placeOrder(contract, parent_order)
            self._ib.placeOrder(contract, tp_order)
            self._ib.placeOrder(contract, sl_order)

            log.info(
                f"IB BRACKET: {ib_side} {req.qty} {req.symbol} "
                f"TP={req.take_profit} SL={req.stop_loss} "
                f"parentId={parent_order.orderId}"
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

    # NYSE observed holidays — extend each year as needed
    _NYSE_HOLIDAYS = frozenset({
        # 2025
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
        "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
        "2025-11-27", "2025-12-25",
        # 2026
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
        "2026-11-26", "2026-12-25",
        # 2027
        "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
        "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
        "2027-11-25", "2027-12-24",
    })

    def get_clock(self) -> Clock:
        """IB doesn't have a direct clock API — derive from wall time (NYSE hours).

        Accounts for weekends and NYSE observed holidays.
        """
        from datetime import datetime, timedelta, date
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        now = datetime.now(ET)
        today = now.date()

        def _is_trading_day(d: date) -> bool:
            if d.weekday() >= 5:  # Saturday=5, Sunday=6
                return False
            if d.isoformat() in self._NYSE_HOLIDAYS:
                return False
            return True

        weekday = now.weekday()
        hour, minute = now.hour, now.minute
        is_open = (
            _is_trading_day(today)
            and (hour > 9 or (hour == 9 and minute >= 30))
            and hour < 16
        )

        # next_close: today 16:00 ET if open, else None
        if is_open:
            next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            next_close = None

        # next_open: next trading day 09:30 ET
        candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if not _is_trading_day(today) or now >= candidate:
            days_ahead = 1
            while True:
                next_day = (now + timedelta(days=days_ahead)).date()
                if _is_trading_day(next_day):
                    next_dt = datetime(next_day.year, next_day.month, next_day.day,
                                       9, 30, 0, tzinfo=ET)
                    candidate = next_dt
                    break
                days_ahead += 1
        next_open = candidate

        return Clock(is_open=is_open, next_open=next_open, next_close=next_close)

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
        order = self.submit_order(req)
        return {"method": "ib_bracket", "symbol": symbol, "order_id": order.id if order else None}

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
        """Resolve IB contract object for a symbol.

        For stocks: tries SMART first, then explicit primaryExch hints.
        Failed symbols get a 30-min cooldown (TTL) so transient IB blips recover.
        """
        retry_after = self._bad_contracts.get(symbol, 0.0)
        if time.time() < retry_after:
            return None

        if asset == "futures":
            return self._contracts.get_contract(symbol)

        if asset == "stock":
            from ib_insync import Stock

            # 0th attempt: hard-coded override (bypasses SMART entirely)
            if symbol in self._SYMBOL_EXCHANGE:
                exch = self._SYMBOL_EXCHANGE[symbol]
                try:
                    c = Stock(symbol, exch, "USD")
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} via override exchange={exch}")
                        return qualified[0]
                except Exception:
                    pass

            # 1st attempt: plain SMART routing
            contract = Stock(symbol, "SMART", "USD")
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
            except Exception:
                pass

            # 2nd attempt: primaryExch hints (SMART + hint)
            for exch in self._STOCK_EXCHANGE_HINTS:
                try:
                    c = Stock(symbol, "SMART", "USD")
                    c.primaryExch = exch
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} with primaryExch={exch}")
                        return qualified[0]
                except Exception:
                    pass

            # 3rd attempt: direct exchange routing (no SMART)
            for exch in self._STOCK_EXCHANGE_HINTS:
                try:
                    c = Stock(symbol, exch, "USD")
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} via direct exchange={exch}")
                        return qualified[0]
                except Exception:
                    pass

            log.warning(
                f"IB: cannot qualify contract for {symbol} "
                f"(tried override + SMART + primaryExch + direct) — skipping for 30 min"
            )
            self._bad_contracts[symbol] = time.time() + 1800  # 30-min cooldown
            return None

        if asset == "crypto":
            from ib_insync import Crypto
            # IB paper only supports BTC and ETH
            base = symbol.split("/")[0] if "/" in symbol else symbol[:-3]
            contract = Crypto(base, "PAXOS", "USD")
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
            except Exception:
                pass
            return contract

        return None

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IB Gateway")
