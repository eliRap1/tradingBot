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
        # ── Fintech / growth (NASDAQ) ──────────────────────────────────
        "HOOD": "NASDAQ",
        "RIVN": "NASDAQ",
        "LCID": "NASDAQ",
        "SOFI": "NASDAQ",
        "DKNG": "NASDAQ",
        "OPEN": "NASDAQ",
        "COIN": "NASDAQ",
        "AFRM": "NASDAQ",
        "GTLB": "NASDAQ",
        "MNST": "NASDAQ",
        "ON":   "NASDAQ",   # ON Semiconductor (reserved word risk)
        # ── Symbols that need explicit NYSE routing ────────────────────
        "UWMC": "NYSE",
        "BILL": "NYSE",
        "PATH": "NYSE",
        # ── NYSE large-caps that fail SMART qualification ──────────────
        "OXY":  "NYSE",
        "MO":   "NYSE",
        "SNOW": "NYSE",
        "NKE":  "NYSE",
        "MPC":  "NYSE",
        "NET":  "NYSE",
        "HPQ":  "NYSE",
        "NOW":  "NYSE",
        "COF":  "NYSE",
        "SO":   "NYSE",
        "LOW":  "NYSE",
        "CMG":  "NYSE",
        "PM":   "NYSE",
        "V":    "NYSE",
        "ZS":   "NASDAQ",
        "OKTA": "NASDAQ",
        # ── NASDAQ large-caps that fail SMART qualification ────────────
        "MELI": "NASDAQ",
        "CSX":  "NASDAQ",
        "PEP":  "NASDAQ",
        "ZM":   "NASDAQ",
        "AMAT": "NASDAQ",
        "IDXX": "NASDAQ",
        # ── ETFs used for regime/breadth checks ───────────────────────
        "SPY":  "ARCA",
        "QQQ":  "NASDAQ",
        "IWM":  "ARCA",
        "DIA":  "ARCA",
        "TLT":  "NASDAQ",
        "GLD":  "ARCA",
        "SLV":  "ARCA",
        "UUP":  "ARCA",
        "XLK":  "ARCA",
        "XLF":  "ARCA",
        "XLV":  "ARCA",
        "XLE":  "ARCA",
        "XLY":  "ARCA",
        "XLP":  "ARCA",
        "XLI":  "ARCA",
        "XLRE": "ARCA",
        "XLU":  "ARCA",
        "RSP":  "ARCA",
    }

    def __init__(self, config: dict):
        ib_cfg = config.get("ib", {})
        self._host = ib_cfg.get("host", "127.0.0.1")
        self._port = ib_cfg.get("port", 4002)
        self._client_id = ib_cfg.get("client_id", 1)
        self._timeout = ib_cfg.get("timeout_sec", 10)
        self._config = config
        self._lock = threading.Lock()
        self._last_position_refresh = 0.0
        self._positions_subscribed = False
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
                self._ib.reqMarketDataType(3)  # delayed-frozen fallback
                self._positions_subscribed = False
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

    @staticmethod
    def _format_order_qty(asset: str, qty: float):
        """IB rejects scientific notation for tiny crypto quantities."""
        if asset == "crypto":
            return format(float(qty), ".8f")
        return qty

    @staticmethod
    def _contract_symbol(contract) -> str:
        sec_type = str(getattr(contract, "secType", "") or "").upper()
        symbol = str(getattr(contract, "symbol", "") or "")
        currency = str(getattr(contract, "currency", "") or "")
        if sec_type == "CRYPTO" and symbol and currency:
            return f"{symbol}/{currency}"
        return (
            getattr(contract, "localSymbol", None)
            or symbol
            or ""
        )

    @staticmethod
    def _position_key(symbol: str) -> str:
        return str(symbol or "").replace("/", "").replace(".", "").replace(" ", "").upper()

    def _refresh_position_cache(self, wait_sec: float = 0.25):
        """Ask IB for a fresh position snapshot without depending on ib.sleep()."""
        now = time.time()
        if now - getattr(self, "_last_position_refresh", 0.0) < 2.0:
            return
        self._last_position_refresh = now

        requested = False
        try:
            client = getattr(self._ib, "client", None)
            req_positions = getattr(client, "reqPositions", None)
            if callable(req_positions):
                req_positions()
                requested = True
                self._positions_subscribed = True
        except Exception as e:
            log.debug(f"IB client.reqPositions() refresh failed: {e}")

        if not requested:
            try:
                req_positions = getattr(self._ib, "reqPositions", None)
                if callable(req_positions):
                    req_positions()
                    requested = True
                    self._positions_subscribed = True
            except Exception as e:
                log.debug(f"IB reqPositions() refresh failed: {e}")

        if not requested or wait_sec <= 0:
            return

        deadline = time.time() + wait_sec
        while time.time() < deadline:
            try:
                if list(self._ib.portfolio()) or list(self._ib.positions()):
                    break
            except Exception:
                break
            time.sleep(0.1)

    def _invalidate_position_cache(self):
        self._last_position_refresh = 0.0

    def _portfolio_item_to_position(self, item) -> Position:
        qty = float(item.position)
        return Position(
            symbol=self._contract_symbol(item.contract),
            qty=qty,
            avg_price=float(getattr(item, "averageCost", 0) or 0),
            market_value=float(getattr(item, "marketValue", 0) or 0),
            unrealized_pl=float(getattr(item, "unrealizedPNL", 0) or 0),
            side="long" if qty > 0 else "short",
        )

    def _account_position_to_position(self, item) -> Position:
        qty = float(item.position)
        avg_price = float(getattr(item, "avgCost", 0) or 0)
        return Position(
            symbol=self._contract_symbol(item.contract),
            qty=qty,
            avg_price=avg_price,
            market_value=avg_price * qty,
            unrealized_pl=0.0,
            side="long" if qty > 0 else "short",
        )

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
        self._refresh_position_cache()
        by_symbol: dict[str, Position] = {}

        for item in self._ib.portfolio():
            qty = float(item.position)
            if qty == 0:
                continue
            pos = self._portfolio_item_to_position(item)
            by_symbol[self._position_key(pos.symbol)] = pos

        try:
            account_positions = self._ib.positions()
        except Exception as e:
            log.debug(f"IB positions() fallback failed: {e}")
            account_positions = []

        for item in account_positions:
            qty = float(item.position)
            if qty == 0:
                continue
            pos = self._account_position_to_position(item)
            by_symbol.setdefault(self._position_key(pos.symbol), pos)

        return list(by_symbol.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        target = self._position_key(symbol)
        for pos in self.get_positions():
            if self._position_key(pos.symbol) == target:
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
        order_qty = self._format_order_qty(asset, req.qty)

        ib_side = "BUY" if req.side == "buy" else "SELL"

        def _apply_tif(order) -> None:
            if asset != "crypto":
                order.tif = (req.time_in_force or "day").upper()

        # IB PAXOS crypto: BUY requires cashQty + GTC; SELL uses totalQuantity + GTC
        def _make_market(side: str, qty) -> "MarketOrder":
            o = MarketOrder(side, 0 if (asset == "crypto" and side == "BUY") else qty)
            if asset == "crypto":
                o.tif = "IOC"
                if side == "BUY":
                    o.cashQty = float(req.notional or (float(qty) * (self.get_quote(req.symbol).mid or 1)))
            else:
                _apply_tif(o)
            return o

        if req.take_profit and req.stop_loss:
            tp_side = "SELL" if ib_side == "BUY" else "BUY"

            # Place parent first (transmit=False); IB assigns orderId via nextOrderId
            parent_order = _make_market(ib_side, order_qty)
            parent_order.transmit = False
            parent_trade = self._ib.placeOrder(contract, parent_order)
            parent_id = parent_trade.order.orderId

            # Children reference parent by its IB-assigned orderId
            tp_order = LimitOrder(tp_side, order_qty, req.take_profit)
            tp_order.parentId = parent_id
            tp_order.transmit = False
            _apply_tif(tp_order)
            self._ib.placeOrder(contract, tp_order)

            sl_order = StopOrder(tp_side, order_qty, req.stop_loss)
            sl_order.parentId = parent_id
            sl_order.transmit = True  # releases all three atomically
            _apply_tif(sl_order)
            self._ib.placeOrder(contract, sl_order)

            log.info(
                f"IB BRACKET: {ib_side} {req.qty} {req.symbol} "
                f"TP={req.take_profit} SL={req.stop_loss} "
                f"parentId={parent_order.orderId}"
            )
            self._invalidate_position_cache()
            return Order(
                id=str(parent_trade.order.orderId),
                symbol=req.symbol,
                qty=req.qty,
                side=req.side,
                order_type="bracket",
                status="submitted",
            )
        else:
            order = _make_market(ib_side, order_qty)
            trade = self._ib.placeOrder(contract, order)
            log.info(f"IB MARKET: {ib_side} {req.qty} {req.symbol}")
            self._invalidate_position_cache()
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
            # Poll instead of self._ib.sleep() — ib.sleep() calls
            # loop.run_until_complete() which raises "event loop already running"
            # when invoked from Discord's async context or any other running loop.
            # ib_insync's background thread keeps updating the ticker regardless.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if (ticker.bid and ticker.bid > 0) or (ticker.last and ticker.last > 0):
                    break
                time.sleep(0.1)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
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

    def submit_market_order(self, symbol: str, qty: int, side: str, notional: float = None):
        req = OrderRequest(symbol=symbol, qty=qty, side=side, notional=notional)
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

            # 0th attempt: hard-coded override with SMART routing + primary exchange hint.
            # This avoids direct-routing rejections from IB precautionary settings.
            if symbol in self._SYMBOL_EXCHANGE:
                exch = self._SYMBOL_EXCHANGE[symbol]
                try:
                    c = Stock(symbol, "SMART", "USD")
                    c.primaryExch = exch
                    qualified = self._ib.qualifyContracts(c)
                    if qualified:
                        log.info(f"Resolved {symbol} via override primaryExch={exch}")
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
            return Crypto(base, "PAXOS", "USD")

        return None

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IB Gateway")
