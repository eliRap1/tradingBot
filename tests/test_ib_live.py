"""Live IB integration tests for connectivity, strategies, orders, and resilience."""

from __future__ import annotations

import concurrent.futures
import numbers
import time

import pytest

pytestmark = pytest.mark.live

ACCEPTED_ORDER_STATUSES = {
    "ApiPending",
    "PendingSubmit",
    "PreSubmitted",
    "Submitted",
    "Filled",
}
CANCELLED_ORDER_STATUSES = {"Cancelled", "Inactive", "ApiCancelled"}
TERMINAL_ORDER_STATUSES = ACCEPTED_ORDER_STATUSES | CANCELLED_ORDER_STATUSES


def _wait_for_value(getter, predicate=None, timeout: float = 10.0, interval: float = 0.25):
    predicate = predicate or (lambda value: value is not None)
    deadline = time.time() + timeout
    last_value = None
    while time.time() < deadline:
        last_value = getter()
        if predicate(last_value):
            return last_value
        time.sleep(interval)
    return last_value


def _find_trade(broker, order_id: str):
    for trade in broker._ib.trades():
        if str(trade.order.orderId) == str(order_id):
            return trade
    return None


def _trade_messages(trade) -> str:
    messages = []
    for entry in getattr(trade, "log", []):
        message = getattr(entry, "message", None)
        if message:
            messages.append(str(message))
    return " | ".join(messages)


def _positions_by_symbol(positions) -> dict[str, object]:
    return {position.symbol: position for position in positions}


def _assert_bar_frame(symbol: str, df) -> None:
    assert df is not None and not df.empty, f"{symbol}: expected non-empty bars"
    for column in ("open", "high", "low", "close", "volume"):
        assert column in df.columns, f"{symbol}: missing column {column}"
    for column in ("open", "high", "low", "close"):
        assert df[column].notna().all(), f"{symbol}: NaN values in {column}"


class TestPhase1Connectivity:
    def test_connection_health(self, ib_session):
        broker, data, config = ib_session
        assert broker._ib.isConnected()
        equity = broker.get_equity()
        assert equity > 0, f"Expected positive equity, got {equity}"

    def test_resolve_all_contracts(self, ib_session):
        from tests.conftest import ALL_SYMBOLS

        broker, data, config = ib_session

        resolved = {}
        failed = []
        for symbol in ALL_SYMBOLS:
            asset = broker.asset_type(symbol)
            contract = broker._resolve_contract(symbol, asset)
            if contract is None:
                failed.append(symbol)
                continue
            resolved[symbol] = {
                "con_id": getattr(contract, "conId", None),
                "exchange": getattr(contract, "exchange", None),
                "primary_exchange": getattr(
                    contract,
                    "primaryExchange",
                    getattr(contract, "primaryExch", None),
                ),
            }

        assert len(resolved) == len(ALL_SYMBOLS), f"Failed to resolve: {failed}"

    def test_get_quotes(self, ib_session):
        from tests.conftest import ALL_SYMBOLS

        broker, data, config = ib_session

        got_price = {}
        no_price = []
        for symbol in ALL_SYMBOLS:
            quote = broker.get_quote(symbol)
            if quote and quote.mid > 0:
                got_price[symbol] = quote.mid
                continue

            price = data.get_latest_price(symbol)
            if price and price > 0:
                got_price[symbol] = price
            else:
                no_price.append(symbol)

        assert len(got_price) >= 20, (
            f"Expected at least 20 symbols with prices, got {len(got_price)}. "
            f"No price for: {no_price}"
        )

    def test_bad_symbol(self, ib_session):
        broker, data, config = ib_session

        broker._bad_contracts.pop("FAKESYM123", None)
        data._bad_contracts.pop("FAKESYM123", None)

        contract = broker._resolve_contract("FAKESYM123", "stock")
        assert contract is None, "Expected None for fake symbol"
        assert "FAKESYM123" in broker._bad_contracts, "Expected fake symbol in bad-contract cache"
        assert broker._bad_contracts["FAKESYM123"] > time.time(), "Expected retry TTL in the future"

        broker._bad_contracts.pop("FAKESYM123", None)
        assert "FAKESYM123" not in broker._bad_contracts

    def test_pacing_rapid_resolution(self, ib_session):
        from tests.conftest import ALL_SYMBOLS

        broker, data, config = ib_session

        broker._bad_contracts.clear()

        results = {}
        for symbol in ALL_SYMBOLS:
            asset = broker.asset_type(symbol)
            contract = broker._resolve_contract(symbol, asset)
            results[symbol] = contract is not None

        failed = [symbol for symbol, ok in results.items() if not ok]
        assert not failed, f"Rapid contract resolution failed for: {failed}"


class TestPhase2Strategies:
    def test_fetch_bars_daily(self, ib_session):
        from tests.conftest import ALL_SYMBOLS

        broker, data, config = ib_session

        bars_dict = data.get_bars(ALL_SYMBOLS, timeframe="1Day", days=30)
        fetched = []
        failed = []
        for symbol in ALL_SYMBOLS:
            df = bars_dict.get(symbol)
            if df is None or df.empty:
                failed.append(symbol)
                continue
            _assert_bar_frame(symbol, df)
            fetched.append(symbol)

        assert len(fetched) >= 20, f"Expected 20+ symbols with daily bars, failed: {failed}"

    def test_fetch_bars_intraday(self, ib_session):
        from tests.conftest import SYMBOLS

        broker, data, config = ib_session

        test_symbols = SYMBOLS["stocks"][:5] + ["SPY"]
        fetched = []
        failed = []
        for symbol in test_symbols:
            df = data.get_intraday_bars(symbol, timeframe="5Min", days=2)
            if df is None or df.empty:
                failed.append(symbol)
                continue
            _assert_bar_frame(symbol, df)
            fetched.append(symbol)

        assert len(fetched) >= 4, f"Expected 4+ symbols with intraday bars, failed: {failed}"

    def test_strategy_signals(self, ib_session):
        from strategies import ALL_STRATEGIES
        from tests.conftest import ALL_SYMBOLS

        broker, data, config = ib_session

        bars_dict = data.get_bars(ALL_SYMBOLS, timeframe="1Day", days=60)
        assert len(bars_dict) >= 20, "Insufficient bar coverage to test strategies"

        errors = []
        signal_counts = {name: 0 for name in ALL_STRATEGIES}

        for strategy_name, strategy_cls in ALL_STRATEGIES.items():
            try:
                strategy = strategy_cls(config)
            except Exception as exc:
                errors.append(f"{strategy_name}: failed to instantiate: {exc}")
                continue

            try:
                signals = strategy.generate_signals(bars_dict)
            except Exception as exc:
                errors.append(f"{strategy_name}: generate_signals crashed: {exc}")
                continue

            assert isinstance(signals, dict), (
                f"{strategy_name}: expected dict, got {type(signals)}"
            )

            for symbol, score in signals.items():
                assert isinstance(score, numbers.Real), (
                    f"{strategy_name}/{symbol}: expected numeric score, got {type(score)}"
                )
                score_value = float(score)
                assert -1.0 <= score_value <= 1.0, (
                    f"{strategy_name}/{symbol}: score {score_value} outside [-1, 1]"
                )
                signal_counts[strategy_name] += 1

        assert not errors, "Strategy failures:\n" + "\n".join(errors)

    def test_strategy_router(self, ib_session):
        from tests.conftest import ALL_SYMBOLS
        from strategy_router import StrategyRouter

        broker, data, config = ib_session

        router = StrategyRouter(config)
        for symbol in ALL_SYMBOLS:
            asset = broker.asset_type(symbol)
            weights = router.get_strategies(asset)
            assert isinstance(weights, dict), f"{symbol}: weights is not a dict"
            assert weights, f"{symbol}: no strategies assigned"
            total = sum(weights.values())
            assert 0.99 <= total <= 1.01, (
                f"{symbol} ({asset}): weights sum to {total}, expected about 1.0"
            )
            for strategy_name, weight in weights.items():
                assert 0.0 < weight <= 1.0, (
                    f"{symbol}/{strategy_name}: weight {weight} outside (0, 1]"
                )


class TestPhase3Orders:
    def test_market_order_stock(self, ib_session):
        from base_broker import OrderRequest

        broker, data, config = ib_session
        market_open = broker.is_market_open()

        order = broker.submit_order(OrderRequest(symbol="AMD", qty=1, side="buy"))
        assert order is not None, "submit_order returned None"
        assert order.id is not None, "order has no id"

        trade = _wait_for_value(lambda: _find_trade(broker, order.id), timeout=5.0)
        assert trade is not None, f"Could not find IB trade for order {order.id}"

        status = _wait_for_value(
            lambda: trade.orderStatus.status,
            predicate=lambda value: value in TERMINAL_ORDER_STATUSES,
            timeout=10.0,
        )
        assert status in ACCEPTED_ORDER_STATUSES, (
            f"Unexpected stock order status: {status} ({_trade_messages(trade)})"
        )

        if not market_open:
            broker.cancel_order(order.id)
            pytest.skip("NYSE is closed; submission validated, fill assertion skipped.")

        position = _wait_for_value(
            lambda: _positions_by_symbol(broker.get_positions()).get("AMD"),
            predicate=lambda value: value is not None and value.qty > 0,
            timeout=10.0,
        )
        assert position is not None, "AMD position did not appear after market buy"

    def test_market_order_etf(self, ib_session):
        from base_broker import OrderRequest

        broker, data, config = ib_session
        market_open = broker.is_market_open()

        order = broker.submit_order(OrderRequest(symbol="SPY", qty=1, side="buy"))
        assert order is not None
        assert order.id is not None

        trade = _wait_for_value(lambda: _find_trade(broker, order.id), timeout=5.0)
        assert trade is not None, f"Could not find IB trade for order {order.id}"

        status = _wait_for_value(
            lambda: trade.orderStatus.status,
            predicate=lambda value: value in TERMINAL_ORDER_STATUSES,
            timeout=10.0,
        )
        assert status in ACCEPTED_ORDER_STATUSES, (
            f"Unexpected ETF order status: {status} ({_trade_messages(trade)})"
        )

        if not market_open:
            broker.cancel_order(order.id)
            pytest.skip("NYSE is closed; submission validated, fill assertion skipped.")

    def test_market_order_futures(self, ib_session):
        from base_broker import OrderRequest

        broker, data, config = ib_session

        order = broker.submit_order(OrderRequest(symbol="NQ", qty=1, side="buy"))
        assert order is not None
        assert order.id is not None

        trade = _wait_for_value(lambda: _find_trade(broker, order.id), timeout=5.0)
        assert trade is not None, f"Could not find IB trade for order {order.id}"

        status = _wait_for_value(
            lambda: trade.orderStatus.status,
            predicate=lambda value: value in TERMINAL_ORDER_STATUSES,
            timeout=10.0,
        )
        messages = _trade_messages(trade).lower()
        if status in CANCELLED_ORDER_STATUSES and (
            "margin" in messages or "insufficient" in messages
        ):
            pytest.skip(f"Futures margin insufficient: {_trade_messages(trade)}")

        assert status in ACCEPTED_ORDER_STATUSES, (
            f"Unexpected futures order status: {status} ({_trade_messages(trade)})"
        )

    def test_market_order_crypto(self, ib_session):
        from base_broker import OrderRequest

        broker, data, config = ib_session

        order = broker.submit_order(
            OrderRequest(symbol="BTC/USD", qty=0.00001, side="buy", notional=1.0)
        )
        assert order is not None
        assert order.id is not None

        trade = _wait_for_value(lambda: _find_trade(broker, order.id), timeout=5.0)
        assert trade is not None, f"Could not find IB trade for order {order.id}"

        status = _wait_for_value(
            lambda: trade.orderStatus.status,
            predicate=lambda value: value in TERMINAL_ORDER_STATUSES,
            timeout=10.0,
        )
        messages = _trade_messages(trade).lower()
        if status in CANCELLED_ORDER_STATUSES and (
            "regulatory" in messages or "not allowed" in messages or "permission" in messages
        ):
            pytest.skip(f"Crypto not enabled on this account: {_trade_messages(trade)}")

        assert getattr(trade.order, "cashQty", None) is not None, "Expected cashQty on BTC/USD buy"
        assert getattr(trade.order, "tif", None) == "IOC", "Expected IOC tif for IB crypto buy"
        assert status in ACCEPTED_ORDER_STATUSES, (
            f"Unexpected crypto order status: {status} ({_trade_messages(trade)})"
        )

    def test_bracket_order(self, ib_session):
        broker, data, config = ib_session

        quote = broker.get_quote("AAPL")
        if not quote or quote.mid <= 0:
            pytest.skip("Cannot get AAPL quote for bracket test")

        price = quote.mid
        take_profit = round(price * 1.05, 2)
        stop_loss = round(price * 0.97, 2)
        order = broker.submit_bracket_order("AAPL", 1, "buy", take_profit, stop_loss)

        assert order is not None, "Bracket order returned None"
        assert order.id is not None, "Bracket order has no id"

        parent_id = int(order.id)
        trades = _wait_for_value(
            lambda: [
                trade
                for trade in broker._ib.trades()
                if trade.order.orderId == parent_id or trade.order.parentId == parent_id
            ],
            predicate=lambda value: len(value) >= 3,
            timeout=10.0,
        )
        assert trades is not None and len(trades) >= 3, (
            f"Expected bracket parent and children for order {parent_id}"
        )

        parent_trade = next((trade for trade in trades if trade.order.orderId == parent_id), None)
        children = [trade for trade in trades if trade.order.parentId == parent_id]

        assert parent_trade is not None, f"Missing parent trade for bracket order {parent_id}"
        assert parent_trade.order.transmit is False, "Parent bracket order should transmit=False"
        assert len(children) >= 2, f"Expected at least two child orders, found {len(children)}"
        assert any(
            child.order.orderType == "LMT" and child.order.transmit is False for child in children
        ), "Expected a take-profit child order with transmit=False"
        assert any(
            child.order.orderType == "STP" and child.order.transmit is True for child in children
        ), "Expected a stop-loss child order with transmit=True"

    def test_cancel_order(self, ib_session):
        from ib_insync import LimitOrder

        broker, data, config = ib_session

        asset = broker.asset_type("MSFT")
        contract = broker._resolve_contract("MSFT", asset)
        assert contract is not None, "Cannot resolve MSFT contract"

        trade = broker._ib.placeOrder(contract, LimitOrder("BUY", 1, 1.0))
        order_id = str(trade.order.orderId)

        time.sleep(1)
        broker.cancel_order(order_id)

        status = _wait_for_value(
            lambda: trade.orderStatus.status,
            predicate=lambda value: value in CANCELLED_ORDER_STATUSES,
            timeout=10.0,
        )
        assert status in CANCELLED_ORDER_STATUSES, f"Expected cancelled status, got {status}"

    def test_close_positions(self, ib_session):
        broker, data, config = ib_session

        positions = broker.get_positions()
        if not positions:
            pytest.skip("No open positions to close; earlier orders may not have filled.")

        for position in positions:
            broker.close_position(position.symbol)

        remaining = _wait_for_value(
            lambda: [position for position in broker.get_positions() if abs(position.qty) > 0],
            predicate=lambda value: value == [],
            timeout=20.0,
        )
        assert remaining == [], f"Failed to close positions: {[position.symbol for position in remaining or []]}"


class TestPhase4Resilience:
    def test_bad_contract_cache_ttl(self, ib_session):
        broker, data, config = ib_session

        broker._bad_contracts.pop("FAKESYM123", None)

        contract = broker._resolve_contract("FAKESYM123", "stock")
        assert contract is None
        assert "FAKESYM123" in broker._bad_contracts

        ttl = broker._bad_contracts["FAKESYM123"]
        now = time.time()
        assert now + 1700 < ttl < now + 1900, f"TTL {ttl} not in expected 30-minute range"

        broker._bad_contracts.pop("FAKESYM123", None)
        assert broker._resolve_contract("FAKESYM123", "stock") is None
        assert "FAKESYM123" in broker._bad_contracts

        broker._bad_contracts.pop("FAKESYM123", None)

    def test_duplicate_orders(self, ib_session):
        from base_broker import OrderRequest

        broker, data, config = ib_session

        order_1 = broker.submit_order(OrderRequest(symbol="AAPL", qty=1, side="buy"))
        order_2 = broker.submit_order(OrderRequest(symbol="AAPL", qty=1, side="buy"))

        assert order_1.id != order_2.id, "Duplicate orders should receive unique IB order IDs"

        time.sleep(2)
        position = _positions_by_symbol(broker.get_positions()).get("AAPL")
        if position and abs(position.qty) > 0:
            broker.close_position("AAPL")
        else:
            broker.cancel_order(order_1.id)
            broker.cancel_order(order_2.id)

    def test_event_loop_safety(self, ib_session):
        broker, data, config = ib_session

        errors = []

        def _run_from_thread():
            try:
                quote = broker.get_quote("SPY")
                if quote is None or quote.mid <= 0:
                    errors.append(f"get_quote returned {quote}")
            except Exception as exc:
                errors.append(f"get_quote error: {exc}")

            try:
                price = data.get_latest_price("SPY")
                if price is None or price <= 0:
                    errors.append(f"get_latest_price returned {price}")
            except Exception as exc:
                errors.append(f"get_latest_price error: {exc}")

            try:
                contract = broker._resolve_contract("SPY", broker.asset_type("SPY"))
                if contract is None:
                    errors.append("_resolve_contract returned None from worker thread")
            except Exception as exc:
                errors.append(f"_resolve_contract error: {exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_from_thread)
            future.result(timeout=15)

        assert not errors, "Event loop safety failures:\n" + "\n".join(errors)

    def test_reconnection(self, ib_session):
        broker, data, config = ib_session

        broker._ib.disconnect()
        assert not broker._ib.isConnected(), "Broker should be disconnected before reconnect test"

        equity = broker.get_equity()
        assert broker._ib.isConnected(), "Broker should reconnect on demand"
        assert equity > 0, f"Expected positive equity after reconnect, got {equity}"

    def test_pacing_limits_bar_fetch(self, ib_session):
        from tests.conftest import ALL_SYMBOLS

        broker, data, config = ib_session

        data.invalidate_cache()
        data._bad_contracts.clear()
        data._no_data_cache.clear()

        bars_dict = data.get_bars(ALL_SYMBOLS, timeframe="1Day", days=30)
        assert len(bars_dict) >= 18, (
            f"Expected at least 18 symbols with bars after fresh fetch, got {len(bars_dict)}"
        )

    def test_fractional_qty_handling(self, ib_session):
        broker, data, config = ib_session

        stock_qty = broker._format_order_qty("stock", 0.5)
        assert stock_qty == 0.5, f"Expected stock qty passthrough, got {stock_qty}"

        crypto_qty = broker._format_order_qty("crypto", 0.00001234)
        assert crypto_qty == "0.00001234", (
            f"Expected fixed-point crypto qty string, got {crypto_qty}"
        )

    def test_missing_data_no_crash(self, ib_session):
        broker, data, config = ib_session

        data._bad_contracts.pop("ZZZZZ", None)
        data._no_data_cache.pop("ZZZZZ", None)

        result = data.get_bars(["ZZZZZ"], timeframe="1Day", days=30)
        assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
        assert "ZZZZZ" not in result or result["ZZZZZ"] is None or result["ZZZZZ"].empty, (
            "Expected no data for nonexistent symbol"
        )
