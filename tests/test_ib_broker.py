import sys
import types
from unittest.mock import MagicMock

from ib_broker import IBBroker


class FakeIBPositions:
    def __init__(self, portfolio_items=None, account_positions=None):
        self._portfolio_items = portfolio_items or []
        self._account_positions = account_positions or []
        self.req_positions_calls = 0

    def isConnected(self):
        return True

    def reqPositions(self):
        self.req_positions_calls += 1

    def portfolio(self):
        return self._portfolio_items

    def positions(self):
        return self._account_positions


def _contract(symbol, local_symbol=None, sec_type="", currency=""):
    return types.SimpleNamespace(
        symbol=symbol,
        localSymbol=local_symbol,
        secType=sec_type,
        currency=currency,
    )


def _portfolio_item(symbol, qty, average_cost, market_value, unrealized_pnl):
    return types.SimpleNamespace(
        contract=_contract(symbol),
        position=qty,
        averageCost=average_cost,
        marketValue=market_value,
        unrealizedPNL=unrealized_pnl,
    )


def _account_position(symbol, qty, avg_cost):
    return types.SimpleNamespace(
        contract=_contract(symbol),
        position=qty,
        avgCost=avg_cost,
    )


def test_format_order_qty_keeps_crypto_out_of_scientific_notation():
    assert IBBroker._format_order_qty("crypto", 0.00001334) == "0.00001334"


def test_format_order_qty_leaves_non_crypto_numeric():
    assert IBBroker._format_order_qty("stock", 10) == 10


def test_get_positions_falls_back_to_account_positions():
    broker = object.__new__(IBBroker)
    broker._last_position_refresh = 0.0
    broker._ib = FakeIBPositions(
        portfolio_items=[],
        account_positions=[_account_position("GLD", 1, 190.0)],
    )

    positions = broker.get_positions()

    assert broker._ib.req_positions_calls == 1
    assert len(positions) == 1
    assert positions[0].symbol == "GLD"
    assert positions[0].qty == 1
    assert positions[0].avg_price == 190.0
    assert positions[0].market_value == 190.0


def test_get_positions_merges_portfolio_with_account_snapshot():
    broker = object.__new__(IBBroker)
    broker._last_position_refresh = 0.0
    broker._ib = FakeIBPositions(
        portfolio_items=[_portfolio_item("AAPL", 2, 170.0, 345.0, 5.0)],
        account_positions=[
            _account_position("AAPL", 2, 170.0),
            _account_position("GLD", 1, 190.0),
        ],
    )

    positions = broker.get_positions()
    by_symbol = {pos.symbol: pos for pos in positions}

    assert set(by_symbol) == {"AAPL", "GLD"}
    assert by_symbol["AAPL"].unrealized_pl == 5.0
    assert by_symbol["GLD"].market_value == 190.0


def test_crypto_position_symbol_normalizes_to_pair():
    broker = object.__new__(IBBroker)
    broker._last_position_refresh = 0.0
    broker._ib = FakeIBPositions(
        portfolio_items=[
            types.SimpleNamespace(
                contract=_contract("BTC", local_symbol="BTC.USD", sec_type="CRYPTO", currency="USD"),
                position=0.01,
                averageCost=100000.0,
                marketValue=1000.0,
                unrealizedPNL=5.0,
            )
        ],
    )

    positions = broker.get_positions()

    assert positions[0].symbol == "BTC/USD"
    assert IBBroker._position_key("BTC.USD") == IBBroker._position_key("BTC/USD")


def test_get_positions_refreshes_existing_subscription():
    broker = object.__new__(IBBroker)
    broker._last_position_refresh = 0.0
    broker._positions_subscribed = True
    broker._ib = FakeIBPositions(
        portfolio_items=[],
        account_positions=[_account_position("GLD", 1, 190.0)],
    )

    broker.get_positions()

    assert broker._ib.req_positions_calls == 1


def test_resolve_contract_uses_smart_primary_exchange_for_overrides(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, exchange, currency):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency
            self.primaryExch = None

    monkeypatch.setitem(sys.modules, "ib_insync", types.SimpleNamespace(Stock=FakeStock))

    broker = object.__new__(IBBroker)
    broker._ib = MagicMock()
    broker._contracts = MagicMock()
    broker._bad_contracts = {}

    seen = []

    def _qualify(contract):
        seen.append(contract)
        return [contract]

    broker._ib.qualifyContracts.side_effect = _qualify

    contract = broker._resolve_contract("GLD", "stock")

    assert contract.exchange == "SMART"
    assert contract.primaryExch == "ARCA"
    assert seen[0].exchange == "SMART"
    assert seen[0].primaryExch == "ARCA"
