from unittest.mock import MagicMock
from base_broker import OrderRequest


def make_routing_broker():
    """Build a RoutingBroker with mock sub-brokers."""
    from routing_broker import RoutingBroker
    from instrument_classifier import InstrumentClassifier

    config = {
        "futures": {"contracts": [
            {"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"}
        ]},
        "screener": {"crypto": ["BTC/USD", "ETH/USD"]},
    }

    ib_broker = MagicMock()
    alpaca_broker = MagicMock()
    clf = InstrumentClassifier(config)

    rb = RoutingBroker.__new__(RoutingBroker)
    rb._ib = ib_broker
    rb._alpaca = alpaca_broker
    rb._clf = clf
    return rb, ib_broker, alpaca_broker


def test_futures_routes_to_ib():
    rb, ib, alpaca = make_routing_broker()
    req = OrderRequest(symbol="NQ", qty=1, side="buy", take_profit=19000, stop_loss=18500)
    ib.submit_order.return_value = MagicMock()
    rb.submit_order(req)
    ib.submit_order.assert_called_once_with(req)
    alpaca.submit_order.assert_not_called()


def test_crypto_routes_to_alpaca():
    rb, ib, alpaca = make_routing_broker()
    req = OrderRequest(symbol="BTC/USD", qty=0.01, side="buy", take_profit=85000, stop_loss=78000)
    alpaca.submit_order.return_value = MagicMock()
    rb.submit_order(req)
    alpaca.submit_order.assert_called_once_with(req)
    ib.submit_order.assert_not_called()


def test_stock_routes_to_ib():
    rb, ib, alpaca = make_routing_broker()
    req = OrderRequest(symbol="AAPL", qty=10, side="buy", take_profit=210, stop_loss=195)
    ib.submit_order.return_value = MagicMock()
    rb.submit_order(req)
    ib.submit_order.assert_called_once_with(req)
    alpaca.submit_order.assert_not_called()


def test_get_positions_aggregates_both():
    rb, ib, alpaca = make_routing_broker()
    ib.get_positions.return_value = [MagicMock(symbol="NQ")]
    alpaca.get_positions.return_value = [MagicMock(symbol="BTC/USD")]
    positions = rb.get_positions()
    assert len(positions) == 2


def test_close_position_routes_correctly():
    rb, ib, alpaca = make_routing_broker()
    rb.close_position("AAPL")
    ib.close_position.assert_called_once_with("AAPL")
    alpaca.close_position.assert_not_called()

    rb.close_position("BTC/USD")
    alpaca.close_position.assert_called_once_with("BTC/USD")


def test_asset_type_delegates_to_classifier():
    rb, ib, alpaca = make_routing_broker()
    assert rb.asset_type("NQ") == "futures"
    assert rb.asset_type("BTC/USD") == "crypto"
    assert rb.asset_type("AAPL") == "stock"
