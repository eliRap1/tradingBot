from unittest.mock import MagicMock

import pandas as pd

from alerts import DiscordBot
from base_broker import Order, Position, Quote


def test_submit_test_buy_uses_quote_to_size_fractional_btc():
    broker = MagicMock()
    broker.asset_type.return_value = "crypto"
    broker.get_quote.return_value = Quote(symbol="BTC/USD", bid=99_999.0, ask=100_001.0)
    broker.submit_market_order.return_value = Order(
        id="btc-test-1",
        symbol="BTC/USD",
        qty=0.00001,
        side="buy",
        order_type="market",
        status="submitted",
    )

    bot = DiscordBot(tracker=MagicMock(), broker=broker)

    msg = bot._submit_test_buy()

    broker.submit_market_order.assert_called_once_with(
        "BTC/USD", 0.00001, "buy", notional=1.0
    )
    assert "Submitted test buy" in msg
    assert "btc-test-1" in msg


def test_submit_test_buy_requires_live_quote():
    broker = MagicMock()
    broker.asset_type.return_value = "crypto"
    broker.get_quote.return_value = None

    bot = DiscordBot(tracker=MagicMock(), broker=broker)

    msg = bot._submit_test_buy()

    broker.submit_market_order.assert_not_called()
    assert "No price available" in msg


def test_submit_test_buy_blocks_stock_when_market_closed():
    broker = MagicMock()
    broker.asset_type.return_value = "stock"

    coordinator = MagicMock()
    coordinator.live_manager.can_trade_symbol.return_value = (False, "premarket_only")

    bot = DiscordBot(tracker=MagicMock(), broker=broker, coordinator=coordinator)

    msg = bot._submit_test_buy(symbol="GLD", notional_usd=10.0)

    coordinator.live_manager.can_trade_symbol.assert_called_once_with("GLD")
    broker.submit_market_order.assert_not_called()
    assert "premarket_only" in msg


def test_submit_test_buy_falls_back_to_daily_bar_cache():
    broker = MagicMock()
    broker.asset_type.return_value = "crypto"
    broker.get_quote.return_value = None
    broker.submit_market_order.return_value = Order(
        id="btc-test-bar-1",
        symbol="BTC/USD",
        qty=0.00001,
        side="buy",
        order_type="market",
        status="submitted",
    )

    coordinator = MagicMock()
    coordinator.live_manager.get_live_price.return_value = (None, "no_price_available")
    coordinator.data.get_intraday_bars.return_value = pd.DataFrame(
        {"close": [100000.0]}
    )

    bot = DiscordBot(tracker=MagicMock(), broker=broker, coordinator=coordinator)

    msg = bot._submit_test_buy(symbol="BTC/USD", notional_usd=1.0)

    coordinator.data.get_intraday_bars.assert_called_once_with(
        "BTC/USD", timeframe="1Day", days=5
    )
    broker.submit_market_order.assert_called_once_with(
        "BTC/USD", 0.00001, "buy", notional=1.0
    )
    assert "Submitted test buy" in msg


def test_positions_message_shows_visible_position():
    broker = MagicMock()
    broker.get_positions.return_value = [
        Position(
            symbol="GLD",
            qty=1,
            avg_price=190.0,
            market_value=192.0,
            unrealized_pl=2.0,
            side="long",
        )
    ]
    broker.get_equity.return_value = 100_002.0

    bot = DiscordBot(tracker=MagicMock(), broker=broker)

    msg = bot._positions_message()

    assert "Open Positions" in msg
    assert "GLD" in msg
    assert "Total unrealized" in msg
    assert "$100,002.00" in msg


def test_positions_message_shows_open_order_when_position_not_visible_yet():
    broker = MagicMock()
    broker.get_positions.return_value = []
    broker.get_open_orders.return_value = [
        Order(
            id="test-order-1",
            symbol="GLD",
            qty=1,
            side="buy",
            order_type="market",
            status="submitted",
        )
    ]
    broker.get_equity.return_value = 100_000.0

    bot = DiscordBot(tracker=MagicMock(), broker=broker)

    msg = bot._positions_message()

    assert "No filled positions are visible yet" in msg
    assert "Open Orders" in msg
    assert "GLD" in msg
    assert "$100,000.00" in msg
