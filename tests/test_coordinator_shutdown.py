import threading
from unittest.mock import MagicMock

from coordinator import Coordinator


def test_wait_or_stop_returns_false_when_timeout_expires():
    coord = Coordinator.__new__(Coordinator)
    coord._shutdown_event = threading.Event()

    assert coord._wait_or_stop(0.01) is False


def test_wait_or_stop_returns_true_when_shutdown_requested():
    coord = Coordinator.__new__(Coordinator)
    coord._shutdown_event = threading.Event()
    coord._shutdown_event.set()

    assert coord._wait_or_stop(10) is True


def test_shutdown_stops_watchers_discord_and_broker_once():
    coord = Coordinator.__new__(Coordinator)
    coord._shutdown_event = threading.Event()
    coord._shutdown_lock = threading.Lock()
    coord._shutdown_complete = False
    coord.stop_watchers = MagicMock()
    coord.discord_bot = MagicMock()
    coord.broker = MagicMock()

    coord.shutdown()
    coord.shutdown()

    assert coord._shutdown_event.is_set() is True
    coord.stop_watchers.assert_called_once_with()
    coord.discord_bot.stop.assert_called_once_with()
    coord.broker.disconnect.assert_called_once_with()
