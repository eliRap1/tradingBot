from unittest.mock import MagicMock

from alerts import DiscordBot


def test_discord_bot_stop_waits_for_close_future():
    bot = DiscordBot(tracker=MagicMock(), broker=MagicMock())
    bot._stop_event.clear()
    bot._loop = MagicMock()
    bot._loop.is_closed.return_value = False
    bot._client = MagicMock()
    bot._thread = MagicMock()
    bot._thread.is_alive.return_value = True

    close_future = MagicMock()

    import alerts
    original = alerts.asyncio.run_coroutine_threadsafe
    alerts.asyncio.run_coroutine_threadsafe = MagicMock(return_value=close_future)
    try:
        bot.stop()
    finally:
        alerts.asyncio.run_coroutine_threadsafe = original

    close_future.result.assert_called_once_with(timeout=5)
    bot._thread.join.assert_called_once_with(timeout=5)
    assert bot._stop_event.is_set() is True
