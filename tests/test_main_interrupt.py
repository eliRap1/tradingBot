import pytest

import main


class _ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


def test_first_interrupt_starts_shutdown_and_raises(monkeypatch):
    calls = []

    class Bot:
        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(main.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(main, "_schedule_forced_exit", lambda timeout_sec=5.0: calls.append("timer"))
    main._shutdown_press_count = 0
    main._force_exit_timer = None

    handler = main._build_interrupt_handler(Bot())

    with pytest.raises(KeyboardInterrupt):
        handler(None, None)

    assert calls == ["shutdown", "timer"]


def test_second_interrupt_forces_exit(monkeypatch):
    calls = []

    class Bot:
        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(main.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    main._shutdown_press_count = 1
    main._force_exit_timer = None

    handler = main._build_interrupt_handler(Bot())

    with pytest.raises(SystemExit) as exc:
        handler(None, None)

    assert exc.value.code == 130
    assert calls == []
