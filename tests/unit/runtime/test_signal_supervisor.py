"""Tests for ``tether_service.runtime.signal_supervisor.SignalSupervisor``.

Synthesis §4 Phase 3 step 31; B6 step 3.

Critical: tests MUST mock ``signal.signal`` and ``os._exit`` so we don't
actually install handlers or terminate the test process. The supervisor's
class-level ``_installed`` flag is reset between tests via a fixture.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from tether_service.runtime.signal_supervisor import (
    DEFAULT_MAX_SHUTDOWN_SEC,
    SignalSupervisor,
)


@pytest.fixture(autouse=True)
def _reset_supervisor_class_state():
    """Reset the class-level ``_installed`` flag between tests."""
    SignalSupervisor._installed = False
    yield
    SignalSupervisor._installed = False


@pytest.fixture
def stub_signals(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Replace ``signal.signal`` and ``signal.getsignal`` so the test
    process never actually has its handlers replaced.

    Returns a dict with:
        - ``installed``: list of (signum, handler) pairs from signal.signal
        - ``original_sigint`` / ``original_sigterm``: the (mock) original
          handlers returned by getsignal
    """
    original_sigint = MagicMock(name="original-sigint")
    original_sigterm = MagicMock(name="original-sigterm")
    installed: List[Tuple[int, Any]] = []

    def fake_getsignal(sig: int) -> Any:
        if sig == signal.SIGINT:
            return original_sigint
        if sig == signal.SIGTERM:
            return original_sigterm
        return None

    def fake_signal(sig: int, handler: Any) -> Any:
        installed.append((sig, handler))
        return None

    monkeypatch.setattr(signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)

    return {
        "installed": installed,
        "original_sigint": original_sigint,
        "original_sigterm": original_sigterm,
    }


@pytest.fixture
def stub_os_exit(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``os._exit`` with a Mock so the test process survives."""
    mock = MagicMock(name="os._exit")
    monkeypatch.setattr(os, "_exit", mock)
    return mock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_signal_supervisor_construction():
    """``__init__`` records max_shutdown_sec and starts with
    shutdown_in_progress=False; default is DEFAULT_MAX_SHUTDOWN_SEC."""
    s = SignalSupervisor()
    assert s.max_shutdown_sec == DEFAULT_MAX_SHUTDOWN_SEC
    assert s.shutdown_in_progress is False

    s2 = SignalSupervisor(max_shutdown_sec=2.5)
    assert s2.max_shutdown_sec == 2.5
    assert s2.shutdown_in_progress is False


def test_signal_supervisor_does_not_install_handlers_in_construction():
    """``__init__`` MUST NOT touch ``signal.signal``; only ``install()`` does."""
    pre_sigint = signal.getsignal(signal.SIGINT)
    pre_sigterm = signal.getsignal(signal.SIGTERM)

    SignalSupervisor()
    SignalSupervisor(max_shutdown_sec=1.0)

    # The actual handlers are unchanged.
    assert signal.getsignal(signal.SIGINT) == pre_sigint
    assert signal.getsignal(signal.SIGTERM) == pre_sigterm

    # And the class-level guard hasn't been set by mere construction.
    assert SignalSupervisor._installed is False


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------


def test_signal_supervisor_install_registers_handlers(stub_signals):
    """install() calls signal.signal for both SIGINT and SIGTERM with the
    supervisor's _handle_signal as the handler."""
    s = SignalSupervisor()
    s.install()

    installed_sigs = {sig for sig, _ in stub_signals["installed"]}
    assert installed_sigs == {signal.SIGINT, signal.SIGTERM}

    # Both registrations point at the supervisor's bound method.
    for _, handler in stub_signals["installed"]:
        assert handler == s._handle_signal


def test_signal_supervisor_install_captures_original_handlers(stub_signals):
    """install() calls signal.getsignal so the original handlers are
    preserved for forwarding."""
    s = SignalSupervisor()
    s.install()

    assert s._original_sigint is stub_signals["original_sigint"]
    assert s._original_sigterm is stub_signals["original_sigterm"]


def test_signal_supervisor_install_idempotent(
    stub_signals, caplog: pytest.LogCaptureFixture
):
    """install() called twice is a no-op on the second call (the second
    SignalSupervisor instance does not re-register handlers and a warning
    is emitted)."""
    s1 = SignalSupervisor()
    s1.install()

    n_after_first = len(stub_signals["installed"])

    with caplog.at_level(logging.WARNING):
        s2 = SignalSupervisor()
        s2.install()  # must be no-op

    assert len(stub_signals["installed"]) == n_after_first
    assert any(
        "already installed" in rec.message for rec in caplog.records
    ), f"expected 'already installed' warning; got {caplog.records!r}"


# ---------------------------------------------------------------------------
# _handle_signal
# ---------------------------------------------------------------------------


def test_signal_supervisor_handles_first_signal(stub_signals, stub_os_exit):
    """On first signal: shutdown_in_progress flips True, start time
    recorded, original handler invoked, os._exit NOT called."""
    s = SignalSupervisor(max_shutdown_sec=10.0)
    s.install()

    s._handle_signal(signal.SIGINT, None)

    assert s.shutdown_in_progress is True
    assert s._shutdown_start_time is not None
    stub_signals["original_sigint"].assert_called_once_with(signal.SIGINT, None)
    stub_os_exit.assert_not_called()


def test_signal_supervisor_first_sigterm_forwards_to_sigterm_handler(
    stub_signals, stub_os_exit
):
    """SIGTERM forwards to the SIGTERM original handler, not SIGINT's."""
    s = SignalSupervisor(max_shutdown_sec=10.0)
    s.install()

    s._handle_signal(signal.SIGTERM, None)

    stub_signals["original_sigterm"].assert_called_once_with(
        signal.SIGTERM, None
    )
    stub_signals["original_sigint"].assert_not_called()
    stub_os_exit.assert_not_called()


def test_signal_supervisor_handles_second_signal_force_exits(
    stub_signals, stub_os_exit
):
    """Second signal causes os._exit(1); the original handler is NOT
    called the second time."""
    s = SignalSupervisor(max_shutdown_sec=10.0)
    s.install()

    s._handle_signal(signal.SIGINT, None)
    stub_signals["original_sigint"].reset_mock()
    stub_os_exit.assert_not_called()

    s._handle_signal(signal.SIGINT, None)

    stub_os_exit.assert_called_once_with(1)
    # The original handler must NOT be called again on the second signal.
    stub_signals["original_sigint"].assert_not_called()


def test_signal_supervisor_swallows_original_handler_exception(
    stub_signals, stub_os_exit, caplog: pytest.LogCaptureFixture
):
    """If the original handler raises (non-SystemExit), the supervisor
    logs and continues without re-raising — uvicorn must not crash on a
    flaky chain."""
    stub_signals["original_sigint"].side_effect = RuntimeError("bad")
    s = SignalSupervisor(max_shutdown_sec=10.0)
    s.install()

    with caplog.at_level(logging.ERROR):
        s._handle_signal(signal.SIGINT, None)  # MUST NOT raise

    assert s.shutdown_in_progress is True


def test_signal_supervisor_propagates_systemexit_from_original(
    stub_signals, stub_os_exit
):
    """``SystemExit`` from the original handler propagates (uvicorn relies
    on this for clean exit on subsequent KeyboardInterrupt)."""
    stub_signals["original_sigint"].side_effect = SystemExit(0)
    s = SignalSupervisor(max_shutdown_sec=10.0)
    s.install()

    with pytest.raises(SystemExit):
        s._handle_signal(signal.SIGINT, None)


# ---------------------------------------------------------------------------
# Force-exit timer
# ---------------------------------------------------------------------------


def test_signal_supervisor_force_exit_timer_fires(
    stub_signals, stub_os_exit
):
    """Time-based force-exit: install with a very short max_shutdown_sec,
    deliver the first signal, sleep past the budget, observe os._exit(1)."""
    s = SignalSupervisor(max_shutdown_sec=0.1)
    s.install()

    s._handle_signal(signal.SIGINT, None)

    # Wait for the timer thread to fire.
    time.sleep(0.3)

    assert stub_os_exit.called, "force-exit timer should have called os._exit"
    # The call must be (1) per the spec.
    stub_os_exit.assert_any_call(1)


def test_signal_supervisor_force_exit_timer_starts_daemon_thread(
    stub_signals, stub_os_exit
):
    """The timer thread is a daemon thread named 'SignalSupervisor-timer'."""
    s = SignalSupervisor(max_shutdown_sec=10.0)  # long so it doesn't fire
    s.install()

    s._handle_signal(signal.SIGINT, None)

    try:
        assert s._timer_thread is not None
        assert s._timer_thread.daemon is True
        assert s._timer_thread.name == "SignalSupervisor-timer"
    finally:
        # CRITICAL: clear _shutdown_started so the daemon thread, when it
        # wakes from time.sleep(10) AFTER monkeypatch has reverted os._exit
        # to the real one, takes the safe branch and skips os._exit(1)
        # instead of killing the test runner. Pre-existing Phase 3 bug
        # surfaced in Phase 7 as suite runtime crept past 10s.
        s._shutdown_started = False


def test_signal_supervisor_force_exit_timer_skipped_if_no_shutdown(
    stub_signals, stub_os_exit
):
    """If _force_exit_timer somehow runs without shutdown started (defensive
    branch in the supervisor), it does NOT call os._exit."""
    s = SignalSupervisor(max_shutdown_sec=0.05)
    s.install()
    # Don't trigger _handle_signal; call _force_exit_timer directly.
    s._force_exit_timer()
    stub_os_exit.assert_not_called()
