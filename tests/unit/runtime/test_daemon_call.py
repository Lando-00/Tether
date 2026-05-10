"""Tests for ``tether.runtime.daemon_call.daemon_thread_call``.

Synthesis §13.4 M1.
"""
from __future__ import annotations

import gc
import threading
import time

import pytest

from tether.runtime.daemon_call import daemon_thread_call


@pytest.fixture(autouse=True)
def _restore_gc():
    """Some tests intentionally disable GC inside the daemon thread (which is
    process-wide in CPython). Restore the prior state so subsequent tests
    aren't polluted.
    """
    was_enabled = gc.isenabled()
    yield
    if was_enabled and not gc.isenabled():
        gc.enable()
    elif not was_enabled and gc.isenabled():
        gc.disable()


def test_daemon_thread_call_returns_value():
    """A normally-returning callable's result is returned to the caller."""

    def fn() -> int:
        return 42

    assert daemon_thread_call(fn, timeout=2.0, gc_disable=False) == 42


def test_daemon_thread_call_timeout():
    """If the callable exceeds the timeout, ``TimeoutError`` is raised and
    the daemon thread is abandoned (still alive but not joined)."""
    started = threading.Event()
    keep_alive = threading.Event()  # never set; thread keeps sleeping

    def fn() -> None:
        started.set()
        # Sleep way past timeout; we'll abandon this thread.
        keep_alive.wait(timeout=10.0)

    with pytest.raises(TimeoutError):
        daemon_thread_call(fn, timeout=0.1, gc_disable=False, label="slow")

    # Confirm fn actually started.
    assert started.is_set()
    # Release it so it can clean up after the test (even though it's daemon).
    keep_alive.set()


def test_daemon_thread_call_propagates_exception():
    """A callable that raises ``ValueError`` causes ``daemon_thread_call`` to
    re-raise ``ValueError`` with its args preserved."""

    def fn() -> None:
        raise ValueError("boom", 7)

    with pytest.raises(ValueError) as ei:
        daemon_thread_call(fn, timeout=2.0, gc_disable=False)
    assert ei.value.args == ("boom", 7)


def test_daemon_thread_call_propagates_keyboard_interrupt():
    """``KeyboardInterrupt`` is a ``BaseException`` (not ``Exception``); the
    helper must still re-raise it (uses ``except BaseException``)."""

    def fn() -> None:
        raise KeyboardInterrupt("user-interrupt")

    with pytest.raises(KeyboardInterrupt):
        daemon_thread_call(fn, timeout=2.0, gc_disable=False)


def test_daemon_thread_call_gc_disable_disables_gc():
    """With ``gc_disable=True``, ``gc.isenabled()`` reads ``False`` inside
    the daemon thread (GC is process-wide in CPython, so it affects the
    interpreter — that's intentional, see daemon_call.py docstring)."""
    observed: dict[str, bool] = {}

    def fn() -> None:
        observed["enabled_inside"] = gc.isenabled()

    # Make sure GC starts enabled so the helper has something to disable.
    if not gc.isenabled():
        gc.enable()

    daemon_thread_call(fn, timeout=2.0, gc_disable=True, label="gctest")
    assert observed["enabled_inside"] is False
    # GC must remain disabled (the rule: NEVER re-enable from the daemon).
    assert gc.isenabled() is False


def test_daemon_thread_call_gc_disable_no_op_if_already_disabled():
    """If GC was already disabled before the call, the helper does not call
    ``gc.disable()`` again (and certainly never re-enables). Behavior is a
    no-op on GC; the callable still runs and the result is returned.
    """
    observed: dict[str, bool] = {}

    def fn() -> int:
        observed["enabled_inside"] = gc.isenabled()
        return 99

    gc.disable()  # caller already disabled GC

    result = daemon_thread_call(
        fn, timeout=2.0, gc_disable=True, label="gctest-already"
    )

    assert result == 99
    # Inside: GC is still disabled (we didn't re-enable, no harm).
    assert observed["enabled_inside"] is False
    # After: GC is still disabled (we never re-enable).
    assert gc.isenabled() is False


def test_daemon_thread_call_gc_disable_false_leaves_gc_enabled():
    """With ``gc_disable=False``, the helper does NOT touch GC."""
    observed: dict[str, bool] = {}

    def fn() -> None:
        observed["enabled_inside"] = gc.isenabled()

    if not gc.isenabled():
        gc.enable()

    daemon_thread_call(fn, timeout=2.0, gc_disable=False, label="nogc")

    assert observed["enabled_inside"] is True
    assert gc.isenabled() is True


def test_daemon_thread_call_thread_name():
    """The daemon thread's name includes the ``label`` parameter."""
    observed: dict[str, str] = {}

    def fn() -> None:
        observed["name"] = threading.current_thread().name

    daemon_thread_call(
        fn, timeout=2.0, gc_disable=False, label="my-special-label"
    )
    assert "my-special-label" in observed["name"]
    assert observed["name"].endswith("-thread")


def test_daemon_thread_call_default_label():
    """Default label is ``"daemon"`` so the thread name is
    ``"daemon-thread"`` (preserves human-readable default)."""
    observed: dict[str, str] = {}

    def fn() -> None:
        observed["name"] = threading.current_thread().name

    daemon_thread_call(fn, timeout=2.0, gc_disable=False)
    assert observed["name"] == "daemon-thread"


def test_daemon_thread_call_completes_quickly():
    """Sanity: a fast callable returns well before the budget expires."""

    def fn() -> str:
        time.sleep(0.01)
        return "ok"

    start = time.time()
    assert daemon_thread_call(fn, timeout=2.0, gc_disable=False) == "ok"
    elapsed = time.time() - start
    assert elapsed < 1.0


def test_daemon_thread_call_default_does_not_disable_gc():
    """Phase 3 follow-up: the ``gc_disable`` default is now ``False``.

    Pre-flip (default ``True``) was a footgun for Phase 4.5 connectors
    that run ``daemon_thread_call`` during process lifetime (e.g. logout
    teardown) — silently leaving GC disabled for the rest of the process.

    With no ``gc_disable`` kwarg, ``fn()`` observes ``gc.isenabled() is
    True`` and the parent thread's GC state is unchanged.

    :class:`HardwareWatchdog.shutdown_all` passes ``gc_disable=True``
    explicitly (see ``runtime/hw_watchdog.py``), so the shutdown
    invariant (R5) is preserved by that explicit kwarg, not by this
    default.
    """
    observed: dict[str, bool] = {}

    def fn() -> str:
        observed["enabled_inside"] = gc.isenabled()
        return "default-gc"

    if not gc.isenabled():
        gc.enable()

    # Call with NO ``gc_disable`` kwarg — exercises the new default.
    result = daemon_thread_call(fn, timeout=2.0, label="default-gc-test")

    assert result == "default-gc"
    # Inside the daemon thread: GC was NOT disabled.
    assert observed["enabled_inside"] is True
    # After: parent thread's GC state unchanged.
    assert gc.isenabled() is True
