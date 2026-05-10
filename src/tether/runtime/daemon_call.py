"""Run a synchronous callable in a daemon thread with optional GC-disable
and bounded timeout. Returns the callable's result, or raises ``TimeoutError``
if it exceeds the budget.

Used by :class:`HardwareWatchdog.shutdown_all` to bound MLC engine teardown,
and by future connectors (Phase 4.5) that wrap blocking native cleanup.
The daemon-thread mechanic ensures the process can exit cleanly even if the
callable hangs in native code (OpenCL destructor, etc.) — daemon threads
are killed on process exit, the OS reclaims the resources.

Synthesis §13.4 M1. Behavior preserved verbatim from
``app/http/api.py:99-167`` ``shutdown_provider_with_timeout``.

CRITICAL: When ``gc_disable=True``, GC is DISABLED in the daemon thread and
NOT re-enabled. This is load-bearing for OpenCL/TVM destructors on
Snapdragon Adreno (.github/copilot-instructions.md "Never re-enable GC in
the daemon shutdown thread" rule). The thread is abandoned on timeout, so
accumulated garbage is reclaimed when the process exits.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Callable, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def daemon_thread_call(
    fn: Callable[[], T],
    *,
    timeout: float,
    gc_disable: bool = False,
    label: str = "daemon",
) -> T:
    """Run ``fn()`` in a daemon thread; return its result if it completes
    within ``timeout``, else raise :class:`TimeoutError`.

    Args:
        fn: nullary callable to execute.
        timeout: max seconds to wait for completion.
        gc_disable: if ``True``, disable Python GC inside the daemon thread
            for the duration of ``fn()`` and DO NOT re-enable. Required for
            MLC/OpenCL teardown paths (synthesis §6 / GC-disable rule). If
            GC was already disabled by the caller, this is a no-op (we
            don't try to re-disable).

            DEFAULT FALSE (Phase 3 follow-up; rubber-duck 1m CONCERN):
            callers like Phase 4.5 connectors that run during process
            lifetime (e.g. logout teardown) MUST NOT disable GC — that
            would silently leak GC state for the rest of the process.
            :class:`HardwareWatchdog.shutdown_all` passes ``True``
            explicitly because the GC-disable rule is load-bearing for
            shutdown ONLY. New callers must opt in.
        label: thread name suffix (for debugging / log messages); the
            spawned thread's name is ``f"{label}-thread"``.

    Returns:
        The return value of ``fn()``.

    Raises:
        TimeoutError: if ``fn()`` doesn't complete within ``timeout``
            seconds. The thread is abandoned (will be killed on process
            exit).
        BaseException: any exception raised by ``fn()`` is re-raised in
            the calling thread, preserving traceback.

    Notes:
        - GC is process-wide in CPython; calling ``gc.disable()`` from this
          daemon thread affects the whole interpreter. That's intentional —
          the destructors that hang are sometimes triggered from other
          threads via the cyclic GC.
        - If the callable raises, GC is left in whatever state it was in
          when the exception happened (we never re-enable, per the rule).
    """
    done = threading.Event()
    result: List[T] = []
    exc: List[BaseException] = []

    def _wrapper() -> None:
        gc_was_enabled = gc.isenabled()
        if gc_disable and gc_was_enabled:
            gc.disable()
            logger.debug(
                "daemon_thread_call(%s): GC disabled in daemon thread", label
            )
        try:
            value = fn()
            result.append(value)
        except BaseException as e:  # capture EVERY exception class
            exc.append(e)
        finally:
            # DO NOT re-enable GC here — synthesis §6 / GC-disable rule.
            # The daemon thread will be killed on process exit; accumulated
            # garbage is fine.
            done.set()

    thread = threading.Thread(
        target=_wrapper, daemon=True, name=f"{label}-thread"
    )
    thread.start()

    start = time.time()
    if not done.wait(timeout=timeout):
        elapsed = time.time() - start
        logger.warning(
            "daemon_thread_call(%s) TIMED OUT after %.2fs (budget=%.2fs); "
            "abandoning. Daemon thread will be killed on process exit.",
            label,
            elapsed,
            timeout,
        )
        raise TimeoutError(
            f"daemon_thread_call({label}) exceeded {timeout:.2f}s"
        )

    if exc:
        # Re-raise the captured exception preserving traceback.
        raise exc[0]

    elapsed = time.time() - start
    logger.debug(
        "daemon_thread_call(%s) completed in %.2fs", label, elapsed
    )
    return result[0]


__all__ = ["daemon_thread_call"]
