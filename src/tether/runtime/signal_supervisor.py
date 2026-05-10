"""SignalSupervisor: SIGINT/SIGTERM handling for ``WatchdogMode.SERVER``.

Closes the gap from B6 §1.2 #1: the legacy
``app/http/api.py:25-65`` ``force_exit_handler`` only fires on a SECOND
signal, not on a TIMEOUT. This module adds a real time-based force-exit
timer.

Behavior on signal:

    1. Mark shutdown in progress; record start time.
    2. Call the original SIGINT/SIGTERM handler (uvicorn's shutdown
       trigger).
    3. Schedule a daemon timer thread: if shutdown isn't complete after
       ``max_shutdown_sec``, force-exit with ``os._exit(1)``.
    4. If a SECOND signal arrives, force-exit immediately.

LIBRARY-mode users do NOT install this — they manage signals themselves
(per :class:`tether_service.runtime.watchdog_mode.WatchdogMode.LIBRARY`
contract). ``p3-lifespan-slim`` will install it from ``__main__.py`` for
SERVER-mode startup.

Synthesis §4 Phase 3 step 31; B6 step 3.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_SHUTDOWN_SEC = 5.0


class SignalSupervisor:
    """Install SIGINT/SIGTERM handlers with bounded force-exit.

    Construction does NOT install handlers; call :meth:`install` once at
    startup. Idempotent: calling :meth:`install` twice is a no-op (logs a
    warning).

    The supervisor is effectively process-wide singleton — only one
    instance should call :meth:`install` per process (we guard with a
    class-level flag).
    """

    _installed = False  # class-level guard so multiple instances don't double-install

    def __init__(
        self, *, max_shutdown_sec: float = DEFAULT_MAX_SHUTDOWN_SEC
    ) -> None:
        self._max_shutdown_sec = max_shutdown_sec
        self._shutdown_started = False
        self._shutdown_start_time: Optional[float] = None
        self._original_sigint: Any = None
        self._original_sigterm: Any = None
        self._timer_thread: Optional[threading.Thread] = None

    @property
    def max_shutdown_sec(self) -> float:
        return self._max_shutdown_sec

    @property
    def shutdown_in_progress(self) -> bool:
        return self._shutdown_started

    def install(self) -> None:
        """Install SIGINT/SIGTERM handlers. Idempotent (warns + skips on
        a second call to any SignalSupervisor instance)."""
        if SignalSupervisor._installed:
            logger.warning(
                "SignalSupervisor.install: already installed; skipping"
            )
            return

        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        SignalSupervisor._installed = True
        logger.info(
            "SignalSupervisor installed: max_shutdown_sec=%.2fs",
            self._max_shutdown_sec,
        )

    def _handle_signal(self, signum: int, frame: Any) -> None:
        if not self._shutdown_started:
            # First signal — initiate graceful shutdown.
            self._shutdown_started = True
            self._shutdown_start_time = time.time()
            logger.warning(
                "SignalSupervisor: signal %d received; initiating shutdown",
                signum,
            )

            # Start the force-exit timer thread (daemon — dies with the
            # process if shutdown completes normally before the timer fires).
            self._timer_thread = threading.Thread(
                target=self._force_exit_timer,
                daemon=True,
                name="SignalSupervisor-timer",
            )
            self._timer_thread.start()

            # Forward to the original handler (uvicorn's shutdown trigger).
            original = (
                self._original_sigint
                if signum == signal.SIGINT
                else self._original_sigterm
            )
            if callable(original):
                try:
                    original(signum, frame)
                except SystemExit:
                    raise
                except Exception:
                    logger.exception(
                        "SignalSupervisor: original signal handler raised"
                    )
        else:
            # Second signal — force-exit immediately.
            elapsed = time.time() - (self._shutdown_start_time or 0)
            logger.error(
                "SignalSupervisor: second signal %d received "
                "(%.2fs into shutdown); FORCE EXIT",
                signum,
                elapsed,
            )
            os._exit(1)

    def _force_exit_timer(self) -> None:
        """Daemon thread that force-exits if shutdown takes too long."""
        time.sleep(self._max_shutdown_sec)
        # If we're still here, shutdown is hung; force-exit.
        if self._shutdown_started:  # nothing else clears this; double-check
            elapsed = time.time() - (self._shutdown_start_time or 0)
            logger.error(
                "SignalSupervisor: shutdown exceeded %.2fs (elapsed=%.2fs); "
                "FORCE EXIT",
                self._max_shutdown_sec,
                elapsed,
            )
            os._exit(1)


__all__ = ["SignalSupervisor", "DEFAULT_MAX_SHUTDOWN_SEC"]
