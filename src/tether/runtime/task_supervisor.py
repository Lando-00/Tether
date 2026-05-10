"""Structured-concurrency wrapper for long-running async tasks.

Synthesis §13.4 M3: connector drain tasks (Phase 6.5 step 66e) need a
shared lifecycle:

* **start** — schedule the coroutine on the running event loop.
* **stop**  — cancel the task, ``await`` up to ``timeout`` seconds for
  a cooperative exit, then abandon the task and return.
* **cleanup** — ensure the task is not leaked into asyncio's
  "Task was destroyed but it is pending!" warning at process exit.

The same shape will be reused by future Gmail polling (Phase 2b) and
any other long-lived asyncio coroutine the engine owns. We deliberately
keep this a 30-line wrapper rather than pulling in ``anyio`` or a full
``asyncio.TaskGroup`` — neither buys us anything for a single coroutine
with a single owner.

Lifecycle rules:

* :meth:`start` is idempotent — a second call while running is a logged
  no-op. Calling :meth:`start` after :meth:`stop` re-arms the
  ``coro_factory`` and runs it again (rare; useful for "restart the
  drain after re-auth").
* :meth:`stop` is idempotent — a second call after the task has
  already completed (or was never started) is a no-op.
* The factory pattern (``Callable[[], Awaitable]``) is required because
  coroutines are one-shot — a second :meth:`start` cannot await the
  same coroutine object. Callers that don't need restart can pass
  ``lambda: my_coro()``.

Citations:

* Synthesis §13.4 M3.
* Connector spec §3.4 (drain task lifecycle).
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from tether.core.logging import logger


class SupervisedTask:
    """Lifecycle wrapper for one long-running asyncio task.

    Args:
        coro_factory: Zero-arg callable producing the coroutine to run.
            A factory (rather than a coroutine) is required so
            :meth:`start` can be called more than once over the
            wrapper's lifetime — coroutines are one-shot.
        name: Human-readable label for logs and ``Task.set_name``.
    """

    def __init__(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        name: str,
    ) -> None:
        self.name = name
        self._coro_factory = coro_factory
        self._task: Optional[asyncio.Task[Any]] = None

    def start(self) -> None:
        """Schedule the task on the running event loop. Idempotent."""
        if self._task is not None and not self._task.done():
            logger.debug(
                "SupervisedTask(%s).start: already running; no-op", self.name
            )
            return
        # Construct a fresh coroutine on each start (coroutines are
        # one-shot; a re-start after stop must call the factory again).
        coro = self._coro_factory()
        self._task = asyncio.create_task(coro, name=self.name)

    async def stop(self, *, timeout: float = 2.0) -> None:
        """Cancel + await up to ``timeout`` seconds, then abandon. Idempotent.

        After ``stop`` returns the task may still be alive in the
        background (the abandoned-after-timeout case), but the wrapper
        considers itself stopped — the next :meth:`start` will create a
        new task and overwrite the slot. ``CancelledError`` and any
        exception raised by the task body are caught + logged so a
        misbehaving drain task cannot break the engine's shutdown.

        Phase 6.5 (synthesis §4 step 66e): connectors get a 2-second
        cooperative budget per the connector spec §3.3 step 6 contract.
        """
        if self._task is None:
            return
        if self._task.done():
            # Already finished (success, exception, or prior cancel).
            self._task = None
            return

        self._task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task), timeout=timeout
            )
        except asyncio.CancelledError:
            # Expected — the cancel propagated cleanly through the
            # coroutine. Logged at DEBUG so it doesn't pollute the
            # normal-shutdown log.
            logger.debug(
                "SupervisedTask(%s) cancelled cleanly", self.name
            )
        except asyncio.TimeoutError:
            logger.warning(
                "SupervisedTask(%s) did not exit within %.1fs of cancel; "
                "abandoning task and continuing shutdown.",
                self.name,
                timeout,
            )
        except Exception as exc:  # noqa: BLE001 - logged + swallowed
            # The drain task raised something other than CancelledError.
            # We still consider stop() complete because the task is no
            # longer running.
            logger.exception(
                "SupervisedTask(%s) raised during stop: %s",
                self.name,
                exc,
            )
        finally:
            self._task = None

    @property
    def done(self) -> bool:
        """``True`` once the task has finished (success, error, or cancel)."""
        return self._task is None or self._task.done()


__all__ = ["SupervisedTask"]
