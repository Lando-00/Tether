"""CancelToken Protocol for orchestrator cancellation.

Synthesis §11.3 R7: replaces the ``asyncio.Event``-typed cancel parameter
with a Protocol so the public API is loop-implementation-agnostic
(Trio-friendly later).

The orchestrator polls ``cancel_token.cancelled()`` at chunk boundaries
and during tool dispatch. Adapters can wrap any "is cancelled" signal:

  - :class:`AsyncEventCancelToken` wraps ``asyncio.Event`` for back-compat
    with ``Engine.stream(cancel_event=...)``.
  - Future: ``TrioCancelToken``, ``threading.Event`` adapter, etc.

Synthesis §3.4 (Library mode) + §3.5 (cancellation contract) +
§11.3 R7 (CancelToken Protocol introduction).
"""
from __future__ import annotations

import asyncio
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CancelToken(Protocol):
    """Read-only cancellation signal.

    The orchestrator calls :meth:`cancelled` repeatedly during streaming
    and tool dispatch. Setting cancellation is the caller's responsibility
    (the token is set externally — e.g., HTTP client disconnect handler
    or library user calling ``token.set()``).
    """

    def cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        ...


class AsyncEventCancelToken:
    """:class:`CancelToken` backed by ``asyncio.Event``.

    Used internally so ``chat.py``'s existing
    ``asyncio.Event + request.is_disconnected()`` wiring keeps working
    without invasive HTTP-layer changes during the Phase 5 migration.
    """

    def __init__(self, event: Optional[asyncio.Event] = None):
        self._event = event if event is not None else asyncio.Event()

    def cancelled(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        """Set the underlying event (cancellation requested)."""
        self._event.set()

    @property
    def event(self) -> asyncio.Event:
        """Expose the underlying event for callers that need direct access."""
        return self._event


__all__ = ["CancelToken", "AsyncEventCancelToken"]
