"""Shared aiosqlite + WAL base for ``tether.context.*`` SQLite stores.

Synthesis §13.4 M2: extracted from ``sqlite_store.py`` during Phase 6.5
when :class:`tether.context.inbox_store.SqliteInbox` needed the same
lifecycle scaffolding. Both :class:`tether.context.sqlite_store.SqliteSessionStore`
and :class:`tether.context.inbox_store.SqliteInbox` subclass
:class:`AsyncSqliteStore` so we don't duplicate connection management,
finalizers, and atexit safety nets.

Lifecycle::

    store = ConcreteSubclass(dsn=...)   # sync; subclass runs migrations
    await store.connect()                # opens aiosqlite + PRAGMAs
    ...                                  # subclass-specific ops
    await store.aclose()                 # closes the connection

``connect()`` is idempotent and lock-protected. ``aclose()`` is also
idempotent. Subclass methods use :meth:`_ensure_connected` to lazily
auto-connect on first call so direct-constructor tests / library
callers that skip ``async with`` still work.

``aiosqlite.Connection`` starts a *non-daemon* worker thread on first
use. If a process exits without ``aclose``-ing, those threads block
process termination. This module installs two safety nets:

* :func:`weakref.finalize` per store — pushes aiosqlite's STOP sentinel
  on garbage collection (catches forgotten ``aclose`` while the
  process is still running).
* Module-level :func:`atexit` handler — at interpreter shutdown, walks
  the WeakSet of live stores and stops any still-open aiosqlite worker
  thread synchronously. Without this, leaked connections in test
  suites block pytest exit beyond the SignalSupervisor's force-exit
  budget.

Production paths still call ``aclose`` via ``Engine.aclose``; these
finalizers are safety nets only.

The constructor parses the DSN and ensures the parent directory exists
but does NOT run migrations. Each subclass owns its migration set:
``SqliteSessionStore`` runs the v1+v2+tool_audit migrations,
``SqliteInbox`` runs the inbox migration. yoyo-migrations is
idempotent and tracks applied versions in ``_yoyo_*`` tables so two
subclasses sharing the same DB file never collide.
"""
from __future__ import annotations

import asyncio
import atexit
import weakref
from pathlib import Path
from typing import Optional

import aiosqlite

from tether.core.logging import logger

# Module-level WeakSet of live stores. Used by the atexit handler to
# stop any aiosqlite worker thread that survives until interpreter
# shutdown (e.g., tests that construct an Engine without ``async with``
# and never call ``aclose``). WeakSet so a properly closed store is
# auto-removed when GC runs.
_LIVE_STORES: "weakref.WeakSet[AsyncSqliteStore]" = weakref.WeakSet()


def _stop_aiosqlite_worker_sync(conn: aiosqlite.Connection) -> None:
    """Push aiosqlite's STOP sentinel onto the worker queue synchronously.

    Touches private aiosqlite internals; gated on attribute presence
    so a future aiosqlite refactor degrades to a logged no-op rather
    than an exception.
    """
    try:
        from aiosqlite.core import _STOP_RUNNING_SENTINEL  # type: ignore[attr-defined]

        if not getattr(conn, "_running", False):
            return
        conn._running = False  # type: ignore[attr-defined]
        # ``_tx`` is a SimpleQueue; ``put_nowait`` is safe from any
        # thread, including the GC thread and the atexit thread.
        conn._tx.put_nowait((None, lambda: _STOP_RUNNING_SENTINEL))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - finalizer must never raise
        # Logged at DEBUG because this fires only on the leaked-store
        # path — production code always calls aclose explicitly.
        logger.debug(
            "Could not stop aiosqlite worker via private API",
            exc_info=True,
        )


def _emergency_close_aiosqlite(conn_ref: "weakref.ref[aiosqlite.Connection]") -> None:
    """``weakref.finalize`` callback for a store's aiosqlite connection."""
    conn = conn_ref()
    if conn is None:
        return
    _stop_aiosqlite_worker_sync(conn)


def _atexit_close_all() -> None:
    """Stop every still-live aiosqlite worker at interpreter shutdown.

    Iterates a snapshot of the WeakSet so concurrent finalization can't
    mutate it under us. Each connection is signalled to stop; the
    non-daemon worker thread exits, allowing the process to terminate.
    """
    # ``list(...)`` snapshots the WeakSet; iterating it directly while
    # finalizers fire would raise RuntimeError.
    for store in list(_LIVE_STORES):
        conn = getattr(store, "_conn", None)
        if conn is not None:
            _stop_aiosqlite_worker_sync(conn)


atexit.register(_atexit_close_all)


class AsyncSqliteStore:
    """Async SQLite store base — aiosqlite + WAL + lifecycle.

    Subclass-specific schema (yoyo migrations) and CRUD ops live in the
    subclass. Construction is sync and only parses the DSN + ensures
    the parent directory exists; the aiosqlite connection opens on
    :meth:`connect` (or lazily via :meth:`_ensure_connected` on first
    method call).
    """

    def __init__(self, dsn: str) -> None:
        # Parse the DSN — same shape as the legacy SqliteSessionStore
        # so existing callers keep working.
        if dsn.startswith("sqlite:///"):
            path = dsn[len("sqlite:///"):]
        else:
            path = dsn

        # Ensure parent directory exists; aiosqlite.connect won't create
        # missing parent dirs and would error noisily otherwise.
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        self._dsn = dsn
        self._path = str(p)
        # Connection holder; populated by connect(). Stays None when
        # aclose() runs so a second aclose() is a no-op.
        self._conn: Optional[aiosqlite.Connection] = None
        # Lock for connect()/aclose() so concurrent fixture setup or
        # lazy auto-connects don't race.
        self._lifecycle_lock = asyncio.Lock()
        # Finalizer slot — populated by connect() so the safety-net
        # only fires when an aiosqlite worker thread is actually live.
        self._finalizer: Optional[weakref.finalize] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the aiosqlite connection and apply PRAGMAs. Idempotent."""
        async with self._lifecycle_lock:
            if self._conn is not None:
                return

            # ``aiosqlite.connect()`` returns a Connection whose worker
            # thread is *created but not yet started*; the thread is
            # only started inside ``Connection.__await__``. We mark the
            # thread daemon BEFORE awaiting so a leaked store (test
            # forgetting ``aclose``) cannot hang process exit. Once the
            # thread is running, ``Thread.daemon`` is read-only.
            conn = aiosqlite.connect(self._path)
            try:
                conn._thread.daemon = True  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - degrade gracefully
                # If a future aiosqlite refactor renames the attribute,
                # we still get a working connection — the atexit /
                # finalize safety nets below cover the leak path.
                logger.debug(
                    "Could not set aiosqlite worker thread daemon=True",
                    exc_info=True,
                )
            conn = await conn
            conn.row_factory = aiosqlite.Row
            # PRAGMAs identical to the legacy _init_pragmas: WAL journal
            # for concurrent readers, NORMAL fsync for performance,
            # foreign_keys ON so the v2 FK constraints actually fire.
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA synchronous=NORMAL;")
            await conn.execute("PRAGMA foreign_keys=ON;")
            await conn.commit()
            self._conn = conn
            # Safety net 1: GC-triggered finalizer per store.
            self._finalizer = weakref.finalize(
                self, _emergency_close_aiosqlite, weakref.ref(conn)
            )
            # Safety net 2: register in the module-level WeakSet so
            # the atexit handler can find this connection if neither
            # aclose nor GC fires before interpreter shutdown.
            _LIVE_STORES.add(self)

    async def aclose(self) -> None:
        """Close the connection. Idempotent."""
        async with self._lifecycle_lock:
            if self._conn is None:
                return
            try:
                await self._conn.close()
            finally:
                self._conn = None
                # The finalizer's job is now done; detach it so the GC
                # path doesn't try to stop an already-stopped worker.
                if self._finalizer is not None:
                    self._finalizer.detach()
                    self._finalizer = None
                # WeakSet auto-removes on GC, but discard now to keep
                # the atexit walk small.
                _LIVE_STORES.discard(self)

    async def __aenter__(self) -> "AsyncSqliteStore":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_connected(self) -> aiosqlite.Connection:
        """Lazy-connect on first use.

        :meth:`connect` is the preferred explicit entry point (and what
        ``Engine.__aenter__`` calls), but tests / library callers that
        construct a store without ``async with`` still expect it to
        "just work" on the first method call. The lock makes the lazy
        path safe for concurrent first-use.
        """
        if self._conn is None:
            await self.connect()
        # connect() may have been called concurrently; re-check.
        assert self._conn is not None
        return self._conn


__all__ = ["AsyncSqliteStore"]
