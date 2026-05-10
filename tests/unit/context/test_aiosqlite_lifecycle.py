"""SqliteSessionStore async lifecycle: connect / aclose / __aenter__ / __aexit__.

Phase 6 step 63 (synthesis §3.6): the store now uses aiosqlite for
true async I/O. Lifecycle invariants this file pins down:

* ``__init__`` does NOT open the underlying connection (DSN parsed +
  yoyo migrations applied; aiosqlite stays untouched).
* ``connect()`` is idempotent — calling it twice yields the same
  Connection object; no second open.
* ``aclose()`` is idempotent — calling it after no connection or
  twice in a row is safe.
* ``async with SqliteSessionStore(...) as store`` opens on entry and
  closes on exit.
* PRAGMAs (``WAL`` / ``synchronous=NORMAL`` / ``foreign_keys=ON``)
  match the prior sync ``_init_pragmas`` exactly.
* aiosqlite worker thread is set to daemon BEFORE start, so a leaked
  connection cannot block process exit (the test-suite exit guarantee
  baked into pyproject.toml's ``-W error`` sweep).

Note: ``asyncio_mode = "auto"`` in pyproject.toml — every async test
function and fixture is auto-handled by pytest-asyncio. No anyio
marker needed.
"""
from __future__ import annotations

import pytest

from tether_service.context.sqlite_store import SqliteSessionStore


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


async def test_construct_does_not_open_connection(tmp_path):
    """``__init__`` stores the DSN + applies migrations but DOES NOT
    open the aiosqlite connection."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqliteSessionStore(dsn=dsn)
    assert store._conn is None
    # No active aiosqlite worker; nothing to aclose.


# ---------------------------------------------------------------------------
# connect() idempotency
# ---------------------------------------------------------------------------


async def test_connect_opens_connection(tmp_path):
    """``connect()`` opens the aiosqlite connection."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqliteSessionStore(dsn=dsn)
    try:
        assert store._conn is None
        await store.connect()
        assert store._conn is not None
    finally:
        await store.aclose()


async def test_connect_idempotent(tmp_path):
    """``connect()`` can be awaited multiple times safely; the second
    call is a no-op and returns the same Connection."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqliteSessionStore(dsn=dsn)
    try:
        await store.connect()
        first_conn = store._conn

        await store.connect()
        second_conn = store._conn

        assert first_conn is second_conn  # same object; no re-open
    finally:
        await store.aclose()


# ---------------------------------------------------------------------------
# aclose() idempotency
# ---------------------------------------------------------------------------


async def test_aclose_without_connect_is_noop(tmp_path):
    """``aclose()`` on a never-connected store is a clean no-op."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqliteSessionStore(dsn=dsn)
    await store.aclose()  # no error
    assert store._conn is None


async def test_aclose_idempotent(tmp_path):
    """``aclose()`` can be awaited multiple times safely."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqliteSessionStore(dsn=dsn)

    await store.connect()
    await store.aclose()
    await store.aclose()  # no error

    assert store._conn is None


# ---------------------------------------------------------------------------
# Lazy auto-connect on first use
# ---------------------------------------------------------------------------


async def test_methods_auto_connect_on_first_use(tmp_path):
    """A method called on a never-connect()'d store transparently
    auto-connects.

    ``connect()`` remains the preferred explicit entry point and is
    what ``Engine.__aenter__`` calls; the lazy path is the convenience
    layer that lets direct-constructor tests skip ``async with``.
    """
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqliteSessionStore(dsn=dsn)
    try:
        assert store._conn is None
        await store.add_user("s-auto", "hello")
        # Auto-connect must have populated _conn.
        assert store._conn is not None
        history = await store.get_history("s-auto")
        assert history == [{"role": "user", "content": "hello"}]
    finally:
        await store.aclose()


# ---------------------------------------------------------------------------
# Async-context-manager support
# ---------------------------------------------------------------------------


async def test_async_context_manager_lifecycle(tmp_path):
    """``async with SqliteSessionStore(...)`` opens on enter, closes on exit."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"

    async with SqliteSessionStore(dsn=dsn) as store:
        assert store._conn is not None
        await store.add_user("s1", "hi")
        history = await store.get_history("s1")
        assert len(history) == 1

    # __aexit__ must have closed the connection.
    assert store._conn is None


# ---------------------------------------------------------------------------
# PRAGMAs run on connect()
# ---------------------------------------------------------------------------


async def test_pragmas_set_after_connect(tmp_path):
    """``connect()`` applies ``PRAGMA journal_mode=WAL``, ``synchronous=NORMAL``,
    ``foreign_keys=ON`` — the same set as the prior sync ``_init_pragmas``."""
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    async with SqliteSessionStore(dsn=dsn) as store:
        async with store._conn.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
        assert row[0].lower() == "wal"

        async with store._conn.execute("PRAGMA synchronous") as cur:
            row = await cur.fetchone()
        # synchronous=NORMAL → integer 1.
        assert row[0] == 1

        async with store._conn.execute("PRAGMA foreign_keys") as cur:
            row = await cur.fetchone()
        # foreign_keys=ON → integer 1.
        assert row[0] == 1


# ---------------------------------------------------------------------------
# Daemon thread guarantee (process-exit safety)
# ---------------------------------------------------------------------------


async def test_aiosqlite_worker_thread_is_daemon(tmp_path):
    """The aiosqlite worker thread MUST be daemon so a leaked store
    cannot block process exit.

    aiosqlite ships the thread as non-daemon by default; we set
    ``daemon=True`` BEFORE the thread is started inside
    ``Connection.__await__``. Once the thread is running, the daemon
    flag is read-only — so this test would fail loudly if the
    pre-await injection broke (e.g., aiosqlite renames ``_thread``).
    """
    dsn = f"sqlite:///{tmp_path / 'test.db'}"
    async with SqliteSessionStore(dsn=dsn) as store:
        thread = store._conn._thread  # type: ignore[attr-defined]
        assert thread.is_alive(), "worker thread should be running"
        assert thread.daemon is True, (
            "aiosqlite worker thread MUST be daemon so leaked stores "
            "cannot block process exit"
        )
