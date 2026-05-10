"""SQLite-backed session store using aiosqlite for true async I/O.

Phase 6 step 63 (synthesis §3.6): the prior implementation wrapped
synchronous ``sqlite3`` calls inside ``async def`` methods, blocking
the event loop on every DB round-trip. This module replaces that with
``aiosqlite``, which dispatches each call to a background worker thread
so the loop remains free.

Lifecycle:

    store = SqliteSessionStore(dsn=...)   # sync; runs migrations only
    await store.connect()                  # opens aiosqlite + PRAGMAs
    ...                                    # use store
    await store.aclose()                   # closes the connection

``connect()`` is idempotent and lock-protected. ``aclose()`` is also
idempotent. For convenience the store auto-connects lazily on first
DB-touching method call, so callers that construct ``Engine`` directly
(without ``async with``) still work — but ``connect()`` is the
preferred explicit entry point and is what ``Engine.__aenter__``
invokes. Synthesis §3.6.

Schema is unchanged from prior step (v1 messages/sessions + v2
turns/tool_calls/raw_events). yoyo migrations still run synchronously
inside ``__init__`` against a short-lived stdlib ``sqlite3`` connection
— that's fine for a one-shot boot-time operation.

Important: ``aiosqlite.Connection`` starts a *non-daemon* worker
thread on first use. If a process exits without ``aclose()``-ing the
store, those threads block process termination. We register two
safety nets:

* ``weakref.finalize`` per store — pushes aiosqlite's STOP sentinel
  on garbage collection (catches forgotten ``aclose`` while the
  process is still running).
* Module-level ``atexit`` handler — at interpreter shutdown, walks
  the WeakSet of live stores and stops any still-open aiosqlite
  worker thread synchronously. Without this, leaked connections in
  test suites (Engine fixtures that don't use ``async with``) block
  pytest exit beyond the SignalSupervisor's force-exit budget.

Production paths still call ``aclose`` via ``Engine.aclose``; these
finalizers are safety nets only.
"""
from __future__ import annotations

import asyncio
import atexit
import datetime
import json
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from tether_service.core.interfaces import SessionStore
from tether_service.core.logging import logger


# Module-level WeakSet of live stores. Used by the atexit handler to
# stop any aiosqlite worker thread that survives until interpreter
# shutdown (e.g., tests that construct ``Engine.from_settings`` without
# ``async with`` and never call ``aclose``). WeakSet so a properly
# closed store is auto-removed when GC runs.
_LIVE_STORES: "weakref.WeakSet[SqliteSessionStore]" = weakref.WeakSet()


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


class SqliteSessionStore(SessionStore):
    """Async SQLite-backed session store.

    Construction does NOT open the aiosqlite connection — the DSN is
    parsed and yoyo migrations are applied synchronously, then the
    object is ready. Async work begins at :meth:`connect`.
    """

    def __init__(self, dsn: str):
        # Apply all pending migrations BEFORE any aiosqlite connection
        # is opened. yoyo is idempotent — calling on an already-current DB
        # is a no-op via its tracking table. Direct construction paths
        # (contract tests, CLI one-shots) get the latest schema
        # automatically. Synthesis §3.6.
        from tether_service.context.migration_runner import (
            apply_pending_migrations,
        )
        apply_pending_migrations(dsn)

        # Parse DSN — same shape as prior step.
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
            # PRAGMAs identical to the prior _init_pragmas: WAL journal
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

    async def __aenter__(self) -> "SqliteSessionStore":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_connected(self) -> aiosqlite.Connection:
        """Lazy-connect on first use.

        ``connect()`` is the preferred explicit entry point (and what
        ``Engine.__aenter__`` calls), but tests / library callers that
        construct an Engine without ``async with`` still expect the
        store to "just work" on the first method call. The lock makes
        the lazy path safe for concurrent first-use.
        """
        if self._conn is None:
            await self.connect()
        # connect() may have been called concurrently; re-check.
        assert self._conn is not None
        return self._conn

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str, created_at: int) -> None:
        """Create a new session."""
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            (session_id, datetime.datetime.fromtimestamp(created_at).isoformat(), "{}"),
        )
        await conn.commit()

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        conn = await self._ensure_connected()
        async with conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "session_id": r["id"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if deleted, False if not found.

        Deletes all child messages first.
        """
        conn = await self._ensure_connected()
        # Delete all messages for this session first.
        await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        # Now delete the session and capture rowcount on the same cursor.
        cur = await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        deleted_count = cur.rowcount
        await cur.close()
        await conn.commit()
        return deleted_count > 0

    async def delete_all_sessions(self) -> int:
        """Delete all sessions and child messages. Returns the count of
        deleted sessions."""
        conn = await self._ensure_connected()
        async with conn.execute("SELECT COUNT(*) FROM sessions") as cur:
            row = await cur.fetchone()
        count = row[0] if row else 0
        await conn.execute("DELETE FROM messages")
        await conn.execute("DELETE FROM sessions")
        await conn.commit()
        return count

    async def _ensure_session(self, session_id: str) -> None:
        conn = await self._ensure_connected()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        await conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            (session_id, now, "{}"),
        )
        await conn.commit()

    # ------------------------------------------------------------------
    # Message writes (v1 + opportunistic v2)
    # ------------------------------------------------------------------

    async def add_user(
        self,
        session_id: str,
        text: str,
        *,
        turn_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        conn = await self._ensure_connected()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        await conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, "user", text, ts),
        )
        await conn.commit()
        # turn_id/seq_start accepted for future step-65 reshape; not yet
        # written to v2 tables (messages.turn_id column not yet added).
        # Synthesis §3.6.
        if turn_id is not None:
            logger.debug(
                "add_user: turn_id=%s seq_start=%s (v2 write deferred to step 65)",
                turn_id, seq_start,
            )

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
        *,
        turn_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        conn = await self._ensure_connected()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        thinking_value = thinking_text if (save_thinking and thinking_text) else None
        await conn.execute(
            "INSERT INTO messages(session_id, role, content, thinking_text, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", text, thinking_value, ts),
        )
        await conn.commit()
        # turn_id/seq_start accepted for future step-65 reshape. Synthesis §3.6.
        if turn_id is not None:
            logger.debug(
                "add_assistant_text: turn_id=%s seq_start=%s (v2 write deferred to step 65)",
                turn_id, seq_start,
            )

    async def add_assistant_toolcall(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        conn = await self._ensure_connected()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # v1 messages row (unchanged shape — synthesis §3.6 anti-scope).
        await conn.execute(
            "INSERT INTO messages(session_id, role, tool_name, args, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "tool", tool_name, json.dumps(args or {}), ts),
        )
        # v2 tool_calls row when both turn_id and tool_call_id are provided.
        if turn_id is not None and tool_call_id is not None:
            await conn.execute(
                "INSERT OR IGNORE INTO tool_calls"
                "(tool_call_id, session_id, turn_id, name, arguments_json, status, call_seq)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    tool_call_id, session_id, turn_id, tool_name,
                    json.dumps(args or {}), "running", seq_start,
                ),
            )
        await conn.commit()

    async def add_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any,
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        seq_start: Optional[int] = None,
        status: str = "ok",
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        conn = await self._ensure_connected()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # v1 messages row (unchanged shape — synthesis §3.6 anti-scope).
        await conn.execute(
            "INSERT INTO messages(session_id, role, tool_name, result, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "tool_result", tool_name, json.dumps(result), ts),
        )
        # v2 tool_calls UPDATE when both ids are provided.
        if turn_id is not None and tool_call_id is not None:
            completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await conn.execute(
                "UPDATE tool_calls SET status=?, result_json=?, error_json=?,"
                " result_seq=?, completed_at=?, duration_ms=?"
                " WHERE tool_call_id=?",
                (
                    status, json.dumps(result), error,
                    seq_start, completed_at, duration_ms,
                    tool_call_id,
                ),
            )
        await conn.commit()

    # ------------------------------------------------------------------
    # History reconstruction (canonical shape — DO NOT change)
    # ------------------------------------------------------------------

    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        """Reconstruct the model-facing history from the messages table.

        Canonical shape (synthesis §3.6):
          - role=user/assistant/system → passthrough.
          - role=tool → assistant message with ``<<function_call>> {...}``.
          - role=tool_result → user message with ``Tool 'name' returned:\\n{...}``.

        This shape is verified by tests/contract/test_session_store_history_contract.py.
        """
        conn = await self._ensure_connected()
        async with conn.execute(
            "SELECT role, content, thinking_text, tool_name, args, result"
            " FROM messages WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        history: List[Dict[str, Any]] = []
        for r in rows:
            role = r["role"]
            if role in ("user", "assistant", "system"):
                content = r["content"] or ""
                if role == "assistant" and include_thinking:
                    thinking = r["thinking_text"] or ""
                    if thinking:
                        content = f"{thinking}{content}"
                history.append({"role": role, "content": content})
            elif role == "tool":
                # Assistant made a tool call - format as assistant message
                # with function_call syntax so the model sees its own prior call.
                tool_name = r["tool_name"]
                args = json.loads(r["args"] or "{}")
                tool_call_json = json.dumps({"name": tool_name, "arguments": args})
                content = f"<<function_call>> {tool_call_json}"
                history.append({"role": "assistant", "content": content})
            elif role == "tool_result":
                # Tool execution result - format as user message so the
                # model can see what it received from the tool.
                tool_name = r["tool_name"]
                result = json.loads(r["result"] or "{}")
                result_text = json.dumps(result, indent=2)
                content = f"Tool '{tool_name}' returned:\n{result_text}"
                history.append({"role": "user", "content": content})
        return history

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        """Seed an empty session with a system-role message."""
        conn = await self._ensure_connected()
        async with conn.execute(
            "SELECT COUNT(1) AS c FROM messages WHERE session_id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        count = row["c"] if row else 0
        if count == 0:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await conn.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, "system", prompt, ts),
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # v2 turn lifecycle (synthesis §3.6 + b1-persistence.md)
    # ------------------------------------------------------------------

    async def start_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        model_name: Optional[str] = None,
    ) -> None:
        """Insert a turns row in 'running' state.

        ``_ensure_session`` guarantees a parent sessions row even when
        the orchestrator calls this before the first message write.
        Synthesis §3.6.
        """
        await self._ensure_session(session_id)
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT OR IGNORE INTO turns(turn_id, session_id, model_name, status)"
            " VALUES (?, ?, ?, ?)",
            (turn_id, session_id, model_name, "running"),
        )
        await conn.commit()

    async def complete_turn(
        self,
        turn_id: str,
        *,
        status: str = "completed",
        stop_reason: Optional[str] = None,
        error_json: Optional[str] = None,
    ) -> None:
        """Stamp the turns row with completed_at + final status.

        ``status`` must satisfy the CHECK constraint:
        ``running | completed | failed | cancelled``. Synthesis §3.6.
        """
        conn = await self._ensure_connected()
        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        await conn.execute(
            "UPDATE turns SET status=?, stop_reason=?, completed_at=?, error_json=?"
            " WHERE turn_id=?",
            (status, stop_reason, completed_at, error_json, turn_id),
        )
        await conn.commit()

    async def record_raw_event(
        self,
        session_id: str,
        turn_id: str,
        seq: int,
        event_type: str,
        payload: Dict[str, Any],
        *,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Insert a raw_events row.

        ``UNIQUE(turn_id, seq)`` violations are logged at WARNING and
        swallowed; the replay log can tolerate sparse gaps.
        Synthesis §3.6.
        """
        import sqlite3 as _sqlite3
        conn = await self._ensure_connected()
        payload_json = json.dumps(payload, default=str)
        try:
            await conn.execute(
                "INSERT INTO raw_events"
                "(session_id, turn_id, seq, type, tool_call_id, payload_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_id, seq, event_type, tool_call_id, payload_json),
            )
            await conn.commit()
        except _sqlite3.IntegrityError:
            # aiosqlite re-raises stdlib sqlite3 exception types unchanged.
            logger.warning(
                "Duplicate raw_event skipped: turn_id=%s seq=%s", turn_id, seq
            )
