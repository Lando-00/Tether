"""Inbound-event inbox: ABC + SQLite-backed implementation.

Phase 6.5 step 66c (synthesis §4 + §13.4 M2; ADR-0009): the inbox
layer that connectors drain ``InboundEvent`` values into. Lives in
the shared ``data/tether.db`` (per ADR-0009 — deviates from connector
spec §3.6 by ratified user decision; one DSN, one connection, one
migration track).

Public surface (per connector spec §3.4 + §10):

* :meth:`InboundInbox.append_many` — idempotent insert
  (``INSERT OR IGNORE`` on ``(connector_id, event_id)``); validates
  ``payload`` JSON size and ``summary`` length against the
  Settings-resolved caps.
* :meth:`InboundInbox.list_unread` — events with ``inbox_seen=0``,
  ordered by ``received_at ASC`` then ``event_id ASC`` (stable when
  many events share the same ISO timestamp).
* :meth:`InboundInbox.list_recent` — newest events first, regardless
  of seen flag.
* :meth:`InboundInbox.mark_seen` — flips ``inbox_seen`` to 1 for the
  listed event ids; idempotent; returns affected count.
* :meth:`InboundInbox.prune_older_than` — deletes events older than
  ``now - retention_days``; returns deleted count.

The seen flag here is the **orchestrator-side** "we've shown this
event to a downstream consumer" flag — distinct from any future
``platform_read`` column that connectors-with-platform-read-state
(Gmail, WhatsApp Web) may add later. Connectors that need to mark
messages read on the upstream platform expose a separate
``*_inbox_mark_seen`` tool (synthesis §10.6).

Citations:

* Synthesis §4 Phase 6.5 step 66c, §13.4 M2.
* Connector spec §3.4 + §3.7 (size caps).
* ADR-0009 (shared-DB decision).
"""
from __future__ import annotations

import abc
import datetime
import json
from typing import List

from tether.connectors.types import InboundEvent
from tether.context._async_sqlite_base import AsyncSqliteStore
from tether.core.logging import logger

# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class InboundInbox(abc.ABC):
    """Inbound event storage contract.

    Concrete implementations persist :class:`InboundEvent` values
    drained from connector ``inbound_stream()`` generators by the
    drain task in :class:`tether.core.connector_registry.ConnectorRegistry`.
    """

    @abc.abstractmethod
    async def append_many(self, events: List[InboundEvent]) -> int:
        """Insert events idempotently.

        Returns the count of NEW rows persisted (rows that already
        existed at ``(connector_id, event_id)`` are silently skipped).
        Raises :class:`ValueError` when an event's payload exceeds the
        configured ``max_payload_bytes`` or its summary exceeds
        ``max_summary_chars``.
        """

    @abc.abstractmethod
    async def list_unread(
        self, connector_id: str, limit: int = 50
    ) -> List[InboundEvent]:
        """Events with ``inbox_seen = 0`` for ``connector_id``.

        Ordered by ``received_at ASC`` then ``event_id ASC`` so the
        order is stable when many events share the same timestamp.
        """

    @abc.abstractmethod
    async def list_recent(
        self, connector_id: str, limit: int = 50
    ) -> List[InboundEvent]:
        """All events for ``connector_id``, newest first."""

    @abc.abstractmethod
    async def mark_seen(
        self, connector_id: str, event_ids: List[str]
    ) -> int:
        """Mark events as seen by the orchestrator.

        Idempotent — events already at ``inbox_seen=1`` count toward
        the affected total only if SQLite reports them updated, which
        in practice it does via ``UPDATE ... WHERE inbox_seen=0``.
        Returns affected count.
        """

    @abc.abstractmethod
    async def prune_older_than(self, retention_days: int) -> int:
        """Delete events whose ``received_at`` is older than now-N days.

        Returns deleted count.
        """


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SqliteInbox(AsyncSqliteStore, InboundInbox):
    """SQLite-backed :class:`InboundInbox` using shared ``data/tether.db``.

    Per ADR-0009: shares the DB file with
    :class:`tether.context.sqlite_store.SqliteSessionStore` so there is
    one DSN, one connection lifecycle, and one migration track. The
    ``inbound_events`` table lives in migration 004; yoyo's
    ``_yoyo_*`` tracking handles "two stores apply the same migration
    set" idempotently.

    Args:
        dsn: The same ``sqlite:///`` DSN used by
            :class:`SqliteSessionStore`. Migrations 001-004 are
            applied at construction (idempotent — the migration runner
            caches per-DSN within the process and yoyo's tracking
            table guards against double-apply across restarts).
        max_payload_bytes: Cap on ``json.dumps(payload)`` bytes per
            row; default 64 KiB. Connectors are also expected to clamp
            before yielding events; this is defense-in-depth at the
            inbox layer (connector spec §3.4 + §3.7).
        max_summary_chars: Cap on ``summary`` characters per row;
            default 512.
    """

    def __init__(
        self,
        dsn: str,
        *,
        max_payload_bytes: int = 64_000,
        max_summary_chars: int = 512,
    ) -> None:
        # yoyo migrations 001-004 — idempotent, cached per-DSN per
        # process, safe to call alongside SqliteSessionStore on the
        # same DSN. Synthesis §3.6 (migration runner contract).
        from tether.context.migration_runner import (
            apply_pending_migrations,
        )
        apply_pending_migrations(dsn)

        # Lifecycle scaffolding (DSN parsing, parent dir, lock,
        # connection holder, finalizer) lives on AsyncSqliteStore —
        # extracted in Phase 6.5 step 66a per synthesis §13.4 M2.
        super().__init__(dsn)

        if max_payload_bytes < 1:
            raise ValueError(
                f"max_payload_bytes must be >= 1, got {max_payload_bytes}"
            )
        if max_summary_chars < 0:
            raise ValueError(
                f"max_summary_chars must be >= 0, got {max_summary_chars}"
            )
        self._max_payload_bytes = max_payload_bytes
        self._max_summary_chars = max_summary_chars

    # ------------------------------------------------------------------
    # InboundInbox
    # ------------------------------------------------------------------

    async def append_many(self, events: List[InboundEvent]) -> int:
        """Insert events idempotently. Returns count of NEW rows."""
        if not events:
            return 0
        # Validate sizes before opening the transaction so a bad event
        # doesn't even reach SQLite. Validation order: payload then
        # summary so the error message points at the first violation.
        rows: list[tuple] = []
        for ev in events:
            self._validate_event(ev)
            payload_json = json.dumps(ev.payload, sort_keys=True)
            received_at_iso = _iso(ev.received_at)
            rows.append(
                (
                    ev.connector_id,
                    ev.event_id,
                    ev.kind,
                    received_at_iso,
                    payload_json,
                    ev.summary,
                )
            )

        conn = await self._ensure_connected()
        # ``executemany`` is one round-trip per event but inside a
        # single transaction; the orchestrator drains events one at a
        # time today, but batch APIs (Phase 2b Gmail polling) will
        # call this with a list. ``INSERT OR IGNORE`` makes
        # ``(connector_id, event_id)`` collisions silent — exactly the
        # idempotency guarantee the spec wants.
        cur = await conn.executemany(
            "INSERT OR IGNORE INTO inbound_events"
            "(connector_id, event_id, kind, received_at, payload, summary)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        new_count = cur.rowcount if cur.rowcount is not None else 0
        await cur.close()
        await conn.commit()
        return max(new_count, 0)

    async def list_unread(
        self, connector_id: str, limit: int = 50
    ) -> List[InboundEvent]:
        """Unread events, oldest first (stable order)."""
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        conn = await self._ensure_connected()
        async with conn.execute(
            "SELECT connector_id, event_id, kind, received_at, payload, summary"
            " FROM inbound_events"
            " WHERE connector_id = ? AND inbox_seen = 0"
            " ORDER BY received_at ASC, event_id ASC"
            " LIMIT ?",
            (connector_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_event(r) for r in rows]

    async def list_recent(
        self, connector_id: str, limit: int = 50
    ) -> List[InboundEvent]:
        """All events for a connector, newest first."""
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        conn = await self._ensure_connected()
        async with conn.execute(
            "SELECT connector_id, event_id, kind, received_at, payload, summary"
            " FROM inbound_events"
            " WHERE connector_id = ?"
            " ORDER BY received_at DESC, event_id DESC"
            " LIMIT ?",
            (connector_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_event(r) for r in rows]

    async def mark_seen(
        self, connector_id: str, event_ids: List[str]
    ) -> int:
        """Flip ``inbox_seen`` to 1; returns affected count."""
        if not event_ids:
            return 0
        conn = await self._ensure_connected()
        # Build a placeholder string of the right length. SQLite has a
        # default 999-parameter limit so callers passing absurdly long
        # lists would 500; we clamp at the inbox layer rather than
        # silently truncating to keep the callsite obvious.
        if len(event_ids) > 900:
            raise ValueError(
                f"mark_seen: at most 900 event_ids per call (got {len(event_ids)})"
            )
        placeholders = ",".join("?" for _ in event_ids)
        cur = await conn.execute(
            f"UPDATE inbound_events SET inbox_seen = 1"
            f" WHERE connector_id = ? AND inbox_seen = 0"
            f" AND event_id IN ({placeholders})",
            (connector_id, *event_ids),
        )
        affected = cur.rowcount if cur.rowcount is not None else 0
        await cur.close()
        await conn.commit()
        return max(affected, 0)

    async def prune_older_than(self, retention_days: int) -> int:
        """Delete events older than ``now - retention_days``."""
        if retention_days < 0:
            raise ValueError(
                f"retention_days must be >= 0, got {retention_days}"
            )
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=retention_days
        )
        cutoff_iso = _iso(cutoff)
        conn = await self._ensure_connected()
        cur = await conn.execute(
            "DELETE FROM inbound_events WHERE received_at < ?",
            (cutoff_iso,),
        )
        deleted = cur.rowcount if cur.rowcount is not None else 0
        await cur.close()
        await conn.commit()
        if deleted > 0:
            logger.info(
                "SqliteInbox.prune_older_than(%d days): deleted %d events",
                retention_days,
                deleted,
            )
        return max(deleted, 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_event(self, ev: InboundEvent) -> None:
        """Enforce size caps. Raises :class:`ValueError` on violation."""
        # Payload byte size measured on the JSON-serialised bytes that
        # will land in SQLite; sort_keys=True makes the size stable
        # across re-orderings of the same dict.
        payload_bytes = len(
            json.dumps(ev.payload, sort_keys=True).encode("utf-8")
        )
        if payload_bytes > self._max_payload_bytes:
            raise ValueError(
                f"InboundEvent payload exceeds max_payload_bytes "
                f"({payload_bytes} > {self._max_payload_bytes}) for "
                f"connector_id={ev.connector_id!r} event_id={ev.event_id!r}"
            )
        if ev.summary is not None and len(ev.summary) > self._max_summary_chars:
            raise ValueError(
                f"InboundEvent summary exceeds max_summary_chars "
                f"({len(ev.summary)} > {self._max_summary_chars}) for "
                f"connector_id={ev.connector_id!r} event_id={ev.event_id!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime.datetime) -> str:
    """ISO-8601 UTC string. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.isoformat()


def _row_to_event(row) -> InboundEvent:
    """Convert an aiosqlite row to an :class:`InboundEvent`.

    ``row["received_at"]`` is parsed back to ``datetime``; ``payload``
    is JSON-decoded to a dict.
    """
    received_at = datetime.datetime.fromisoformat(row["received_at"])
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=datetime.timezone.utc)
    payload = json.loads(row["payload"]) if row["payload"] else {}
    return InboundEvent(
        event_id=row["event_id"],
        connector_id=row["connector_id"],
        kind=row["kind"],
        received_at=received_at,
        payload=payload,
        summary=row["summary"],
    )


__all__ = ["InboundInbox", "SqliteInbox"]
