"""Unit tests for :class:`tether.context.inbox_store.SqliteInbox`.

Phase 6.5 step 66h (synthesis §4): exercise the public surface of the
inbox layer end-to-end against a real SQLite file. ``pytest-asyncio``
``mode='auto'`` (configured in pyproject.toml) means each ``async def``
test runs on its own loop.

Coverage matrix:

* ``append_many`` — happy path, idempotent, payload-size cap, summary
  cap, empty list.
* ``list_unread`` — ordering (received_at ASC, event_id ASC), per-
  connector isolation, limit clamp.
* ``list_recent`` — newest-first ordering.
* ``mark_seen`` — flips the flag, idempotent on already-seen ids,
  returns affected count, large-list rejection.
* ``prune_older_than`` — deletes by cutoff, returns count, retains
  recent rows.
"""
from __future__ import annotations

import datetime
import json

import pytest

from tether.connectors.types import InboundEvent
from tether.context.inbox_store import SqliteInbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def inbox(tmp_path):
    """Fresh :class:`SqliteInbox` rooted at ``tmp_path/inbox.db``."""
    db_path = (tmp_path / "inbox.db").as_posix()
    store = SqliteInbox(f"sqlite:///{db_path}")
    await store.connect()
    try:
        yield store
    finally:
        await store.aclose()


def _event(
    *,
    event_id: str,
    connector_id: str = "echo",
    kind: str = "echo.test",
    payload=None,
    summary: str | None = None,
    received_at=None,
) -> InboundEvent:
    return InboundEvent(
        event_id=event_id,
        connector_id=connector_id,
        kind=kind,
        received_at=received_at
        or datetime.datetime.now(datetime.timezone.utc),
        payload=payload if payload is not None else {"i": 0},
        summary=summary,
    )


# ---------------------------------------------------------------------------
# append_many
# ---------------------------------------------------------------------------


async def test_append_many_inserts_new_rows(inbox):
    n = await inbox.append_many([_event(event_id="e1"), _event(event_id="e2")])
    assert n == 2
    rows = await inbox.list_recent("echo")
    assert len(rows) == 2


async def test_append_many_empty_list_is_noop(inbox):
    n = await inbox.append_many([])
    assert n == 0


async def test_append_many_is_idempotent_on_duplicate_event_ids(inbox):
    e1 = _event(event_id="dup")
    n1 = await inbox.append_many([e1])
    assert n1 == 1

    # Second insert with the SAME (connector_id, event_id) — ignored.
    e1_redux = _event(event_id="dup", payload={"i": 99})
    n2 = await inbox.append_many([e1_redux])
    assert n2 == 0

    # The original payload survives — INSERT OR IGNORE keeps the first row.
    rows = await inbox.list_recent("echo")
    assert len(rows) == 1
    assert rows[0].payload == {"i": 0}


async def test_append_many_raises_when_payload_exceeds_cap(tmp_path):
    db_path = (tmp_path / "inbox.db").as_posix()
    store = SqliteInbox(
        f"sqlite:///{db_path}", max_payload_bytes=128, max_summary_chars=512
    )
    try:
        await store.connect()
        # Build a payload whose JSON is well over 128 bytes.
        big_payload = {"data": "x" * 400}
        ev = _event(event_id="big", payload=big_payload)
        with pytest.raises(ValueError, match="max_payload_bytes"):
            await store.append_many([ev])
    finally:
        await store.aclose()


async def test_append_many_raises_when_summary_exceeds_cap(tmp_path):
    db_path = (tmp_path / "inbox.db").as_posix()
    store = SqliteInbox(
        f"sqlite:///{db_path}",
        max_payload_bytes=64_000,
        max_summary_chars=10,
    )
    try:
        await store.connect()
        ev = _event(event_id="long-summary", summary="this is way too long")
        with pytest.raises(ValueError, match="max_summary_chars"):
            await store.append_many([ev])
    finally:
        await store.aclose()


async def test_append_many_round_trips_payload_summary(inbox):
    """Stored payload + summary survive the JSON round-trip unchanged."""
    nested = {"outer": {"inner": [1, 2, 3]}, "bool": True, "null": None}
    ev = _event(event_id="round", payload=nested, summary="hello")
    await inbox.append_many([ev])
    rows = await inbox.list_recent("echo")
    assert len(rows) == 1
    assert rows[0].payload == nested
    assert rows[0].summary == "hello"


# ---------------------------------------------------------------------------
# list_unread / list_recent
# ---------------------------------------------------------------------------


async def test_list_unread_ordered_received_at_asc_event_id_asc(inbox):
    """Stable order: received_at ASC, then event_id ASC for ties."""
    base = datetime.datetime(2026, 5, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
    # Same timestamp for e1/e2 — event_id breaks the tie alphabetically.
    e1 = _event(event_id="e1", received_at=base)
    e2 = _event(event_id="e2", received_at=base)
    e3 = _event(
        event_id="e3", received_at=base + datetime.timedelta(seconds=10)
    )
    # Insert in non-sorted order on purpose.
    await inbox.append_many([e3, e2, e1])

    rows = await inbox.list_unread("echo")
    assert [r.event_id for r in rows] == ["e1", "e2", "e3"]


async def test_list_unread_per_connector_isolation(inbox):
    """Events from different connectors do not bleed into each other's view."""
    await inbox.append_many(
        [
            _event(event_id="a1", connector_id="alpha"),
            _event(event_id="b1", connector_id="bravo"),
        ]
    )
    alpha = await inbox.list_unread("alpha")
    bravo = await inbox.list_unread("bravo")
    assert [r.event_id for r in alpha] == ["a1"]
    assert [r.event_id for r in bravo] == ["b1"]


async def test_list_unread_excludes_already_seen_events(inbox):
    e1 = _event(event_id="seen")
    e2 = _event(event_id="unseen")
    await inbox.append_many([e1, e2])
    await inbox.mark_seen("echo", ["seen"])

    unread = await inbox.list_unread("echo")
    assert [r.event_id for r in unread] == ["unseen"]


async def test_list_unread_respects_limit(inbox):
    base = datetime.datetime(2026, 5, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
    events = [
        _event(event_id=f"e{i}", received_at=base + datetime.timedelta(seconds=i))
        for i in range(5)
    ]
    await inbox.append_many(events)
    rows = await inbox.list_unread("echo", limit=2)
    assert len(rows) == 2
    # Oldest 2 events.
    assert [r.event_id for r in rows] == ["e0", "e1"]


async def test_list_recent_newest_first(inbox):
    base = datetime.datetime(2026, 5, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
    events = [
        _event(event_id=f"e{i}", received_at=base + datetime.timedelta(seconds=i))
        for i in range(3)
    ]
    await inbox.append_many(events)
    rows = await inbox.list_recent("echo")
    assert [r.event_id for r in rows] == ["e2", "e1", "e0"]


# ---------------------------------------------------------------------------
# mark_seen
# ---------------------------------------------------------------------------


async def test_mark_seen_returns_affected_count(inbox):
    await inbox.append_many(
        [_event(event_id="m1"), _event(event_id="m2"), _event(event_id="m3")]
    )
    affected = await inbox.mark_seen("echo", ["m1", "m2"])
    assert affected == 2

    unread = await inbox.list_unread("echo")
    assert [r.event_id for r in unread] == ["m3"]


async def test_mark_seen_is_idempotent(inbox):
    await inbox.append_many([_event(event_id="m1")])
    first = await inbox.mark_seen("echo", ["m1"])
    second = await inbox.mark_seen("echo", ["m1"])
    assert first == 1
    # Second call: row already inbox_seen=1; UPDATE WHERE inbox_seen=0
    # matches no rows.
    assert second == 0


async def test_mark_seen_unknown_event_id_returns_zero(inbox):
    affected = await inbox.mark_seen("echo", ["nonexistent"])
    assert affected == 0


async def test_mark_seen_empty_list_is_noop(inbox):
    affected = await inbox.mark_seen("echo", [])
    assert affected == 0


async def test_mark_seen_rejects_oversize_id_list(inbox):
    huge = [f"e{i}" for i in range(901)]
    with pytest.raises(ValueError, match="at most 900"):
        await inbox.mark_seen("echo", huge)


# ---------------------------------------------------------------------------
# prune_older_than
# ---------------------------------------------------------------------------


async def test_prune_older_than_deletes_old_events(inbox):
    now = datetime.datetime.now(datetime.timezone.utc)
    old = _event(event_id="old", received_at=now - datetime.timedelta(days=40))
    fresh = _event(event_id="fresh", received_at=now - datetime.timedelta(days=5))
    await inbox.append_many([old, fresh])

    deleted = await inbox.prune_older_than(retention_days=30)
    assert deleted == 1
    rows = await inbox.list_recent("echo")
    assert [r.event_id for r in rows] == ["fresh"]


async def test_prune_older_than_zero_days_deletes_all(inbox):
    now = datetime.datetime.now(datetime.timezone.utc)
    await inbox.append_many(
        [
            _event(event_id="a", received_at=now - datetime.timedelta(seconds=10)),
            _event(event_id="b", received_at=now - datetime.timedelta(seconds=5)),
        ]
    )
    deleted = await inbox.prune_older_than(retention_days=0)
    # Both rows are older than ``now`` (cutoff = now - 0 days = now).
    # SQLite ISO-8601 string compare excludes anything strictly less
    # than now; rows inserted moments ago are < now, so all deleted.
    assert deleted == 2


async def test_prune_older_than_negative_raises(inbox):
    with pytest.raises(ValueError, match="retention_days"):
        await inbox.prune_older_than(retention_days=-1)


async def test_prune_older_than_no_events_returns_zero(inbox):
    deleted = await inbox.prune_older_than(retention_days=30)
    assert deleted == 0
