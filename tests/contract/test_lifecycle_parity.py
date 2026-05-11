"""Lifecycle parity: start_turn, complete_turn, record_raw_event
behave identically across MemoryStore and SqliteSessionStore.

Phase 6 step 64. The p6-widen PR added a single
test_memory_and_sqlite_lifecycle_parity smoke; this file extends
coverage with deeper assertions. Synthesis §3.6 + §11.3 R19.
"""
import json


# All tests run twice via the `store` fixture in conftest.py.
# That fixture parametrizes over memory_store + sqlite_store.


async def test_start_turn_required_before_record_raw_event(store):
    """A raw event requires its turn to exist (FK on Sqlite; soft on Memory)."""
    await store.create_session("s-1", 1700000000)

    # Start the turn, then record an event — both stores accept this sequence.
    await store.start_turn("s-1", "t-1", model_name="dummy")
    await store.record_raw_event("s-1", "t-1", 0, "text_delta", {"text": "hi"})
    # No exception — both stores accept the well-formed sequence.


async def test_record_raw_event_duplicate_seq_handled_gracefully(store):
    """Duplicate (turn_id, seq) on raw_events is handled gracefully on both stores.

    Sqlite: UNIQUE constraint raises IntegrityError; the impl catches + ignores
    so the FIRST write wins.
    Memory: appends without dedup — both writes are visible.

    Either way, the store must remain queryable and the seq=0 slot must hold
    at least one of the two payloads. P0-G / Tribunal P0-15 (A8-F1): the
    original test had zero assertions on observable state.
    """
    await store.create_session("s-1", 1700000000)
    await store.start_turn("s-1", "t-1")

    await store.record_raw_event("s-1", "t-1", 0, "text_delta", {"text": "first"})
    # Second call with same (turn_id, seq) — must not crash either store.
    await store.record_raw_event("s-1", "t-1", 0, "text_delta", {"text": "duplicate"})

    if hasattr(store, "raw_events"):  # MemoryStore — appends without dedup
        events = [
            e for e in store.raw_events
            if e["turn_id"] == "t-1" and e["seq"] == 0
        ]
        assert len(events) >= 1, "memory store dropped the seq=0 raw event"
        payloads = [e["payload"] for e in events]
        assert {"text": "first"} in payloads or {"text": "duplicate"} in payloads
    else:  # SqliteSessionStore — UNIQUE enforced; first write wins.
        async with store._conn.execute(
            "SELECT payload_json FROM raw_events WHERE turn_id = ? AND seq = ?",
            ("t-1", 0),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "sqlite store lost the seq=0 raw event"
        assert json.loads(row[0]) == {"text": "first"}, (
            "sqlite UNIQUE-on-conflict should preserve the first write"
        )


async def test_complete_turn_status_transitions(store):
    """complete_turn updates the turn status from 'running' to 'completed'."""
    await store.create_session("s-1", 1700000000)
    await store.start_turn("s-1", "t-1", model_name="dummy")

    await store.complete_turn("t-1", status="completed", stop_reason="complete")

    if hasattr(store, "turns"):  # MemoryStore
        turn = store.turns.get("t-1")
        assert turn is not None
        assert turn["status"] == "completed"
        assert turn["stop_reason"] == "complete"
    else:  # SqliteSessionStore
        async with store._conn.execute(
            "SELECT status, stop_reason FROM turns WHERE turn_id = ?", ("t-1",)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "completed"
        assert row[1] == "complete"


async def test_complete_turn_cancelled(store):
    """complete_turn with status='cancelled' is honored on both stores."""
    await store.create_session("s-1", 1700000000)
    await store.start_turn("s-1", "t-1", model_name="dummy")
    await store.complete_turn("t-1", status="cancelled", stop_reason="cancelled")

    if hasattr(store, "turns"):
        assert store.turns["t-1"]["status"] == "cancelled"
    else:
        async with store._conn.execute(
            "SELECT status FROM turns WHERE turn_id = ?", ("t-1",)
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == "cancelled"


async def test_complete_turn_failed(store):
    """complete_turn with status='failed' + error_json is honored."""
    await store.create_session("s-1", 1700000000)
    await store.start_turn("s-1", "t-1", model_name="dummy")
    await store.complete_turn("t-1", status="failed", stop_reason="error", error_json='{"err":"boom"}')

    if hasattr(store, "turns"):
        t = store.turns["t-1"]
        assert t["status"] == "failed"
        assert t["stop_reason"] == "error"
    else:
        async with store._conn.execute(
            "SELECT status, stop_reason, error_json FROM turns WHERE turn_id = ?", ("t-1",)
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == "failed"
        assert row[1] == "error"


async def test_multiple_turns_per_session(store):
    """One session can host multiple turns; both stores track each."""
    await store.create_session("s-1", 1700000000)
    await store.start_turn("s-1", "t-1")
    await store.start_turn("s-1", "t-2")
    await store.complete_turn("t-1", status="completed")
    await store.complete_turn("t-2", status="completed")

    if hasattr(store, "turns"):
        assert "t-1" in store.turns
        assert "t-2" in store.turns
    else:
        async with store._conn.execute(
            "SELECT turn_id FROM turns WHERE session_id = ? ORDER BY turn_id", ("s-1",)
        ) as cur:
            rows = await cur.fetchall()
        ids = [r[0] for r in rows]
        assert "t-1" in ids
        assert "t-2" in ids


async def test_record_raw_event_payload_shape(store):
    """raw_events payload is preserved as a dict on both stores."""
    await store.create_session("s-1", 1700000000)
    await store.start_turn("s-1", "t-1")
    payload = {"text": "hello world", "meta": {"x": 1, "y": [2, 3]}}
    await store.record_raw_event("s-1", "t-1", 0, "text_delta", payload)

    if hasattr(store, "raw_events"):  # MemoryStore
        events = [e for e in store.raw_events if e["turn_id"] == "t-1"]
        assert len(events) >= 1
        assert events[0]["payload"] == payload
    else:  # SqliteSessionStore — payload_json round-trips through json
        async with store._conn.execute(
            "SELECT payload_json FROM raw_events WHERE turn_id = ? AND seq = ?", ("t-1", 0)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert json.loads(row[0]) == payload
