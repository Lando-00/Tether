"""Contract: when callers provide turn_id/tool_call_id/seq_start to
SessionStore.add_* methods, v2 tables (turns, tool_calls, raw_events) get
populated. Legacy callers without these kwargs continue to write v1-only.

Tests exercise both SqliteSessionStore (real SQL assertions) and MemoryStore
(ABC parity — no errors, state accessible via .turns / .tool_calls / .raw_events).

Synthesis §3.6 + b1-persistence.md v2 table design.
Phase 6 step 62 (p6-widen-store-api).
"""
import json
import sqlite3

import pytest

from tether_service.context.memory_store import MemoryStore
from tether_service.context.sqlite_store import SqliteSessionStore

# Note: asyncio_mode = "auto" in pyproject.toml — no anyio marker needed.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_sqlite(tmp_path):
    """Isolated SqliteSessionStore + raw db_path per test."""
    db_path = tmp_path / "v2_pop.db"
    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    yield store, db_path


@pytest.fixture
async def memory_store_v2():
    """Fresh MemoryStore per test."""
    yield MemoryStore()


# ---------------------------------------------------------------------------
# SqliteSessionStore: start_turn / complete_turn
# ---------------------------------------------------------------------------


async def test_start_turn_inserts_turns_row(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.start_turn("s1", "t1", model_name="dummy")

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT turn_id, session_id, model_name, status FROM turns"
    ).fetchall()
    conn.close()
    assert rows == [("t1", "s1", "dummy", "running")]


async def test_start_turn_creates_session_row(fresh_sqlite):
    """start_turn must call _ensure_session so sessions FK is satisfied."""
    store, db_path = fresh_sqlite
    await store.start_turn("s-auto", "t-auto", model_name=None)

    conn = sqlite3.connect(str(db_path))
    sessions = conn.execute(
        "SELECT id FROM sessions WHERE id='s-auto'"
    ).fetchone()
    conn.close()
    assert sessions is not None


async def test_complete_turn_updates_status(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.start_turn("s1", "t1", model_name="dummy")
    await store.complete_turn("t1", status="completed", stop_reason="complete")

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT status, stop_reason FROM turns WHERE turn_id=?", ("t1",)
    ).fetchone()
    conn.close()
    assert row == ("completed", "complete")


async def test_complete_turn_sets_completed_at(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.start_turn("s1", "t1")
    await store.complete_turn("t1", status="failed", stop_reason="error")

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT completed_at FROM turns WHERE turn_id=?", ("t1",)
    ).fetchone()
    conn.close()
    assert row[0] is not None


# ---------------------------------------------------------------------------
# SqliteSessionStore: add_assistant_toolcall populates tool_calls
# ---------------------------------------------------------------------------


async def test_add_assistant_toolcall_with_ids_populates_tool_calls(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.create_session("s1", 1700000000)
    await store.start_turn("s1", "t1", model_name="dummy")
    await store.add_assistant_toolcall(
        "s1", "now", {"tz": "UTC"},
        turn_id="t1", tool_call_id="call-001", seq_start=2,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT tool_call_id, turn_id, name, arguments_json, status, call_seq"
        " FROM tool_calls"
    ).fetchone()
    conn.close()
    assert row[0] == "call-001"
    assert row[1] == "t1"
    assert row[2] == "now"
    assert json.loads(row[3]) == {"tz": "UTC"}
    assert row[4] == "running"
    assert row[5] == 2


async def test_add_assistant_toolcall_without_ids_skips_v2(fresh_sqlite):
    """No tool_call_id → tool_calls table must stay empty."""
    store, db_path = fresh_sqlite
    await store.add_assistant_toolcall("s1", "now", {})

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# SqliteSessionStore: add_tool_result updates tool_calls
# ---------------------------------------------------------------------------


async def test_add_tool_result_updates_tool_calls_row(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.create_session("s1", 1700000000)
    await store.start_turn("s1", "t1", model_name="dummy")
    await store.add_assistant_toolcall(
        "s1", "now", {}, turn_id="t1", tool_call_id="call-001", seq_start=2
    )
    await store.add_tool_result(
        "s1", "now", {"now": "noon"},
        turn_id="t1", tool_call_id="call-001", seq_start=3,
        status="ok", duration_ms=42,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT status, result_json, result_seq, duration_ms"
        " FROM tool_calls WHERE tool_call_id=?",
        ("call-001",),
    ).fetchone()
    conn.close()
    assert row[0] == "ok"
    assert json.loads(row[1]) == {"now": "noon"}
    assert row[2] == 3
    assert row[3] == 42


# ---------------------------------------------------------------------------
# SqliteSessionStore: record_raw_event
# ---------------------------------------------------------------------------


async def test_record_raw_event_inserts_row(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.create_session("s1", 1700000000)
    await store.start_turn("s1", "t1", model_name="dummy")
    await store.record_raw_event("s1", "t1", 0, "text_delta", {"text": "hello"})

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT session_id, turn_id, seq, type, payload_json FROM raw_events"
    ).fetchone()
    conn.close()
    assert row == ("s1", "t1", 0, "text_delta", '{"text": "hello"}')


async def test_record_raw_event_duplicate_is_skipped(fresh_sqlite):
    """UNIQUE(turn_id, seq) duplicate must NOT raise — logged + skipped."""
    store, db_path = fresh_sqlite
    await store.create_session("s1", 1700000000)
    await store.start_turn("s1", "t1")
    await store.record_raw_event("s1", "t1", 0, "text_delta", {"text": "a"})
    await store.record_raw_event("s1", "t1", 0, "text_delta", {"text": "b"})  # duplicate

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
    first = conn.execute(
        "SELECT payload_json FROM raw_events WHERE turn_id='t1' AND seq=0"
    ).fetchone()
    conn.close()
    assert count == 1  # second insert was silently dropped
    assert json.loads(first[0]) == {"text": "a"}  # first value preserved


async def test_record_raw_event_with_tool_call_id(fresh_sqlite):
    store, db_path = fresh_sqlite
    await store.create_session("s1", 1700000000)
    await store.start_turn("s1", "t1")
    await store.add_assistant_toolcall(
        "s1", "now", {}, turn_id="t1", tool_call_id="call-x"
    )
    await store.record_raw_event(
        "s1", "t1", 5, "tool_call", {"name": "now"},
        tool_call_id="call-x",
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT tool_call_id FROM raw_events WHERE seq=5"
    ).fetchone()
    conn.close()
    assert row[0] == "call-x"


# ---------------------------------------------------------------------------
# Legacy back-compat: no v2 writes when ids not provided
# ---------------------------------------------------------------------------


async def test_legacy_calls_without_turn_id_still_work(fresh_sqlite):
    """add_* without the new kwargs writes v1 only; v2 tables stay empty."""
    store, db_path = fresh_sqlite
    await store.add_user("s1", "hi")
    await store.add_assistant_text("s1", "hello")
    await store.add_assistant_toolcall("s1", "now", {})
    await store.add_tool_result("s1", "now", {"r": 1})

    conn = sqlite3.connect(str(db_path))
    msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    tools = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
    conn.close()

    assert msgs == 4   # all v1 inserts landed
    assert turns == 0  # no start_turn called
    assert tools == 0  # no tool_call_id provided
    assert events == 0  # no record_raw_event called


# ---------------------------------------------------------------------------
# MemoryStore: ABC parity
# ---------------------------------------------------------------------------


async def test_memory_store_start_turn(memory_store_v2):
    store = memory_store_v2
    await store.start_turn("sx", "tx", model_name="dummy")
    assert store.turns["tx"]["status"] == "running"
    assert store.turns["tx"]["session_id"] == "sx"


async def test_memory_store_complete_turn(memory_store_v2):
    store = memory_store_v2
    await store.start_turn("sx", "tx")
    await store.complete_turn("tx", status="completed", stop_reason="complete")
    assert store.turns["tx"]["status"] == "completed"
    assert store.turns["tx"]["stop_reason"] == "complete"


async def test_memory_store_tool_calls_populated(memory_store_v2):
    store = memory_store_v2
    await store.start_turn("sx", "tx")
    await store.add_assistant_toolcall(
        "sx", "now", {"tz": "UTC"}, turn_id="tx", tool_call_id="call-m1", seq_start=0
    )
    assert "call-m1" in store.tool_calls
    assert store.tool_calls["call-m1"]["status"] == "running"


async def test_memory_store_tool_calls_updated_on_result(memory_store_v2):
    store = memory_store_v2
    await store.start_turn("sx", "tx")
    await store.add_assistant_toolcall(
        "sx", "now", {}, turn_id="tx", tool_call_id="call-m2"
    )
    await store.add_tool_result(
        "sx", "now", {"r": 1},
        turn_id="tx", tool_call_id="call-m2", status="ok", duration_ms=10,
    )
    assert store.tool_calls["call-m2"]["status"] == "ok"
    assert store.tool_calls["call-m2"]["duration_ms"] == 10


async def test_memory_store_raw_events(memory_store_v2):
    store = memory_store_v2
    await store.start_turn("sx", "tx")
    await store.record_raw_event("sx", "tx", 0, "text_delta", {"text": "hi"})
    await store.record_raw_event("sx", "tx", 1, "message_stop", {})
    assert len(store.raw_events) == 2
    assert store.raw_events[0]["seq"] == 0
    assert store.raw_events[1]["type"] == "message_stop"


# ---------------------------------------------------------------------------
# Full lifecycle: both stores, no errors
# ---------------------------------------------------------------------------


async def test_full_lifecycle_sqlite(fresh_sqlite):
    """Complete turn lifecycle works end-to-end on SqliteSessionStore."""
    store, _ = fresh_sqlite
    await store.create_session("sx", 1700000000)
    await store.start_turn("sx", "tx", model_name="dummy")
    await store.add_user("sx", "hi", turn_id="tx", seq_start=0)
    await store.add_assistant_toolcall(
        "sx", "now", {}, turn_id="tx", tool_call_id="call-lc1", seq_start=1
    )
    await store.add_tool_result(
        "sx", "now", {"r": 1},
        turn_id="tx", tool_call_id="call-lc1", seq_start=2, status="ok",
    )
    await store.add_assistant_text("sx", "done", turn_id="tx", seq_start=3)
    await store.record_raw_event("sx", "tx", 4, "message_stop", {})
    await store.complete_turn("tx", status="completed", stop_reason="complete")
    # Must not raise.


async def test_full_lifecycle_memory(memory_store_v2):
    """Complete turn lifecycle works end-to-end on MemoryStore."""
    store = memory_store_v2
    await store.create_session("sx", 1700000000)
    await store.start_turn("sx", "tx", model_name="dummy")
    await store.add_user("sx", "hi", turn_id="tx", seq_start=0)
    await store.add_assistant_toolcall(
        "sx", "now", {}, turn_id="tx", tool_call_id="call-lc2", seq_start=1
    )
    await store.add_tool_result(
        "sx", "now", {"r": 1},
        turn_id="tx", tool_call_id="call-lc2", seq_start=2, status="ok",
    )
    await store.add_assistant_text("sx", "done", turn_id="tx", seq_start=3)
    await store.record_raw_event("sx", "tx", 4, "message_stop", {})
    await store.complete_turn("tx", status="completed", stop_reason="complete")
    # Must not raise.
