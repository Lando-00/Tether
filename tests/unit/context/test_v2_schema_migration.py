"""Tests for the 002_v2_auxiliary_tables migration.

Phase 6 step 61: ADDITIVE — adds turns, tool_calls, raw_events.
v1 sessions and messages remain unchanged.
"""
import sqlite3
from pathlib import Path

import pytest

from tether.context.migration_runner import apply_pending_migrations

# yoyo 8.x uses datetime.utcnow() internally — suppress its DeprecationWarning
# so that `-W error::DeprecationWarning` sweeps stay clean.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:yoyo")


@pytest.fixture
def tmp_dsn(tmp_path: Path) -> str:
    """Yoyo-compatible SQLite DSN."""
    db_path = tmp_path / "v2_test.db"
    return f"sqlite:///{db_path.as_posix()}"


def _connect(dsn: str) -> sqlite3.Connection:
    path = dsn.replace("sqlite:///", "")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def test_v2_tables_created(tmp_dsn: str) -> None:
    """Fresh DB applies both 001 and 002; 3 new tables exist."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # v1 tables still present:
        assert "sessions" in tables
        assert "messages" in tables
        # v2 tables added:
        assert "turns" in tables
        assert "tool_calls" in tables
        assert "raw_events" in tables
    finally:
        conn.close()


def test_v2_indexes_created(tmp_dsn: str) -> None:
    """6 v2 indexes are present after 002 migration."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )}
        for required in (
            "idx_turns_session_started",
            "idx_tool_calls_turn",
            "idx_tool_calls_session_name",
            "idx_raw_events_turn_seq",
            "idx_raw_events_session_turn",
            "idx_raw_events_tool_call",
        ):
            assert required in indexes, f"index {required} missing"
    finally:
        conn.close()


def test_turn_timeline_view_exists(tmp_dsn: str) -> None:
    """The turn_timeline view is created and queryable."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        views = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        )}
        assert "turn_timeline" in views

        # Empty view returns 0 rows on a fresh DB
        rows = conn.execute("SELECT * FROM turn_timeline").fetchall()
        assert rows == []
    finally:
        conn.close()


def test_v2_tables_accept_inserts(tmp_dsn: str) -> None:
    """End-to-end: insert into turns, tool_calls, raw_events; query the view."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        # 1. Create a session in the v1 sessions table (FK target — even
        #    though we deferred the FK declaration on turns, sessions
        #    must exist for downstream conventions).
        conn.execute(
            "INSERT INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            ("test-sess", "2026-01-01T00:00:00Z", "{}"),
        )

        # 2. Create a turn
        conn.execute(
            "INSERT INTO turns(turn_id, session_id, model_name) VALUES (?, ?, ?)",
            ("t-001", "test-sess", "dummy"),
        )

        # 3. Create a tool call associated with the turn
        conn.execute(
            "INSERT INTO tool_calls(tool_call_id, session_id, turn_id, name, status) VALUES (?, ?, ?, ?, ?)",
            ("call-001", "test-sess", "t-001", "now", "ok"),
        )

        # 4. Create raw events: one tool_call, one tool_result, one text_delta
        conn.execute(
            "INSERT INTO raw_events(session_id, turn_id, seq, type, tool_call_id, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("test-sess", "t-001", 0, "tool_call", "call-001", '{"name":"now"}'),
        )
        conn.execute(
            "INSERT INTO raw_events(session_id, turn_id, seq, type, tool_call_id, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("test-sess", "t-001", 1, "tool_result", "call-001", '{"result":"noon"}'),
        )
        conn.execute(
            "INSERT INTO raw_events(session_id, turn_id, seq, type, payload_json) VALUES (?, ?, ?, ?, ?)",
            ("test-sess", "t-001", 2, "text_delta", '{"text":"hi"}'),
        )
        conn.commit()

        # 5. Query the view: 3 events, ordered by seq
        rows = conn.execute(
            "SELECT * FROM turn_timeline WHERE turn_id = ? ORDER BY seq",
            ("t-001",),
        ).fetchall()
        assert len(rows) == 3
        assert rows[0]["seq"] == 0
        assert rows[0]["type"] == "tool_call"
        assert rows[0]["tool_name"] == "now"
        assert rows[1]["seq"] == 1
        assert rows[1]["type"] == "tool_result"
        assert rows[2]["seq"] == 2
        assert rows[2]["type"] == "text_delta"
        assert rows[2]["tool_name"] is None  # No tool_call_id on text events
    finally:
        conn.close()


def test_raw_events_unique_turn_seq(tmp_dsn: str) -> None:
    """raw_events(turn_id, seq) UNIQUE — duplicate seq within same turn rejected."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        conn.execute("INSERT INTO sessions(id, created_at, metadata) VALUES ('s', '2026', '{}')")
        conn.execute("INSERT INTO turns(turn_id, session_id) VALUES ('t', 's')")
        conn.execute(
            "INSERT INTO raw_events(session_id, turn_id, seq, type, payload_json) VALUES ('s', 't', 0, 'a', '{}')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO raw_events(session_id, turn_id, seq, type, payload_json) VALUES ('s', 't', 0, 'a', '{}')"
            )
    finally:
        conn.close()


def test_tool_calls_status_check(tmp_dsn: str) -> None:
    """tool_calls.status CHECK constraint rejects invalid values."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        conn.execute("INSERT INTO sessions(id, created_at, metadata) VALUES ('s', '2026', '{}')")
        conn.execute("INSERT INTO turns(turn_id, session_id) VALUES ('t', 's')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO tool_calls(tool_call_id, session_id, turn_id, name, status) "
                "VALUES ('tc', 's', 't', 'x', 'bogus_status')"
            )
    finally:
        conn.close()


def test_v1_inserts_still_work(tmp_dsn: str) -> None:
    """v1 sessions + messages tables continue to accept inserts."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        conn.execute(
            "INSERT INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            ("v1-sess", "2026-01-01T00:00:00Z", "{}"),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            ("v1-sess", "user", "hello", "2026-01-01T00:00:01Z"),
        )
        conn.commit()

        rows = conn.execute("SELECT role, content FROM messages WHERE session_id = ?", ("v1-sess",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["role"] == "user"
        assert rows[0]["content"] == "hello"
    finally:
        conn.close()
