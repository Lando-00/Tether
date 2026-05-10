"""Tests for 003_tool_audit migration.

Phase 7 step 73: append-only tool_audit log table.
Synthesis §3.6 + B5 step 7.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tether_service.context.migration_runner import apply_pending_migrations

# yoyo 8.x uses datetime.utcnow() internally — suppress its DeprecationWarning
# so that `-W error::DeprecationWarning` sweeps stay clean.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:yoyo")


@pytest.fixture
def tmp_dsn(tmp_path: Path) -> str:
    db_path = tmp_path / "audit.db"
    return f"sqlite:///{db_path.as_posix()}"


def _connect(dsn: str) -> sqlite3.Connection:
    path = dsn.replace("sqlite:///", "")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def test_tool_audit_table_created(tmp_dsn: str) -> None:
    """003 migration creates the tool_audit table with all expected columns."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "tool_audit" in tables

        columns = {r[1] for r in conn.execute("PRAGMA table_info(tool_audit)")}
        for required in (
            "audit_id", "correlation_id", "session_id", "turn_id",
            "tool_call_id", "tool_name", "args_sha256", "args_json",
            "capabilities", "status", "error_kind", "duration_ms",
            "started_at", "completed_at",
        ):
            assert required in columns, f"tool_audit.{required} missing"
    finally:
        conn.close()


def test_tool_audit_indexes(tmp_dsn: str) -> None:
    """All 4 indexes are present after migration."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_tool_audit_%'"
        )}
        assert indexes >= {
            "idx_tool_audit_session_started",
            "idx_tool_audit_turn",
            "idx_tool_audit_tool_name_started",
            "idx_tool_audit_correlation",
        }
    finally:
        conn.close()


def test_tool_audit_status_check(tmp_dsn: str) -> None:
    """Inserting an invalid status raises IntegrityError."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO tool_audit("
                "correlation_id, session_id, turn_id, tool_name, args_sha256, status"
                ") VALUES ('cid', 'sid', 'tid', 'now', 'hash', 'invalid_status')"
            )
    finally:
        conn.close()


def test_tool_audit_accepts_valid_statuses(tmp_dsn: str) -> None:
    """All three valid status values insert cleanly."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        for status in ("ok", "error", "cancelled"):
            conn.execute(
                "INSERT INTO tool_audit("
                "correlation_id, session_id, turn_id, tool_name, args_sha256, status"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (f"cid-{status}", "sid", "tid", "now", "hash", status),
            )
        conn.commit()

        rows = conn.execute("SELECT status FROM tool_audit").fetchall()
        assert {r[0] for r in rows} == {"ok", "error", "cancelled"}
    finally:
        conn.close()


def test_tool_audit_default_capabilities_empty_list(tmp_dsn: str) -> None:
    """capabilities defaults to '[]' (empty JSON list)."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        conn.execute(
            "INSERT INTO tool_audit("
            "correlation_id, session_id, turn_id, tool_name, args_sha256, status"
            ") VALUES ('cid', 'sid', 'tid', 'now', 'hash', 'ok')"
        )
        row = conn.execute("SELECT capabilities FROM tool_audit").fetchone()
        assert row[0] == "[]"
    finally:
        conn.close()


def test_tool_audit_args_json_optional(tmp_dsn: str) -> None:
    """args_json is optional (NULL by default)."""
    apply_pending_migrations(tmp_dsn)

    conn = _connect(tmp_dsn)
    try:
        conn.execute(
            "INSERT INTO tool_audit("
            "correlation_id, session_id, turn_id, tool_name, args_sha256, status"
            ") VALUES ('cid', 'sid', 'tid', 'now', 'hash', 'ok')"
        )
        row = conn.execute("SELECT args_json FROM tool_audit").fetchone()
        assert row[0] is None
    finally:
        conn.close()


def test_settings_audit_log_default_false() -> None:
    """SecuritySettings.audit_log.store_args defaults to False (privacy-by-default).

    Tests the sub-model directly to avoid needing a providers arg.
    Synthesis §3.6 + B5 step 7.
    """
    from tether_service.config.settings import AuditLogSettings, SecuritySettings

    s = SecuritySettings()
    assert s.audit_log.store_args is False

    # Explicit override works
    s2 = SecuritySettings(audit_log={"store_args": True})
    assert s2.audit_log.store_args is True
