"""Unit tests for the yoyo-backed migration runner.

Verifies fresh-apply, idempotency, and that the baseline schema contains
all columns that SqliteSessionStore expects.

Phase 6 step 59. Synthesis §3.6, B1 step 2.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tether.context.migration_runner import apply_pending_migrations

# yoyo 8.x uses datetime.utcnow() internally — suppress its DeprecationWarning
# so that `-W error::DeprecationWarning` sweeps stay clean. The warning is in
# third-party code; our code does not use datetime.utcnow().
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:yoyo")


@pytest.fixture
def tmp_dsn(tmp_path: Path) -> str:
    """Yoyo-compatible SQLite DSN pointing at a fresh tmp file."""
    db_path = tmp_path / "test.db"
    return f"sqlite:///{db_path.as_posix()}"


def test_fresh_db_applies_baseline(tmp_dsn: str) -> None:
    """A fresh DB applies all pending migrations and reports the count."""
    applied = apply_pending_migrations(tmp_dsn)
    assert applied >= 1, "At least 001_current_schema must be applied"

    db_path = tmp_dsn.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "sessions" in tables
        assert "messages" in tables
    finally:
        conn.close()


def test_idempotent_second_apply(tmp_dsn: str) -> None:
    """Applying twice is a no-op the second time — both via in-process
    cache (fast path) and via yoyo's own tracking table (slow path)."""
    import tether.context.migration_runner as _runner

    first = apply_pending_migrations(tmp_dsn)
    assert first >= 1

    # Fast path: in-process cache returns 0 without hitting yoyo.
    second = apply_pending_migrations(tmp_dsn)
    assert second == 0

    # Slow path: clear the in-process cache and verify yoyo also returns 0
    # (its _yoyo_migration tracking table already lists 001_current_schema).
    _runner._MIGRATED_DSNS.discard(_runner._normalize_dsn(tmp_dsn))
    third = apply_pending_migrations(tmp_dsn)
    assert third == 0


def test_baseline_creates_required_columns(tmp_dsn: str) -> None:
    """The baseline schema includes all columns SqliteSessionStore expects."""
    apply_pending_migrations(tmp_dsn)

    db_path = tmp_dsn.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        msg_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        for required in (
            "id",
            "session_id",
            "role",
            "content",
            "thinking_text",
            "tool_name",
            "args",
            "result",
            "ts",
        ):
            assert required in msg_cols, f"messages.{required} missing from baseline"

        sess_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        for required in ("id", "created_at", "metadata"):
            assert required in sess_cols, f"sessions.{required} missing"
    finally:
        conn.close()


def test_baseline_creates_idx_session_ts(tmp_dsn: str) -> None:
    """The baseline creates the idx_session_ts index on messages."""
    apply_pending_migrations(tmp_dsn)

    db_path = tmp_dsn.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "idx_session_ts" in indexes
    finally:
        conn.close()


def test_lazy_import_via_subprocess() -> None:
    """Importing migration_runner (without calling it) does NOT pull yoyo.

    Uses a subprocess so we get a truly fresh interpreter. The acceptance
    command (A12) validates the same invariant at the tether_service level.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tether.context.migration_runner; "
                "assert 'yoyo' not in sys.modules, "
                "'yoyo eagerly imported by migration_runner'; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Lazy-import invariant violated:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
