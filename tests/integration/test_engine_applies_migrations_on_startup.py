"""Engine.from_settings applies pending migrations before the store opens.

Phase 6 step 59: the schema lives in tether_service/context/migrations/.
Engine.from_settings calls apply_pending_migrations BEFORE constructing
SqliteSessionStore, so the store opens against a current-schema DB.

Synthesis §3.6, B1 step 2.
"""
from __future__ import annotations

import sqlite3

import pytest

from tether.config.settings import Settings
from tether.engine import Engine

# yoyo 8.x uses datetime.utcnow() internally; suppress its DeprecationWarning
# so that `-W error::DeprecationWarning` sweeps stay clean.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:yoyo")


def _settings(db_path: str) -> Settings:
    """Minimal Settings dict using DummyProvider + SqliteSessionStore."""
    return Settings.model_validate(
        {
            "system": {"prompt": "migration-test-prompt"},
            "providers": {
                "model": {
                    "impl": "tether.providers.dummy.provider.DummyProvider",
                    "args": {},
                },
                "parser": {
                    "impl": "tether.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether.context.sqlite_store.SqliteSessionStore",
                    "args": {"dsn": f"sqlite:///{db_path}"},
                },
            },
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


def test_engine_from_settings_creates_schema(tmp_path) -> None:
    """Engine.from_settings on a fresh DB applies the baseline migration so
    the SqliteSessionStore can immediately query messages/sessions tables.
    """
    db_path = (tmp_path / "engine_test.db").as_posix()
    settings = _settings(db_path)

    engine = Engine.from_settings(settings)
    assert engine is not None

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
        # yoyo's tracking table must exist (proves migrations ran)
        assert any("yoyo" in t for t in tables), (
            f"No yoyo tracking table found; tables={tables}"
        )
    finally:
        conn.close()


def test_engine_idempotent_second_construct(tmp_path) -> None:
    """Constructing Engine twice against the same DB is a no-op the second time
    (idempotency guarantee from apply_pending_migrations).
    """
    db_path = (tmp_path / "idem_test.db").as_posix()
    settings = _settings(db_path)

    engine1 = Engine.from_settings(settings)
    engine2 = Engine.from_settings(settings)

    assert engine1 is not None
    assert engine2 is not None

    # Verify both constructions share the same DB and its schema is intact
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
