"""Greenfield rebuild smoke: empty data dir → Engine.from_settings →
clean schema → end-to-end round-trip.

Phase 6 step 66. Verifies the post-reset bootstrap works: yoyo migrations
apply cleanly on a fresh DB and the full v2 store API is functional.

Citations: synthesis §3.6 (schema persistence), §4 Phase 6 step 66.
"""
from __future__ import annotations

import sqlite3

import pytest


def _settings(dsn: str):
    """Return minimal Settings using DummyProvider + SqliteSessionStore."""
    from tether_service.config.settings import Settings

    return Settings.model_validate(
        {
            "system": {"prompt": "greenfield-test"},
            "providers": {
                "model": {
                    "impl": "tether_service.providers.dummy.provider.DummyProvider",
                    "args": {},
                },
                "parser": {
                    "impl": "tether_service.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether_service.context.sqlite_store.SqliteSessionStore",
                    "args": {"dsn": dsn},
                },
            },
            "storage": {"sqlite": {"dsn": dsn}},
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


@pytest.mark.anyio
async def test_greenfield_engine_from_settings_works(tmp_path):
    """Empty data dir -> Engine.from_settings constructs cleanly."""
    from tether_service.engine import Engine

    db_path = tmp_path / "data" / "fresh.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dsn = f"sqlite:///{db_path.as_posix()}"

    async with Engine.from_settings(_settings(dsn)):
        pass

    assert db_path.exists()

    # Verify the schema was applied (sessions + messages + yoyo tracking tables)
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "sessions" in tables
        assert "messages" in tables
        assert any("yoyo" in t for t in tables), (
            f"No yoyo tracking table; tables={tables}"
        )
    finally:
        conn.close()


@pytest.mark.anyio
async def test_greenfield_full_round_trip(tmp_path):
    """Empty DB -> create session -> add user -> add assistant w/ thinking ->
    get_history returns canonical shape."""
    from tether_service.engine import Engine

    db_path = tmp_path / "data" / "fresh.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dsn = f"sqlite:///{db_path.as_posix()}"

    async with Engine.from_settings(_settings(dsn)) as engine:
        await engine.store.create_session("s-greenfield", 1700000000)
        await engine.store.add_user("s-greenfield", "What is 2+2?")
        await engine.store.add_assistant_text(
            "s-greenfield",
            "It equals 4.",
            thinking_text="Adding two and two: 4.",
        )

        history = await engine.store.get_history("s-greenfield", include_thinking=True)

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "What is 2+2?"}
    assert history[1] == {
        "role": "assistant",
        "content": "Adding two and two: 4.It equals 4.",  # thinking prepended
    }


@pytest.mark.anyio
async def test_greenfield_v2_tables_populated(tmp_path):
    """Empty DB -> tool round-trip populates v2 tables (turns, tool_calls)."""
    from tether_service.engine import Engine

    db_path = tmp_path / "data" / "fresh.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dsn = f"sqlite:///{db_path.as_posix()}"

    async with Engine.from_settings(_settings(dsn)) as engine:
        await engine.store.create_session("s-tool", 1700000000)
        await engine.store.start_turn("s-tool", "t-1", model_name="dummy")
        await engine.store.add_assistant_toolcall(
            "s-tool",
            "now",
            {},
            turn_id="t-1",
            tool_call_id="call-1",
            seq_start=0,
        )
        await engine.store.add_tool_result(
            "s-tool",
            "now",
            {"now": "noon"},
            turn_id="t-1",
            tool_call_id="call-1",
            seq_start=1,
            status="ok",
        )
        await engine.store.complete_turn("t-1", status="completed", stop_reason="complete")

        # Query v2 tables while the connection is still open
        conn = engine.store._conn
        async with conn.execute(
            "SELECT status FROM turns WHERE turn_id = ?", ("t-1",)
        ) as cur:
            turn_row = await cur.fetchone()

        async with conn.execute(
            "SELECT name, status FROM tool_calls WHERE tool_call_id = ?", ("call-1",)
        ) as cur:
            tool_row = await cur.fetchone()

    assert turn_row is not None
    assert turn_row[0] == "completed"
    assert tool_row is not None
    assert tool_row[0] == "now"
    assert tool_row[1] == "ok"


@pytest.mark.anyio
async def test_greenfield_thinking_as_row(tmp_path):
    """Empty DB -> assistant w/ thinking -> role='thinking' row exists in messages."""
    from tether_service.engine import Engine

    db_path = tmp_path / "data" / "fresh.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dsn = f"sqlite:///{db_path.as_posix()}"

    async with Engine.from_settings(_settings(dsn)) as engine:
        await engine.store.add_assistant_text(
            "s-thinking",
            "answer",
            thinking_text="reason",
        )

        conn = engine.store._conn
        async with conn.execute(
            "SELECT role FROM messages WHERE session_id = ? ORDER BY id ASC",
            ("s-thinking",),
        ) as cur:
            rows = await cur.fetchall()

    roles = [r[0] for r in rows]
    assert "thinking" in roles
    assert "assistant" in roles
    assert roles.index("thinking") < roles.index("assistant")
