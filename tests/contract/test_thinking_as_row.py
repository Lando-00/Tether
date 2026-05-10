"""Thinking persistence: stored as separate role='thinking' rows, but
output from get_history() preserves the canonical shape (assistant
content optionally prepended with thinking).

Phase 6 step 65 — verifies the row-vs-column storage refactor doesn't
change the contract. Synthesis §3.6.

Note: asyncio_mode="auto" (pyproject.toml) handles async test execution;
no pytestmark.anyio needed here.
"""
import pytest


async def test_thinking_creates_thinking_row(store):
    """Calling add_assistant_text with thinking_text creates a role='thinking' row."""
    await store.add_user("s1", "q")
    await store.add_assistant_text("s1", "answer", thinking_text="reasoning")

    if hasattr(store, "sessions"):  # MemoryStore
        rows = store.sessions["s1"]
        thinking_rows = [r for r in rows if r.get("role") == "thinking"]
        assistant_rows = [r for r in rows if r.get("role") == "assistant"]
        assert len(thinking_rows) == 1
        assert thinking_rows[0]["content"] == "reasoning"
        assert len(assistant_rows) == 1
        assert assistant_rows[0]["content"] == "answer"
    else:  # SqliteSessionStore
        async with store._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            ("s1",),
        ) as cur:
            rows = await cur.fetchall()
        roles = [r["role"] for r in rows]
        assert "thinking" in roles
        assert roles.index("thinking") < roles.index("assistant")


async def test_save_thinking_false_no_thinking_row(store):
    """When save_thinking=False, NO thinking row is created — only assistant."""
    await store.add_assistant_text("s1", "answer", thinking_text="reasoning", save_thinking=False)

    if hasattr(store, "sessions"):
        rows = store.sessions.get("s1", [])
        thinking_rows = [r for r in rows if r.get("role") == "thinking"]
        assert thinking_rows == []
    else:
        async with store._conn.execute(
            "SELECT role FROM messages WHERE session_id = ?", ("s1",)
        ) as cur:
            rows = await cur.fetchall()
        roles = [r["role"] for r in rows]
        assert "thinking" not in roles


async def test_no_thinking_text_no_thinking_row(store):
    """When thinking_text is None or empty, NO thinking row created."""
    await store.add_assistant_text("s1", "answer", thinking_text=None)
    await store.add_assistant_text("s1", "answer2", thinking_text="")

    if hasattr(store, "sessions"):
        rows = store.sessions.get("s1", [])
        thinking_rows = [r for r in rows if r.get("role") == "thinking"]
        assert thinking_rows == []
    else:
        async with store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = ?",
            ("s1", "thinking"),
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 0


async def test_get_history_excludes_thinking_by_default(store):
    """include_thinking=False (default) drops thinking from output."""
    await store.add_user("s1", "q")
    await store.add_assistant_text("s1", "answer", thinking_text="reasoning")

    history = await store.get_history("s1")

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "q"}
    assert history[1] == {"role": "assistant", "content": "answer"}


async def test_get_history_includes_thinking_when_flagged(store):
    """include_thinking=True prepends thinking to assistant content."""
    await store.add_user("s1", "q")
    await store.add_assistant_text("s1", "answer", thinking_text="reasoning")

    history = await store.get_history("s1", include_thinking=True)

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "q"}
    assert history[1] == {"role": "assistant", "content": "reasoninganswer"}


async def test_multiple_thinking_assistant_pairs(store):
    """Multiple thinking-assistant pairs each merge correctly."""
    await store.add_user("s1", "q1")
    await store.add_assistant_text("s1", "a1", thinking_text="t1")
    await store.add_user("s1", "q2")
    await store.add_assistant_text("s1", "a2", thinking_text="t2")

    history = await store.get_history("s1", include_thinking=True)

    assert len(history) == 4
    assert history[0]["content"] == "q1"
    assert history[1]["content"] == "t1a1"
    assert history[2]["content"] == "q2"
    assert history[3]["content"] == "t2a2"


async def test_thinking_dropped_when_no_following_assistant(store):
    """A trailing thinking row (no assistant following) is dropped from history.

    This rare case can happen during cancellation. The contract is to drop
    the orphan thinking content.
    """
    await store.add_user("s1", "q")

    if hasattr(store, "sessions"):
        store.sessions.setdefault("s1", []).append({"role": "thinking", "content": "orphan"})
    else:
        await store._conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            ("s1", "thinking", "orphan", "2099-01-01T00:00:00Z"),
        )
        await store._conn.commit()

    history = await store.get_history("s1", include_thinking=True)
    # Orphan thinking is dropped; only the user row appears.
    assert len(history) == 1
    assert history[0]["content"] == "q"


async def test_get_history_preserves_v1_thinking_text_column_back_compat(store):
    """If a legacy v1 row still has thinking_text on the assistant column
    (no separate thinking row), get_history merges it correctly. SQL-only
    test; MemoryStore doesn't have this back-compat path."""
    if not hasattr(store, "_conn"):
        pytest.skip("Sqlite-only back-compat path")

    await store.add_user("s1", "q")
    await store._conn.execute(
        "INSERT INTO messages(session_id, role, content, thinking_text, ts) VALUES (?, ?, ?, ?, ?)",
        ("s1", "assistant", "answer", "legacy-thinking", "2099-01-01T00:00:00Z"),
    )
    await store._conn.commit()

    history = await store.get_history("s1", include_thinking=True)
    assert len(history) == 2
    assert history[1] == {"role": "assistant", "content": "legacy-thinkinganswer"}
