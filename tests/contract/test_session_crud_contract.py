"""Session-CRUD contract: MemoryStore + SqliteSessionStore behave
identically for create / list / delete operations.

Phase 6 step 64. Builds on the freeze-gate (history-reconstruction
contract) by extending parametrized coverage to the session-management
lifecycle. Synthesis §3.6 + §11.3 R19.
"""


# All tests run twice via the `store` fixture in conftest.py.
# That fixture parametrizes over memory_store + sqlite_store.


async def test_create_session_then_list(store):
    """A created session appears in list_sessions."""
    await store.create_session("s-1", 1700000000)
    sessions = await store.list_sessions()
    assert any(s["session_id"] == "s-1" for s in sessions)


async def test_create_session_idempotent(store):
    """Creating the same session_id twice does not raise (INSERT OR IGNORE)."""
    await store.create_session("s-1", 1700000000)
    await store.create_session("s-1", 1700000001)  # different timestamp
    sessions = await store.list_sessions()
    assert len([s for s in sessions if s["session_id"] == "s-1"]) == 1


async def test_list_sessions_ordered_newest_first(store):
    """list_sessions returns sessions ordered by created_at DESC."""
    await store.create_session("s-old", 1700000000)
    await store.create_session("s-new", 1700001000)
    sessions = await store.list_sessions()
    if len(sessions) >= 2:
        ids = [s["session_id"] for s in sessions]
        i_old = ids.index("s-old") if "s-old" in ids else -1
        i_new = ids.index("s-new") if "s-new" in ids else -1
        if i_old >= 0 and i_new >= 0:
            assert i_new < i_old, f"newest first: {ids}"


async def test_delete_session_existing_returns_true(store):
    """delete_session returns True for an existing session."""
    await store.create_session("s-1", 1700000000)
    result = await store.delete_session("s-1")
    assert result is True


async def test_delete_session_nonexistent_returns_false(store):
    """delete_session returns False for a session that doesn't exist."""
    result = await store.delete_session("does-not-exist")
    assert result is False


async def test_delete_session_removes_messages(store):
    """delete_session also removes the session's messages (cascade-ish)."""
    await store.create_session("s-1", 1700000000)
    await store.add_user("s-1", "hello")
    await store.delete_session("s-1")

    history = await store.get_history("s-1")
    assert history == []


async def test_delete_all_sessions_empty_returns_zero(store):
    """delete_all_sessions on an empty store returns 0."""
    n = await store.delete_all_sessions()
    assert n == 0


async def test_delete_all_sessions_returns_count(store):
    """delete_all_sessions returns the number of sessions deleted."""
    await store.create_session("s-1", 1700000000)
    await store.create_session("s-2", 1700000001)
    n = await store.delete_all_sessions()
    assert n == 2
    sessions = await store.list_sessions()
    assert sessions == []


async def test_get_history_nonexistent_session_empty(store):
    """get_history on a session that doesn't exist returns [] (no error)."""
    history = await store.get_history("never-created")
    assert history == []
