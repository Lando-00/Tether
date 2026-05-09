import pytest

from tether_service.context.sqlite_store import SqliteSessionStore


@pytest.mark.anyio
async def test_thinking_persisted_optionally(tmp_path):
    db_path = tmp_path / "session.db"
    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")

    session_id = "session-think"
    await store.add_user(session_id, "Hello")

    thinking_block = "<think>Reasoning...</think>\n"
    await store.add_assistant_text(
        session_id,
        "Final answer",
        thinking_text=thinking_block,
        save_thinking=True,
    )

    history_without_thinking = await store.get_history(session_id)
    assert history_without_thinking[-1]["content"] == "Final answer"

    history_with_thinking = await store.get_history(
        session_id, include_thinking=True
    )
    assert history_with_thinking[-1]["content"].startswith(thinking_block)
    assert history_with_thinking[-1]["content"].endswith("Final answer")


@pytest.mark.anyio
async def test_thinking_can_be_skipped(tmp_path):
    db_path = tmp_path / "session.db"
    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")

    session_id = "session-no-think"
    await store.add_user(session_id, "Question")

    await store.add_assistant_text(
        session_id,
        "Result",
        thinking_text="<think>Hidden</think>",
        save_thinking=False,
    )

    history = await store.get_history(session_id, include_thinking=True)
    assert history[-1]["content"] == "Result"
