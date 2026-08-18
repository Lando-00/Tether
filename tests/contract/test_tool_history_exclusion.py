"""Runtime tool disabling and its effect on model-facing context.

Disabling a tool must do three things at once, because doing only the first
leaves a small model both distracted and able to keep trying:

1. stop advertising the tool (it disappears from the orchestrator's tool map);
2. stop it being dispatchable;
3. drop its past calls/results from the model-facing history.

This module covers (3) against both store implementations via the shared
parametrized ``store`` fixture in ``conftest.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List


async def _seed(store) -> str:
    """One turn that used two different tools."""
    session_id = "s-tools"
    await store.add_user(session_id, "give me a local briefing please")
    await store.add_assistant_toolcall(session_id, "time", {"timezone": "UTC"})
    await store.add_tool_result(session_id, "time", {"clock": "12:00"})
    await store.add_assistant_toolcall(session_id, "weather", {"location": "Dublin"})
    await store.add_tool_result(session_id, "weather", {"tempC": 11})
    await store.add_assistant_text(session_id, "It is noon and 11 degrees.")
    return session_id


def _mentions(history: List[Dict[str, Any]], tool_name: str) -> bool:
    return any(tool_name in (m.get("content") or "") for m in history)


async def test_excluded_tool_rows_leave_the_model_facing_history(store) -> None:
    session_id = await _seed(store)

    full = await store.get_history(session_id)
    pruned = await store.get_history(session_id, exclude_tools={"weather"})

    assert _mentions(full, "weather")
    assert not _mentions(pruned, "weather")
    # The other tool, the user turn and the answer all survive.
    assert _mentions(pruned, "time")
    assert any(m["role"] == "user" and "briefing" in m["content"] for m in pruned)
    assert any(m["role"] == "assistant" and "noon" in m["content"] for m in pruned)
    # Exactly the two weather rows (call + result) were dropped.
    assert len(pruned) == len(full) - 2


async def test_excluding_every_used_tool_leaves_only_conversation(store) -> None:
    session_id = await _seed(store)

    pruned = await store.get_history(
        session_id, exclude_tools={"time", "weather"}
    )

    assert [m["role"] for m in pruned] == ["user", "assistant"]


async def test_excluding_nothing_is_a_no_op(store) -> None:
    session_id = await _seed(store)
    baseline = await store.get_history(session_id)

    assert await store.get_history(session_id, exclude_tools=set()) == baseline
    assert await store.get_history(session_id, exclude_tools=None) == baseline


async def test_unknown_tool_name_changes_nothing(store) -> None:
    session_id = await _seed(store)
    baseline = await store.get_history(session_id)

    assert await store.get_history(session_id, exclude_tools={"nope"}) == baseline


async def test_exclusion_is_view_only_not_deletion(store) -> None:
    """The transcript is truth; exclusion only shapes what the model sees."""
    session_id = await _seed(store)

    await store.get_history(session_id, exclude_tools={"weather"})

    assert _mentions(await store.get_history(session_id), "weather")
