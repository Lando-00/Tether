"""The orchestrator must fit the prompt to the provider's context window.

Providers silently evict the front of an over-long context and still return
HTTP 200. Measured on GenieX (`--nctx 4096`, `Qwen3-8B:Q4_0`) with a sentinel
instruction in the system prompt:

    prompt_tokens=2329 -> system prompt honoured
    prompt_tokens=4489 -> system prompt LOST

For marker-only providers the system prompt is the only statement of the
``<<function_call>>`` convention, so front-eviction silently disables tool
calling in long conversations. These tests assert the orchestrator trims the
middle so the provider never has to.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tests.golden.conftest import MinimalMemoryStore
from tether.core.interfaces import ModelProvider
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.context_budget import estimate_messages_tokens
from tether.protocol.orchestration.policies import LoopLimitPolicy, ToolErrorPolicy
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _CapturingProvider(ModelProvider):
    """Records the message list it was handed, then answers trivially."""

    def __init__(self, context_window: int = 4096):
        self.seen_messages: List[List[Dict[str, Any]]] = []
        self._context_window = context_window

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        self.seen_messages.append(list(messages))
        yield "ok"

    def list_models(self) -> List[str]:
        return ["capturing"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return self._context_window


class _NoWindowProvider(_CapturingProvider):
    """Provider whose context window cannot be determined."""

    def get_context_window(self, model_name: str) -> int:
        raise NotImplementedError("no window available")


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        max_tool_loops=1,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
    )


def _build(provider: ModelProvider, store) -> ChattyAgentOrchestrator:
    return ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="CALLING CONVENTION: emit <<function_call>> to call a tool.",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )


async def _seed_long_history(store, session_id: str, turns: int = 40) -> None:
    """Fill a session with enough history to blow past a 4096-token window."""
    await store.ensure_system_prompt(
        session_id, "CALLING CONVENTION: emit <<function_call>> to call a tool."
    )
    filler = "word " * 400  # ~2000 chars per message
    for i in range(turns):
        await store.add_user(session_id, f"{filler} question {i}")
        await store.add_assistant_text(session_id, f"{filler} answer {i}")


@pytest.mark.anyio
async def test_long_history_is_trimmed_before_reaching_the_provider():
    store = MinimalMemoryStore()
    session_id = "sess-long"
    await _seed_long_history(store, session_id)

    provider = _CapturingProvider(context_window=4096)
    orch = _build(provider, store)

    async for _ in orch.run(
        session_id=session_id, prompt="What is 2+2?", model_name="capturing"
    ):
        pass

    assert provider.seen_messages, "provider was never called"
    sent = provider.seen_messages[0]
    assert estimate_messages_tokens(sent) <= 4096


@pytest.mark.anyio
async def test_system_prompt_survives_trimming():
    """The regression this exists to prevent."""
    store = MinimalMemoryStore()
    session_id = "sess-sys"
    await _seed_long_history(store, session_id)

    provider = _CapturingProvider(context_window=4096)
    orch = _build(provider, store)

    async for _ in orch.run(
        session_id=session_id, prompt="What is 2+2?", model_name="capturing"
    ):
        pass

    sent = provider.seen_messages[0]
    system_text = " ".join(
        str(m.get("content", "")) for m in sent if m.get("role") == "system"
    )
    assert "<<function_call>>" in system_text


@pytest.mark.anyio
async def test_newest_prompt_survives_trimming():
    store = MinimalMemoryStore()
    session_id = "sess-recent"
    await _seed_long_history(store, session_id)

    provider = _CapturingProvider(context_window=4096)
    orch = _build(provider, store)

    async for _ in orch.run(
        session_id=session_id, prompt="UNIQUE-FINAL-QUESTION", model_name="capturing"
    ):
        pass

    sent = provider.seen_messages[0]
    combined = " ".join(str(m.get("content", "")) for m in sent)
    assert "UNIQUE-FINAL-QUESTION" in combined


@pytest.mark.anyio
async def test_short_conversation_is_not_trimmed():
    store = MinimalMemoryStore()
    session_id = "sess-short"
    await store.ensure_system_prompt(session_id, "You are helpful.")
    await store.add_user(session_id, "hello")
    await store.add_assistant_text(session_id, "hi there")

    provider = _CapturingProvider(context_window=4096)
    orch = _build(provider, store)

    async for _ in orch.run(
        session_id=session_id, prompt="and again?", model_name="capturing"
    ):
        pass

    sent = provider.seen_messages[0]
    combined = " ".join(str(m.get("content", "")) for m in sent)
    assert "hello" in combined
    assert "hi there" in combined


@pytest.mark.anyio
async def test_provider_without_a_window_is_left_alone():
    """An unknown window must not break the turn."""
    store = MinimalMemoryStore()
    session_id = "sess-nowindow"
    await _seed_long_history(store, session_id, turns=10)

    provider = _NoWindowProvider()
    orch = _build(provider, store)

    async for _ in orch.run(
        session_id=session_id, prompt="still works?", model_name="capturing"
    ):
        pass

    assert provider.seen_messages, "turn should still complete"
