"""Integration tests for the cancellation contract.

Synthesis §3.5:

  1. On cancel, stop iterating the provider stream.
  2. Cancel the in-flight tool task with **250 ms grace**.
  3. Persist partial assistant text with **200 ms write timeout**.
  4. Finalize the parser.
  5. Emit ONE :class:`MessageStop(stop_reason='cancelled')`.

Plus :class:`Orchestrator._audit_tool_call` is invoked with
``status='cancelled'`` when cancellation arrives during tool execution
(synthesis §11.3 R3).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether_service.core.interfaces import ModelProvider, Tool
from tether_service.core.types import OrchestratorConfig, ToolExecutionContext
from tether_service.protocol.orchestration.cancel import AsyncEventCancelToken
from tether_service.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether_service.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.protocol.wire.events import (
    MessageStop,
    TextDelta,
    ToolCall,
    ToolResult,
)

from tests.golden.conftest import MinimalMemoryStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        max_tool_loops=3,
        auto_reload_on_fatal_error=False,
        save_thinking=True,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
    )


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _CancelMidStreamProvider(ModelProvider):
    """Yields two long chunks then sets ``cancel_token`` and yields a
    third chunk that the orchestrator should never consume."""

    def __init__(self, token: AsyncEventCancelToken):
        self._token = token

    async def stream(
        self, model_name, messages, tools=None
    ) -> AsyncGenerator[str, None]:
        yield "Hello world this is the first chunk to flush parser. "
        yield "And this is the second chunk before cancel arrives now. "
        self._token.set()
        yield "third chunk should never be consumed"

    def list_models(self):
        return ["scripted"]

    def unload_model(self, model_name):
        return True

    def get_context_window(self, model_name):
        return 4096


class _ToolCallProvider(ModelProvider):
    """Emits a tool call on the first stream call."""

    async def stream(
        self, model_name, messages, tools=None
    ) -> AsyncGenerator[str, None]:
        yield (
            "Long enough preamble to flush parser overlap. "
            '<<function_call>> {"name": "slow", "arguments": {}}'
        )

    def list_models(self):
        return ["scripted"]

    def unload_model(self, model_name):
        return True

    def get_context_window(self, model_name):
        return 4096


class _SlowTool(Tool):
    """Tool that sleeps long enough to be cancellable.

    Records whether it was cancelled (via ``CancelledError`` propagating
    through ``asyncio.sleep``)."""

    def __init__(self, sleep_sec: float = 5.0):
        self._sleep_sec = sleep_sec
        self.cancelled_ran_at: Optional[float] = None
        self.completed = False

    @property
    def name(self) -> str:
        return "slow"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "slow",
            "description": "sleeps",
            "parameters": {"type": "object", "properties": {}},
        }

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        try:
            await asyncio.sleep(self._sleep_sec)
            self.completed = True
            return {"ok": True}
        except asyncio.CancelledError:
            self.cancelled_ran_at = time.monotonic()
            raise


class _SlowStore(MinimalMemoryStore):
    """SessionStore whose ``add_assistant_text`` sleeps for ``store_sleep``."""

    def __init__(self, store_sleep: float = 0.0):
        super().__init__()
        self._store_sleep = store_sleep
        self.persist_calls: List[Dict[str, Any]] = []
        self.persist_completed = 0

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        self.persist_calls.append({"text": text, "thinking": thinking_text})
        if self._store_sleep > 0:
            await asyncio.sleep(self._store_sleep)
        await super().add_assistant_text(
            session_id, text, thinking_text, save_thinking
        )
        self.persist_completed += 1


# ---------------------------------------------------------------------------
# Cancellation during streaming → MessageStop(cancelled)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancel_during_stream_emits_message_stop():
    """Cancel arrives mid-text; the final event is a single
    ``MessageStop(stop_reason='cancelled')``."""
    token = AsyncEventCancelToken()
    provider = _CancelMidStreamProvider(token)
    store = _SlowStore()

    orch = ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    events = []
    async for evt in orch.run(
        session_id="sid-cancel-stream",
        prompt="hi",
        model_name="scripted",
        cancel_token=token,
    ):
        events.append(evt)

    # Single terminal event, of type cancelled.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[-1].stop_reason == "cancelled"


@pytest.mark.anyio
async def test_cancel_persists_partial_text_within_200ms():
    """Cancel mid-stream: partial assistant text is persisted via the
    finally block (within the 200 ms write budget)."""
    token = AsyncEventCancelToken()
    provider = _CancelMidStreamProvider(token)
    store = _SlowStore()  # no artificial slowdown

    orch = ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    async for _ in orch.run(
        session_id="sid-persist",
        prompt="hi",
        model_name="scripted",
        cancel_token=token,
    ):
        pass

    # Exactly one persist call (from finally) and it completed.
    assert len(store.persist_calls) == 1
    assert store.persist_completed == 1
    persisted_text = store.persist_calls[0]["text"]
    assert "Hello world this is the first chunk" in persisted_text


@pytest.mark.anyio
async def test_cancel_persist_timeout_swallowed_when_store_too_slow():
    """If the store can't persist within 200 ms, the timeout is logged
    and swallowed — the orchestrator still emits one MessageStop."""
    token = AsyncEventCancelToken()
    provider = _CancelMidStreamProvider(token)
    # Store sleeps 1.0s, much longer than the 200ms budget.
    store = _SlowStore(store_sleep=1.0)

    orch = ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    start = time.monotonic()
    events = []
    async for evt in orch.run(
        session_id="sid-slow-store",
        prompt="hi",
        model_name="scripted",
        cancel_token=token,
    ):
        events.append(evt)
    elapsed = time.monotonic() - start

    # Final event is still MessageStop(cancelled). The 200ms timeout
    # bounds the finally block so total runtime < ~600ms (even though
    # the store would have taken 1s).
    assert isinstance(events[-1], MessageStop)
    assert events[-1].stop_reason == "cancelled"
    assert elapsed < 0.9, (
        f"Persist timeout should bound runtime; observed {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Cancellation during tool dispatch → ToolResult(cancelled) + audit row
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancel_during_tool_emits_tool_result_cancelled():
    """Cancel arrives while a tool is running: the orchestrator yields
    ``ToolResult(status='error', error_kind='cancelled')`` then
    ``MessageStop(cancelled)``."""
    token = AsyncEventCancelToken()
    slow_tool = _SlowTool(sleep_sec=2.0)
    tools = {"slow": slow_tool}
    orch = ChattyAgentOrchestrator(
        provider=_ToolCallProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner(tools, timeout_sec=10),
    )

    # Cancel after we know the tool has started. Use a small task to
    # poll for the ToolCall event, then set the token.
    events = []

    async def consumer():
        async for evt in orch.run(
            session_id="sid-cancel-tool",
            prompt="hi",
            model_name="scripted",
            cancel_token=token,
        ):
            events.append(evt)
            if isinstance(evt, ToolCall):
                # Schedule cancel after the tool has started.
                asyncio.get_running_loop().call_later(0.1, token.set)

    await asyncio.wait_for(consumer(), timeout=3.0)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].status == "error"
    assert tool_results[0].error_kind == "cancelled"

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[0].stop_reason == "cancelled"


@pytest.mark.anyio
async def test_cancel_grace_for_tool_task_250ms():
    """Tool task is cancelled with grace, not killed instantly: the
    SlowTool's ``except CancelledError`` handler runs (records timestamp)
    before the orchestrator finalizes."""
    token = AsyncEventCancelToken()
    slow_tool = _SlowTool(sleep_sec=2.0)
    tools = {"slow": slow_tool}
    orch = ChattyAgentOrchestrator(
        provider=_ToolCallProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner(tools, timeout_sec=10),
    )

    async def consumer():
        async for evt in orch.run(
            session_id="sid-grace",
            prompt="hi",
            model_name="scripted",
            cancel_token=token,
        ):
            if isinstance(evt, ToolCall):
                asyncio.get_running_loop().call_later(0.1, token.set)

    await asyncio.wait_for(consumer(), timeout=3.0)

    # The tool's CancelledError handler ran (the orchestrator gave it
    # grace before considering the task abandoned).
    assert slow_tool.cancelled_ran_at is not None
    assert slow_tool.completed is False


@pytest.mark.anyio
async def test_audit_call_status_cancelled():
    """``_audit_tool_call`` is invoked with ``status='cancelled'`` when
    cancel happens during tool execution (synthesis §11.3 R3)."""
    token = AsyncEventCancelToken()
    slow_tool = _SlowTool(sleep_sec=2.0)
    tools = {"slow": slow_tool}
    orch = ChattyAgentOrchestrator(
        provider=_ToolCallProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner(tools, timeout_sec=10),
    )

    audit_calls: List[Dict[str, Any]] = []
    original = orch._audit_tool_call

    async def spy(**kwargs):
        audit_calls.append(kwargs)
        await original(**kwargs)

    orch._audit_tool_call = spy  # type: ignore[assignment]

    async def consumer():
        async for evt in orch.run(
            session_id="sid-audit-cancel",
            prompt="hi",
            model_name="scripted",
            cancel_token=token,
        ):
            if isinstance(evt, ToolCall):
                asyncio.get_running_loop().call_later(0.1, token.set)

    await asyncio.wait_for(consumer(), timeout=3.0)

    cancelled_calls = [c for c in audit_calls if c.get("status") == "cancelled"]
    assert len(cancelled_calls) == 1
    assert cancelled_calls[0]["error_kind"] == "cancelled"
    assert cancelled_calls[0]["tool_name"] == "slow"
