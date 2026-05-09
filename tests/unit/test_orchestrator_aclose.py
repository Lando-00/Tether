"""Tests that provider stream async generators are properly aclose()'d.

Verifies F1: orchestrator wraps provider.stream() in contextlib.aclosing(),
ensuring the generator's finally block (and thus native resource cleanup for
real providers) runs promptly when the orchestrator breaks out of the inner
loop — regardless of exit reason.

Cited: gpt-5.5 rubber-duck [BLOCKING] — p2-followups.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from tether_service.core.interfaces import ModelProvider, SessionStore, StreamParser, Tool
from tether_service.core.types import OrchestratorConfig
from tether_service.protocol.orchestration.orchestrator import orchestrate
from tether_service.protocol.orchestration.tool_runner import ToolRunner


# ---------------------------------------------------------------------------
# Helper: a scripted provider that records whether its finally block ran
# ---------------------------------------------------------------------------

def _make_scripted_provider(chunks: List[str], *, raise_after: Optional[int] = None):
    """Build a ModelProvider whose stream records aclose() via a flag dict."""
    closed_flag = {"finally_ran": False}

    class ScriptedProvider(ModelProvider):
        async def stream(
            self,
            model_name: str,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
        ) -> AsyncGenerator[str, None]:
            try:
                for i, chunk in enumerate(chunks):
                    if raise_after is not None and i >= raise_after:
                        raise RuntimeError("scripted provider error")
                    yield chunk
            finally:
                closed_flag["finally_ran"] = True

        def list_models(self) -> List[str]:
            return ["test-model"]

        def unload_model(self, model_name: str) -> bool:
            return True

        def get_context_window(self, model_name: str) -> int:
            return 4096

    return ScriptedProvider(), closed_flag


# ---------------------------------------------------------------------------
# Minimal fakes for orchestrate deps
# ---------------------------------------------------------------------------

class _NullStore(SessionStore):
    async def create_session(self, session_id: str, created_at: int) -> None: pass
    async def list_sessions(self): return []
    async def delete_session(self, session_id: str) -> bool: return False
    async def delete_all_sessions(self) -> int: return 0
    async def add_user(self, session_id: str, text: str) -> None: pass
    async def add_assistant_text(self, session_id, text, thinking_text=None, save_thinking=True): pass
    async def add_assistant_toolcall(self, session_id, tool_name, args): pass
    async def add_tool_result(self, session_id, tool_name, result): pass
    async def get_history(self, session_id, include_thinking=False): return []
    async def ensure_system_prompt(self, session_id, prompt): pass


class _PassthroughParser(StreamParser):
    """Emits every chunk as a TEXT event with the chunk as delta."""

    def feed(self, chunk: str):
        return [{"type": "text", "data": {"delta": chunk}}]

    def finalize(self):
        return []


class _ToolCallParser(StreamParser):
    """On the second call to feed(), emits a TOOL_COMPLETE event."""

    def __init__(self):
        self._count = 0

    def feed(self, chunk: str):
        self._count += 1
        if self._count == 2:
            return [
                {"type": "tool_started", "data": {}},
                {
                    "type": "tool_complete",
                    "data": {"tool_name": "time", "tool_args": {}},
                },
            ]
        return [{"type": "text", "data": {"delta": chunk}}]

    def finalize(self):
        return []


def _config(**overrides) -> OrchestratorConfig:
    defaults = dict(
        max_tool_loops=2,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
    )
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Test 1: aclose on cancel_event break
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_provider_stream_aclosed_on_break():
    """cancel_event break → provider generator finally block runs."""
    # Two chunks, cancel after first via cancel_event set inline
    provider, closed_flag = _make_scripted_provider(["hello ", "world"])

    cancel_event = asyncio.Event()

    class _CancelAfterFirst(_PassthroughParser):
        def feed(self, chunk):
            cancel_event.set()  # trigger cancellation on first chunk
            return super().feed(chunk)

    events = []
    async for chunk in orchestrate(
        session_id="sess-break",
        prompt="hi",
        model_name="test-model",
        provider=provider,
        parser=_CancelAfterFirst(),
        store=_NullStore(),
        tools={},
        system_prompt="",
        config=_config(),
        tool_runner=ToolRunner({}),
        cancel_event=cancel_event,
    ):
        events.append(chunk)

    assert closed_flag["finally_ran"], "Provider generator finally block did not run after cancel_event break"


# ---------------------------------------------------------------------------
# Test 2: aclose on provider exception
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_provider_stream_aclosed_on_exception():
    """Provider raises after 2 chunks → its finally block still runs."""
    # raise_after=2 means exception on the 3rd iteration (index 2)
    provider, closed_flag = _make_scripted_provider(
        ["chunk0", "chunk1", "chunk2"], raise_after=2
    )

    events = []
    async for chunk in orchestrate(
        session_id="sess-exc",
        prompt="hi",
        model_name="test-model",
        provider=provider,
        parser=_PassthroughParser(),
        store=_NullStore(),
        tools={},
        system_prompt="",
        config=_config(),
        tool_runner=ToolRunner({}),
        cancel_event=None,
    ):
        events.append(chunk)

    assert closed_flag["finally_ran"], "Provider generator finally block did not run after provider exception"


# ---------------------------------------------------------------------------
# Test 3: aclose when orchestrator breaks to execute a tool call
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_provider_stream_aclosed_on_tool_call():
    """Tool call break → provider generator finally block runs before tool execution."""
    # Three chunks: parser will emit TOOL_COMPLETE on chunk index 1, causing a break
    provider, closed_flag = _make_scripted_provider(["chunk0", "chunk1", "chunk2"])

    # Minimal tool fake so ToolRunner doesn't crash
    fake_time_tool = MagicMock()
    fake_time_tool.schema = {"name": "time", "description": "t", "parameters": {"type": "object", "properties": {}, "required": []}}
    fake_time_tool.run = AsyncMock(return_value={"time": "12:00"})
    tools = {"time": fake_time_tool}

    events = []
    async for chunk in orchestrate(
        session_id="sess-tool",
        prompt="hi",
        model_name="test-model",
        provider=provider,
        parser=_ToolCallParser(),
        store=_NullStore(),
        tools=tools,
        system_prompt="",
        config=_config(max_tool_loops=1),
        tool_runner=ToolRunner(tools),
        cancel_event=None,
    ):
        events.append(chunk)

    assert closed_flag["finally_ran"], "Provider generator finally block did not run after tool-call break"
