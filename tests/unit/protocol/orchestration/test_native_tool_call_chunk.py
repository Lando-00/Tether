"""P0-E regression: list-shaped provider chunks must not crash the parser.

Tribunal §3 P0-10 (A11-F1, A1-F4).

``MLCProvider.stream`` declares ``AsyncGenerator[str | List[Dict[str, Any]],
None]`` (core/interfaces.py:28). When the model emits MLC-native
``delta.tool_calls`` (provider.py:792-795) the chunk is a ``list[dict]``.
The orchestrator used to feed every chunk into ``SlidingParser.feed``
which does ``self.buf += chunk`` and TypeErrors on a list. Today
``marker_only_tools=true`` suppresses the list shape, but a single
config flip would crash the orchestrator. This test pins the guard
that dispatches list chunks before the parser ever sees them.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether.core.interfaces import ModelProvider
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    ToolCall,
    ToolResult,
    WireEvent,
)
from tether.tools.base import BaseTool

from tests.golden.conftest import MinimalMemoryStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _NativeToolProvider(ModelProvider):
    """Yields ONE list-shaped native tool_call chunk on the first
    ``stream()`` call, then a plain text chunk on the second call so
    the loop terminates cleanly after the tool result feeds back.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        self.calls += 1
        if self.calls == 1:
            # MLC-native shape — exactly what
            # ``provider.py:794`` emits via ``tc.model_dump()``.
            yield [
                {
                    "id": "call_native_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": json.dumps({"value": "hi"}),
                    },
                }
            ]
            return
        # Second turn: plain text only — no further tool calls so the
        # orchestrator loop exits cleanly via MessageStop.
        yield "Done with the echo. Goodbye for now."

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _EchoTool(BaseTool):
    """Trivial tool that echoes its argument back."""

    def __init__(self) -> None:
        super().__init__()

    async def run(self, value: str = "", **kwargs: Any) -> dict:
        """Echo the value back.

        Args:
            value: The string to echo.
        """
        return {"echoed": value}


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        max_tool_loops=3,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
    )


def _build_orch() -> tuple[ChattyAgentOrchestrator, _NativeToolProvider]:
    provider = _NativeToolProvider()
    tools: Dict[str, Any] = {"echo": _EchoTool()}
    orch = ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="You are helpful.",
        config=_config(),
        tool_runner=ToolRunner(tools, timeout_sec=5),
    )
    return orch, provider


@pytest.mark.anyio
async def test_list_chunk_does_not_raise_typeerror() -> None:
    """A list-shaped provider chunk must be dispatched, not concatenated.

    Without the P0-E guard, ``SlidingParser.feed(chunk)`` runs
    ``self.buf += chunk`` against a ``list`` and raises
    ``TypeError: can only concatenate str (not "list") to str``.
    """
    orch, provider = _build_orch()
    events: List[WireEvent] = []
    async for evt in orch.run(
        session_id="sid-p0e",
        prompt="please echo hi",
        model_name="scripted",
    ):
        events.append(evt)

    # Provider was called twice: once for the native tool_call, once
    # for the post-tool model turn.
    assert provider.calls == 2, (
        f"Expected 2 provider.stream calls, got {provider.calls}"
    )
    # A ToolCall AND a ToolResult were emitted from the native chunk.
    types = [type(e).__name__ for e in events]
    assert "ToolCall" in types, (
        f"No ToolCall wire event from native list chunk: {types}"
    )
    assert "ToolResult" in types, (
        f"No ToolResult wire event from native list chunk: {types}"
    )
    # And the run terminated normally (MessageStop, not Error).
    assert isinstance(events[-1], MessageStop)
    # The dispatched ToolCall carries the name decoded from the list chunk.
    tool_calls = [e for e in events if isinstance(e, ToolCall)]
    assert tool_calls and tool_calls[0].name == "echo"
    # The ToolResult corresponds to the same tool and succeeded.
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert tool_results and tool_results[0].name == "echo"
    assert tool_results[0].status == "ok"


@pytest.mark.anyio
async def test_empty_list_chunk_is_skipped() -> None:
    """An empty list chunk must not crash and must not emit a tool call."""

    class _EmptyListProvider(ModelProvider):
        async def stream(  # type: ignore[override]
            self,
            model_name: str,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
            *,
            request_id: Optional[str] = None,
        ) -> AsyncGenerator[Any, None]:
            yield []  # empty list — must be skipped, not crash
            yield "Hello there, this is a long enough chunk to flush."

        def list_models(self) -> List[str]:
            return ["scripted"]

        def unload_model(self, model_name: str) -> bool:
            return True

        def get_context_window(self, model_name: str) -> int:
            return 4096

    tools: Dict[str, Any] = {}
    orch = ChattyAgentOrchestrator(
        provider=_EmptyListProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="You are helpful.",
        config=_config(),
        tool_runner=ToolRunner(tools, timeout_sec=5),
    )
    events: List[WireEvent] = []
    async for evt in orch.run(
        session_id="sid-p0e-empty",
        prompt="hi",
        model_name="scripted",
    ):
        events.append(evt)

    types = [type(e).__name__ for e in events]
    assert "ToolCall" not in types, f"Unexpected ToolCall: {types}"
    assert isinstance(events[-1], MessageStop)


def test_native_tool_call_from_chunk_helper_shapes() -> None:
    """Direct unit coverage of the chunk → :class:`PToolCallParsed` adapter."""
    helper = ChattyAgentOrchestrator._native_tool_call_from_chunk

    # Empty chunk → None.
    assert helper([]) is None

    # JSON-string arguments are parsed.
    evt = helper(
        [
            {
                "id": "x1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": json.dumps({"city": "Paris"}),
                },
            }
        ]
    )
    assert evt is not None
    assert evt.name == "weather"
    assert evt.tool_call_id == "x1"
    assert evt.arguments == {"city": "Paris"}

    # Dict arguments pass through.
    evt = helper(
        [{"function": {"name": "t", "arguments": {"k": 1}}}]
    )
    assert evt is not None
    assert evt.arguments == {"k": 1}
    # Auto-generated id when missing.
    assert evt.tool_call_id.startswith("call-")

    # Malformed JSON falls back to ``_raw``.
    evt = helper(
        [{"function": {"name": "t", "arguments": "not-json"}}]
    )
    assert evt is not None
    assert evt.arguments == {"_raw": "not-json"}

    # Missing name → None (treated as dropped delta).
    assert helper([{"function": {"arguments": "{}"}}]) is None
