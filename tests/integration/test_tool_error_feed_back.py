"""Integration tests for :class:`ToolErrorPolicy`.

Synthesis §3.5: under ``FEED_BACK_TO_MODEL`` (default) tool errors no
longer break the loop — they're persisted as ``tool_result`` rows with
the error payload and the orchestrator continues so the model can
recover. ``BREAK_LOOP`` retains the legacy single-turn shape for
deterministic tests.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import patch

import pytest

from tether.core.interfaces import ModelProvider, Tool
from tether.core.types import OrchestratorConfig, ToolExecutionContext
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStart,
    MessageStop,
    TextDelta,
    ToolCall,
    ToolResult,
)

from tests.golden.conftest import MinimalMemoryStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _ToolThenDoneProvider(ModelProvider):
    """Iter 1: emit a tool call. Iter 2: emit final answer text."""

    def __init__(self, tool_name: str = "noop"):
        self._calls = 0
        self._tool_name = tool_name

    async def stream(
        self, model_name, messages, tools=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        self._calls += 1
        if self._calls == 1:
            yield (
                "Long enough preamble to flush parser overlap. "
                f'<<function_call>> {{"name": "{self._tool_name}", "arguments": {{}}}}'
            )
        else:
            yield "Final answer after the tool ran successfully indeed."

    def list_models(self):
        return ["scripted"]

    def unload_model(self, model_name):
        return True

    def get_context_window(self, model_name):
        return 4096


class _AlwaysToolProvider(ModelProvider):
    """Every iter emits a tool call (used to stress the loop limit)."""

    def __init__(self, tool_name: str = "noop"):
        self._tool_name = tool_name

    async def stream(
        self, model_name, messages, tools=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        yield (
            "Long enough preamble to flush parser overlap. "
            f'<<function_call>> {{"name": "{self._tool_name}", "arguments": {{}}}}'
        )

    def list_models(self):
        return ["scripted"]

    def unload_model(self, model_name):
        return True

    def get_context_window(self, model_name):
        return 4096


class _RaisingTool(Tool):
    """Raises ValueError on every call."""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "noop",
            "description": "always raises",
            "parameters": {"type": "object", "properties": {}},
        }

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        raise ValueError("simulated tool failure")


class _SlowTool(Tool):
    """Sleeps longer than the runner timeout to trigger TimeoutError."""

    def __init__(self, sleep_sec: float = 5.0):
        self._sleep_sec = sleep_sec

    @property
    def name(self) -> str:
        return "noop"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "noop",
            "description": "sleeps forever",
            "parameters": {"type": "object", "properties": {}},
        }

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        await asyncio.sleep(self._sleep_sec)
        return {"ok": True}


def _config(
    *, tool_error_policy: ToolErrorPolicy = ToolErrorPolicy.FEED_BACK_TO_MODEL
) -> OrchestratorConfig:
    return OrchestratorConfig(
        max_tool_loops=3,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=tool_error_policy,
    )


def _build(provider, tools, config):
    return ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="sys",
        config=config,
        tool_runner=ToolRunner(tools, timeout_sec=1),
    )


# ---------------------------------------------------------------------------
# FEED_BACK_TO_MODEL: error -> ToolResult error then continue loop
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_feed_back_to_model_continues_loop():
    """Iter 1 raises; orchestrator yields ToolResult(status=error) and
    continues to iter 2 where the model emits a final answer."""
    tool = _RaisingTool()
    tools = {"noop": tool}
    orch = _build(
        _ToolThenDoneProvider("noop"),
        tools,
        _config(tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL),
    )

    events = []
    async for evt in orch.run(
        session_id="sid-fb", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    # Find the ToolResult error
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].status == "error"
    assert tool_results[0].error_kind == "execution"
    assert "simulated tool failure" in (tool_results[0].error or "")

    # Loop continued: a final TextDelta arrives and MessageStop is "complete".
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert any("Final answer" in e.text for e in text_deltas), (
        f"Expected the final answer text after error feed-back; got: "
        f"{[e.text for e in text_deltas]}"
    )

    stop = events[-1]
    assert isinstance(stop, MessageStop)
    assert stop.stop_reason == "complete"


@pytest.mark.anyio
async def test_break_loop_terminates():
    """``BREAK_LOOP`` policy: error → MessageStop(stop_reason='error')
    immediately, no further model calls."""
    tool = _RaisingTool()
    tools = {"noop": tool}
    orch = _build(
        _AlwaysToolProvider("noop"),
        tools,
        _config(tool_error_policy=ToolErrorPolicy.BREAK_LOOP),
    )

    events = []
    async for evt in orch.run(
        session_id="sid-bl", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].status == "error"

    stop = events[-1]
    assert isinstance(stop, MessageStop)
    assert stop.stop_reason == "error"


# ---------------------------------------------------------------------------
# Timeout error_kind
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_timeout_yields_timeout_kind():
    """ToolRunner timeout produces ToolResult.error_kind='timeout'."""
    tool = _SlowTool(sleep_sec=10.0)
    tools = {"noop": tool}
    orch = _build(
        _ToolThenDoneProvider("noop"),
        tools,
        _config(tool_error_policy=ToolErrorPolicy.BREAK_LOOP),
    )

    events = []
    async for evt in orch.run(
        session_id="sid-to", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].status == "error"
    assert tool_results[0].error_kind == "timeout"


# ---------------------------------------------------------------------------
# Audit hook called with status=error
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_audit_call_with_status_error():
    """`_audit_tool_call` is invoked once with status='error' on tool error.
    Phase 7 step 73 will swap the body for a real INSERT; for now we
    just verify the call site fires.
    """
    tool = _RaisingTool()
    tools = {"noop": tool}
    orch = _build(
        _AlwaysToolProvider("noop"),
        tools,
        _config(tool_error_policy=ToolErrorPolicy.BREAK_LOOP),
    )

    audit_calls: List[Dict[str, Any]] = []
    original = orch._audit_tool_call

    async def spy(**kwargs):
        audit_calls.append(kwargs)
        await original(**kwargs)

    orch._audit_tool_call = spy  # type: ignore[assignment]

    async for _ in orch.run(
        session_id="sid-audit", prompt="hi", model_name="scripted"
    ):
        pass

    error_calls = [c for c in audit_calls if c.get("status") == "error"]
    assert len(error_calls) == 1
    assert error_calls[0]["error_kind"] == "execution"
    assert error_calls[0]["tool_name"] == "noop"
    # args_sha256 is a 64-char hex string.
    assert isinstance(error_calls[0]["args_sha256"], str)
    assert len(error_calls[0]["args_sha256"]) == 64
