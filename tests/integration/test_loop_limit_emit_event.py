"""Integration tests for :class:`LoopLimitPolicy`.

Synthesis §3.5 (user-ratified default): ``EMIT_LIMIT_EVENT`` yields a
:class:`LoopLimitReached` wire event followed by
:class:`MessageStop(stop_reason='tool_loop_exhausted')`. ``RAISE``
raises :class:`tether.core.errors.LoopLimitReached`.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

import pytest

from tests.golden.conftest import MinimalMemoryStore
from tether.core.errors import LoopLimitReached as LoopLimitReachedError
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
    LoopLimitReached as LoopLimitReachedWire,
)
from tether.protocol.wire.events import (
    MessageStop,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Test fakes — provider that always emits a tool call, tool that always succeeds
# ---------------------------------------------------------------------------


class _AlwaysToolProvider(ModelProvider):
    async def stream(
        self, model_name, messages, tools=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        yield (
            "Long enough preamble to flush parser overlap. "
            '<<function_call>> {"name": "noop", "arguments": {}}'
        )

    def list_models(self):
        return ["scripted"]

    def unload_model(self, model_name):
        return True

    def get_context_window(self, model_name):
        return 4096


class _AlwaysOkTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "noop",
            "description": "always ok",
            "parameters": {"type": "object", "properties": {}},
        }

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        return {"ok": True}


def _build(*, max_tool_loops: int, loop_limit_policy: LoopLimitPolicy):
    tools = {"noop": _AlwaysOkTool()}
    return ChattyAgentOrchestrator(
        provider=_AlwaysToolProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="sys",
        config=OrchestratorConfig(
            max_tool_loops=max_tool_loops,
            auto_reload_on_fatal_error=False,
            save_thinking=False,
            include_thinking_in_history=False,
            loop_limit_policy=loop_limit_policy,
            tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
        ),
        tool_runner=ToolRunner(tools, timeout_sec=5),
    )


# ---------------------------------------------------------------------------
# EMIT_LIMIT_EVENT: yield LoopLimitReached + MessageStop(tool_loop_exhausted)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_emit_limit_event_yields_event():
    """``EMIT_LIMIT_EVENT`` policy: orchestrator yields ``LoopLimitReached``
    after exhausting ``max_tool_loops``."""
    orch = _build(
        max_tool_loops=2, loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT
    )

    events = []
    async for evt in orch.run(
        session_id="sid-llr", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    limit_events = [e for e in events if isinstance(e, LoopLimitReachedWire)]
    assert len(limit_events) == 1
    assert limit_events[0].loops == 2


@pytest.mark.anyio
async def test_emit_limit_event_then_message_stop():
    """``LoopLimitReached`` is followed by
    ``MessageStop(stop_reason='tool_loop_exhausted')``."""
    orch = _build(
        max_tool_loops=2, loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT
    )

    events = []
    async for evt in orch.run(
        session_id="sid-llr2", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    # Find the indices.
    limit_index = next(
        i for i, e in enumerate(events) if isinstance(e, LoopLimitReachedWire)
    )
    stop_index = next(
        i for i, e in enumerate(events) if isinstance(e, MessageStop)
    )
    assert limit_index < stop_index, (
        "LoopLimitReached must come before the terminal MessageStop"
    )
    assert events[stop_index].stop_reason == "tool_loop_exhausted"


# ---------------------------------------------------------------------------
# RAISE: raises core.errors.LoopLimitReached
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_raise_policy_raises():
    """``RAISE`` policy: ``run()`` raises ``core.errors.LoopLimitReached``."""
    orch = _build(max_tool_loops=2, loop_limit_policy=LoopLimitPolicy.RAISE)

    with pytest.raises(LoopLimitReachedError):
        async for _ in orch.run(
            session_id="sid-raise", prompt="hi", model_name="scripted"
        ):
            pass
