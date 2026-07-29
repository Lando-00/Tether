"""``LoopLimitPolicy.RAISE`` must yield ``MessageStop`` before raising.

Phase 5 followups F5 (rubber-duck review): the previous behavior raised
``LoopLimitReachedError`` before the post-finally ``MessageStop`` could
yield, breaking the ABC contract in synthesis §3.5 that "every terminal
path emits exactly one MessageStop". Mirrors the F2 cancellation-path
fix.

Synthesis §3.5; briefing §2 Seam B.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

import pytest

from tests.golden.conftest import MinimalMemoryStore
from tether.core.errors import (
    LoopLimitReached as LoopLimitReachedError,
)
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


class _AlwaysToolProvider(ModelProvider):
    """Provider whose stream always emits a tool call so the orchestrator
    never reaches a "no tool, exit loop" state.
    """

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


def _build(*, max_tool_loops: int) -> ChattyAgentOrchestrator:
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
            loop_limit_policy=LoopLimitPolicy.RAISE,
            tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
        ),
        tool_runner=ToolRunner(tools, timeout_sec=5),
    )


@pytest.mark.anyio
async def test_raise_policy_yields_message_stop_then_raises():
    """``LoopLimitPolicy.RAISE``:
      1. yields ``MessageStop(stop_reason='tool_loop_exhausted')``
      2. raises ``LoopLimitReachedError`` AFTER the yield.
    """
    orch = _build(max_tool_loops=2)

    events: list = []
    raised: Optional[LoopLimitReachedError] = None
    try:
        async for ev in orch.run(
            session_id="sid-raise-stop",
            prompt="hi",
            model_name="scripted",
        ):
            events.append(ev)
    except LoopLimitReachedError as e:
        raised = e

    # Error was raised.
    assert raised is not None, "expected LoopLimitReachedError to be raised"

    # MessageStop was emitted as the LAST event with the right
    # stop_reason. There must be no LoopLimitReached wire event on the
    # RAISE path (that's only EMIT_LIMIT_EVENT's job).
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, (
        f"expected 1 MessageStop on RAISE policy, got "
        f"{[type(e).__name__ for e in events]}"
    )
    assert stops[0].stop_reason == "tool_loop_exhausted"
    assert events[-1] is stops[0], (
        "MessageStop must be the last event yielded on RAISE policy"
    )

    limit_events = [e for e in events if isinstance(e, LoopLimitReachedWire)]
    assert limit_events == [], (
        "LoopLimitReached wire event is reserved for EMIT_LIMIT_EVENT policy"
    )
