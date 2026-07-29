"""Outer ``task.cancel()`` during a long tool run must cancel the
in-flight tool task with the 250 ms grace, not leak it.

Phase 5 followups F3 (rubber-duck review by xhigh + gpt-5.5):
``_dispatch_tools.finally`` was clearing the holder but never calling
``task.cancel()`` — the tool ran unbounded after the orchestrator's
outer iteration was interrupted by an outer ``task.cancel()``.

Synthesis §3.5: tool task cancellation grace = 250 ms.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, Optional

import pytest

from tests.golden.conftest import MinimalMemoryStore
from tether.core.interfaces import ModelProvider, Tool
from tether.core.types import OrchestratorConfig, ToolExecutionContext
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import ToolCall


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


class _ToolCallProvider(ModelProvider):
    """Emits a tool call on every stream call."""

    async def stream(
        self, model_name, messages, tools=None, **kwargs
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


class _SlowCancellableTool(Tool):
    """Tool that sleeps for ``sleep_sec`` and records when CancelledError
    arrives.
    """

    def __init__(self, sleep_sec: float = 5.0):
        self._sleep_sec = sleep_sec
        self.cancel_observed = asyncio.Event()
        self.completed = False

    @property
    def name(self) -> str:
        return "slow"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "slow",
            "description": "sleeps; records cancel",
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
            self.cancel_observed.set()
            raise


@pytest.mark.anyio
async def test_outer_task_cancel_cancels_tool_task():
    """A 5 s tool — outer ``task.cancel()`` while the tool is running
    must cancel the tool task within ~250 ms (the
    :data:`_TOOL_CANCEL_GRACE_SEC` budget). Without F3, the tool task
    leaks until natural completion.
    """
    slow_tool = _SlowCancellableTool(sleep_sec=5.0)
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

    saw_tool_call = asyncio.Event()

    async def consume():
        try:
            async for ev in orch.run(
                session_id="sid-tool-leak",
                prompt="hi",
                model_name="scripted",
            ):
                if isinstance(ev, ToolCall):
                    saw_tool_call.set()
                    # Schedule self-cancel — outer task.cancel(), not
                    # cancel_token.set(). This is the F3 path.
                    asyncio.current_task().cancel()
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consume())

    # Wait until the tool call has been issued (so the tool task is
    # running). Then the inline self-cancel triggers.
    await asyncio.wait_for(saw_tool_call.wait(), timeout=2.0)

    # Bound the consumer's wall-clock — without F3, the tool would run
    # for the full 5 s.
    start = time.monotonic()
    await asyncio.wait_for(consumer_task, timeout=3.0)
    elapsed = time.monotonic() - start

    # The tool's CancelledError handler observed the cancel.
    assert slow_tool.cancel_observed.is_set(), (
        "tool task was NOT cancelled when outer task was cancelled — F3 leak"
    )
    assert slow_tool.completed is False, (
        "tool task should not have completed naturally"
    )
    # Total wall-clock from cancel onwards is well under the 5 s sleep.
    # Allow a generous bound — the 250 ms grace + scheduler latency.
    assert elapsed < 1.5, (
        f"consumer took {elapsed:.2f}s to drain post-cancel; expected <1.5s"
    )


@pytest.mark.anyio
async def test_outer_cancel_does_not_leak_tool_task_after_consumer_done():
    """After outer cancel + consumer drain, no tool task is left running
    in the event loop (defense against the leak being merely "delayed"
    rather than "fixed").
    """
    slow_tool = _SlowCancellableTool(sleep_sec=5.0)
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

    saw_tool_call = asyncio.Event()

    async def consume():
        try:
            async for ev in orch.run(
                session_id="sid-no-leak",
                prompt="hi",
                model_name="scripted",
            ):
                if isinstance(ev, ToolCall):
                    saw_tool_call.set()
                    asyncio.current_task().cancel()
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consume())
    await asyncio.wait_for(saw_tool_call.wait(), timeout=2.0)
    await asyncio.wait_for(consumer_task, timeout=3.0)

    # Yield once so any scheduled tool-task callbacks finalize.
    await asyncio.sleep(0)

    # No "tool:slow" task is still pending.
    pending_tool_tasks = [
        t
        for t in asyncio.all_tasks()
        if t.get_name().startswith("tool:") and not t.done()
    ]
    assert pending_tool_tasks == [], (
        f"leaked tool tasks after outer cancel: {pending_tool_tasks}"
    )
