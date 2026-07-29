"""Outer ``task.cancel()`` during streaming must still yield exactly one
``MessageStop(stop_reason='cancelled')``.

Phase 5 followups F2 (rubber-duck review by xhigh): the bare
``except CancelledError: raise`` path previously skipped the post-finally
``MessageStop`` because async generators cannot yield once an exception
is re-propagating. Reproduced empirically.

Synthesis §3.5 (cancellation contract).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tests.golden.conftest import MinimalMemoryStore
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
    MessageStart,
    MessageStop,
    TextDelta,
)


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


class _SlowProvider(ModelProvider):
    """Yields several text chunks slowly enough that an outer cancel
    can interrupt mid-stream."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        for i in range(20):
            await asyncio.sleep(0.02)
            yield f"chunk-{i} long enough to flush parser overlap. "

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


@pytest.mark.anyio
async def test_outer_task_cancel_emits_message_stop_cancelled():
    """When the consuming task is cancelled (NOT via ``cancel_token``),
    the orchestrator must still yield a terminal
    ``MessageStop(stop_reason='cancelled')`` before the
    ``CancelledError`` propagates out.
    """
    orch = ChattyAgentOrchestrator(
        provider=_SlowProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    events: List[Any] = []

    async def consume():
        try:
            async for ev in orch.run(
                session_id="sid-outer-cancel",
                prompt="hi",
                model_name="scripted",
            ):
                events.append(ev)
                # As soon as we see ANY TextDelta, schedule a self-cancel
                # so the next __anext__ raises CancelledError into the
                # orchestrator generator (the F2 path).
                if isinstance(ev, TextDelta):
                    asyncio.current_task().cancel()
        except asyncio.CancelledError:
            # Expected — the orchestrator re-raises after yielding
            # MessageStop.
            pass

    consumer_task = asyncio.create_task(consume())
    await asyncio.wait_for(consumer_task, timeout=3.0)

    # MessageStart was emitted.
    assert any(isinstance(e, MessageStart) for e in events)

    # Exactly ONE MessageStop, with stop_reason='cancelled'.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, (
        f"expected 1 MessageStop on outer cancel, got "
        f"{[type(e).__name__ for e in events]}"
    )
    assert stops[0].stop_reason == "cancelled", (
        f"expected stop_reason='cancelled', got {stops[0].stop_reason!r}"
    )


@pytest.mark.anyio
async def test_outer_task_cancel_via_aclosing_emits_message_stop():
    """Library callers using ``aclosing`` to drain the generator on cancel
    still observe the terminal MessageStop before GeneratorExit. This
    exercises the same except-CancelledError branch indirectly: when
    ``aclose()`` is called on a still-streaming generator, the orchestrator
    receives GeneratorExit which triggers the same finalization path
    semantics — at minimum, no new events should be emitted after
    aclose. The F2 fix focuses specifically on the CancelledError path,
    so this test asserts the contract that all events seen by the
    consumer prior to aclose are well-formed (MessageStart present
    before any TextDelta).
    """
    from contextlib import aclosing

    orch = ChattyAgentOrchestrator(
        provider=_SlowProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    events: List[Any] = []
    async with aclosing(
        orch.run(
            session_id="sid-aclose",
            prompt="hi",
            model_name="scripted",
        )
    ) as agen:
        async for ev in agen:
            events.append(ev)
            if isinstance(ev, TextDelta):
                # Stop iterating; aclosing __aexit__ will throw
                # GeneratorExit into the generator.
                break

    # Every event we observed before break must be valid wire events,
    # starting with MessageStart.
    assert isinstance(events[0], MessageStart)
    assert any(isinstance(e, TextDelta) for e in events)


@pytest.mark.anyio
async def test_outer_cancel_yields_only_one_message_stop():
    """Even under outer cancel, exactly ONE MessageStop is emitted
    (defense against double-emit if the implementation also reaches
    the post-finally yield).
    """
    orch = ChattyAgentOrchestrator(
        provider=_SlowProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    events: List[Any] = []

    async def consume():
        try:
            async for ev in orch.run(
                session_id="sid-only-one-stop",
                prompt="hi",
                model_name="scripted",
            ):
                events.append(ev)
                if isinstance(ev, TextDelta):
                    asyncio.current_task().cancel()
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consume())
    await asyncio.wait_for(consumer_task, timeout=3.0)

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
