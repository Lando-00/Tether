"""Tests for the orchestrator's :class:`ToolExecutionContext` plumbing.

Verifies the orchestrator builds a ``ToolExecutionContext`` per tool call
from its turn state and forwards it to ``ToolRunner.run`` (Phase 4 step
41a). Specifically:

- ``session_id`` mirrors the orchestrator's ``session_id``
- ``last_user_message`` mirrors the ``prompt`` argument
- ``user_confirmed_send`` is unconditionally False (synthesis §10.8 #4 —
  the regex classifier ships in Phase 2a/2b connector sessions)
- ``turn_id`` is stable across all tool calls in one ``orchestrate()``
  call and distinct across separate ``orchestrate()`` calls

Synthesis §4 Phase 4 step 41a; §10.8 #4; connector spec §4 footer.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from tether.core.interfaces import ModelProvider, Tool
from tether.core.types import OrchestratorConfig, ToolExecutionContext
from tether.protocol.orchestration.orchestrator import orchestrate
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser

from tests.golden.conftest import MinimalMemoryStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _config(max_tool_loops: int = 2) -> OrchestratorConfig:
    return OrchestratorConfig(
        max_tool_loops=max_tool_loops,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
    )


# ---------------------------------------------------------------------------
# Helper providers
# ---------------------------------------------------------------------------


class _OneToolThenDoneProvider(ModelProvider):
    """Iteration 1: long preamble + ``<<function_call>>`` for ``noop``.
    Iteration 2: a final assistant text (no tool call) that exits the loop.
    """

    def __init__(self):
        self._calls = 0

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        self._calls += 1
        if self._calls == 1:
            yield (
                "Long enough preamble before the tool call to flush parser. "
                '<<function_call>> {"name": "noop", "arguments": {}}'
            )
        else:
            yield "All done with the request, no further tool needed."

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _TwoToolsThenDoneProvider(ModelProvider):
    """Iteration 1: tool call A. Iteration 2: tool call B. Iteration 3: done.

    Used to verify ``turn_id`` is stable across iterations within the same
    ``orchestrate()`` call.
    """

    def __init__(self):
        self._calls = 0

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        self._calls += 1
        if self._calls == 1:
            yield (
                "Long enough preamble for the parser to flush before the call. "
                '<<function_call>> {"name": "noop", "arguments": {"step": 1}}'
            )
        elif self._calls == 2:
            yield (
                "Long enough preamble for the parser to flush before the call. "
                '<<function_call>> {"name": "noop", "arguments": {"step": 2}}'
            )
        else:
            yield "All tools have run; here is the final answer."

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _NoopTool(Tool):
    """Minimal Tool ABC implementation with a recordable invoke()."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "noop"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "noop",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        self.calls.append({"args": args, "context": context})
        return {"ok": True}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestratorPassesContext:
    @pytest.mark.anyio
    async def test_orchestrator_passes_context_to_tool_runner(self):
        """When the model emits a tool call, the orchestrator forwards a
        :class:`ToolExecutionContext` populated from turn state to
        ``ToolRunner.run``.
        """
        store = MinimalMemoryStore()
        provider = _OneToolThenDoneProvider()
        tool = _NoopTool()

        runner = ToolRunner({"noop": tool}, timeout_sec=5)
        # Spy on runner.run to capture the context kwarg the orchestrator
        # passes; delegate to the real implementation for everything else.
        original_run = runner.run
        captured: List[Dict[str, Any]] = []

        async def spy_run(name, args, *, context=None, tool_call_id=None):
            captured.append({"name": name, "args": args, "context": context})
            return await original_run(name, args, context=context, tool_call_id=tool_call_id)

        runner.run = spy_run  # type: ignore[assignment]

        async for _ in orchestrate(
            session_id="sid-ctx",
            prompt="please do the thing",
            model_name="m",
            provider=provider,
            parser=SlidingParser(),
            store=store,
            tools={"noop": tool},
            system_prompt="sys",
            config=_config(),
            tool_runner=runner,
        ):
            pass

        assert len(captured) == 1, f"expected 1 tool call, got {len(captured)}"
        ctx = captured[0]["context"]
        assert isinstance(ctx, ToolExecutionContext)
        assert ctx.session_id == "sid-ctx"
        assert ctx.last_user_message == "please do the thing"
        assert ctx.user_confirmed_send is False
        assert isinstance(ctx.turn_id, str) and ctx.turn_id  # non-empty

    @pytest.mark.anyio
    async def test_orchestrator_user_confirmed_send_always_false(self):
        """Synthesis §10.8 #4: in this refactor the orchestrator does NOT
        invoke any regex/heuristic classifier — ``user_confirmed_send`` is
        unconditionally False, regardless of the user prompt's wording.
        """
        store = MinimalMemoryStore()
        provider = _OneToolThenDoneProvider()
        tool = _NoopTool()
        runner = ToolRunner({"noop": tool}, timeout_sec=5)

        # Prompt strongly suggesting confirmation; the orchestrator must
        # still pass user_confirmed_send=False.
        async for _ in orchestrate(
            session_id="sid-confirm",
            prompt="YES SEND IT NOW I CONFIRM",
            model_name="m",
            provider=provider,
            parser=SlidingParser(),
            store=store,
            tools={"noop": tool},
            system_prompt="sys",
            config=_config(),
            tool_runner=runner,
        ):
            pass

        assert len(tool.calls) == 1
        ctx = tool.calls[0]["context"]
        assert ctx is not None
        assert ctx.user_confirmed_send is False

    @pytest.mark.anyio
    async def test_orchestrator_turn_id_stable_within_one_orchestrate(self):
        """``turn_id`` is identical for every tool call inside a single
        ``orchestrate()`` invocation.
        """
        store = MinimalMemoryStore()
        provider = _TwoToolsThenDoneProvider()
        tool = _NoopTool()
        runner = ToolRunner({"noop": tool}, timeout_sec=5)

        async for _ in orchestrate(
            session_id="sid-stable",
            prompt="run twice",
            model_name="m",
            provider=provider,
            parser=SlidingParser(),
            store=store,
            tools={"noop": tool},
            system_prompt="sys",
            config=_config(max_tool_loops=4),
            tool_runner=runner,
        ):
            pass

        assert len(tool.calls) == 2, (
            f"expected 2 tool calls, got {len(tool.calls)}: {tool.calls}"
        )
        ctx1 = tool.calls[0]["context"]
        ctx2 = tool.calls[1]["context"]
        assert ctx1 is not None and ctx2 is not None
        assert ctx1.turn_id == ctx2.turn_id, (
            f"turn_id should be stable within one orchestrate(); "
            f"got {ctx1.turn_id!r} and {ctx2.turn_id!r}"
        )

    @pytest.mark.anyio
    async def test_orchestrator_turn_id_distinct_across_orchestrate_calls(self):
        """Each call to ``orchestrate()`` generates a fresh ``turn_id``."""
        store = MinimalMemoryStore()
        tool = _NoopTool()
        runner = ToolRunner({"noop": tool}, timeout_sec=5)

        # First call
        async for _ in orchestrate(
            session_id="sid-distinct",
            prompt="first turn",
            model_name="m",
            provider=_OneToolThenDoneProvider(),
            parser=SlidingParser(),
            store=store,
            tools={"noop": tool},
            system_prompt="sys",
            config=_config(),
            tool_runner=runner,
        ):
            pass

        # Second call — same session, different prompt.
        async for _ in orchestrate(
            session_id="sid-distinct",
            prompt="second turn",
            model_name="m",
            provider=_OneToolThenDoneProvider(),
            parser=SlidingParser(),
            store=store,
            tools={"noop": tool},
            system_prompt="sys",
            config=_config(),
            tool_runner=runner,
        ):
            pass

        assert len(tool.calls) == 2
        ctx1 = tool.calls[0]["context"]
        ctx2 = tool.calls[1]["context"]
        assert ctx1 is not None and ctx2 is not None
        assert ctx1.turn_id != ctx2.turn_id, (
            f"turn_id should differ across orchestrate() calls; "
            f"got {ctx1.turn_id!r} twice"
        )
        # Each call mirrors its own prompt
        assert ctx1.last_user_message == "first turn"
        assert ctx2.last_user_message == "second turn"
