"""P0-C regression: ``parser.finalize()`` runs on every exit path.

Tribunal §3 P0-05 / A2-F1.

Before this fix, ``ChattyAgentOrchestrator.run`` only invoked
``self.parser.finalize()`` on the success path (just before the
``except`` blocks). The cancel branch yielded ``MessageStop`` and
re-raised without ever calling ``finalize()`` — directly contradicting
the 5-step cancellation-contract docstring.

These tests pin the new behaviour: ``finalize()`` is called from the
``finally:`` block on success, cancel, exception, and loop-limit raise.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from tests.golden.conftest import MinimalMemoryStore
from tether.core.interfaces import ModelProvider, StreamParser
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _ScriptedProvider(ModelProvider):
    """Yields scripted string chunks. One script per stream() call."""

    def __init__(self, scripts: List[List[str]]):
        self._scripts = list(scripts)
        self._call_index = 0

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self._call_index >= len(self._scripts):
            raise RuntimeError(
                f"_ScriptedProvider exhausted after {self._call_index} calls"
            )
        chunks = self._scripts[self._call_index]
        self._call_index += 1
        for chunk in chunks:
            yield chunk

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _RaisingProvider(ModelProvider):
    """Raises ``RuntimeError`` after the first chunk to drive the
    ``except Exception`` branch in ``run()``."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "partial text "
        raise RuntimeError("provider exploded")

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


def _config(**overrides) -> OrchestratorConfig:
    defaults = dict(
        max_tool_loops=3,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
    )
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)


def _spy_parser() -> StreamParser:
    """Wrap a real :class:`SlidingParser` so ``finalize`` calls are
    counted by ``MagicMock`` while real behaviour is preserved."""
    real = SlidingParser()
    spy = MagicMock(wraps=real)
    return spy


def _build_orch(
    *,
    provider: ModelProvider,
    parser: StreamParser,
    config: Optional[OrchestratorConfig] = None,
) -> ChattyAgentOrchestrator:
    return ChattyAgentOrchestrator(
        provider=provider,
        parser=parser,
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="You are helpful.",
        config=config or _config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )


# ---------------------------------------------------------------------------
# P0-C: finalize() is called on every exit path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_finalize_called_on_success_path():
    """Happy path: ``finalize`` is called exactly once after the stream
    completes normally."""
    parser = _spy_parser()
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["A long enough chunk to flush past the parser overlap buffer."]]
        ),
        parser=parser,
    )
    async for _ in orch.run(
        session_id="sid-success", prompt="hi", model_name="scripted"
    ):
        pass
    assert parser.finalize.call_count == 1


@pytest.mark.anyio
async def test_finalize_called_on_cancel_path():
    """Outer-cancel path: ``finalize`` is still called exactly once when
    the consumer aborts mid-stream via ``aclose()``."""
    parser = _spy_parser()
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["A long enough chunk to flush past the parser overlap buffer."]]
        ),
        parser=parser,
    )
    gen = orch.run(
        session_id="sid-cancel", prompt="hi", model_name="scripted"
    )
    # Consume one event then abort.
    async for _ in gen:
        break
    await gen.aclose()
    assert parser.finalize.call_count == 1


@pytest.mark.anyio
async def test_finalize_called_on_exception_path():
    """Provider error: ``finalize`` is still called exactly once."""
    parser = _spy_parser()
    orch = _build_orch(
        provider=_RaisingProvider(),
        parser=parser,
    )
    # Provider error is captured by ``except Exception`` and surfaced as
    # an Error event; the run completes without re-raising.
    async for _ in orch.run(
        session_id="sid-exc", prompt="x", model_name="scripted"
    ):
        pass
    assert parser.finalize.call_count == 1


@pytest.mark.anyio
async def test_finalize_called_on_loop_limit_raise_path():
    """``LoopLimitPolicy.RAISE``: ``finalize`` is still called exactly once
    even though the orchestrator re-raises ``LoopLimitReached``."""
    from tether.core.errors import LoopLimitReached

    # Force the for-loop to fall through by setting max_tool_loops=0 so the
    # ``else`` branch fires immediately and the RAISE policy raises.
    parser = _spy_parser()
    orch = _build_orch(
        provider=_ScriptedProvider([["unused"]]),
        parser=parser,
        config=_config(
            max_tool_loops=0,
            loop_limit_policy=LoopLimitPolicy.RAISE,
        ),
    )
    with pytest.raises(LoopLimitReached):
        async for _ in orch.run(
            session_id="sid-llr", prompt="x", model_name="scripted"
        ):
            pass
    assert parser.finalize.call_count == 1


@pytest.mark.anyio
async def test_finalize_failure_does_not_mask_original_exception():
    """A ``finalize()`` that itself raises must not swallow the original
    error from the provider."""
    parser = _spy_parser()
    parser.finalize.side_effect = RuntimeError("finalize boom")
    orch = _build_orch(
        provider=_RaisingProvider(),
        parser=parser,
    )
    # The provider error becomes an Error wire-event (no re-raise), but
    # finalize still runs in the finally and its exception must be
    # swallowed by the guard so the generator can complete normally.
    async for _ in orch.run(
        session_id="sid-finboom", prompt="x", model_name="scripted"
    ):
        pass
    assert parser.finalize.call_count == 1


@pytest.mark.anyio
async def test_finalize_exception_swallowed_on_cancel_path():
    """A throwing ``finalize()`` on the cancel path must not mask the
    propagating ``CancelledError`` (the guard inside ``finally`` swallows
    parser exceptions and lets the original one continue to propagate)."""
    parser = _spy_parser()
    parser.finalize.side_effect = RuntimeError("finalize boom on cancel")
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["A long enough chunk to flush past the parser overlap buffer."]]
        ),
        parser=parser,
    )
    gen = orch.run(
        session_id="sid-finboom-cancel", prompt="hi", model_name="scripted"
    )
    async for _ in gen:
        break
    # aclose() must complete without raising the parser's RuntimeError.
    await gen.aclose()
    assert parser.finalize.call_count == 1
