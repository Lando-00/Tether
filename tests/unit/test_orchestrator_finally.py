"""Tests for the outer ``try/finally`` partial-text persistence in
``orchestrate()``.

Per A5 orchestrator investigation + _synthesis.md §4 Phase 2 step 23. The
finally block must persist any in-progress assistant text on:
  - mid-stream cancellation (cancel_event)
  - unexpected exception inside the loop body
… while NOT double-persisting on the happy path or the existing streaming-
error path (those set ``text_persisted = True`` after their inner write).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether_service.core.interfaces import ModelProvider
from tether_service.core.types import OrchestratorConfig
from tether_service.protocol.orchestration.orchestrator import orchestrate
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.protocol.parsers.sliding import SlidingParser

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
    )


class _CountingStore(MinimalMemoryStore):
    """MinimalMemoryStore that counts ``add_assistant_text`` calls."""

    def __init__(self):
        super().__init__()
        self.persist_calls: List[Dict[str, Any]] = []

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        self.persist_calls.append(
            {
                "text": text,
                "thinking": thinking_text,
                "save_thinking": save_thinking,
            }
        )
        await super().add_assistant_text(
            session_id, text, thinking_text, save_thinking
        )


class _CancellingProvider(ModelProvider):
    """Yields two long chunks then sets ``cancel_event`` and yields a third
    chunk (which the orchestrator should not consume after seeing the cancel).
    Chunks are >OVERLAP (16) so SlidingParser emits TEXT deltas immediately."""

    def __init__(self, cancel_event: asyncio.Event):
        self._cancel_event = cancel_event

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "Hello world this is the first chunk. "
        yield "And this is the second chunk before cancel. "
        # After two chunks, request cancellation.
        self._cancel_event.set()
        yield "third chunk should never be consumed by orchestrator"

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _CompleteProvider(ModelProvider):
    """Yields a single complete chunk (>OVERLAP=16 chars) and exits cleanly."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "All done with the request, no tool needed."

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _RaisingProvider(ModelProvider):
    """Yields a long partial chunk (>OVERLAP), then raises a streaming error."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "Partial reply text before the model exploded."
        raise RuntimeError("model exploded")

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _DeliberateRaiseStore(MinimalMemoryStore):
    """get_history raises after we have ensured-prompt + add-user, simulating
    an unexpected exception inside the orchestration loop body but AFTER we
    would have any in-progress text. Combined with a small accumulator we
    pre-seed via a custom provider call sequence — see test below."""

    def __init__(self):
        super().__init__()
        self._first = True

    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        if self._first:
            self._first = False
            return await super().get_history(session_id, include_thinking)
        raise RuntimeError("history exploded")


@pytest.mark.anyio
async def test_orchestrator_persists_on_cancel():
    """Mid-stream cancel: finally persists accumulated text; one done event."""
    store = _CountingStore()
    cancel = asyncio.Event()
    provider = _CancellingProvider(cancel)

    events: List[bytes] = []
    async for chunk in orchestrate(
        session_id="sid-cancel",
        prompt="hi",
        model_name="m",
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
        cancel_event=cancel,
    ):
        events.append(chunk)

    # Exactly one persistence call (from the finally block).
    assert len(store.persist_calls) == 1
    persisted = store.persist_calls[0]
    # The two chunks emit TEXT events; finally persists the accumulated text.
    # Trailing "OVERLAP" chars stay buffered in SlidingParser, so we just
    # check a substantial prefix made it through.
    assert "Hello world this is the first chunk." in persisted["text"]

    # Exactly one done event emitted at the end.
    decoded = [c.decode("utf-8").strip() for c in events if c.strip()]
    done_events = [d for d in decoded if '"type": "done"' in d or '"type":"done"' in d]
    assert len(done_events) == 1


@pytest.mark.anyio
async def test_orchestrator_does_not_double_persist():
    """Happy path (no tool, no error) persists once via the inner success
    path; finally must skip thanks to ``text_persisted = True``."""
    store = _CountingStore()
    provider = _CompleteProvider()

    async for _ in orchestrate(
        session_id="sid-happy",
        prompt="hi",
        model_name="m",
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    ):
        pass

    assert len(store.persist_calls) == 1
    assert "All done with the request" in store.persist_calls[0]["text"]


@pytest.mark.anyio
async def test_orchestrator_persists_on_provider_error():
    """Provider raises mid-stream → existing inner error path persists once;
    finally skips."""
    store = _CountingStore()
    provider = _RaisingProvider()

    async for _ in orchestrate(
        session_id="sid-err",
        prompt="hi",
        model_name="m",
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
    ):
        pass

    assert len(store.persist_calls) == 1
    assert "Partial reply text" in store.persist_calls[0]["text"]


@pytest.mark.anyio
async def test_orchestrator_finally_runs_even_on_unexpected_exception():
    """An unexpected exception INSIDE the loop body (before the inner persist
    sites fire) — finally must still persist what's accumulated.

    We simulate this by letting iteration 1 stream + persist normally (a
    happy completion would set text_persisted=True), but BEFORE persistence
    we raise. Easiest construction: a provider that yields one chunk, then
    a store whose ``add_assistant_toolcall`` would never be called (no tool
    here), and we raise from a custom provider after enough text is
    accumulated but before inner persist runs.
    """
    store = _CountingStore()

    class _PartialThenRaiseProvider(ModelProvider):
        async def stream(
            self,
            model_name: str,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
            yield "Half-written reply"
            # raise an exception that propagates as the streaming error path
            # — but make it look non-streaming-related so the inner except
            # catches it; the inner except path persists once. To exercise
            # the *outer* finally, re-raise from inside the inner except by
            # using a non-Exception-derived raise is impossible. Instead,
            # we use a CustomError that the inner except DOES catch (it
            # catches ``Exception``), which is the realistic path. The inner
            # path persists, sets text_persisted=True; finally sees True
            # and skips. So this case mirrors test_orchestrator_persists_on_provider_error.
            #
            # To genuinely test outer-finally on unexpected exception, we
            # need an exception raised from a code path NOT inside the
            # inner try (e.g. from get_history before the try block on a
            # later iteration). See _DeliberateRaiseStore below.
            raise RuntimeError("won't reach here")

    # Use a provider that emits a tool call so we'll loop into iteration 2,
    # where get_history will raise (outside the inner try/except). The text
    # chunk before the marker must be > OVERLAP (16) chars so SlidingParser
    # actually emits TEXT events that mirror into last_response_text.
    class _OneToolThenDoneProvider(ModelProvider):
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
                # Iteration 1: emit a long preamble + a tool call (no inner persist of preamble)
                yield (
                    "This is a long enough preamble before the tool call to flush the parser. "
                    '<<function_call>> {"name": "noop", "arguments": {}}'
                )
            else:
                # Iteration 2 never executes — get_history raises first.
                yield "should never run"

        def list_models(self) -> List[str]:
            return []

        def unload_model(self, model_name: str) -> bool:
            return True

        def get_context_window(self, model_name: str) -> int:
            return 4096

    # Add a noop tool for the runner so the call doesn't error out.
    class _NoopTool:
        schema = {"name": "noop", "description": "", "parameters": {}}

        async def invoke(self, args):
            return {"ok": True}

    raising_store = _DeliberateRaiseStore()
    # Re-instrument raising_store so we can count persist calls too.
    persist_calls: List[Dict[str, Any]] = []
    original_persist = raising_store.add_assistant_text

    async def counting_persist(
        session_id, text, thinking_text=None, save_thinking=True
    ):
        persist_calls.append({"text": text, "thinking": thinking_text})
        await original_persist(session_id, text, thinking_text, save_thinking)

    raising_store.add_assistant_text = counting_persist  # type: ignore[assignment]

    async for _ in orchestrate(
        session_id="sid-unexpected",
        prompt="hi",
        model_name="m",
        provider=_OneToolThenDoneProvider(),
        parser=SlidingParser(),
        store=raising_store,
        tools={"noop": _NoopTool()},  # type: ignore[dict-item]
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({"noop": _NoopTool()}, timeout_sec=5),  # type: ignore[dict-item]
    ):
        pass

    # In iteration 2, get_history raises. That's caught by the OUTER except,
    # then finally fires. Iteration 1's text was "Reply preamble " (before
    # the tool marker). That text was never persisted (existing behavior:
    # tool-call iterations don't persist preamble text). So the finally has
    # no in-progress text from iteration 2 to persist (we never started it).
    # The mirror accumulators contain iteration 1's text, however, which is
    # what we want to recover on error.
    assert len(persist_calls) >= 1
    assert any("preamble" in p["text"] for p in persist_calls)
