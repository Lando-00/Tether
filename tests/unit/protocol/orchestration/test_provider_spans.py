"""Provider streaming spans tests.

Phase 7 step 72. Verifies that ``provider.stream.start``,
``provider.stream.chunk`` (sampled), ``provider.stream.end``, and
``provider.stream.error`` are emitted by
:meth:`ChattyAgentOrchestrator._run_one_turn_until_tool_or_end` with the
correct fields.

Uses ``structlog.testing.capture_logs()`` — same pattern as
``test_tool_spans.py``. The ``merge_contextvars`` processor (included by
``capture_logs`` by default) makes contextvars-bound values like
``request_id`` and ``turn_id`` automatically appear in captured log dicts.

Synthesis §3 (observability), §4 Phase 7 step 72.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
import structlog.contextvars
from structlog.testing import capture_logs

from tether_service.core.interfaces import ModelProvider, SessionStore
from tether_service.core.types import OrchestratorConfig
from tether_service.protocol.orchestration.cancel import CancelToken
from tether_service.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether_service.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.protocol.parsers.sliding import SlidingParser

from tests.golden.conftest import MinimalMemoryStore

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Test provider stubs
# ---------------------------------------------------------------------------

class _FixedChunkProvider(ModelProvider):
    """Yields a configurable number of fixed text chunks."""

    def __init__(self, chunks: int = 3, text: str = "A" * 40):
        self._n = chunks
        self._text = text

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        for _ in range(self._n):
            yield self._text

    def list_models(self) -> List[str]:
        return ["fixed"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096

    @property
    def kind(self) -> str:
        return "fixed"


class _ErrorAfterOneChunkProvider(ModelProvider):
    """Yields one chunk then raises a RuntimeError."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "A" * 40
        raise RuntimeError("provider kaboom")

    def list_models(self) -> List[str]:
        return ["error"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096

    @property
    def kind(self) -> str:
        return "error"


class _CancelAfterOneChunkProvider(ModelProvider):
    """Yields one chunk then raises asyncio.CancelledError."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "A" * 40
        raise asyncio.CancelledError("simulated cancellation")

    def list_models(self) -> List[str]:
        return ["cancel"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096

    @property
    def kind(self) -> str:
        return "cancel"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> OrchestratorConfig:
    defaults = dict(
        max_tool_loops=1,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
    )
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)


def _build_orch(
    provider: ModelProvider,
    config: Optional[OrchestratorConfig] = None,
) -> ChattyAgentOrchestrator:
    tools: Dict = {}
    return ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="",
        config=config or _config(),
        tool_runner=ToolRunner(tools, timeout_sec=5),
    )


async def _drain(orch: ChattyAgentOrchestrator, session_id: str = "sid") -> None:
    """Consume all events from a single run() call."""
    async for _ in orch.run(
        session_id=session_id,
        prompt="hi",
        model_name="model",
    ):
        pass


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_happy_path_start_and_end_emitted():
    """Happy path: exactly one provider.stream.start and one provider.stream.end."""
    orch = _build_orch(_FixedChunkProvider(chunks=3))

    with capture_logs() as cap:
        await _drain(orch)

    events = [e["event"] for e in cap]
    assert events.count("provider.stream.start") == 1
    assert events.count("provider.stream.end") == 1
    assert "provider.stream.error" not in events


async def test_happy_path_end_has_chunks_emitted():
    """provider.stream.end carries chunks_emitted == number of yielded chunks."""
    orch = _build_orch(_FixedChunkProvider(chunks=3))

    with capture_logs() as cap:
        await _drain(orch)

    end_logs = [e for e in cap if e.get("event") == "provider.stream.end"]
    assert len(end_logs) == 1
    assert end_logs[0]["chunks_emitted"] == 3


async def test_happy_path_end_has_duration_ms():
    """provider.stream.end carries duration_ms >= 0."""
    orch = _build_orch(_FixedChunkProvider(chunks=3))

    with capture_logs() as cap:
        await _drain(orch)

    end_logs = [e for e in cap if e.get("event") == "provider.stream.end"]
    assert len(end_logs) == 1
    assert isinstance(end_logs[0]["duration_ms"], int)
    assert end_logs[0]["duration_ms"] >= 0


async def test_happy_path_start_has_model_id():
    """provider.stream.start carries model_id field."""
    orch = _build_orch(_FixedChunkProvider(chunks=3))

    with capture_logs() as cap:
        await _drain(orch)

    start_logs = [e for e in cap if e.get("event") == "provider.stream.start"]
    assert len(start_logs) == 1
    assert start_logs[0].get("model_id") == "model"


# ---------------------------------------------------------------------------
# Chunk sampling
# ---------------------------------------------------------------------------

async def test_chunk_sampling_with_sample_1():
    """With sample=1, every chunk is logged."""
    orch = _build_orch(_FixedChunkProvider(chunks=3), config=_config(provider_chunk_log_sample=1))

    with capture_logs() as cap:
        await _drain(orch)

    chunk_logs = [e for e in cap if e.get("event") == "provider.stream.chunk"]
    assert len(chunk_logs) == 3


async def test_chunk_sampling_default_50_with_100_chunks():
    """Default sample=50 with 100 chunks yields approximately 3 chunk events.

    Expected: chunk 1 (first) + chunk 50 + chunk 100 = 3 events.
    Allow 2-3 to handle edge cases in counting. Synthesis §4 Phase 7 step 72.
    """
    orch = _build_orch(
        _FixedChunkProvider(chunks=100),
        config=_config(provider_chunk_log_sample=50),
    )

    with capture_logs() as cap:
        await _drain(orch)

    chunk_logs = [e for e in cap if e.get("event") == "provider.stream.chunk"]
    assert 2 <= len(chunk_logs) <= 3, (
        f"Expected 2-3 chunk events with 100 chunks and sample=50, "
        f"got {len(chunk_logs)}: {[e.get('chunk_index') for e in chunk_logs]}"
    )


async def test_chunk_sampling_disabled_with_0():
    """With sample=0, no provider.stream.chunk events are logged."""
    orch = _build_orch(_FixedChunkProvider(chunks=5), config=_config(provider_chunk_log_sample=0))

    with capture_logs() as cap:
        await _drain(orch)

    chunk_logs = [e for e in cap if e.get("event") == "provider.stream.chunk"]
    assert len(chunk_logs) == 0


async def test_chunk_log_has_chunk_index_and_size_bytes():
    """provider.stream.chunk events carry chunk_index and size_bytes."""
    orch = _build_orch(_FixedChunkProvider(chunks=1, text="hello"), config=_config(provider_chunk_log_sample=1))

    with capture_logs() as cap:
        await _drain(orch)

    chunk_logs = [e for e in cap if e.get("event") == "provider.stream.chunk"]
    assert len(chunk_logs) >= 1
    first = chunk_logs[0]
    assert isinstance(first.get("chunk_index"), int) and first["chunk_index"] >= 1
    assert isinstance(first.get("size_bytes"), int) and first["size_bytes"] >= 0


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

async def test_error_path_emits_provider_error_kind():
    """Provider exception → provider.stream.error with error_kind='provider_error'."""
    orch = _build_orch(_ErrorAfterOneChunkProvider())

    with capture_logs() as cap:
        await _drain(orch)  # error is swallowed by the orchestrator's error path

    error_logs = [e for e in cap if e.get("event") == "provider.stream.error"]
    assert len(error_logs) == 1
    log = error_logs[0]
    assert log.get("error_kind") == "provider_error"
    assert log.get("error_class") == "RuntimeError"
    assert isinstance(log.get("duration_ms"), int)
    assert log.get("chunks_emitted") == 1


async def test_error_path_no_stream_end():
    """On provider error, provider.stream.end must NOT be emitted."""
    orch = _build_orch(_ErrorAfterOneChunkProvider())

    with capture_logs() as cap:
        await _drain(orch)

    assert "provider.stream.end" not in [e["event"] for e in cap]


# ---------------------------------------------------------------------------
# Cancellation path
# ---------------------------------------------------------------------------

async def test_cancellation_path_emits_cancelled_kind():
    """asyncio.CancelledError from provider → provider.stream.error error_kind='cancelled'."""
    orch = _build_orch(_CancelAfterOneChunkProvider())

    with capture_logs() as cap:
        try:
            await _drain(orch)
        except asyncio.CancelledError:
            pass  # CancelledError propagates from the orchestrator; expected

    error_logs = [e for e in cap if e.get("event") == "provider.stream.error"]
    assert len(error_logs) == 1
    log = error_logs[0]
    assert log.get("error_kind") == "cancelled"
    assert isinstance(log.get("duration_ms"), int)


async def test_cancellation_path_cancelled_error_propagates():
    """CancelledError from provider propagates through orchestrator to caller.

    The orchestrator's outer CancelledError handler catches the error, yields
    MessageStop(stop_reason='cancelled'), then re-raises the CancelledError.
    The caller therefore sees: [MessageStart, ..., MessageStop] followed by
    CancelledError. Synthesis §3.5 cancellation contract.
    """
    orch = _build_orch(_CancelAfterOneChunkProvider())

    events = []
    with pytest.raises(asyncio.CancelledError):
        async for evt in orch.run(
            session_id="cancel-sid",
            prompt="hi",
            model_name="model",
        ):
            events.append(evt)

    from tether_service.protocol.wire.events import MessageStop
    stop_events = [e for e in events if isinstance(e, MessageStop)]
    assert len(stop_events) == 1, f"Expected 1 MessageStop, got {len(stop_events)}"
    assert stop_events[0].stop_reason == "cancelled"


# ---------------------------------------------------------------------------
# request_id propagation via contextvars
# ---------------------------------------------------------------------------

async def test_request_id_appears_in_span_events():
    """Span events carry request_id from structlog contextvars.

    RequestIdMiddleware binds ``request_id`` to contextvars before calling
    the orchestrator. Passing ``merge_contextvars`` as a processor to
    ``capture_logs`` makes it visible in captured log dicts.
    Phase 7 step 72; synthesis §3 (observability).
    """
    import structlog.contextvars as _slcv

    known_rid = "test-req-abc123"
    _slcv.bind_contextvars(request_id=known_rid)
    try:
        orch = _build_orch(_FixedChunkProvider(chunks=2))

        with capture_logs(processors=[_slcv.merge_contextvars]) as cap:
            await _drain(orch)
    finally:
        _slcv.unbind_contextvars("request_id")

    span_events = [
        e for e in cap
        if e.get("event") in ("provider.stream.start", "provider.stream.end")
    ]
    assert len(span_events) >= 2, "Expected at least start + end span events"
    for evt in span_events:
        assert evt.get("request_id") == known_rid, (
            f"span event {evt['event']!r} missing request_id={known_rid!r}; "
            f"got {evt.get('request_id')!r}"
        )
