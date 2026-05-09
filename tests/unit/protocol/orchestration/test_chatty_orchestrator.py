"""Unit tests for :class:`ChattyAgentOrchestrator` (synthesis §3.5).

Verifies:

  - Class structure: required kwargs, named seams.
  - Happy path: yields ``MessageStart`` first, then ``TextDelta`` /
    ``MessageStop`` last.
  - Envelope fields: every event has session_id, turn_id, monotonic
    seq, UTC timestamp.
  - turn_id is stable within one ``run()`` invocation, distinct across.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether_service.core.interfaces import ModelProvider, SessionStore, StreamParser
from tether_service.core.types import OrchestratorConfig
from tether_service.protocol.orchestration.cancel import AsyncEventCancelToken
from tether_service.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether_service.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.protocol.parsers.events import (
    PText,
    PToolCallParsed,
    ParserEvent,
)
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.protocol.wire.events import (
    MessageStart,
    MessageStop,
    TextDelta,
    WireEvent,
)

from tests.golden.conftest import MinimalMemoryStore


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


def _build_orch(
    *,
    provider: ModelProvider,
    store: Optional[SessionStore] = None,
    parser: Optional[StreamParser] = None,
    tools: Optional[Dict[str, Any]] = None,
    config: Optional[OrchestratorConfig] = None,
) -> ChattyAgentOrchestrator:
    tools = tools or {}
    return ChattyAgentOrchestrator(
        provider=provider,
        parser=parser or SlidingParser(),
        store=store or MinimalMemoryStore(),
        tools=tools,
        system_prompt="You are helpful.",
        config=config or _config(),
        tool_runner=ToolRunner(tools, timeout_sec=5),
    )


# ---------------------------------------------------------------------------
# Constructor / required-kwargs structure
# ---------------------------------------------------------------------------


def test_orchestrator_class_constructor_accepts_all_kwargs():
    """All required kwargs accepted; instance built without error."""
    orch = _build_orch(provider=_ScriptedProvider([["hello world"]]))
    assert orch.provider is not None
    assert orch.parser is not None
    assert orch.store is not None
    assert orch.tools == {}
    assert orch.system_prompt == "You are helpful."
    assert orch.config is not None
    assert orch.tool_runner is not None
    assert orch.hw_watchdog is None  # optional


def test_orchestrator_class_rejects_missing_required():
    """Missing any required kwarg raises TypeError (Python ``__init__``)."""
    with pytest.raises(TypeError):
        ChattyAgentOrchestrator()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Named seams exist
# ---------------------------------------------------------------------------


def test_orchestrator_seams_exist():
    """Synthesis §3.5: 5-7 named seams as instance methods."""
    expected = [
        "_seed_history",
        "_run_one_turn_until_tool_or_end",
        "_dispatch_tools",
        "_persist_partial",
        "_classify_outcome",
        "_wire",
    ]
    for name in expected:
        assert hasattr(ChattyAgentOrchestrator, name), f"Orchestrator missing seam: {name}"


# ---------------------------------------------------------------------------
# run() yields WireEvent
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orchestrator_run_yields_wire_event():
    """With a one-chunk text-only stream, run() yields TextDelta + MessageStop."""
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["A long enough chunk to flush parser overlap here."]]
        )
    )
    events: List[WireEvent] = []
    async for evt in orch.run(
        session_id="sid-yields",
        prompt="hi",
        model_name="scripted",
    ):
        events.append(evt)

    assert len(events) >= 2  # MessageStart + at least 1 TextDelta + MessageStop
    assert all(isinstance(e, WireEvent.__args__[0].__args__[0].__class__.__base__.__base__) or isinstance(e, (MessageStart, MessageStop, TextDelta)) for e in events)
    # Tighter check: no event is a raw bytes / dict.
    for e in events:
        assert not isinstance(e, (bytes, dict))


@pytest.mark.anyio
async def test_orchestrator_run_message_start_first():
    """First yielded event is MessageStart with available_tools list."""
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["Long enough text chunk to flush past the parser overlap buffer."]]
        )
    )
    first_event = None
    async for evt in orch.run(
        session_id="sid-first", prompt="hi", model_name="scripted"
    ):
        first_event = evt
        break

    assert isinstance(first_event, MessageStart)
    assert first_event.available_tools == []  # no tools registered


@pytest.mark.anyio
async def test_orchestrator_run_message_start_includes_tools():
    """``available_tools`` reflects ``self.tools``."""
    from tether_service.tools.time_tool import TimeTool

    tool = TimeTool()
    orch = _build_orch(
        provider=_ScriptedProvider([["Long enough chunk to flush parser."]]),
        tools={"time_tool": tool},
    )

    async for evt in orch.run(
        session_id="sid-mst", prompt="hi", model_name="scripted"
    ):
        if isinstance(evt, MessageStart):
            assert len(evt.available_tools) == 1
            descriptor = evt.available_tools[0]
            assert descriptor.name in ("time_tool", "time")  # auto-schema vs registry
            break
    else:
        pytest.fail("No MessageStart yielded")


@pytest.mark.anyio
async def test_orchestrator_run_message_stop_last():
    """Last yielded event is MessageStop; no further events after."""
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["Long enough text chunk to flush past the parser overlap buffer."]]
        )
    )
    events: List[WireEvent] = []
    async for evt in orch.run(
        session_id="sid-last", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    assert isinstance(events[-1], MessageStop)
    assert events[-1].stop_reason == "complete"


@pytest.mark.anyio
async def test_orchestrator_run_envelope_fields():
    """Every event has session_id, turn_id, monotonic seq, UTC ts."""
    orch = _build_orch(
        provider=_ScriptedProvider(
            [["Long enough text chunk to flush past the parser overlap buffer."]]
        )
    )
    events: List[WireEvent] = []
    async for evt in orch.run(
        session_id="sid-env", prompt="hi", model_name="scripted"
    ):
        events.append(evt)

    assert len(events) >= 2
    turn_id = events[0].turn_id
    assert turn_id, "turn_id should be non-empty"
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs), f"seq must be monotonically increasing; got {seqs}"
    assert len(set(seqs)) == len(seqs), f"seq values must be unique: {seqs}"
    for e in events:
        assert e.session_id == "sid-env"
        assert e.turn_id == turn_id
        assert isinstance(e.ts, datetime)
        assert e.ts.tzinfo is not None  # timezone-aware (UTC)


@pytest.mark.anyio
async def test_orchestrator_turn_id_distinct_across_runs():
    """Two ``run()`` invocations produce distinct turn_ids."""
    orch = _build_orch(
        provider=_ScriptedProvider(
            [
                ["Long chunk one to flush past parser overlap buffer area."],
                ["Long chunk two to flush past parser overlap buffer area."],
            ]
        )
    )
    turn_ids: List[str] = []
    for prompt in ("first", "second"):
        async for evt in orch.run(
            session_id="sid-multi", prompt=prompt, model_name="scripted"
        ):
            if isinstance(evt, MessageStart):
                turn_ids.append(evt.turn_id)
                break

    assert len(turn_ids) == 2
    assert turn_ids[0] != turn_ids[1]


# ---------------------------------------------------------------------------
# _classify_outcome
# ---------------------------------------------------------------------------


def test_classify_outcome_cancelled_wins():
    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    assert (
        orch._classify_outcome(cancelled=True, final_stop_reason="complete")
        == "cancelled"
    )


def test_classify_outcome_uses_final_when_set():
    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    assert (
        orch._classify_outcome(cancelled=False, final_stop_reason="error")
        == "error"
    )
    assert (
        orch._classify_outcome(
            cancelled=False, final_stop_reason="tool_loop_exhausted"
        )
        == "tool_loop_exhausted"
    )


def test_classify_outcome_default_complete():
    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    assert (
        orch._classify_outcome(cancelled=False, final_stop_reason=None)
        == "complete"
    )


# ---------------------------------------------------------------------------
# _wire (parser-event → WireEvent translation)
# ---------------------------------------------------------------------------


def test_wire_translates_ptext_to_text_delta():
    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    envelope = {
        "session_id": "s",
        "turn_id": "t",
        "seq": 0,
        "ts": datetime.now(timezone.utc),
    }
    wire = orch._wire(PText(text="hello"), envelope)
    assert isinstance(wire, TextDelta)
    assert wire.text == "hello"


def test_wire_returns_none_for_tool_call_detected():
    """PToolCallDetected has no v2 wire equivalent — caller skips."""
    from tether_service.protocol.parsers.events import PToolCallDetected

    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    envelope = {
        "session_id": "s",
        "turn_id": "t",
        "seq": 0,
        "ts": datetime.now(timezone.utc),
    }
    assert orch._wire(PToolCallDetected(), envelope) is None


def test_wire_returns_none_for_tool_call_parsed():
    """PToolCallParsed is consumed by main loop — _wire returns None."""
    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    envelope = {
        "session_id": "s",
        "turn_id": "t",
        "seq": 0,
        "ts": datetime.now(timezone.utc),
    }
    parsed = PToolCallParsed(tool_call_id="x", name="t", arguments={})
    assert orch._wire(parsed, envelope) is None


# ---------------------------------------------------------------------------
# _audit_tool_call no-op
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_audit_tool_call_no_op():
    """Phase 7 step 73 hook is a no-op stub today."""
    orch = _build_orch(provider=_ScriptedProvider([["x"]]))
    # Should not raise / hang.
    await orch._audit_tool_call(
        session_id="s",
        turn_id="t",
        tool_name="time",
        args_sha256="abc",
        status="ok",
    )


# ---------------------------------------------------------------------------
# Cancellation pre-stream
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orchestrator_cancel_before_first_loop():
    """Cancel token already set → orchestrator emits MessageStart then
    MessageStop(cancelled) without consuming the provider."""
    token = AsyncEventCancelToken()
    token.set()  # cancelled before run starts

    orch = _build_orch(provider=_ScriptedProvider([["irrelevant"]]))
    events: List[WireEvent] = []
    async for evt in orch.run(
        session_id="sid-cancel-pre",
        prompt="hi",
        model_name="scripted",
        cancel_token=token,
    ):
        events.append(evt)

    assert isinstance(events[-1], MessageStop)
    assert events[-1].stop_reason == "cancelled"

