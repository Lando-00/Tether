"""Tests for :func:`v0_compat_serialize` (synthesis §3.4 streaming + §11.3 R1).

These tests pin the WireEvent → legacy v0 NDJSON dict mapping. The
existing tests that observe the v0 byte stream depend on this mapping
producing the same dict shape (modulo timestamps) the legacy
``orchestrate()`` emitter produced.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tether.protocol.orchestration.emitter import (
    NdjsonEmitter,
    v0_compat_serialize,
)
from tether.protocol.wire.events import (
    Error,
    HwReset,
    LoopLimitReached,
    MessageStart,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolDescriptor,
    ToolResult,
)


def _envelope(**overrides):
    """Common envelope kwargs for constructing WireEvent instances in tests."""
    base = {
        "session_id": "sid-test",
        "turn_id": "abc123def456",
        "seq": 0,
        "ts": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return base


def _decode(raw: bytes) -> dict:
    assert raw.endswith(b"\n"), "v0 NDJSON output must end with \\n"
    return json.loads(raw.decode("utf-8").strip())


# ---------------------------------------------------------------------------
# MessageStart absorbed (no v0 equivalent)
# ---------------------------------------------------------------------------


def test_message_start_emits_nothing():
    evt = MessageStart(
        **_envelope(),
        available_tools=[ToolDescriptor(name="time", description="t")],
    )
    assert v0_compat_serialize(evt) == b""


# ---------------------------------------------------------------------------
# MessageStop -> done
# ---------------------------------------------------------------------------


def test_message_stop_complete_to_done():
    evt = MessageStop(**_envelope(), stop_reason="complete")
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "done"
    assert out["session_id"] == "sid-test"
    assert out["data"] == {}


def test_message_stop_cancelled_to_done():
    """Legacy was reason-agnostic — every MessageStop maps to ``done``."""
    evt = MessageStop(**_envelope(), stop_reason="cancelled")
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "done"
    assert out["data"] == {}


def test_message_stop_tool_loop_exhausted_to_done():
    evt = MessageStop(**_envelope(), stop_reason="tool_loop_exhausted")
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "done"


# ---------------------------------------------------------------------------
# TextDelta / ThinkingDelta -> text / think
# ---------------------------------------------------------------------------


def test_text_delta_to_text():
    evt = TextDelta(**_envelope(), text="hello world")
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "text"
    assert out["data"] == {"delta": "hello world"}


def test_thinking_delta_to_think():
    evt = ThinkingDelta(**_envelope(), text="reasoning step")
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "think"
    assert out["data"] == {"delta": "reasoning step"}


# ---------------------------------------------------------------------------
# ToolCall / ToolResult mapping
# ---------------------------------------------------------------------------


def test_tool_call_to_tool_started():
    evt = ToolCall(
        **_envelope(),
        tool_call_id="call-x",
        name="time",
        arguments={"timezone": "Europe/Dublin"},
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "tool_started"
    assert out["data"] == {
        "tool_name": "time",
        "tool_args": {"timezone": "Europe/Dublin"},
    }


def test_tool_result_ok_to_tool_completed():
    evt = ToolResult(
        **_envelope(),
        tool_call_id="call-x",
        name="time",
        status="ok",
        result={"now": "2026-05-09T19:52:55"},
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "tool_completed"
    assert out["data"] == {
        "tool_name": "time",
        "tool_result": {"now": "2026-05-09T19:52:55"},
    }


def test_tool_result_ok_with_none_result_to_empty_dict():
    """Legacy emitted ``tool_result: {}`` when result was missing."""
    evt = ToolResult(
        **_envelope(),
        tool_call_id="call-x",
        name="time",
        status="ok",
        result=None,
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "tool_completed"
    assert out["data"]["tool_result"] == {}


def test_tool_result_error_to_tool_error():
    evt = ToolResult(
        **_envelope(),
        tool_call_id="call-x",
        name="time",
        status="error",
        error_kind="execution",
        error="ValueError: bad arg",
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "tool_error"
    assert out["data"] == {"tool_name": "time", "error": "ValueError: bad arg"}


def test_tool_result_error_with_none_error_falls_back():
    """Defensive: if ``status='error'`` but ``error`` is None, emit
    'unknown' so v0 ``tool_error.data.error`` is always non-empty."""
    evt = ToolResult(
        **_envelope(),
        tool_call_id="call-x",
        name="time",
        status="error",
        error_kind="cancelled",
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "tool_error"
    assert out["data"]["error"] == "unknown"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


def test_error_event_to_error():
    evt = Error(
        **_envelope(),
        message="boom",
        error_type="RuntimeError",
        is_fatal=True,
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "error"
    assert out["data"]["message"] == "boom"
    assert out["data"]["error_type"] == "RuntimeError"
    assert out["data"]["is_fatal"] is True
    # Legacy wire kept ``recoverable`` always False.
    assert out["data"]["recoverable"] is False


def test_error_event_default_is_fatal_false():
    evt = Error(
        **_envelope(),
        message="boom",
        error_type="ValueError",
    )
    out = _decode(v0_compat_serialize(evt))
    assert out["data"]["is_fatal"] is False


# ---------------------------------------------------------------------------
# LoopLimitReached
# ---------------------------------------------------------------------------


def test_loop_limit_reached_to_loop_limit_reached():
    evt = LoopLimitReached(**_envelope(), loops=5)
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "loop_limit_reached"
    assert out["data"] == {"loops": 5}


# ---------------------------------------------------------------------------
# HwReset
# ---------------------------------------------------------------------------


def test_hw_reset_to_info_message():
    evt = HwReset(**_envelope(), model_name="Qwen3-4B-q4f16_0-MLC")
    out = _decode(v0_compat_serialize(evt))
    assert out["type"] == "info"
    assert "Qwen3-4B-q4f16_0-MLC" in out["data"]["message"]
    assert "HardwareWatchdog" in out["data"]["message"]


# ---------------------------------------------------------------------------
# NDJSON line terminator + envelope shape
# ---------------------------------------------------------------------------


def test_v0_output_always_ends_with_newline():
    cases = [
        TextDelta(**_envelope(), text="hi"),
        MessageStop(**_envelope(), stop_reason="complete"),
        ToolCall(
            **_envelope(),
            tool_call_id="x",
            name="t",
            arguments={},
        ),
        Error(**_envelope(), message="m", error_type="E"),
        LoopLimitReached(**_envelope(), loops=3),
    ]
    for evt in cases:
        out = v0_compat_serialize(evt)
        assert out.endswith(b"\n"), f"{type(evt).__name__} missing newline"


def test_v0_envelope_includes_session_id_and_ts():
    evt = TextDelta(**_envelope(session_id="sid-xyz"), text="hi")
    out = _decode(v0_compat_serialize(evt))
    assert out["session_id"] == "sid-xyz"
    assert "ts" in out
    # Should parse as ISO 8601.
    datetime.fromisoformat(out["ts"])


# ---------------------------------------------------------------------------
# NdjsonEmitter (legacy dict shim during transition) still works
# ---------------------------------------------------------------------------


def test_ndjson_emitter_dict_shim_still_works():
    """The transitional dict-shape NdjsonEmitter must keep working until
    p5-cutover-c removes it (the legacy ``orchestrate()`` function still
    uses it during this transitional commit)."""
    em = NdjsonEmitter()
    out = em.emit({"type": "text", "session_id": "s", "data": {"delta": "hi"}})
    assert out.endswith(b"\n")
    decoded = json.loads(out.decode("utf-8").strip())
    assert decoded["type"] == "text"
    assert decoded["session_id"] == "s"
    assert decoded["data"] == {"delta": "hi"}
    assert "ts" in decoded

# ---------------------------------------------------------------------------
# Phase 9.8: research-mode events over the legacy v0 wire
# ---------------------------------------------------------------------------


def test_notebook_progress_events_are_suppressed_not_errors():
    """v0 clients must not receive ``unknown_wire_event`` for research events."""
    from tether.protocol.wire.events import (
        NotebookFactAdded,
        NotebookLimitReached,
        NotebookNoFacts,
        NotebookPhaseProgress,
        NotebookPhaseStart,
        NotebookQueryAdded,
    )

    events = [
        NotebookPhaseStart(**_envelope(), phase="plan", iteration=0),
        NotebookPhaseProgress(**_envelope(), phase="plan", iteration=0, elapsed_ms=10),
        NotebookFactAdded(
            **_envelope(), fact_text="f", source_query="q", total_facts=1
        ),
        NotebookQueryAdded(**_envelope(), query="q", queue_depth=1),
        NotebookLimitReached(**_envelope(), limit_kind="max_facts", count=1),
        NotebookNoFacts(**_envelope(), queries_attempted=1, iterations=1),
    ]

    for evt in events:
        assert v0_compat_serialize(evt) == b"", f"{type(evt).__name__} must be dropped"


def test_notebook_clarification_maps_to_v0_text():
    from tether.protocol.wire.events import NotebookClarificationRequested

    evt = NotebookClarificationRequested(
        **_envelope(),
        reason="ambiguous_correction",
        message="Which earlier term should this correction replace?",
        candidates=["Irelend"],
    )

    payload = _decode(v0_compat_serialize(evt))

    assert payload["type"] == "text"
    assert "Which earlier term" in payload["data"]["delta"]
    assert "- Irelend" in payload["data"]["delta"]
