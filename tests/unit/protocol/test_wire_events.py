"""Unit tests for :mod:`tether.protocol.wire.events`.

Phase 5 step 49: WireEvent discriminated union + ``_Base`` envelope.

Synthesis §4 Phase 5 step 49; §11.3 R1 (``stop_reason``); §11.3 R6
(``ToolResult.error_kind``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from tether.protocol.wire.events import (
    PROTOCOL_VERSION,
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
    WireEvent,
)


def _envelope(**overrides):
    base = {
        "session_id": "sess-1",
        "turn_id": "turn-1",
        "seq": 0,
        "ts": datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Constants + simple constructions
# ---------------------------------------------------------------------------


def test_protocol_version_constant():
    assert PROTOCOL_VERSION == "1.0"


def test_message_start_constructs():
    ev = MessageStart(
        **_envelope(),
        available_tools=[ToolDescriptor(name="time", description="Get time")],
    )
    assert ev.type == "message_start"
    assert ev.protocol_version == "1.0"
    assert ev.session_id == "sess-1"
    assert ev.turn_id == "turn-1"
    assert ev.seq == 0
    assert len(ev.available_tools) == 1
    assert ev.available_tools[0].name == "time"


def test_message_stop_complete():
    ev = MessageStop(**_envelope(seq=5), stop_reason="complete")
    assert ev.type == "message_stop"
    assert ev.stop_reason == "complete"


def test_message_stop_cancelled():
    """Synthesis §11.3 R1: ``cancelled`` is a valid stop_reason."""
    ev = MessageStop(**_envelope(seq=5), stop_reason="cancelled")
    assert ev.stop_reason == "cancelled"


def test_message_stop_client_disconnect():
    """Synthesis §11.3 R1: ``client_disconnect`` is a valid stop_reason."""
    ev = MessageStop(**_envelope(seq=5), stop_reason="client_disconnect")
    assert ev.stop_reason == "client_disconnect"


def test_message_stop_tool_loop_exhausted():
    ev = MessageStop(**_envelope(seq=5), stop_reason="tool_loop_exhausted")
    assert ev.stop_reason == "tool_loop_exhausted"


def test_message_stop_error_reason():
    ev = MessageStop(**_envelope(seq=5), stop_reason="error")
    assert ev.stop_reason == "error"


def test_message_stop_invalid_reason():
    with pytest.raises(ValidationError):
        MessageStop(**_envelope(seq=5), stop_reason="x")  # type: ignore[arg-type]


def test_text_delta_string():
    ev = TextDelta(**_envelope(seq=2), text="hello")
    assert ev.type == "text_delta"
    assert ev.text == "hello"


def test_thinking_delta_string():
    ev = ThinkingDelta(**_envelope(seq=2), text="thought...")
    assert ev.type == "thinking_delta"
    assert ev.text == "thought..."


def test_tool_call_with_args():
    ev = ToolCall(
        **_envelope(seq=3),
        tool_call_id="call-1",
        name="time",
        arguments={"timezone": "UTC"},
    )
    assert ev.type == "tool_call"
    assert ev.tool_call_id == "call-1"
    assert ev.name == "time"
    assert ev.arguments == {"timezone": "UTC"}


def test_tool_call_default_arguments_empty():
    ev = ToolCall(**_envelope(seq=3), tool_call_id="c", name="t")
    assert ev.arguments == {}


def test_tool_result_ok():
    ev = ToolResult(
        **_envelope(seq=4),
        tool_call_id="call-1",
        name="time",
        status="ok",
        result={"time": "2026-05-09T12:00:00Z"},
    )
    assert ev.type == "tool_result"
    assert ev.status == "ok"
    assert ev.result == {"time": "2026-05-09T12:00:00Z"}
    assert ev.error_kind is None
    assert ev.missing_capabilities == []


def test_tool_result_error_permission():
    """Synthesis §11.3 R6: capabilities-denied path uses
    ``error_kind="permission"`` + ``missing_capabilities``."""
    ev = ToolResult(
        **_envelope(seq=4),
        tool_call_id="call-1",
        name="web_search",
        status="error",
        error_kind="permission",
        missing_capabilities=["network"],
        error="Capability 'network' not granted",
    )
    assert ev.status == "error"
    assert ev.error_kind == "permission"
    assert ev.missing_capabilities == ["network"]


@pytest.mark.parametrize("kind", ["permission", "execution", "timeout", "cancelled"])
def test_tool_result_error_kinds(kind):
    """Synthesis §11.3 R6: each error_kind enum value is accepted."""
    ev = ToolResult(
        **_envelope(seq=4),
        tool_call_id="call-1",
        name="t",
        status="error",
        error_kind=kind,
        error="boom",
    )
    assert ev.error_kind == kind


def test_tool_result_invalid_error_kind():
    with pytest.raises(ValidationError):
        ToolResult(
            **_envelope(seq=4),
            tool_call_id="c",
            name="t",
            status="error",
            error_kind="bogus",  # type: ignore[arg-type]
        )


def test_tool_result_invalid_status():
    with pytest.raises(ValidationError):
        ToolResult(
            **_envelope(seq=4),
            tool_call_id="c",
            name="t",
            status="weird",  # type: ignore[arg-type]
        )


def test_error_event():
    ev = Error(
        **_envelope(seq=10),
        message="something broke",
        error_type="ProviderError",
        is_fatal=True,
    )
    assert ev.type == "error"
    assert ev.message == "something broke"
    assert ev.error_type == "ProviderError"
    assert ev.is_fatal is True


def test_loop_limit_reached_carries_loops():
    ev = LoopLimitReached(**_envelope(seq=99), loops=5)
    assert ev.type == "loop_limit_reached"
    assert ev.loops == 5


def test_hw_reset_carries_model_name():
    ev = HwReset(**_envelope(seq=11), model_name="Qwen3-4B-q4f16_0-MLC")
    assert ev.type == "hw_reset"
    assert ev.model_name == "Qwen3-4B-q4f16_0-MLC"


# ---------------------------------------------------------------------------
# Envelope contract
# ---------------------------------------------------------------------------


def test_envelope_required_fields_session_id():
    payload = _envelope()
    payload.pop("session_id")
    with pytest.raises(ValidationError):
        TextDelta(**payload, text="x")


def test_envelope_required_fields_turn_id():
    payload = _envelope()
    payload.pop("turn_id")
    with pytest.raises(ValidationError):
        TextDelta(**payload, text="x")


def test_envelope_required_fields_seq():
    payload = _envelope()
    payload.pop("seq")
    with pytest.raises(ValidationError):
        TextDelta(**payload, text="x")


def test_envelope_required_fields_ts():
    payload = _envelope()
    payload.pop("ts")
    with pytest.raises(ValidationError):
        TextDelta(**payload, text="x")


def test_envelope_seq_must_be_non_negative():
    with pytest.raises(ValidationError):
        TextDelta(**_envelope(seq=-1), text="x")


def test_envelope_extra_forbid():
    with pytest.raises(ValidationError):
        TextDelta(**_envelope(), text="x", unknown_field="boom")  # type: ignore[call-arg]


def test_envelope_frozen():
    ev = TextDelta(**_envelope(), text="x")
    with pytest.raises((ValueError, TypeError)):
        ev.text = "mutated"  # type: ignore[misc]


def test_envelope_default_protocol_version():
    ev = TextDelta(**_envelope(), text="x")
    assert ev.protocol_version == PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


def test_wire_event_discriminated_union_text_delta():
    adapter = TypeAdapter(WireEvent)
    payload = {
        "type": "text_delta",
        "protocol_version": "1.0",
        "session_id": "s",
        "turn_id": "t",
        "seq": 1,
        "ts": "2026-05-09T12:00:00+00:00",
        "text": "hi",
    }
    ev = adapter.validate_python(payload)
    assert isinstance(ev, TextDelta)
    assert ev.text == "hi"


def test_wire_event_discriminated_union_message_stop():
    adapter = TypeAdapter(WireEvent)
    payload = {
        "type": "message_stop",
        "protocol_version": "1.0",
        "session_id": "s",
        "turn_id": "t",
        "seq": 99,
        "ts": "2026-05-09T12:00:00+00:00",
        "stop_reason": "complete",
    }
    ev = adapter.validate_python(payload)
    assert isinstance(ev, MessageStop)
    assert ev.stop_reason == "complete"


def test_wire_event_discriminated_union_tool_result_permission():
    """Round-trip a capability-denied ToolResult through the union."""
    adapter = TypeAdapter(WireEvent)
    payload = {
        "type": "tool_result",
        "protocol_version": "1.0",
        "session_id": "s",
        "turn_id": "t",
        "seq": 4,
        "ts": "2026-05-09T12:00:00+00:00",
        "tool_call_id": "c",
        "name": "web_search",
        "status": "error",
        "error_kind": "permission",
        "missing_capabilities": ["network"],
        "error": "denied",
    }
    ev = adapter.validate_python(payload)
    assert isinstance(ev, ToolResult)
    assert ev.error_kind == "permission"
    assert ev.missing_capabilities == ["network"]


def test_wire_event_discriminated_union_invalid_type():
    adapter = TypeAdapter(WireEvent)
    payload = {
        "type": "unknown",
        "session_id": "s",
        "turn_id": "t",
        "seq": 0,
        "ts": "2026-05-09T12:00:00+00:00",
    }
    with pytest.raises(ValidationError):
        adapter.validate_python(payload)


def test_wire_event_round_trip_json():
    """Each variant: model_dump_json -> TypeAdapter parse -> equal."""
    adapter = TypeAdapter(WireEvent)
    events = [
        MessageStart(**_envelope(seq=0)),
        MessageStop(**_envelope(seq=99), stop_reason="complete"),
        TextDelta(**_envelope(seq=1), text="hello"),
        ThinkingDelta(**_envelope(seq=1), text="thinking"),
        ToolCall(**_envelope(seq=2), tool_call_id="c", name="t", arguments={"k": "v"}),
        ToolResult(
            **_envelope(seq=3),
            tool_call_id="c",
            name="t",
            status="ok",
            result={"r": 1},
        ),
        Error(
            **_envelope(seq=10),
            message="x",
            error_type="E",
            is_fatal=False,
        ),
        LoopLimitReached(**_envelope(seq=20), loops=5),
        HwReset(**_envelope(seq=21), model_name="Qwen3-4B"),
    ]
    for ev in events:
        raw = ev.model_dump_json()
        parsed = adapter.validate_json(raw)
        assert parsed == ev


# ---------------------------------------------------------------------------
# ToolDescriptor
# ---------------------------------------------------------------------------


def test_tool_descriptor_defaults():
    td = ToolDescriptor(name="time")
    assert td.name == "time"
    assert td.description == ""
    assert td.parameters == {}


def test_tool_descriptor_extra_forbid():
    with pytest.raises(ValidationError):
        ToolDescriptor(name="t", bogus=1)  # type: ignore[call-arg]
