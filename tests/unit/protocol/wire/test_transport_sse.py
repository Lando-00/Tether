"""Unit tests for transport_sse — SSE wire transport.

Synthesis §3.4 (HTTP transport); §4 Phase 5 step 53.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator, List

import pytest

from tether_service.protocol.wire.events import (
    MessageStart,
    MessageStop,
    TextDelta,
    ToolDescriptor,
    WireEvent,
)
from tether_service.protocol.wire.transport_sse import transport_sse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
_SID = "s1"
_TID = "t1"


def _text_delta(text: str, seq: int = 0) -> TextDelta:
    return TextDelta(session_id=_SID, turn_id=_TID, seq=seq, ts=_NOW, text=text)


def _message_stop(seq: int = 0, stop_reason: str = "complete") -> MessageStop:
    return MessageStop(
        session_id=_SID,
        turn_id=_TID,
        seq=seq,
        ts=_NOW,
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


def _message_start(seq: int = 0) -> MessageStart:
    return MessageStart(
        session_id=_SID,
        turn_id=_TID,
        seq=seq,
        ts=_NOW,
        available_tools=[
            ToolDescriptor(name="web_search", description="Search the web")
        ],
    )


async def _as_iter(events: List[WireEvent]) -> AsyncIterator[WireEvent]:
    for e in events:
        yield e


async def _collect(events: List[WireEvent]) -> str:
    """Collect all SSE chunks into a single string for assertions."""
    parts: List[bytes] = []
    async for chunk in transport_sse(_as_iter(events)):
        parts.append(chunk)
    return b"".join(parts).decode("utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sse_emits_id_event_data():
    """Each SSE block contains id:, event:, data: fields and a blank line."""
    body = await _collect([_text_delta("hi", 0)])
    assert "id: 0" in body
    assert "event: text_delta" in body
    assert "data: {" in body
    assert body.endswith("\n\n")


@pytest.mark.anyio
async def test_sse_id_is_seq():
    """The id field equals the WireEvent seq number."""
    body = await _collect([_text_delta("x", 7)])
    lines = body.splitlines()
    id_line = next(l for l in lines if l.startswith("id: "))
    assert id_line == "id: 7"


@pytest.mark.anyio
async def test_sse_event_is_type():
    """The event field equals the WireEvent type discriminator."""
    body = await _collect([_message_stop(2)])
    lines = body.splitlines()
    event_line = next(l for l in lines if l.startswith("event: "))
    assert event_line == "event: message_stop"


@pytest.mark.anyio
async def test_sse_data_is_compact_json():
    """The data field is a single-line JSON object (no embedded newlines)."""
    body = await _collect([_text_delta("hello world", 0)])
    data_line = next(l for l in body.splitlines() if l.startswith("data: "))
    json_str = data_line[len("data: "):]
    # Must parse as valid JSON
    obj = json.loads(json_str)
    assert obj["type"] == "text_delta"
    assert obj["text"] == "hello world"
    # No embedded newlines in the data value itself
    assert "\n" not in json_str


@pytest.mark.anyio
async def test_sse_blank_line_terminator():
    """Each SSE block ends with a blank line (\\n\\n)."""
    events = [_text_delta("a", 0)]
    chunks: List[bytes] = []
    async for chunk in transport_sse(_as_iter(events)):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].endswith(b"\n\n")


@pytest.mark.anyio
async def test_sse_multiple_events_separated():
    """Three events produce three SSE blocks each ending with \\n\\n."""
    events = [_text_delta("a", 0), _text_delta("b", 1), _message_stop(2)]
    body = await _collect(events)
    # Count blank-line terminators
    double_newline_count = body.count("\n\n")
    assert double_newline_count == 3


@pytest.mark.anyio
async def test_sse_handles_text_delta():
    """TextDelta(text='hi', seq=5) produces correct id, event, data fields."""
    events = [_text_delta("hi", 5)]
    body = await _collect(events)
    lines = body.splitlines()
    assert "id: 5" in lines
    assert "event: text_delta" in lines
    data_line = next(l for l in lines if l.startswith("data: "))
    obj = json.loads(data_line[len("data: "):])
    assert obj["type"] == "text_delta"
    assert obj["text"] == "hi"
    assert obj["seq"] == 5


@pytest.mark.anyio
async def test_sse_handles_message_stop():
    """MessageStop produces event: message_stop with stop_reason in data."""
    events = [_message_stop(10, stop_reason="complete")]
    body = await _collect(events)
    assert "event: message_stop" in body
    data_line = next(l for l in body.splitlines() if l.startswith("data: "))
    obj = json.loads(data_line[len("data: "):])
    assert obj["type"] == "message_stop"
    assert obj["stop_reason"] == "complete"


@pytest.mark.anyio
async def test_sse_handles_message_start():
    """MessageStart with available_tools produces correct SSE frame."""
    events = [_message_start(0)]
    body = await _collect(events)
    assert "event: message_start" in body
    data_line = next(l for l in body.splitlines() if l.startswith("data: "))
    obj = json.loads(data_line[len("data: "):])
    assert obj["type"] == "message_start"
    assert len(obj["available_tools"]) == 1
    assert obj["available_tools"][0]["name"] == "web_search"


@pytest.mark.anyio
async def test_sse_empty_stream():
    """Empty input → zero bytes yielded."""

    async def _empty():
        return
        yield  # pragma: no cover

    chunks: List[bytes] = []
    async for chunk in transport_sse(_empty()):
        chunks.append(chunk)
    assert chunks == []
