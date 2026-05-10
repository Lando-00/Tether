"""Unit tests for transport_ndjson — v2 NDJSON wire transport.

Synthesis §3.4 (HTTP transport); §4 Phase 5 step 53.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator, List

import pytest

from tether.protocol.wire.events import (
    MessageStop,
    TextDelta,
    ToolCall,
    WireEvent,
)
from tether.protocol.wire.transport_ndjson import transport_ndjson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
_SID = "s1"
_TID = "t1"


def _text_delta(text: str, seq: int = 0) -> TextDelta:
    return TextDelta(session_id=_SID, turn_id=_TID, seq=seq, ts=_NOW, text=text)


def _message_stop(seq: int = 0) -> MessageStop:
    return MessageStop(
        session_id=_SID, turn_id=_TID, seq=seq, ts=_NOW, stop_reason="complete"
    )


def _tool_call(name: str, seq: int = 0) -> ToolCall:
    return ToolCall(
        session_id=_SID,
        turn_id=_TID,
        seq=seq,
        ts=_NOW,
        tool_call_id="tc1",
        name=name,
        arguments={"q": "test"},
    )


async def _as_iter(events: List[WireEvent]) -> AsyncIterator[WireEvent]:
    for e in events:
        yield e


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ndjson_emits_one_line_per_event():
    """Three events → three NDJSON lines."""
    events = [_text_delta("a", 0), _text_delta("b", 1), _message_stop(2)]
    chunks: List[bytes] = []
    async for chunk in transport_ndjson(_as_iter(events)):
        chunks.append(chunk)
    assert len(chunks) == 3


@pytest.mark.anyio
async def test_ndjson_lines_are_v2_json():
    """Each chunk is parseable JSON carrying v2 type names."""
    events = [_text_delta("hi", 0), _message_stop(1)]
    types = []
    async for chunk in transport_ndjson(_as_iter(events)):
        obj = json.loads(chunk.decode("utf-8"))
        types.append(obj["type"])
    assert types == ["text_delta", "message_stop"]


@pytest.mark.anyio
async def test_ndjson_lines_terminated_with_newline():
    """Each yielded chunk ends with b'\\n'."""
    events = [_text_delta("hi", 0), _message_stop(1)]
    async for chunk in transport_ndjson(_as_iter(events)):
        assert chunk.endswith(b"\n"), f"Expected trailing newline, got: {chunk!r}"


@pytest.mark.anyio
async def test_ndjson_empty_stream():
    """Empty input → zero bytes yielded."""

    async def _empty():
        return
        yield  # pragma: no cover

    chunks: List[bytes] = []
    async for chunk in transport_ndjson(_empty()):
        chunks.append(chunk)
    assert chunks == []


@pytest.mark.anyio
async def test_ndjson_handles_text_delta():
    """TextDelta event → JSON with type=text_delta and the correct text."""
    events = [_text_delta("hi", 5)]
    async for chunk in transport_ndjson(_as_iter(events)):
        obj = json.loads(chunk.decode("utf-8"))
        assert obj["type"] == "text_delta"
        assert obj["text"] == "hi"
        assert obj["seq"] == 5


@pytest.mark.anyio
async def test_ndjson_handles_tool_call():
    """ToolCall event → JSON with type=tool_call, name, and arguments."""
    events = [_tool_call("web_search", 3)]
    async for chunk in transport_ndjson(_as_iter(events)):
        obj = json.loads(chunk.decode("utf-8"))
        assert obj["type"] == "tool_call"
        assert obj["name"] == "web_search"
        assert obj["arguments"] == {"q": "test"}
