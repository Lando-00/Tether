"""Protocol introspection endpoints.

``GET /api/v1/protocol/schema``
    JSON Schema for :data:`WireEvent` (generated via Pydantic
    :class:`TypeAdapter`).

``GET /api/v1/protocol/example``
    Canonical NDJSON recording of one full turn (with tool execution).

Phase 5 foundation. ``p5-cutover-*`` will use the schema endpoint for
content negotiation. Phase 8 step 91 will checkpoint a frozen JSON
artifact at ``docs/protocol/events.schema.json`` with a CI freshness
check; for now both responses are generated dynamically.

Synthesis §4 Phase 5 step 50.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import TypeAdapter

from tether.protocol.wire.events import (
    PROTOCOL_VERSION,
    MessageStart,
    MessageStop,
    TextDelta,
    ToolCall,
    ToolDescriptor,
    ToolResult,
    WireEvent,
)

router = APIRouter(prefix="/protocol", tags=["protocol"])


@router.get("/schema")
def get_schema() -> Dict[str, Any]:
    """Return the JSON Schema for :data:`WireEvent`.

    Generated dynamically from the Pydantic types via
    :class:`TypeAdapter`. ``p5-cutover-*`` will use this for content
    negotiation; Phase 8 step 91 will land a checked-in artifact at
    ``docs/protocol/events.schema.json`` with a CI freshness check.
    """
    adapter = TypeAdapter(WireEvent)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "schema": adapter.json_schema(),
    }


@router.get("/example", response_class=PlainTextResponse)
def get_example() -> str:
    """Return a canonical NDJSON recording of one full turn.

    Static for now (Phase 8 will move this into a checked-in fixture).
    Demonstrates: ``message_start`` -> ``text_delta`` -> ``tool_call`` ->
    ``tool_result`` -> ``text_delta`` -> ``message_stop``.
    """
    sid = "<example_session>"
    tid = "<example_turn>"
    now = datetime.now(timezone.utc)

    events: List[WireEvent] = [
        MessageStart(
            session_id=sid,
            turn_id=tid,
            seq=0,
            ts=now,
            available_tools=[
                ToolDescriptor(name="time", description="Get current time"),
                ToolDescriptor(name="weather", description="Get weather forecast"),
            ],
        ),
        TextDelta(
            session_id=sid,
            turn_id=tid,
            seq=1,
            ts=now,
            text="Let me check the time. ",
        ),
        ToolCall(
            session_id=sid,
            turn_id=tid,
            seq=2,
            ts=now,
            tool_call_id="call-001",
            name="time",
            arguments={"timezone": "UTC"},
        ),
        ToolResult(
            session_id=sid,
            turn_id=tid,
            seq=3,
            ts=now,
            tool_call_id="call-001",
            name="time",
            status="ok",
            result={"time": "2026-05-09T12:00:00+00:00"},
        ),
        TextDelta(
            session_id=sid,
            turn_id=tid,
            seq=4,
            ts=now,
            text="The current time is noon UTC.",
        ),
        MessageStop(
            session_id=sid,
            turn_id=tid,
            seq=5,
            ts=now,
            stop_reason="complete",
        ),
    ]

    lines = [event.model_dump_json() for event in events]
    return "\n".join(lines) + "\n"


__all__ = ["router"]
