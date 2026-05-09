"""Integration tests for protocol introspection endpoints.

``GET /api/v1/protocol/schema``  -> JSON Schema for WireEvent
``GET /api/v1/protocol/example`` -> canonical NDJSON turn recording

Phase 5 step 50 (synthesis §4).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from tether_service.app.http.api import create_app
from tether_service.protocol.wire.events import (
    PROTOCOL_VERSION,
    MessageStart,
    MessageStop,
    ToolCall,
    ToolResult,
    WireEvent,
)


@pytest.fixture
def client():
    """Use the lifespan-aware client so engine startup/shutdown run."""
    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/v1/protocol/schema
# ---------------------------------------------------------------------------


def test_schema_returns_json(client):
    r = client.get("/api/v1/protocol/schema")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert "protocol_version" in body
    assert "schema" in body


def test_schema_protocol_version_matches(client):
    r = client.get("/api/v1/protocol/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["protocol_version"] == "1.0"
    assert body["protocol_version"] == PROTOCOL_VERSION


def test_schema_includes_all_event_types(client):
    """The schema's $defs (or top-level oneOf/anyOf) must reference every
    WireEvent variant by class name."""
    r = client.get("/api/v1/protocol/schema")
    assert r.status_code == 200
    schema = r.json()["schema"]
    raw = json.dumps(schema)
    # Each Pydantic model name should appear somewhere in the schema.
    expected = [
        "MessageStart",
        "MessageStop",
        "TextDelta",
        "ThinkingDelta",
        "ToolCall",
        "ToolResult",
        "Error",
        "LoopLimitReached",
        "HwReset",
    ]
    for name in expected:
        assert name in raw, f"WireEvent variant {name!r} missing from schema"


def test_schema_includes_discriminator(client):
    """Schema should declare 'type' as the discriminator (Pydantic emits a
    'discriminator' object with 'propertyName' = 'type')."""
    r = client.get("/api/v1/protocol/schema")
    schema = r.json()["schema"]
    raw = json.dumps(schema)
    assert "discriminator" in raw
    assert "propertyName" in raw
    assert '"type"' in raw


def test_schema_lists_each_stop_reason(client):
    """Synthesis §11.3 R1: cancelled and client_disconnect are valid
    stop_reasons. Verify they appear in the schema."""
    r = client.get("/api/v1/protocol/schema")
    raw = json.dumps(r.json()["schema"])
    for reason in [
        "complete",
        "tool_loop_exhausted",
        "cancelled",
        "client_disconnect",
        "error",
    ]:
        assert reason in raw, f"stop_reason {reason!r} missing from schema"


def test_schema_lists_each_error_kind(client):
    """Synthesis §11.3 R6: error_kind must include permission."""
    r = client.get("/api/v1/protocol/schema")
    raw = json.dumps(r.json()["schema"])
    for kind in ["permission", "execution", "timeout", "cancelled"]:
        assert kind in raw, f"error_kind {kind!r} missing from schema"


# ---------------------------------------------------------------------------
# /api/v1/protocol/example
# ---------------------------------------------------------------------------


def test_example_returns_ndjson(client):
    r = client.get("/api/v1/protocol/example")
    assert r.status_code == 200
    # PlainTextResponse default content-type is text/plain.
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "\n" in body
    lines = [ln for ln in body.splitlines() if ln.strip()]
    assert len(lines) >= 2


def test_example_starts_with_message_start(client):
    body = client.get("/api/v1/protocol/example").text
    first = body.splitlines()[0]
    parsed = json.loads(first)
    assert parsed["type"] == "message_start"


def test_example_ends_with_message_stop(client):
    body = client.get("/api/v1/protocol/example").text
    lines = [ln for ln in body.splitlines() if ln.strip()]
    last = lines[-1]
    parsed = json.loads(last)
    assert parsed["type"] == "message_stop"


def test_example_lines_validate_against_wireevent(client):
    """Every line in the example must round-trip through TypeAdapter(WireEvent)."""
    body = client.get("/api/v1/protocol/example").text
    adapter = TypeAdapter(WireEvent)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    parsed_events = [adapter.validate_json(ln) for ln in lines]
    assert len(parsed_events) >= 2
    assert isinstance(parsed_events[0], MessageStart)
    assert isinstance(parsed_events[-1], MessageStop)


def test_example_demonstrates_tool_round_trip(client):
    """Example must show a tool_call followed by a tool_result that
    references the same tool_call_id."""
    body = client.get("/api/v1/protocol/example").text
    adapter = TypeAdapter(WireEvent)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    events = [adapter.validate_json(ln) for ln in lines]

    call_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolCall))
    result_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolResult))
    assert call_idx < result_idx
    call: ToolCall = events[call_idx]  # type: ignore[assignment]
    result: ToolResult = events[result_idx]  # type: ignore[assignment]
    assert call.tool_call_id == result.tool_call_id
    assert call.name == result.name
    assert result.status == "ok"


def test_example_seq_is_strictly_increasing(client):
    """seq within a turn is monotonically increasing per envelope contract."""
    body = client.get("/api/v1/protocol/example").text
    adapter = TypeAdapter(WireEvent)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    events = [adapter.validate_json(ln) for ln in lines]
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_example_protocol_version_on_every_event(client):
    body = client.get("/api/v1/protocol/example").text
    lines = [ln for ln in body.splitlines() if ln.strip()]
    for ln in lines:
        obj = json.loads(ln)
        assert obj.get("protocol_version") == "1.0"
