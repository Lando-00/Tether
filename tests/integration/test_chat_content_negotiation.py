"""Integration tests for /api/v1/chat/stream content negotiation.

Default (no Accept header): application/x-ndjson v2 typed events (NEW DEFAULT
after p5-cutover-c-flip-default). v0 is legacy opt-in via version=0 parameter.
Accept: text/event-stream: SSE-framed v2 typed events.

All responses carry X-Tether-Protocol-Version: 1.0. v0 legacy responses
additionally carry Warning: 299 per RFC 9110 §5.6.7.

Uses the same minimal app pattern as test_lifespan_starts_engine and
test_stream_request_bounds: an Engine built with DummyProvider + in-memory
session store, wired directly into the FastAPI lifespan so lifespan
startup/shutdown run correctly.

Synthesis §4 Phase 5 step 53; §11.3 R18 (p5-cutover-c-flip-default).
"""
from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.api import lifespan
from tether.app.http.routers.chat import router as chat_router
from tether.app.http.routers.health import router as health_router
from tether.engine import Engine
from tether.protocol.parsers.sliding import SlidingParser
from tether.providers.dummy.provider import DummyProvider

# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app with the chat router + DummyProvider.

    Uses the real lifespan so Engine.__aenter__ runs (tools startup, etc.).
    Engine.aclose() is patched to a no-op so the DummyProvider's no-op
    hardware teardown doesn't produce confusing log noise, but we could also
    leave it real since DummyProvider has no native handles.
    """
    engine = Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="You are a helpful assistant.",
    )

    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine

    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(chat_router)
    v1.include_router(health_router)
    app.include_router(v1)
    return app


@pytest.fixture
def client():
    """TestClient with lifespan — Engine __aenter__/__aexit__ run."""
    with TestClient(_build_test_app()) as c:
        # DummyProvider needs a live session; create one via the engine directly.
        # We patch the session store so create_session calls work.
        yield c


def _post(client: TestClient, *, accept: str | None = None, session_id: str = "test-s") -> Any:
    """POST to /api/v1/chat/stream and return the response."""
    headers = {"Accept": accept} if accept else {}
    return client.post(
        "/api/v1/chat/stream",
        json={"session_id": session_id, "prompt": "hi", "model_name": "dummy"},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_is_ndjson_v2(client):
    """Default (no Accept) returns application/x-ndjson with v2 typed vocab (NEW DEFAULT)."""
    resp = _post(client, session_id="test-default")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert resp.headers["x-tether-protocol-version"] == "1.0"
    body = resp.text
    lines = [line for line in body.splitlines() if line.strip()]
    # At least one v2 line
    assert len(lines) >= 1
    # v2 vocabulary in use (not v0)
    assert "text_delta" in body
    assert "message_start" in body
    assert "message_stop" in body


def test_explicit_ndjson_no_version_is_v2(client):
    """Accept: application/x-ndjson (no version param) is v2 (same as default)."""
    resp = _post(client, accept="application/x-ndjson", session_id="test-ndjson")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert resp.headers["x-tether-protocol-version"] == "1.0"
    body = resp.text
    lines = [line for line in body.splitlines() if line.strip()]
    assert len(lines) >= 1
    # v2 vocabulary in use
    assert "text_delta" in body


def test_sse_returns_text_event_stream(client):
    """Accept: text/event-stream returns SSE-framed v2 events."""
    resp = _post(client, accept="text/event-stream", session_id="test-sse")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-tether-protocol-version"] == "1.0"
    body = resp.text
    # SSE framing present
    assert "id: " in body
    assert "event: " in body
    assert "data: {" in body
    assert "\n\n" in body  # blank-line terminator


def test_sse_event_types_are_v2(client):
    """SSE stream uses v2 vocabulary (message_start, text_delta, message_stop)."""
    resp = _post(client, accept="text/event-stream", session_id="test-sse-v2")
    assert resp.status_code == 200
    body = resp.text
    assert "event: message_start" in body
    assert "event: text_delta" in body
    assert "event: message_stop" in body


def test_sse_id_field_is_monotonic(client):
    """SSE id fields are monotonically increasing seq numbers starting at 0."""
    resp = _post(client, accept="text/event-stream", session_id="test-sse-seq")
    assert resp.status_code == 200
    body = resp.text
    ids = [int(m.group(1)) for m in re.finditer(r"^id: (\d+)$", body, re.MULTILINE)]
    assert len(ids) >= 2, f"Expected at least 2 SSE events, got ids={ids}"
    assert ids == sorted(ids), f"id fields not monotonic: {ids}"
    assert ids == list(range(len(ids))), f"id fields have gaps: {ids}"


def test_protocol_version_header_on_both_paths(client):
    """Both NDJSON and SSE responses carry X-Tether-Protocol-Version: 1.0."""
    for accept in ["application/x-ndjson", "text/event-stream"]:
        resp = _post(client, accept=accept, session_id="test-hdr")
        assert resp.headers.get("x-tether-protocol-version") == "1.0", (
            f"Missing X-Tether-Protocol-Version: 1.0 on Accept={accept!r}; "
            f"got headers={dict(resp.headers)}"
        )
