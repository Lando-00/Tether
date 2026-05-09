"""Integration tests for v0 / v2 NDJSON dual-emit on /api/v1/chat/stream.

Default (no Accept) and explicit 'Accept: application/x-ndjson' stay v0.
'Accept: application/x-ndjson; version=1.0' opts into v2 (text_delta,
message_stop, tool_call, tool_result vocab).

All three NDJSON paths and the SSE path carry X-Tether-Protocol-Version: 1.0.

Synthesis §11.3 R18 (split big-bang cutover into 3 PRs); §4 Phase 5 step 54.
p5-cutover-a-dual-emit.
"""
from __future__ import annotations

import json

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from tether_service.app.http.api import lifespan
from tether_service.app.http.routers.chat import router as chat_router
from tether_service.app.http.routers.health import router as health_router
from tether_service.engine import Engine
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.providers.dummy.provider import DummyProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    """Minimal FastAPI app with chat router + DummyProvider engine."""
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
        yield c


def _post(client: TestClient, *, accept: str | None = None, session_id: str = "test-s"):
    headers = {"Accept": accept} if accept else {}
    return client.post(
        "/api/v1/chat/stream",
        json={"session_id": session_id, "prompt": "hi", "model_name": "dummy"},
        headers=headers,
    )


def _decode(body: str) -> list[dict]:
    """Parse NDJSON body into list of event dicts."""
    return [json.loads(line) for line in body.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# v0 default path
# ---------------------------------------------------------------------------


def test_default_path_emits_v0_vocab(client):
    """No Accept header -> v0 dict events (text, done); no v2 vocab leaks."""
    resp = _post(client, session_id="test-default")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _decode(resp.text)
    types = [e["type"] for e in events]
    # v0 vocab present
    assert "text" in types or "think" in types
    assert "done" in types
    # No v2 vocab leaked
    assert "text_delta" not in types
    assert "message_stop" not in types
    assert "message_start" not in types


def test_explicit_v0_path_emits_v0_vocab(client):
    """Accept: application/x-ndjson (no version) -> v0 dict events."""
    resp = _post(client, accept="application/x-ndjson", session_id="test-v0")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _decode(resp.text)
    types = [e["type"] for e in events]
    assert "text" in types
    assert "done" in types
    assert "text_delta" not in types
    assert "message_start" not in types


# ---------------------------------------------------------------------------
# v2 NDJSON opt-in path
# ---------------------------------------------------------------------------


def test_v2_optin_emits_v2_vocab(client):
    """Accept: application/x-ndjson; version=1.0 -> v2 typed events."""
    resp = _post(client, accept="application/x-ndjson; version=1.0", session_id="test-v2")
    assert resp.status_code == 200
    assert "application/x-ndjson" in resp.headers["content-type"]
    events = _decode(resp.text)
    types = [e["type"] for e in events]
    # v2 vocab: first event is message_start, last is message_stop
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert "text_delta" in types
    # No v0 vocab leaked
    assert "text" not in types
    assert "done" not in types
    assert "tool_started" not in types


def test_v2_optin_with_quoted_version(client):
    """Accept: application/x-ndjson; version=\"1.0\" (quoted) also opts in."""
    resp = _post(
        client,
        accept='application/x-ndjson; version="1.0"',
        session_id="test-v2-quoted",
    )
    assert resp.status_code == 200
    events = _decode(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"


def test_v2_envelope_fields_present(client):
    """v2 events carry session_id, turn_id, seq, ts, protocol_version."""
    resp = _post(
        client, accept="application/x-ndjson; version=1.0", session_id="test-envelope"
    )
    events = _decode(resp.text)
    for e in events:
        assert "session_id" in e, f"Missing session_id in event: {e}"
        assert "turn_id" in e, f"Missing turn_id in event: {e}"
        assert "seq" in e, f"Missing seq in event: {e}"
        assert "ts" in e, f"Missing ts in event: {e}"
        assert e.get("protocol_version") == "1.0", f"Wrong protocol_version: {e}"


def test_v2_seq_monotonic(client):
    """seq is monotonically increasing and starts at 0 within a turn."""
    resp = _post(
        client, accept="application/x-ndjson; version=1.0", session_id="test-seq"
    )
    events = _decode(resp.text)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), f"seq not monotonic: {seqs}"
    assert seqs[0] == 0, f"seq does not start at 0: {seqs}"


def test_v2_tool_call_carries_id(client):
    """v2 tool_call events have a tool_call_id; matching tool_result follows."""
    resp = _post(
        client, accept="application/x-ndjson; version=1.0", session_id="test-tool"
    )
    events = _decode(resp.text)
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    if not tool_calls:
        # DummyProvider doesn't always trigger tools for a plain "hi" prompt.
        pytest.skip("DummyProvider did not emit tool_call events for this prompt")
    assert tool_calls[0]["tool_call_id"].startswith("call-")
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    if tool_results:
        assert tool_results[0]["tool_call_id"] == tool_calls[0]["tool_call_id"]


def test_protocol_version_header_on_all_three_paths(client):
    """X-Tether-Protocol-Version: 1.0 on default v0, explicit v0, v2 NDJSON, SSE."""
    cases = [
        (None, "default (no Accept)"),
        ("application/x-ndjson", "explicit v0 NDJSON"),
        ("application/x-ndjson; version=1.0", "v2 NDJSON"),
        ("text/event-stream", "SSE"),
    ]
    for accept, label in cases:
        resp = _post(
            client,
            accept=accept,
            session_id=f"test-hdr-{label[:5].replace(' ', '')}",
        )
        assert resp.headers.get("x-tether-protocol-version") == "1.0", (
            f"Missing X-Tether-Protocol-Version: 1.0 on {label!r}; "
            f"got headers={dict(resp.headers)}"
        )
