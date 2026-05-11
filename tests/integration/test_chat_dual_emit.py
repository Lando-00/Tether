"""Integration tests for v0 / v2 NDJSON dual-emit on /api/v1/chat/stream.

Default (no Accept) is now v2 NDJSON after p5-cutover-c-flip-default.
v0 is legacy opt-in via 'Accept: application/x-ndjson; version=0'.
'Accept: application/x-ndjson; version=1.0' is explicit v2 (same as default).

All four NDJSON paths and the SSE path carry X-Tether-Protocol-Version: 1.0.
v0 legacy responses additionally carry Warning: 299 per RFC 9110 §5.6.7.

Synthesis §11.3 R18 (split big-bang cutover into 3 PRs); §4 Phase 5 step 56.
p5-cutover-c-flip-default.
"""
from __future__ import annotations

import json

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from tether.app.http.api import lifespan
from tether.app.http.routers.chat import router as chat_router
from tether.app.http.routers.health import router as health_router
from tether.engine import Engine
from tether.protocol.parsers.sliding import SlidingParser
from tether.providers.dummy.provider import DummyProvider


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


def _post(
    client: TestClient,
    *,
    accept: str | None = None,
    session_id: str = "test-s",
):
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
# v2 default path (NEW DEFAULT after p5-cutover-c-flip-default)
# ---------------------------------------------------------------------------


def test_default_path_emits_v2_vocab(client):
    """No Accept header -> v2 typed events (message_start, text_delta, message_stop)."""
    resp = _post(client, session_id="test-default")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _decode(resp.text)
    types = [e["type"] for e in events]
    # v2 vocab present
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert "text_delta" in types
    # No v0 vocab leaked
    assert "text" not in types
    assert "done" not in types
    assert "tool_started" not in types


# ---------------------------------------------------------------------------
# v0 legacy opt-in path (DEPRECATED; version=0 explicit)
# ---------------------------------------------------------------------------


def test_v0_legacy_optin_emits_v0_vocab(client):
    """Accept: application/x-ndjson; version=0 -> v0 dict events (legacy opt-in)."""
    resp = _post(client, accept="application/x-ndjson; version=0", session_id="test-v0")
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
    """v2 events carry session_id, turn_id, seq, ts, protocol_version.

    Both default (no Accept) and explicit version=1.0 yield v2 envelopes.
    """
    for accept, label in [
        (None, "default"),
        ("application/x-ndjson; version=1.0", "explicit v2"),
    ]:
        resp = _post(client, accept=accept, session_id=f"test-envelope-{label[:3]}")
        events = _decode(resp.text)
        for e in events:
            assert "session_id" in e, f"[{label}] Missing session_id in event: {e}"
            assert "turn_id" in e, f"[{label}] Missing turn_id in event: {e}"
            assert "seq" in e, f"[{label}] Missing seq in event: {e}"
            assert "ts" in e, f"[{label}] Missing ts in event: {e}"
            assert e.get("protocol_version") == "1.0", f"[{label}] Wrong protocol_version: {e}"


def test_v2_seq_monotonic(client):
    """seq is monotonically increasing and starts at 0 within a turn."""
    resp = _post(
        client, accept="application/x-ndjson; version=1.0", session_id="test-seq"
    )
    events = _decode(resp.text)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), f"seq not monotonic: {seqs}"
    assert seqs[0] == 0, f"seq does not start at 0: {seqs}"


def test_v2_tool_call_carries_id():
    """v2 tool_call events have a tool_call_id; matching tool_result follows.

    The shared ``client`` fixture wires the Engine to an AsyncMock store, so
    DummyProvider receives MagicMock messages instead of the real prompt and
    can never produce a ``<<function_call>>`` marker. This test instead builds
    its own app with a real in-memory store (MinimalMemoryStore from the
    golden conftest) and sends a prompt that already contains a function-call
    marker — DummyProvider echoes it back, the SlidingParser detects it, and
    the orchestrator emits the v2 tool_call wire envelope. P0-G / Tribunal
    P0-16 (A8-F2): the previous version masked a missing tool_call with
    ``pytest.skip``, so the assertion never ran.
    """
    from tests.golden.conftest import MinimalMemoryStore

    class _StoreWithAudit(MinimalMemoryStore):
        async def audit_tool_call(self, **kwargs):
            return None

    fc_marker = '<<function_call>> {"name":"weather","arguments":{}}'

    engine = Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=_StoreWithAudit(),
        tools={},
        system_prompt="You are a helpful assistant.",
    )
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(chat_router)
    v1.include_router(health_router)
    app.include_router(v1)

    with TestClient(app) as c:
        resp = c.post(
            "/api/v1/chat/stream",
            json={
                "session_id": "test-tool",
                "prompt": fc_marker,
                "model_name": "dummy",
            },
            headers={"Accept": "application/x-ndjson; version=1.0"},
        )
    events = _decode(resp.text)
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    assert tool_calls, (
        "expected at least one tool_call event but stream produced none — "
        "either the scripted DummyProvider stopped echoing the prompt or "
        "the orchestrator stopped emitting tool_call wire events "
        "(Tribunal P0-16 / A8-F2)"
    )
    assert tool_calls[0]["tool_call_id"].startswith("call-")
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    if tool_results:
        assert tool_results[0]["tool_call_id"] == tool_calls[0]["tool_call_id"]


def test_protocol_version_header_on_all_four_paths(client):
    """X-Tether-Protocol-Version: 1.0 on default v2, v0 legacy, explicit v2 NDJSON, SSE."""
    cases = [
        (None, "default (no Accept) -> v2"),
        ("application/x-ndjson; version=0", "v0 legacy NDJSON"),
        ("application/x-ndjson; version=1.0", "explicit v2 NDJSON"),
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


# ---------------------------------------------------------------------------
# Warning header tests
# ---------------------------------------------------------------------------


def test_v0_legacy_response_has_warning_header(client):
    """v0 legacy Accept opt-in includes RFC 9110 §5.6.7 Warning: 299 ..."""
    resp = _post(
        client, accept="application/x-ndjson; version=0", session_id="test-warn"
    )
    assert resp.status_code == 200
    warning = resp.headers.get("warning", "")
    assert warning.startswith("299"), f"missing or malformed Warning header: {warning!r}"
    assert "deprecated" in warning.lower()
    assert "version=1.0" in warning  # tells callers how to migrate


def test_default_no_warning_header(client):
    """v2 default response does NOT carry the deprecation Warning."""
    resp = _post(client, session_id="test-no-warn")
    assert resp.status_code == 200
    assert "warning" not in {k.lower() for k in resp.headers.keys()}
