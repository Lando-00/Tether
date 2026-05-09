"""Error paths in ``/api/v1/chat/stream`` must include a terminal event
after the error frame so consumers don't block on a missing ``MessageStop``
/ ``done``.

Phase 5 followups F7 (rubber-duck review): previously, error handlers
emitted only the error frame; v2 clients would block on MessageStop
until their socket timeout. Each transport now synthesizes a terminal
frame (``done`` for v0; ``message_stop`` with stop_reason='error' for
v2 / SSE).

Synthesis §3.5; §11.3 R18.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether_service.app.http.api import lifespan
from tether_service.app.http.routers.chat import router as chat_router


# ---------------------------------------------------------------------------
# Failing engine stub — chat() / stream() raise mid-iteration so the chat
# router's outer try/except fires (the F7 path).
# ---------------------------------------------------------------------------


class _FailingEngine:
    """Stub Engine whose chat() and stream() raise mid-iteration.

    The chat router routes through ``engine.chat`` (NDJSON v2 + SSE) and
    ``engine.stream`` (NDJSON v0 legacy); both must surface the F7
    terminal events.
    """

    def __init__(self):
        self._orchestrator_registry: Dict[str, str] = {
            "chat": (
                "tether_service.protocol.orchestration.chatty"
                ".ChattyAgentOrchestrator"
            ),
            "research": (
                "tether_service.protocol.orchestration.notebook"
                ".NotebookOrchestrator"
            ),
        }

    async def chat(self, **kwargs) -> AsyncGenerator[Any, None]:
        # Make this an async generator that raises before yielding.
        if False:
            yield  # pragma: no cover
        raise RuntimeError("boom-chat")

    async def stream(self, **kwargs) -> AsyncGenerator[bytes, None]:
        if False:
            yield  # pragma: no cover
        raise RuntimeError("boom-stream")

    # No-op lifecycle so the FastAPI lifespan can __aenter__/__aexit__.
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _build_failing_app() -> FastAPI:
    """Build a minimal FastAPI app whose engine raises on every chat call."""
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = _FailingEngine()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(chat_router)
    app.include_router(v1)
    return app


@pytest.fixture
def failing_client():
    with TestClient(_build_failing_app()) as c:
        yield c


def _post(client: TestClient, *, accept: str | None = None) -> Any:
    headers = {"Accept": accept} if accept else {}
    return client.post(
        "/api/v1/chat/stream",
        json={"session_id": "sid-f7", "prompt": "hi", "model_name": "dummy"},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# v0 — error frame followed by ``done``
# ---------------------------------------------------------------------------


def test_v0_error_path_emits_error_then_done(failing_client):
    """v0 NDJSON error response includes both 'error' and 'done' events,
    and the error frame's ``ts`` is an ISO timestamp (not None).
    """
    resp = _post(failing_client, accept="application/x-ndjson; version=0")
    assert resp.status_code == 200
    lines = [json.loads(l) for l in resp.text.splitlines() if l.strip()]
    types = [l["type"] for l in lines]
    assert "error" in types
    assert types[-1] == "done", f"v0 last event must be 'done', got types={types}"

    # F7: error frame's ``ts`` is an ISO timestamp, not None.
    err = next(l for l in lines if l["type"] == "error")
    assert err["ts"] is not None
    assert isinstance(err["ts"], str) and "T" in err["ts"], (
        f"expected ISO timestamp on v0 error.ts, got {err['ts']!r}"
    )

    # F7: terminal ``done`` frame is well-formed.
    done = next(l for l in lines if l["type"] == "done")
    assert done["session_id"] == "sid-f7"
    assert isinstance(done["ts"], str) and "T" in done["ts"]


# ---------------------------------------------------------------------------
# v2 NDJSON — error frame followed by ``message_stop`` (stop_reason='error')
# ---------------------------------------------------------------------------


def test_v2_error_path_emits_error_then_message_stop(failing_client):
    """v2 NDJSON error response: 'error' frame then 'message_stop'
    (stop_reason='error') so consumers don't block on missing terminal.
    """
    resp = _post(failing_client, accept="application/x-ndjson")
    assert resp.status_code == 200
    lines = [json.loads(l) for l in resp.text.splitlines() if l.strip()]
    types = [l["type"] for l in lines]
    assert "error" in types
    assert types[-1] == "message_stop", (
        f"v2 last event must be 'message_stop', got types={types}"
    )

    stop = next(l for l in lines if l["type"] == "message_stop")
    assert stop["stop_reason"] == "error"
    # v2 envelope fields are present.
    assert stop["protocol_version"] == "1.0"
    assert stop["session_id"] == "sid-f7"
    assert isinstance(stop["seq"], int)


# ---------------------------------------------------------------------------
# SSE — error event followed by message_stop event
# ---------------------------------------------------------------------------


def test_sse_error_path_emits_error_then_message_stop(failing_client):
    """SSE error response includes ``event: error`` and ``event: message_stop``
    frames so SSE consumers see a terminal event."""
    resp = _post(failing_client, accept="text/event-stream")
    assert resp.status_code == 200
    body = resp.text
    assert "event: error" in body
    assert "event: message_stop" in body
    # message_stop should appear AFTER error.
    err_idx = body.index("event: error")
    stop_idx = body.index("event: message_stop")
    assert err_idx < stop_idx, (
        f"message_stop must follow error (err@{err_idx}, stop@{stop_idx})"
    )
    # The message_stop data carries stop_reason='error'.
    # Find the data line right after "event: message_stop".
    after_stop = body[stop_idx:]
    # Each SSE frame is "event: X\ndata: {...}\n\n".
    data_line = next(
        (
            l
            for l in after_stop.splitlines()
            if l.startswith("data: ")
        ),
        None,
    )
    assert data_line is not None
    payload = json.loads(data_line[len("data: "):])
    assert payload.get("stop_reason") == "error"
