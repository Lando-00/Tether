"""Integration tests for mode field routing in /api/v1/chat/stream.

Verifies:
- Default mode ("chat") works end-to-end.
- Explicit mode="chat" works.
- mode="research" returns 501 before streaming (both SSE and NDJSON).
- Unknown mode returns 422 (Pydantic Literal validation).
- Omitting mode defaults to "chat" via Pydantic default.
- engine.stream(mode="research") dispatches via registry and surfaces
  NotImplementedError (not silently falling back to ChattyAgent).

Uses the same minimal-app pattern as test_chat_content_negotiation:
Engine with DummyProvider + AsyncMock session store.

Briefing §2 Seam B item 4; synthesis §3.5.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.api import lifespan
from tether.app.http.routers.chat import StreamRequest, router as chat_router
from tether.app.http.routers.health import router as health_router
from tether.config.settings import Settings
from tether.engine import Engine
from tether.protocol.parsers.sliding import SlidingParser
from tether.providers.dummy.provider import DummyProvider


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    """Minimal FastAPI app with DummyProvider engine (same pattern as
    test_chat_content_negotiation)."""
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
    with TestClient(_build_test_app()) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(client: TestClient, *, mode: str | None = None, accept: str | None = None,
          session_id: str = "test-s") -> object:
    """POST to /api/v1/chat/stream."""
    headers = {"Accept": accept} if accept else {}
    body: dict = {"session_id": session_id, "prompt": "hi", "model_name": "dummy"}
    if mode is not None:
        body["mode"] = mode
    return client.post("/api/v1/chat/stream", json=body, headers=headers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_mode_is_chat(client):
    """Omitting mode defaults to 'chat' → 200 (ChattyAgentOrchestrator runs)."""
    resp = _post(client, session_id="test-default-mode")
    assert resp.status_code == 200


def test_explicit_mode_chat(client):
    """mode='chat' → 200 OK (ChattyAgentOrchestrator selected from registry)."""
    resp = _post(client, mode="chat", session_id="test-explicit-chat")
    assert resp.status_code == 200


def test_research_mode_returns_501_ndjson(client):
    """mode='research' → 501 Not Implemented (before streaming begins).

    NotebookOrchestrator.is_implemented=False so the router short-circuits
    before returning a StreamingResponse. The 501 body mentions the doc path.
    """
    resp = _post(client, mode="research", session_id="test-research-ndjson")
    assert resp.status_code == 501
    detail = resp.json().get("detail", "")
    assert "docs/research/06_context_strategies.md" in detail


def test_research_mode_returns_501_sse(client):
    """mode='research' + Accept: text/event-stream → 501 (not a streamed error).

    The 501 is returned before any streaming body begins, regardless of
    which wire format the client requested.
    """
    resp = _post(
        client,
        mode="research",
        accept="text/event-stream",
        session_id="test-research-sse",
    )
    assert resp.status_code == 501
    detail = resp.json().get("detail", "")
    assert "docs/research/06_context_strategies.md" in detail


def test_unknown_mode_returns_422(client):
    """mode='invalid' → 422 Unprocessable Entity (Pydantic Literal validation).

    FastAPI's default validation error status for body schema violations is 422.
    """
    resp = _post(client, mode="invalid", session_id="test-invalid-mode")
    assert resp.status_code == 422


def test_default_mode_field_is_chat():
    """StreamRequest.mode defaults to 'chat' without sending the field."""
    r = StreamRequest(session_id="s", prompt="p", model_name="m")
    assert r.mode == "chat"


# ---------------------------------------------------------------------------
# Engine.stream mode-dispatch contract (library path)
# ---------------------------------------------------------------------------


def _minimal_settings(tmp_path) -> Settings:
    """Minimal Settings using DummyProvider + in-memory sqlite store."""
    db = tmp_path / "mode_routing_test.db"
    return Settings.model_validate({
        "system": {"prompt": "test"},
        "providers": {
            "model": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "parser": {
                "impl": "tether.protocol.parsers.sliding.SlidingParser",
                "args": {},
            },
            "session_store": {
                "impl": "tether.context.sqlite_store.SqliteSessionStore",
                "args": {},
            },
        },
        "storage": {"sqlite": {"dsn": f"sqlite:///{db}"}},
        "tools": {
            "registry": [],
            "enabled": ["web_search"],
            "disabled": ["time", "weather", "forecast"],
        },
        # ADR-0020 §D6: research mode is opt-in. Explicitly register it
        # so this test's mode='research' call hits NotebookOrchestrator.
        "orchestrator": {
            "registry": {
                "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
                "research": "tether.protocol.orchestration.notebook.NotebookOrchestrator",
            }
        },
    })


@pytest.mark.anyio
async def test_engine_stream_research_mode_raises_not_implemented(tmp_path):
    """Engine.stream(mode='research') must dispatch to NotebookOrchestrator
    and surface its NotImplementedError.

    This is the library-path contract: callers cannot bypass mode dispatch
    via the v0 NDJSON path. Engine.stream routes through Engine.chat which
    uses the registry, so mode='research' hits NotebookOrchestrator.run()
    which raises NotImplementedError. Briefing §2 Seam B item 4.
    """
    settings = _minimal_settings(tmp_path)
    engine = Engine.from_settings(settings)

    with pytest.raises(NotImplementedError):
        async for _ in engine.stream(
            session_id="s",
            prompt="p",
            model_name="dummy",
            mode="research",
        ):
            pass


@pytest.fixture
def anyio_backend():
    return "asyncio"
