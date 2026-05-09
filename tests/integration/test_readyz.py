"""
Integration tests for the /readyz health endpoint.

Acceptance A1: /readyz returns ready=true under DummyProvider.
Synthesis §6 row 2 / B6 §1.2 #4.
"""
import pytest
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient
from typing import Any, AsyncGenerator, Dict, List, Optional

from tether_service.app.http.routers.health import router as health_router
from tether_service.core.interfaces import ModelProvider, SessionStore
from tether_service.protocol.service.generation_service import GenerationService


# ---------------------------------------------------------------------------
# Minimal fakes — enough for the readyz endpoint only
# ---------------------------------------------------------------------------

class _MinimalStore(SessionStore):
    """In-memory store: implements only what readyz needs (get_history)."""

    async def create_session(self, session_id: str, created_at: int) -> None:
        pass

    async def list_sessions(self) -> List[Dict[str, Any]]:
        return []

    async def delete_session(self, session_id: str) -> bool:
        return False

    async def delete_all_sessions(self) -> int:
        return 0

    async def add_user(self, session_id: str, text: str) -> None:
        pass

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        pass

    async def add_assistant_toolcall(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        pass

    async def add_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        pass

    async def get_history(self, session_id: str, include_thinking: bool = False) -> List[Dict[str, Any]]:
        return []

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        pass


class _TwoModelProvider(ModelProvider):
    """Returns exactly two dummy model names — used as the happy-path provider."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    def list_models(self) -> List[str]:
        return ["dummy-a", "dummy-b"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _EmptyModelProvider(ModelProvider):
    """list_models() returns [] — simulates no models available."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    def list_models(self) -> List[str]:
        return []

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _BrokenProvider(ModelProvider):
    """list_models() raises — simulates a crashed provider."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError("provider down")
        yield  # pragma: no cover

    def list_models(self) -> List[str]:
        raise RuntimeError("provider down")

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 0


class _BrokenStore(_MinimalStore):
    """get_history() raises — simulates a broken DB."""

    async def get_history(self, session_id: str, include_thinking: bool = False) -> List[Dict[str, Any]]:
        raise RuntimeError("db connection failed")


def _make_app(provider: ModelProvider, store: SessionStore) -> FastAPI:
    """Build a minimal FastAPI app wired to the given provider and store."""
    from tether_service.protocol.parsers.sliding import SlidingParser

    gen_svc = GenerationService(
        provider=provider,
        parser=SlidingParser(),
        session_store=store,
        tools={},
        system_prompt="",
    )
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health_router)
    app.include_router(v1)
    app.state.gen_svc = gen_svc
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_readyz_returns_ready_true_with_dummy_provider():
    """A1: /readyz → ready=true when store and provider are healthy."""
    client = TestClient(_make_app(_TwoModelProvider(), _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "ready": True,
        "store": True,
        "provider": True,
        "models_available": 2,
    }


def test_readyz_returns_ready_false_when_no_models():
    """A1 neg: /readyz → ready=false when list_models returns []."""
    client = TestClient(_make_app(_EmptyModelProvider(), _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is True
    assert body["provider"] is False
    assert "no models available" in body["error"]


def test_readyz_returns_ready_false_when_provider_raises():
    """Provider exception → ready=false, store=true."""
    client = TestClient(_make_app(_BrokenProvider(), _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is True
    assert body["provider"] is False
    assert "provider down" in body["error"]


def test_readyz_returns_ready_false_when_store_raises():
    """Store exception → ready=false, store=false, provider=None."""
    client = TestClient(_make_app(_TwoModelProvider(), _BrokenStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is False
    assert body["provider"] is None
    assert "db connection failed" in body["error"]
