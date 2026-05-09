"""
Integration tests for the /readyz health endpoint.

Phase 3 step 37 (synthesis §6 row 2 / B6 §1.2 #4 / §4 Phase 3):
``/readyz`` now uses :class:`HardwareWatchdog.health_summary()` when the
engine carries a watchdog (always true via ``Engine.from_settings``); falls
back to ``provider.list_models()`` otherwise.
"""
import pytest
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient
from typing import Any, AsyncGenerator, Dict, List, Optional

from tether_service.app.http.routers.health import router as health_router
from tether_service.core.interfaces import ModelProvider, SessionStore
from tether_service.engine import Engine
from tether_service.providers.hw import HardwareLifecycle, HwErrorClass, HwHealth
from tether_service.runtime.hw_watchdog import HardwareWatchdog


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


class _DummyProvider(ModelProvider):
    """Non-HW provider — does NOT implement HardwareLifecycle."""

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
    """list_models() returns [] — only used in the fallback path."""

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


class _BrokenStore(_MinimalStore):
    """get_history() raises — simulates a broken DB."""

    async def get_history(self, session_id: str, include_thinking: bool = False) -> List[Dict[str, Any]]:
        raise RuntimeError("db connection failed")


class _FakeHWProvider(ModelProvider, HardwareLifecycle):
    """Minimal HardwareLifecycle implementation for /readyz tests.

    Returns the configured ``HwHealth`` from ``hw_health()``. Other
    HardwareLifecycle members are stubs the readyz path does not exercise.
    """

    def __init__(self, health: HwHealth):
        self._health = health

    # ModelProvider
    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    def list_models(self) -> List[str]:
        return ["hw-a"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096

    # HardwareLifecycle
    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        return HwErrorClass.TRANSIENT

    async def hw_reset(self, model_name: str) -> None:
        return None

    async def hw_health(self) -> HwHealth:
        return self._health

    @property
    def hw_shutdown_budget_sec(self) -> float:
        return 3.0

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        return 0.75


def _make_app(provider: ModelProvider, store: SessionStore, *, with_watchdog: bool = True) -> FastAPI:
    """Build a minimal FastAPI app wired to the given provider and store.

    ``with_watchdog=True`` (default) wraps the provider in a
    :class:`HardwareWatchdog`; non-HW providers are filtered out at
    construction so the watchdog reports an empty ``providers`` list.
    ``with_watchdog=False`` exercises the legacy fallback path
    (``list_models``) — useful when checking that an Engine constructed
    without ``from_settings`` still works.
    """
    from tether_service.protocol.parsers.sliding import SlidingParser

    watchdog: Optional[HardwareWatchdog] = (
        HardwareWatchdog([provider]) if with_watchdog else None
    )

    gen_svc = Engine(
        provider=provider,
        parser=SlidingParser(),
        session_store=store,
        tools={},
        system_prompt="",
        hw_watchdog=watchdog,
    )
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health_router)
    app.include_router(v1)
    app.state.gen_svc = gen_svc
    return app


# ---------------------------------------------------------------------------
# Tests — Phase 3 step 37 shape
# ---------------------------------------------------------------------------

def test_readyz_with_dummy_provider():
    """DummyProvider isn't HardwareLifecycle → watchdog has zero HW
    providers → health_summary returns {"providers": [], "overall":
    "healthy"}. ready=true, hw_health on the wire."""
    client = TestClient(_make_app(_DummyProvider(), _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["store"] is True
    assert body["provider"] is True
    assert body["hw_health"]["overall"] == "healthy"
    assert body["hw_health"]["providers"] == []
    # Old field shouldn't appear on the watchdog path.
    assert "models_available" not in body


def test_readyz_store_failure():
    """Store throws → ready=false, store=false, provider=None.
    Same shape regardless of watchdog presence."""
    client = TestClient(_make_app(_DummyProvider(), _BrokenStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is False
    assert body["provider"] is None
    assert "db connection failed" in body["error"]


def test_readyz_with_fake_hw_provider_healthy():
    """A HardwareLifecycle provider reporting ``healthy`` produces
    ready=true, hw_health.overall='healthy', a single provider entry."""
    provider = _FakeHWProvider(HwHealth(status="healthy", details={"loaded_models": 2}))
    client = TestClient(_make_app(provider, _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["store"] is True
    assert body["provider"] is True
    assert body["hw_health"]["overall"] == "healthy"
    assert len(body["hw_health"]["providers"]) == 1
    assert body["hw_health"]["providers"][0]["status"] == "healthy"


def test_readyz_with_fake_hw_provider_degraded_is_ready():
    """``degraded`` is acceptable for /readyz — cold-cache MLC providers
    report degraded until a model is loaded."""
    provider = _FakeHWProvider(HwHealth(status="degraded", details={"loaded_models": 0}))
    client = TestClient(_make_app(provider, _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    assert body["ready"] is True
    assert body["hw_health"]["overall"] == "degraded"


def test_readyz_with_fake_hw_provider_error():
    """Provider reporting ``error`` makes /readyz return ready=false."""
    provider = _FakeHWProvider(HwHealth(status="error", details={"reason": "all engines crashed"}))
    client = TestClient(_make_app(provider, _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is True
    assert body["provider"] is False
    assert body["error"] == "hw_health: error"
    assert body["hw_health"]["overall"] == "error"


# ---------------------------------------------------------------------------
# Tests — fallback path (hw_watchdog=None)
# ---------------------------------------------------------------------------

def test_readyz_no_watchdog_fallback():
    """Engine built directly with hw_watchdog=None falls back to the
    list_models() probe. Old wire shape (models_available)."""
    client = TestClient(
        _make_app(_DummyProvider(), _MinimalStore(), with_watchdog=False)
    )
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["store"] is True
    assert body["provider"] is True
    assert body["models_available"] == 2
    assert "hw_health" not in body


def test_readyz_no_watchdog_empty_models():
    """Fallback path with empty list_models() → ready=false."""
    client = TestClient(
        _make_app(_EmptyModelProvider(), _MinimalStore(), with_watchdog=False)
    )
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is True
    assert body["provider"] is False
    assert "no models available" in body["error"]
