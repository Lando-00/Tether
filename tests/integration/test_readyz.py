"""
Integration tests for the /readyz health endpoint.

Phase 3 step 37 (synthesis §6 row 2 / B6 §1.2 #4 / §4 Phase 3):
``/readyz`` now uses :class:`HardwareWatchdog.health_summary()` when the
engine carries a watchdog (always true via ``Engine.from_settings``); falls
back to ``provider.list_models()`` otherwise.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.routers.health import router as health_router
from tether.core.interfaces import ModelProvider, SessionStore
from tether.engine import Engine
from tether.providers.hw import HardwareLifecycle, HwErrorClass, HwHealth
from tether.runtime.abandoned_tasks import get_notebook_abandoned_task_tracker
from tether.runtime.hw_watchdog import HardwareWatchdog

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

    async def add_user(self, session_id: str, text: str, *, turn_id=None, seq_start=None) -> None:
        pass

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
        *,
        turn_id=None,
        seq_start=None,
    ) -> None:
        pass

    async def add_assistant_toolcall(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        *,
        turn_id=None,
        tool_call_id=None,
        seq_start=None,
    ) -> None:
        pass

    async def add_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any,
        *,
        turn_id=None,
        tool_call_id=None,
        seq_start=None,
        status="ok",
        error=None,
        duration_ms=None,
    ) -> None:
        pass

    async def get_history(self, session_id: str, include_thinking: bool = False) -> List[Dict[str, Any]]:
        return []

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        pass

    async def start_turn(self, session_id: str, turn_id: str, *, model_name=None) -> None:
        pass

    async def complete_turn(self, turn_id: str, *, status="completed", stop_reason=None, error_json=None) -> None:
        pass

    async def record_raw_event(self, session_id, turn_id, seq, event_type, payload, *, tool_call_id=None) -> None:
        pass


class _DummyProvider(ModelProvider):
    """Non-HW provider — does NOT implement HardwareLifecycle."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
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
        *,
        request_id: Optional[str] = None,
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
        self.reset_calls = 0

    # ModelProvider
    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
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
        self.reset_calls += 1
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
    from tether.protocol.parsers.sliding import SlidingParser

    watchdog: Optional[HardwareWatchdog] = HardwareWatchdog([provider]) if with_watchdog else None

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


@pytest.mark.parametrize(
    ("task_count", "expected_status"),
    [(0, "healthy"), (8, "degraded"), (16, "error")],
)
def test_readyz_notebook_cleanup_is_informational_only(task_count, expected_status):
    """All cleanup states are visible but do not affect provider readiness."""
    tracker = get_notebook_abandoned_task_tracker()
    tracker._reset_for_tests()

    class _PendingTask:
        def add_done_callback(self, callback):
            self.callback = callback

    try:
        for _ in range(task_count):
            tracker.track(_PendingTask(), kind="anext")
        provider = _FakeHWProvider(HwHealth(status="healthy", details={}))
        body = TestClient(_make_app(provider, _MinimalStore())).get("/api/v1/readyz").json()
        cleanup = body["operational_health"]["notebook_cleanup"]
        assert body["ready"] is True
        assert body["provider"] is True
        assert cleanup["status"] == expected_status
        assert cleanup["count"] == task_count
        assert cleanup["overflowed"] is False
        assert provider.reset_calls == 0
    finally:
        tracker._reset_for_tests()


def test_readyz_overflowed_notebook_cleanup_remains_informational():
    tracker = get_notebook_abandoned_task_tracker()
    tracker._reset_for_tests()

    class _PendingTask:
        def add_done_callback(self, callback):
            self.callback = callback

    try:
        for _ in range(33):
            tracker.track(_PendingTask(), kind="anext")
        provider = _FakeHWProvider(HwHealth(status="healthy", details={}))
        body = TestClient(_make_app(provider, _MinimalStore())).get("/api/v1/readyz").json()
        cleanup = body["operational_health"]["notebook_cleanup"]
        assert body["ready"] is True
        assert body["provider"] is True
        assert cleanup["status"] == "error"
        assert cleanup["overflowed"] is True
        assert cleanup["count"] == 32
        assert provider.reset_calls == 0
    finally:
        tracker._reset_for_tests()


def test_readyz_store_failure():
    """Store throws → ready=false, store=false.
    ADR-0021: provider field tracks registry health (True when the provider
    is constructed) even when the store is down. Same shape regardless of
    watchdog presence."""
    client = TestClient(_make_app(_DummyProvider(), _BrokenStore()))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is False
    # ADR-0021: provider tracks any-healthy-provider; True because the Engine
    # has a healthy provider in its registry even when the store is broken.
    assert body["provider"] is True
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
    """Provider reporting ``error`` makes /readyz return ready=false.
    ADR-0021: ``provider`` bool tracks registry health independent of HW
    state — the provider IS in the registry and healthy at the registry
    level even when its HW layer reports error."""
    provider = _FakeHWProvider(HwHealth(status="error", details={"reason": "all engines crashed"}))
    client = TestClient(_make_app(provider, _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is True
    # ADR-0021: provider=True because the registry has a healthy entry;
    # HW errors are surfaced via hw_health, not the provider bool.
    assert body["provider"] is True
    assert body["error"] == "hw_health: error"
    assert body["hw_health"]["overall"] == "error"


# ---------------------------------------------------------------------------
# Tests — fallback path (hw_watchdog=None)
# ---------------------------------------------------------------------------


def test_readyz_no_watchdog_fallback():
    """Engine built directly with hw_watchdog=None falls back to the
    list_models() probe. Old wire shape (models_available)."""
    client = TestClient(_make_app(_DummyProvider(), _MinimalStore(), with_watchdog=False))
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
    client = TestClient(_make_app(_EmptyModelProvider(), _MinimalStore(), with_watchdog=False))
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    assert body["ready"] is False
    assert body["store"] is True
    assert body["provider"] is False
    assert "no models available" in body["error"]


# ---------------------------------------------------------------------------
# Tests — Phase 4.5 step 47e: /readyz includes a connectors block
# ---------------------------------------------------------------------------

from typing import AsyncIterator  # noqa: E402

from tether.connectors.base import Connector  # noqa: E402
from tether.connectors.types import (  # noqa: E402
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether.core.connector_registry import ConnectorRegistry  # noqa: E402
from tether.core.interfaces import Tool  # noqa: E402


class _StubConnectorTool(Tool):
    """Trivial Tool used by readyz fake connectors. startup/shutdown are
    no-ops so :func:`tools.lifecycle.startup_all` accepts them."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        return {"name": self._name, "parameters": {"type": "object"}}

    async def invoke(self, args: Dict[str, Any], *, context: Any = None) -> Any:
        return None

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


class _ReadyzConnector(Connector):
    id = "readyz_test"

    def __init__(
        self,
        *,
        health_state: ConnectorState = ConnectorState.READY,
        detail: Optional[str] = "all good",
    ) -> None:
        self._health_state = health_state
        self._detail = detail

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def logout(self) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(state=self._health_state, detail=self._detail)

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=self._health_state)

    async def begin_login(self) -> LoginPrompt:
        return LoginPrompt(kind="url", payload="https://example.com")

    async def complete_login(self, *, payload: Dict[str, Any]) -> LoginContinueResult:
        return LoginContinueResult(state=ConnectorState.READY)

    def tools(self) -> Dict[str, Tool]:
        return {"readyz_test_tool": _StubConnectorTool("readyz_test_tool")}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


def _make_app_with_connector(
    provider: ModelProvider,
    store: SessionStore,
    connector: Optional[Connector],
    *,
    with_watchdog: bool = True,
) -> FastAPI:
    """Variant of _make_app that attaches a ConnectorRegistry."""
    from tether.protocol.parsers.sliding import SlidingParser

    watchdog: Optional[HardwareWatchdog] = HardwareWatchdog([provider]) if with_watchdog else None
    registry = ConnectorRegistry([connector] if connector is not None else [], data_dir=None)
    gen_svc = Engine(
        provider=provider,
        parser=SlidingParser(),
        session_store=store,
        tools=dict(registry.aggregate_tools()),
        system_prompt="",
        hw_watchdog=watchdog,
        connector_registry=registry,
    )
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health_router)
    app.include_router(v1)
    app.state.gen_svc = gen_svc
    return app


def test_readyz_with_connector_registry():
    """A connector registered → /readyz body carries a ``connectors``
    array with each connector's ``{id, state, detail}`` snapshot."""
    conn = _ReadyzConnector(health_state=ConnectorState.READY, detail="ok")
    client = TestClient(_make_app_with_connector(_DummyProvider(), _MinimalStore(), conn))
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert "connectors" in body
    assert body["connectors"] == [{"id": "readyz_test", "state": "ready", "detail": "ok"}]


def test_readyz_with_unconfigured_connector_still_ready():
    """An UNCONFIGURED connector does NOT flip ``ready`` to false —
    that's the expected steady state until the user logs in (connector
    spec §3.3)."""
    conn = _ReadyzConnector(health_state=ConnectorState.UNCONFIGURED, detail="needs login")
    client = TestClient(_make_app_with_connector(_DummyProvider(), _MinimalStore(), conn))
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    assert body["ready"] is True
    assert body["connectors"] == [
        {
            "id": "readyz_test",
            "state": "unconfigured",
            "detail": "needs login",
        }
    ]


def test_readyz_no_connector_registry_field_is_empty_list():
    """Engine without a connector_registry → block is absent (not a
    crash). The minimal _make_app helper used by older tests passes
    ``connector_registry=None`` implicitly; verify nothing breaks."""
    client = TestClient(_make_app(_DummyProvider(), _MinimalStore()))
    resp = client.get("/api/v1/readyz")
    body = resp.json()
    # Body either omits 'connectors' OR carries [] — either is fine; it
    # MUST NOT crash and MUST NOT flip ready.
    assert body["ready"] is True
    assert body.get("connectors", []) == []
