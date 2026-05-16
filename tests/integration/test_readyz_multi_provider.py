"""ADR-0021 Phase 2.A — /readyz multi-provider regression tests.

Asserts:
  - Legacy top-level keys are preserved (additive guarantee).
  - New `providers` map has the contracted shape (healthy / kind /
    source / error per id).
  - `default_provider_id` is present.
  - `ready` is True when ≥1 provider is healthy (one down, one up).
  - `ready` is False when every provider is unhealthy.
  - Legacy `provider: bool` field tracks "any healthy".

Uses the same minimal-FastAPI helper pattern as `tests/integration/test_readyz.py`.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.routers.health import router as health_router
from tether.core.interfaces import ModelProvider, SessionStore
from tether.engine import Engine
from tether.runtime.hw_watchdog import HardwareWatchdog


class _MinStore(SessionStore):
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

    async def record_raw_event(
        self,
        session_id,
        turn_id,
        seq,
        event_type,
        payload,
        *,
        tool_call_id=None,
    ) -> None:
        pass


class _HealthyProvider(ModelProvider):
    """Minimal healthy provider — declares ``kind``/``source`` so
    list_provider_health populates them correctly."""

    @property
    def kind(self) -> str:
        return "fake-healthy"

    @property
    def source(self) -> str:
        return "remote"

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
        return ["fake-1"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


def _build_app(
    providers: Dict[str, ModelProvider],
    default_pid: str,
    failures: Dict[str, str],
) -> FastAPI:
    from tether.protocol.parsers.sliding import SlidingParser

    watchdog = HardwareWatchdog(list(providers.values()))
    engine = Engine(
        providers=providers,
        default_provider_id=default_pid,
        provider_start_failures=failures,
        parser=SlidingParser(),
        session_store=_MinStore(),
        tools={},
        system_prompt="",
        hw_watchdog=watchdog,
    )
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health_router)
    app.include_router(v1)
    app.state.gen_svc = engine
    return app


def test_readyz_legacy_keys_preserved():
    """Every pre-existing top-level key must still appear in the response
    (additive guarantee)."""
    providers = {"good": _HealthyProvider()}
    app = _build_app(providers, default_pid="good", failures={"dead": "boom"})
    client = TestClient(app)
    body = client.get("/api/v1/readyz").json()
    # Legacy keys.
    for k in ("ready", "store", "provider", "hw_health", "connectors",
              "connector_start_failures"):
        assert k in body, f"legacy key {k!r} missing from /readyz body"


def test_readyz_new_providers_map_shape():
    providers = {"good": _HealthyProvider()}
    app = _build_app(providers, default_pid="good", failures={"dead": "AuthError: boom"})
    client = TestClient(app)
    body = client.get("/api/v1/readyz").json()
    assert "providers" in body
    pmap = body["providers"]
    assert "good" in pmap and "dead" in pmap
    for entry in pmap.values():
        assert set(entry.keys()) == {"healthy", "kind", "source", "error"}
    assert pmap["good"]["healthy"] is True
    assert pmap["good"]["kind"] == "fake-healthy"
    assert pmap["good"]["source"] == "remote"
    assert pmap["good"]["error"] is None
    assert pmap["dead"]["healthy"] is False
    assert pmap["dead"]["error"] == "AuthError: boom"


def test_readyz_default_provider_id_present():
    providers = {"good": _HealthyProvider()}
    app = _build_app(providers, default_pid="good", failures={})
    body = TestClient(app).get("/api/v1/readyz").json()
    assert body.get("default_provider_id") == "good"


def test_readyz_ready_true_with_one_provider_down():
    """Store ok + one provider down + one up → ready=true."""
    providers = {"good": _HealthyProvider()}
    app = _build_app(providers, default_pid="good", failures={"dead": "boom"})
    body = TestClient(app).get("/api/v1/readyz").json()
    assert body["ready"] is True


def test_readyz_ready_false_when_all_providers_down():
    """No healthy providers in `providers` map → ready=false. We can't
    construct such an Engine via from_settings (that raises ConfigError),
    so we build it directly with an empty registry-style state — but
    Engine requires at least one provider. Instead, simulate the
    post-warm-up demoted case: a single registered provider that the
    health summary reports as unhealthy via the failures map after
    construction."""
    # Strategy: construct with one healthy provider + one failure, then
    # mutate engine state to demote the healthy one into failures (as
    # __aenter__ would do on warm_up failure). This mirrors the
    # contracted lifecycle without standing up a real warm_up flow.
    providers = {"good": _HealthyProvider()}
    app = _build_app(providers, default_pid="good", failures={})
    engine = app.state.gen_svc
    # Demote the only healthy provider.
    engine._provider_start_failures["good"] = "warm_up: boom"
    engine.providers.pop("good")
    body = TestClient(app).get("/api/v1/readyz").json()
    assert body["ready"] is False
    assert body["provider"] is False
    assert body["providers"]["good"]["healthy"] is False


def test_readyz_provider_bool_legacy_field_tracks_any_healthy():
    """Legacy `provider: bool` is True iff ≥1 entry in the providers
    map is healthy."""
    # Case 1: at least one healthy → True.
    providers = {"good": _HealthyProvider()}
    app = _build_app(providers, default_pid="good", failures={"dead": "x"})
    body = TestClient(app).get("/api/v1/readyz").json()
    assert body["provider"] is True

    # Case 2: zero healthy (demote-then-probe) → False.
    engine = app.state.gen_svc
    engine._provider_start_failures["good"] = "warm_up: lost"
    engine.providers.pop("good")
    body2 = TestClient(app).get("/api/v1/readyz").json()
    assert body2["provider"] is False
