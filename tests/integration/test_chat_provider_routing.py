"""Integration tests for provider_id routing on POST /api/v1/chat/stream.

Covers ADR-0021 Phase 2.B HTTP-layer routing:
  - omitted provider_id → default_provider_id is used (pid passed to engine.chat)
  - explicit provider_id → that pid passed to engine.chat
  - unknown provider_id → 422 BEFORE streaming begins
  - known-but-unhealthy provider_id → 503 BEFORE streaming begins
  - reasoning_effort validation scoped to the chosen provider_id

Uses a _FakeEngine that:
  - exposes providers / default_provider_id / _provider_start_failures (Phase 2.A
    contract, simulated without modifying engine.py)
  - records the provider_id kwarg passed to chat() so the HTTP layer's kwarg
    threading can be asserted

Phase 2.A adds provider_id routing inside Engine; Phase 2.B (this module)
adds it at the HTTP boundary. Tests here verify the HTTP layer only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.api import lifespan
from tether.app.http.routers.chat import router as chat_router
from tether.app.http.routers.health import router as health_router
from tether.app.http.routers.models import router as models_router
from tether.core.interfaces import ModelProvider
from tether.providers.types import ModelDetails

# ---------------------------------------------------------------------------
# Fake engine — records provider_id kwarg from HTTP layer
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal Engine-compatible object for HTTP routing tests.

    Exposes the Phase 2.A multi-provider contract attributes directly so
    the HTTP layer can validate and thread provider_id without needing
    engine.py to be updated first.

    chat() and stream() record the `provider_id` kwarg they receive, then
    yield a minimal valid response. This lets the routing tests assert:
      1. Provider validation (422/503) happens BEFORE yield.
      2. The resolved pid is forwarded to engine.chat/engine.stream.
    """

    def __init__(
        self,
        providers: Dict[str, "ModelProvider"],
        default_pid: str,
        failures: Optional[Dict[str, str]] = None,
        model_infos: Optional[List[ModelDetails]] = None,
    ) -> None:
        self.providers = providers
        self.default_provider_id = default_pid
        self._provider_start_failures = failures or {}
        # Back-compat: real Engine exposes self.provider (singular).
        self.provider = providers[default_pid]
        self._orchestrator_registry = {
            "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
            "research": "tether.protocol.orchestration.notebook.NotebookOrchestrator",
        }
        self._orchestrator_default_mode = "chat"
        self._audit_store_args = False
        self._confirm_intent_classifier = None
        self.tools = {}
        self.store = AsyncMock()
        self.system_prompt = "test"
        self.orchestrator_config = None
        self.tool_runner = None
        self.hw_watchdog = None
        self.connector_registry = None
        self.inbox = None
        self._connector_start_tasks = []
        self._connector_start_failures = []
        self._closed = False

        # Records: list of provider_id kwargs received by chat() / stream().
        self.captured_chat_provider_ids: List[Optional[str]] = []
        self.captured_stream_provider_ids: List[Optional[str]] = []

        # Optional overrides for list_model_info (per-test metadata).
        self._model_infos = model_infos

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def list_models(self) -> List[str]:
        result = []
        for pid, provider in self.providers.items():
            if pid not in self._provider_start_failures:
                result.extend(provider.list_models())
        return result

    def list_model_info(self) -> List[ModelDetails]:
        if self._model_infos is not None:
            return self._model_infos
        result = []
        for pid, provider in self.providers.items():
            if pid not in self._provider_start_failures:
                for info in provider.list_model_info():
                    result.append(info.model_copy(update={"provider_id": pid}))
        return result

    def unload_model(self, model_name: str) -> bool:
        return False

    async def chat(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        mode: Optional[str] = None,
        cancel_token=None,
        reasoning_effort: Optional[str] = None,
        provider_id: Optional[str] = None,
        **kwargs,
    ):
        """Record provider_id and yield a minimal valid v2 event stream."""
        self.captured_chat_provider_ids.append(provider_id)
        from datetime import datetime, timezone

        from tether.protocol.wire.events import MessageStart, MessageStop, TextDelta

        now = datetime.now(timezone.utc)
        yield MessageStart(
            session_id=session_id,
            turn_id="t1",
            seq=0,
            ts=now,
            available_tools=[],
        )
        yield TextDelta(session_id=session_id, turn_id="t1", seq=1, ts=now, text="ok")
        yield MessageStop(
            session_id=session_id,
            turn_id="t1",
            seq=2,
            ts=now,
            stop_reason="complete",
        )

    async def stream(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        mode: Optional[str] = None,
        cancel_event=None,
        reasoning_effort: Optional[str] = None,
        provider_id: Optional[str] = None,
        **kwargs,
    ):
        """Record provider_id and yield a minimal NDJSON v0 bytes chunk."""
        import json

        self.captured_stream_provider_ids.append(provider_id)
        payload = {
            "type": "text",
            "session_id": session_id,
            "data": {"text": "ok"},
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        yield (json.dumps(payload) + "\n").encode()


# ---------------------------------------------------------------------------
# Fake providers (for list_model_info metadata only)
# ---------------------------------------------------------------------------


class _SimpleProvider(ModelProvider):
    kind = "fake-simple"

    def __init__(self, label: str, model_infos: List[ModelDetails]) -> None:
        self.label = label
        self._model_infos = model_infos

    async def stream(self, model_name, messages, tools=None, *, request_id=None, reasoning_effort=None):  # type: ignore
        yield "ok"

    def list_models(self) -> List[str]:
        return [m.id for m in self._model_infos]

    def list_model_info(self) -> List[ModelDetails]:
        return list(self._model_infos)

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 4096


def _make_reasoning_infos(pid: str, *, supports: bool) -> List[ModelDetails]:
    return [
        ModelDetails(
            id="smart-model",
            provider_id=pid,
            provider_kind="fake-simple",
            source="local",
            context_window=128_000,
            supports_thinking=False,
            supports_reasoning_effort=supports,
            reasoning_efforts=["low", "high"] if supports else None,
        )
    ]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _build_app(engine: _FakeEngine) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(chat_router)
    v1.include_router(models_router)
    v1.include_router(health_router)
    app.include_router(v1)
    return app


def _chat_body(**overrides) -> dict:
    body = {
        "session_id": "test-s",
        "prompt": "hello",
        "model_name": "smart-model",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_missing_provider_id_uses_default():
    """Omitting provider_id → the default_provider_id is forwarded to engine.chat."""
    provider_a = _SimpleProvider("a", _make_reasoning_infos("a", supports=True))
    provider_b = _SimpleProvider("b", _make_reasoning_infos("b", supports=False))
    engine = _FakeEngine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.post("/api/v1/chat/stream", json=_chat_body())
        assert resp.status_code == 200
        _ = resp.text
        # HTTP layer must have forwarded default_provider_id to chat().
        assert engine.captured_chat_provider_ids == ["a"]


def test_explicit_provider_id_routes_correctly():
    """Passing provider_id='b' → 'b' is forwarded to engine.chat."""
    provider_a = _SimpleProvider("a", _make_reasoning_infos("a", supports=True))
    provider_b = _SimpleProvider("b", _make_reasoning_infos("b", supports=False))
    engine = _FakeEngine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(provider_id="b"),
        )
        assert resp.status_code == 200
        _ = resp.text
        # HTTP layer must have forwarded 'b' to engine.chat().
        assert engine.captured_chat_provider_ids == ["b"]


def test_namespaced_quantized_model_id_routes_correctly():
    """A GenieX ``org/repo:quant`` model ID passes HTTP validation unchanged."""
    provider_a = _SimpleProvider("a", _make_reasoning_infos("a", supports=True))
    provider_b = _SimpleProvider("b", _make_reasoning_infos("b", supports=False))
    engine = _FakeEngine(
        {"a": provider_a, "geniex": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(
                model_name="unsloth/Qwen3-1.7B-GGUF:Q4_0",
                provider_id="geniex",
            ),
        )
        assert resp.status_code == 200
        _ = resp.text
        assert engine.captured_chat_provider_ids == ["geniex"]


def test_unknown_provider_id_returns_422():
    """provider_id not in engine.providers and not in failures → 422 before streaming."""
    provider_a = _SimpleProvider("a", _make_reasoning_infos("a", supports=True))
    engine = _FakeEngine({"a": provider_a}, default_pid="a")
    with TestClient(_build_app(engine)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(provider_id="nonexistent"),
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "Unknown provider_id 'nonexistent'" in body["detail"]
        # Engine.chat must NOT have been called.
        assert engine.captured_chat_provider_ids == []


def test_unhealthy_provider_id_returns_503():
    """provider_id in _provider_start_failures → 503 with message before streaming."""
    provider_a = _SimpleProvider("a", _make_reasoning_infos("a", supports=True))
    engine = _FakeEngine(
        {"a": provider_a},
        default_pid="a",
        failures={"broken": "AuthError: missing token"},
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(provider_id="broken"),
        )
        assert resp.status_code == 503
        body = resp.json()
        # Code-review follow-up: server no longer leaks raw exception text
        # (which could contain paths / tokens from provider __init__).
        # The generic message still names the failing provider_id and
        # points operators at the /readyz health map.
        assert "broken" in body["detail"]
        assert "AuthError" not in body["detail"]
        assert "missing token" not in body["detail"]
        assert "/api/v1/readyz" in body["detail"]
        assert engine.captured_chat_provider_ids == []


def test_reasoning_effort_validated_against_chosen_provider():
    """reasoning_effort validation uses the metadata for the chosen provider_id.

    'smart-model' exists under both 'a' (supports reasoning) and 'b' (doesn't).
    - provider_id='a' + reasoning_effort='low' → 200 (supported)
    - provider_id='b' + reasoning_effort='low' → 422 (unsupported on 'b')
    """
    provider_a = _SimpleProvider("a", _make_reasoning_infos("a", supports=True))
    provider_b = _SimpleProvider("b", _make_reasoning_infos("b", supports=False))
    engine = _FakeEngine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        # Provider 'a' supports reasoning → should pass validation
        resp_ok = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(reasoning_effort="low", provider_id="a"),
        )
        assert resp_ok.status_code == 200

        # Provider 'b' does NOT support reasoning → 422
        resp_fail = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(reasoning_effort="low", provider_id="b"),
        )
        assert resp_fail.status_code == 422
        detail = resp_fail.json()["detail"]
        assert "does not support reasoning_effort" in detail
        assert "provider 'b'" in detail
