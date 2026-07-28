"""Integration tests for /api/v1/models and /api/v1/models/details with
multiple providers (ADR-0021 Phase 2.B).

Covers:
  - /models returns the merged list[str] union from all healthy providers
  - /models/details rows carry provider_id (not the sentinel "_unwrapped_")
  - unhealthy-provider models are hidden from both endpoints
  - duplicate model names across providers remain raw and are disambiguated by
    ``provider_id`` in /models/details

Uses the same minimal-app + fake-Engine pattern as test_chat_provider_routing.py.
Phase 2.A guarantees engine.providers / engine.default_provider_id /
engine._provider_start_failures; simulated here by setting them post-construction.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.api import lifespan
from tether.app.http.routers.chat import router as chat_router
from tether.app.http.routers.health import router as health_router
from tether.app.http.routers.models import router as models_router
from tether.core.interfaces import ModelProvider
from tether.engine import Engine
from tether.protocol.parsers.sliding import SlidingParser
from tether.providers.types import ModelDetails

# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class _SimpleProvider(ModelProvider):
    """Provider that exposes a configurable list of models."""

    kind = "fake-simple"

    def __init__(self, label: str, models: List[str]) -> None:
        self.label = label
        self._models = models

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    def list_models(self) -> List[str]:
        return list(self._models)

    def list_model_info(self) -> List[ModelDetails]:
        return [
            ModelDetails(
                id=name,
                provider_id=self.label,
                provider_kind="fake-simple",
                source="local",
                context_window=4096,
                supports_thinking=False,
                supports_reasoning_effort=False,
            )
            for name in self._models
        ]

    def unload_model(self, model_name: str) -> bool:
        return model_name in self._models

    def get_context_window(self, model_name: str) -> int:
        return 4096


# ---------------------------------------------------------------------------
# App factory helpers
# ---------------------------------------------------------------------------


def _build_multi_provider_engine(
    providers: Dict[str, ModelProvider],
    default_pid: str,
    failures: Optional[Dict[str, str]] = None,
) -> Engine:
    """Build a real multi-provider Engine for endpoint integration tests."""
    failures = failures or {}
    healthy_providers = {
        provider_id: provider for provider_id, provider in providers.items() if provider_id not in failures
    }
    engine = Engine(
        providers=healthy_providers,
        default_provider_id=default_pid,
        provider_start_failures=failures,
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="You are a helpful assistant.",
    )
    return engine


def _build_app(engine: Engine) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(chat_router)
    v1.include_router(models_router)
    v1.include_router(health_router)
    app.include_router(v1)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_models_legacy_returns_strings_union():
    """/models returns the merged list[str] across all healthy providers."""
    provider_a = _SimpleProvider("a", ["model-a1", "model-a2"])
    provider_b = _SimpleProvider("b", ["model-b1"])
    engine = _build_multi_provider_engine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert all(isinstance(m, str) for m in body)
        assert set(body) == {"model-a1", "model-a2", "model-b1"}


def test_models_details_includes_provider_id():
    """/models/details rows carry provider_id matching a registry key, not the sentinel."""
    provider_a = _SimpleProvider("a", ["model-a1"])
    provider_b = _SimpleProvider("b", ["model-b1"])
    engine = _build_multi_provider_engine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.get("/api/v1/models/details")
        assert resp.status_code == 200
        rows = resp.json()
        # Every row must have provider_id set to a real registry key (not sentinel).
        by_id = {(r["provider_id"], r["id"]): r for r in rows}
        assert ("a", "model-a1") in by_id
        assert ("b", "model-b1") in by_id
        for row in rows:
            assert row["provider_id"] != "_unwrapped_", (
                f"Sentinel '_unwrapped_' leaked into response for model {row['id']}"
            )


def test_models_hides_unhealthy_provider_models():
    """Models from a provider in _provider_start_failures are absent from both endpoints."""
    provider_a = _SimpleProvider("a", ["model-a1"])
    provider_bad = _SimpleProvider("bad", ["model-bad"])
    engine = _build_multi_provider_engine(
        {"a": provider_a, "bad": provider_bad},
        default_pid="a",
        failures={"bad": "ConnectionError: service unreachable"},
    )
    with TestClient(_build_app(engine)) as client:
        # /models
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert "model-bad" not in models
        assert "model-a1" in models

        # /models/details
        resp2 = client.get("/api/v1/models/details")
        assert resp2.status_code == 200
        detail_ids = [(r["provider_id"], r["id"]) for r in resp2.json()]
        assert ("bad", "model-bad") not in detail_ids
        assert ("a", "model-a1") in detail_ids


def test_models_duplicate_names_are_disambiguated_in_details():
    """Duplicate names remain raw; /models/details carries the provider ID."""
    provider_a = _SimpleProvider("a", ["shared-model", "unique-a"])
    provider_b = _SimpleProvider("b", ["shared-model", "unique-b"])
    engine = _build_multi_provider_engine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )
    with TestClient(_build_app(engine)) as client:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        model_set = set(models)
        assert "shared-model" in model_set
        assert list(models).count("shared-model") == 1
        assert "unique-a" in model_set
        assert "unique-b" in model_set

        # /models/details should have two separate rows for shared-model.
        resp2 = client.get("/api/v1/models/details")
        rows = resp2.json()
        shared_rows = [r for r in rows if r["id"] == "shared-model"]
        assert len(shared_rows) == 2
        provider_ids_for_shared = {r["provider_id"] for r in shared_rows}
        assert provider_ids_for_shared == {"a", "b"}


def test_unload_duplicate_model_requires_provider_id():
    provider_a = _SimpleProvider("a", ["shared-model"])
    provider_b = _SimpleProvider("b", ["shared-model"])
    engine = _build_multi_provider_engine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )

    with TestClient(_build_app(engine)) as client:
        response = client.post(
            "/api/v1/models/unload",
            json={"model_name": "shared-model"},
        )

    assert response.status_code == 422
    assert "multiple providers" in response.json()["detail"]


def test_unload_duplicate_model_honors_explicit_provider():
    provider_a = _SimpleProvider("a", ["shared-model"])
    provider_b = _SimpleProvider("b", ["shared-model"])
    engine = _build_multi_provider_engine(
        {"a": provider_a, "b": provider_b},
        default_pid="a",
    )

    with TestClient(_build_app(engine)) as client:
        response = client.post(
            "/api/v1/models/unload",
            json={"model_name": "shared-model", "provider_id": "a"},
        )

    assert response.status_code == 200
    assert response.json()["provider_id"] == "a"
