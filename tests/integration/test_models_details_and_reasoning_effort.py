"""Integration tests for ``GET /api/v1/models/details`` and the
``reasoning_effort`` field on ``POST /api/v1/chat/stream``.

Covers:
- ``GET /api/v1/models`` still returns ``list[str]`` (back-compat).
- ``GET /api/v1/models/details`` returns the :class:`ModelDetails` shape.
- A provider that does NOT advertise reasoning support → 422 with a
  message that points at /models/details.
- A provider that DOES advertise reasoning support → 200 + the value
  is forwarded to the provider's ``stream(...)``.
- An advertised model + a value outside its whitelist → 422.

Uses the same minimal FastAPI app pattern as test_chat_content_negotiation:
no live MLC, no live SDK. Two fake providers (no-reasoning + reasoning-
advertising) drive both branches.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
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


class _NoReasoningProvider(ModelProvider):
    """Provider that exposes a single model with no reasoning support.

    Inherits the ABC's default :meth:`list_model_info` implementation so
    we also exercise the synthesised shape (``supports_reasoning_effort``
    defaults to False).
    """

    kind = "fake-local"

    def __init__(self) -> None:
        self.captured_reasoning_effort: list[Optional[str]] = []

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        self.captured_reasoning_effort.append(reasoning_effort)
        yield "ok"

    def list_models(self) -> List[str]:
        return ["fake-local-7b"]

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _ReasoningProvider(ModelProvider):
    """Provider that advertises reasoning_effort support for one model."""

    kind = "fake-remote"

    def __init__(self) -> None:
        self.captured_reasoning_effort: list[Optional[str]] = []

    @property
    def source(self) -> str:
        return "remote"

    def default_model(self) -> Optional[str]:
        return "fake-gpt-5"

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        self.captured_reasoning_effort.append(reasoning_effort)
        yield "ok"

    def list_models(self) -> List[str]:
        return ["fake-gpt-5", "fake-no-reasoning"]

    def list_model_info(self) -> List[ModelDetails]:
        return [
            ModelDetails(
                id="fake-gpt-5",
                provider_kind="fake-remote",
                source="remote",
                context_window=128_000,
                supports_thinking=True,
                supports_reasoning_effort=True,
                reasoning_efforts=["low", "high"],
                is_default=True,
            ),
            ModelDetails(
                id="fake-no-reasoning",
                provider_kind="fake-remote",
                source="remote",
                context_window=128_000,
                supports_thinking=False,
                supports_reasoning_effort=False,
                reasoning_efforts=None,
                is_default=False,
            ),
        ]

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 128_000


def _build_app(provider: ModelProvider) -> FastAPI:
    engine = Engine(
        provider=provider,
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="You are a helpful assistant.",
    )
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(chat_router)
    v1.include_router(models_router)
    v1.include_router(health_router)
    app.include_router(v1)
    return app


# ---------------------------------------------------------------------------
# /models and /models/details
# ---------------------------------------------------------------------------


def test_models_endpoint_still_returns_list_of_strings():
    """Back-compat: ``GET /api/v1/models`` is unchanged (list[str])."""
    with TestClient(_build_app(_NoReasoningProvider())) as client:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body == ["fake-local-7b"]
        assert all(isinstance(m, str) for m in body)


def test_models_details_default_provider_has_no_reasoning():
    """ABC default ``list_model_info`` synthesises non-reasoning entries."""
    with TestClient(_build_app(_NoReasoningProvider())) as client:
        resp = client.get("/api/v1/models/details")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        info = body[0]
        assert info["id"] == "fake-local-7b"
        assert info["provider_kind"] == "fake-local"
        assert info["source"] == "local"
        assert info["context_window"] == 4096
        assert info["supports_thinking"] is False
        assert info["supports_reasoning_effort"] is False
        assert info["reasoning_efforts"] is None
        assert info["is_default"] is False


def test_models_details_reasoning_provider_advertises_efforts():
    with TestClient(_build_app(_ReasoningProvider())) as client:
        resp = client.get("/api/v1/models/details")
        assert resp.status_code == 200
        body = resp.json()
        by_id = {d["id"]: d for d in body}

        assert by_id["fake-gpt-5"]["supports_reasoning_effort"] is True
        assert by_id["fake-gpt-5"]["reasoning_efforts"] == ["low", "high"]
        assert by_id["fake-gpt-5"]["is_default"] is True
        assert by_id["fake-gpt-5"]["source"] == "remote"
        assert by_id["fake-no-reasoning"]["supports_reasoning_effort"] is False
        assert by_id["fake-no-reasoning"]["is_default"] is False


# ---------------------------------------------------------------------------
# Validation: 422 BEFORE streaming starts
# ---------------------------------------------------------------------------


def _chat_body(**overrides):
    body = {
        "session_id": "test-s",
        "prompt": "hi",
        "model_name": "fake-gpt-5",
    }
    body.update(overrides)
    return body


def test_reasoning_effort_rejected_for_unsupported_model():
    """Model exists but doesn't advertise reasoning → 422 from
    ``_validate_reasoning_effort`` BEFORE any streaming."""
    provider = _ReasoningProvider()
    with TestClient(_build_app(provider)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(
                model_name="fake-no-reasoning",
                reasoning_effort="high",
            ),
        )
        assert resp.status_code == 422
        body = resp.json()
        detail = body.get("detail", "")
        assert "does not support reasoning_effort" in detail
        # And the provider's stream() should NOT have been invoked
        assert provider.captured_reasoning_effort == []


def test_reasoning_effort_rejected_when_value_not_in_whitelist():
    provider = _ReasoningProvider()
    with TestClient(_build_app(provider)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(reasoning_effort="ludicrous"),
        )
        assert resp.status_code == 422
        body = resp.json()
        detail = body.get("detail", "")
        assert "not accepted by model" in detail
        assert "['low', 'high']" in detail
        assert provider.captured_reasoning_effort == []


def test_reasoning_effort_pattern_rejected_at_pydantic():
    """Pydantic-level pattern validation rejects obviously bad shapes
    BEFORE the handler runs, so the handler never sees them."""
    with TestClient(_build_app(_ReasoningProvider())) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(reasoning_effort="x" * 100),
        )
        # FastAPI/Pydantic returns 422 on pattern mismatch.
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Threading: when accepted, reasoning_effort reaches the provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["low", "high"])
def test_reasoning_effort_threaded_to_provider_stream(effort: str):
    provider = _ReasoningProvider()
    with TestClient(_build_app(provider)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(reasoning_effort=effort),
        )
        assert resp.status_code == 200
        # Drain the body so the streaming response fully completes
        # (TestClient buffers it).
        _ = resp.text
        assert provider.captured_reasoning_effort == [effort]


def test_no_reasoning_effort_means_provider_sees_none():
    """The orchestrator must not forward ``reasoning_effort`` when the
    client omits it, even for providers that accept the kwarg."""
    provider = _ReasoningProvider()
    with TestClient(_build_app(provider)) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            json=_chat_body(),  # no reasoning_effort
        )
        assert resp.status_code == 200
        _ = resp.text
        assert provider.captured_reasoning_effort == [None]
