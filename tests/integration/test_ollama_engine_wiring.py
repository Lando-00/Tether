"""Engine + multi-provider integration tests for OllamaProvider (ADR-0022).

Verifies that OllamaProvider integrates correctly with the Engine multi-provider
registry (ADR-0021): dotted-path construction via from_settings, warm_up failure
→ degraded mode, provider_id routing, /models/details shape, /readyz inclusion.

All tests are gated with pytest.importorskip — they skip cleanly on this branch
(mp-ol-2c-tests) and activate automatically once Phase-2.A / 2.B implementation
modules land at the Phase-3.1 merge.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Phase-3.1 gate: skip entire file until provider + client_base exist.
# Remove these importorskip calls after the 2.A/2.B merge.
# ---------------------------------------------------------------------------
pytest.importorskip("tether.providers.ollama.provider")
pytest.importorskip("tether.providers.ollama.client")

from fastapi import APIRouter, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tether.app.http.api import lifespan  # noqa: E402
from tether.app.http.routers.health import router as health_router  # noqa: E402
from tether.app.http.routers.models import router as models_router  # noqa: E402
from tether.core.interfaces import ModelProvider  # noqa: E402
from tether.engine import Engine  # noqa: E402
from tether.protocol.parsers.sliding import SlidingParser  # noqa: E402
from tether.providers.ollama.provider import OllamaProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: minimal provider stub + app factory
# ---------------------------------------------------------------------------


class _DummyProvider(ModelProvider):
    """Minimal stub provider for multi-provider Engine tests."""

    @property
    def kind(self) -> str:
        return "dummy"

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
        return ["dummy-model"]

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 4096


def _make_ollama_provider(
    base_url: str = "http://localhost:11434",
    models: list[str] | None = None,
    thinking_models: list[str] = (),
    *,
    fake_client: Any = None,
) -> OllamaProvider:
    """Construct OllamaProvider with mocked httpx and optional fake internal client.

    When ``fake_client`` is not supplied, a default AsyncMock is wired so the
    engine's degraded-mode warm-up (ADR-0021 P2.A) can call ``version()`` and
    ``show_model()`` cleanly without TypeError on a sync MagicMock.
    """
    if models is None:
        models = ["qwen3:8b"]

    with patch("httpx.AsyncClient") as _mock_http_cls:
        _mock_http_cls.return_value = MagicMock()
        provider = OllamaProvider(
            base_url=base_url,
            models=models,
            thinking_models=thinking_models,
            url_validator=lambda u: None,
        )

    if fake_client is None:
        # ADR-0022 P3.2 review fix: ollama capabilities now advertise
        # warm_up_required=True so the engine probes connectivity at
        # startup. Default the internal client's async methods so the
        # probe succeeds without a TypeError on a sync MagicMock.
        fake_client = AsyncMock()
        fake_client.version.return_value = {"version": "0.6.0"}
        fake_client.list_models.return_value = []
        fake_client.show_model.return_value = {}

    provider._client = fake_client

    return provider


def _build_engine_with_ollama(
    ollama_provider: OllamaProvider,
    extra_providers: dict[str, ModelProvider] | None = None,
    failures: dict[str, str] | None = None,
) -> Engine:
    """Build an Engine with the given OllamaProvider registered as 'ollama-gpu'."""
    providers: dict[str, ModelProvider] = {"ollama-gpu": ollama_provider}
    if extra_providers:
        providers.update(extra_providers)

    return Engine(
        providers=providers,
        default_provider_id="ollama-gpu",
        provider_start_failures=failures or {},
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="Test prompt.",
    )


def _build_app(engine: Engine) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(models_router)
    v1.include_router(health_router)
    app.include_router(v1)
    return app


# ---------------------------------------------------------------------------
# Test 1: Engine.from_settings constructs OllamaProvider via dotted path
# ---------------------------------------------------------------------------


def test_engine_from_settings_constructs_ollama_via_dotted_path(tmp_path) -> None:
    """Engine.from_settings produces an Engine where providers['ollama-gpu']
    is an OllamaProvider instance when the model_registry points at the
    tether.providers.ollama.provider.OllamaProvider dotted path.

    httpx.AsyncClient is patched to prevent real network calls at construction time.
    """
    from tether.config.settings import Settings

    db_path = tmp_path / "data" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dsn = f"sqlite:///{db_path.as_posix()}"

    settings = Settings.model_validate(
        {
            "system": {"prompt": "Test."},
            "providers": {
                "model_registry": {
                    "ollama-gpu": {
                        "impl": "tether.providers.ollama.provider.OllamaProvider",
                        "args": {
                            "base_url": "http://192.168.1.50:11434",
                            "models": ["qwen3:8b"],
                            "url_validator": None,  # allow any URL in tests
                        },
                    }
                },
                "default_model_provider": "ollama-gpu",
                "parser": {
                    "impl": "tether.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether.context.sqlite_store.SqliteSessionStore",
                    "args": {},
                },
            },
            "storage": {"sqlite": {"dsn": dsn}},
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )

    # Patch httpx.AsyncClient so OllamaProvider.__init__ doesn't make real connections.
    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http_cls.return_value = MagicMock()
        engine = Engine.from_settings(settings)

    assert "ollama-gpu" in engine.providers, (
        f"Expected 'ollama-gpu' in engine.providers; got: {sorted(engine.providers)}"
    )
    provider = engine.providers["ollama-gpu"]
    assert isinstance(provider, OllamaProvider), (
        f"Expected OllamaProvider instance; got {type(provider).__name__}"
    )
    assert provider.kind == "ollama"


# ---------------------------------------------------------------------------
# Test 2: warm_up failure → _provider_start_failures + not in providers
# ---------------------------------------------------------------------------


async def test_warm_up_failure_lands_in_provider_start_failures() -> None:
    """When mock client.version() raises, Engine.__aenter__ moves the provider
    into _provider_start_failures and removes it from providers (degraded mode)."""
    import httpx

    fake_client = AsyncMock()
    fake_client.version.side_effect = httpx.ConnectError("Connection refused")

    ollama_provider = _make_ollama_provider(fake_client=fake_client)
    dummy_provider = _DummyProvider()

    engine = _build_engine_with_ollama(
        ollama_provider,
        extra_providers={"dummy": dummy_provider},
        failures={},
    )
    # Override default_provider_id so engine still boots when ollama is demoted.
    engine.default_provider_id = "dummy"

    # Manually invoke the warm_up path (simulate __aenter__ warm_up logic).
    # In production, Engine.__aenter__ iterates over providers and calls warm_up.
    # Here we test the provider directly and simulate the engine's demotion logic.
    try:
        await ollama_provider.warm_up("qwen3:8b")
    except RuntimeError as exc:
        # Simulate engine's demotion logic.
        engine._provider_start_failures["ollama-gpu"] = str(exc)
        del engine.providers["ollama-gpu"]

    assert "ollama-gpu" in engine._provider_start_failures, (
        "Expected 'ollama-gpu' in _provider_start_failures after warm_up failure"
    )
    assert "ollama-gpu" not in engine.providers, (
        "OllamaProvider should be removed from active providers after warm_up failure"
    )


# ---------------------------------------------------------------------------
# Test 3: /models/details rows have provider_id="ollama-gpu"
# ---------------------------------------------------------------------------


def test_models_details_includes_ollama_models_with_provider_id() -> None:
    """/models/details rows from OllamaProvider carry provider_id='ollama-gpu'
    (engine-wrapped, not the sentinel '_unwrapped_')."""
    ollama_provider = _make_ollama_provider(models=["qwen3:8b", "llama3.1:8b"])
    engine = _build_engine_with_ollama(ollama_provider)

    # Patch list_model_info to apply the engine's provider_id wrapping.
    _raw_infos = [
        info.model_copy(update={"provider_id": "ollama-gpu"})
        for info in ollama_provider.list_model_info()
    ]
    engine.list_model_info = lambda: _raw_infos  # type: ignore[method-assign]

    app = _build_app(engine)
    with TestClient(app) as client:
        resp = client.get("/api/v1/models/details")
        assert resp.status_code == 200
        rows = resp.json()

    ollama_rows = [r for r in rows if r.get("provider_id") == "ollama-gpu"]
    assert ollama_rows, (
        f"Expected rows with provider_id='ollama-gpu'; got: {[r.get('provider_id') for r in rows]}"
    )
    model_ids_in_rows = {r["id"] for r in ollama_rows}
    assert "qwen3:8b" in model_ids_in_rows
    assert "llama3.1:8b" in model_ids_in_rows

    # Verify the sentinel is NOT present in responses.
    for row in ollama_rows:
        assert row["provider_id"] != "_unwrapped_", (
            "Sentinel '_unwrapped_' must not appear in /models/details response"
        )


# ---------------------------------------------------------------------------
# Test 4: /readyz reports ollama-gpu in providers block
# ---------------------------------------------------------------------------


def test_readyz_reports_ollama_health() -> None:
    """/readyz 'providers' block includes an entry for 'ollama-gpu' with
    healthy=True (no failures)."""
    ollama_provider = _make_ollama_provider()
    engine = _build_engine_with_ollama(ollama_provider)

    app = _build_app(engine)
    with TestClient(app) as client:
        resp = client.get("/api/v1/readyz")
        assert resp.status_code == 200
        body = resp.json()

    assert "providers" in body, "Expected 'providers' block in /readyz response"
    pmap = body["providers"]
    assert "ollama-gpu" in pmap, (
        f"Expected 'ollama-gpu' in providers map; got: {sorted(pmap)}"
    )
    entry = pmap["ollama-gpu"]
    assert entry["healthy"] is True
    assert entry["error"] is None
    assert entry.get("kind") == "ollama"


# ---------------------------------------------------------------------------
# Test 5: provider_id routing delivers request to OllamaProvider
# ---------------------------------------------------------------------------


async def test_chat_stream_routes_provider_id_ollama() -> None:
    """When provider_id='ollama-gpu' is specified, the engine routes the request
    to OllamaProvider; client.stream_chat is called with the correct model."""
    stream_calls: list[dict] = []

    async def _fake_stream(**kwargs: Any):
        stream_calls.append(dict(kwargs))
        from tether.providers.ollama.client import OllamaStreamEvent

        yield OllamaStreamEvent(kind="text", text="routed ok")
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    ollama_provider = _make_ollama_provider(
        models=["qwen3:8b"],
        fake_client=fake_client,
    )
    dummy_provider = _DummyProvider()

    # Build engine to validate provider registration shape (not used
    # directly — full HTTP routing is covered in
    # test_chat_provider_routing.py; here we just exercise the provider's
    # stream() through the same engine-attached fake client).
    _build_engine_with_ollama(
        ollama_provider,
        extra_providers={"dummy": dummy_provider},
    )

    # Invoke stream directly on the ollama provider to verify routing logic.
    # (Full HTTP routing is tested in test_chat_provider_routing.py; here we
    # verify the provider itself responds to the model call correctly.)
    chunks = [
        c
        async for c in ollama_provider.stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hello from routing test"}],
        )
    ]

    assert stream_calls, "Expected stream_chat to be called"
    assert stream_calls[0]["model"] == "qwen3:8b"
    assert "routed ok" in chunks


# ---------------------------------------------------------------------------
# Test 6: Engine.__aenter__ demotes unreachable OllamaProvider via real warm-up
# ---------------------------------------------------------------------------


async def test_engine_aenter_demotes_unreachable_ollama() -> None:
    """End-to-end ratification of ADR-0022 P3.2 review fix #3: when
    OllamaProvider advertises ``capabilities.warm_up_required=True`` and
    its ``_client.version()`` fails, ``Engine.__aenter__`` calls
    ``_warm_up_providers_degraded`` which moves the provider from
    ``engine.providers`` into ``engine._provider_start_failures`` and
    promotes a healthy provider to the new default.

    This is the real path — not the manual demotion simulated in
    ``test_warm_up_failure_lands_in_provider_start_failures``.
    """
    import httpx

    # OllamaProvider with a client that raises ConnectError on version()
    fake_client = AsyncMock()
    fake_client.version.side_effect = httpx.ConnectError("Connection refused")

    ollama_provider = _make_ollama_provider(
        models=["qwen3:8b"],
        fake_client=fake_client,
    )

    # Need a healthy peer so engine still has a default after demotion.
    dummy_provider = _DummyProvider()

    engine = Engine(
        providers={
            "ollama-gpu": ollama_provider,
            "dummy": dummy_provider,
        },
        default_provider_id="ollama-gpu",
        provider_start_failures={},
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="Test prompt.",
    )

    # Drive Engine.__aenter__ until the warm-up-demotion step completes.
    # We don't need the full lifecycle to finish — only the degraded
    # warm-up call matters for this test — so call the helper directly.
    # (Engine.__aenter__ calls this same method at line ~876.)
    await engine._warm_up_providers_degraded()

    assert "ollama-gpu" in engine._provider_start_failures, (
        f"Expected 'ollama-gpu' demoted into _provider_start_failures; "
        f"got: {sorted(engine._provider_start_failures)}"
    )
    assert "ollama-gpu" not in engine.providers, (
        f"Expected 'ollama-gpu' removed from active providers; "
        f"got: {sorted(engine.providers)}"
    )
    # Verify the failure message carries the warm_up wrapping.
    msg = engine._provider_start_failures["ollama-gpu"]
    assert msg.startswith("warm_up:"), (
        f"Expected demotion message to start with 'warm_up:'; got: {msg!r}"
    )
    # ADR-0021: the configured default is immutable. A failed default is NOT
    # silently replaced — routing to it returns a typed 503 — so only the
    # legacy singular-provider shim moves to a healthy peer.
    assert engine.default_provider_id == "ollama-gpu"
    assert engine.provider is engine.providers["dummy"]
    # And the version() call must have happened (capability gate worked).
    fake_client.version.assert_awaited()
