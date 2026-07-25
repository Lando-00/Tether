"""Integration test: GenieX engine wiring and degraded-startup behaviour.

Covers:
  - Configured-but-unavailable GenieX gets demoted (provider_start_failures).
  - Healthy MLC (simulated by DummyProvider) still works alongside failed GenieX.
  - Model ID / provider_id routing.
  - /readyz provider map shape.
  - Engine model detail shape includes geniex entries when healthy.

Uses Engine.from_settings with a _FailingGenieXProvider for the unavailable case
and a minimal inline GenieXProvider mock for the healthy case.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from tether import Engine
from tether.config.settings import Settings
from tether.core.errors import ProviderUnhealthyError, UnknownProviderError
from tether.core.interfaces import ModelProvider
from tether.providers.geniex.provider import GenieXProvider
from tether.providers.types import ModelDetails, ProviderCapabilities

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _settings_dict(tmp_db: str, registry: dict, default: str) -> dict:
    """Minimal settings dict for engine construction."""
    return {
        "system": {"prompt": "test-prompt"},
        "providers": {
            "model_registry": registry,
            "default_model_provider": default,
            "parser": {
                "impl": "tether.protocol.parsers.sliding.SlidingParser",
                "args": {},
            },
            "session_store": {
                "impl": "tether.context.sqlite_store.SqliteSessionStore",
                "args": {},
            },
        },
        "storage": {"sqlite": {"dsn": f"sqlite:///{tmp_db}"}},
        "tools": {
            "registry": [],
            "enabled": [],
            "disabled": ["time", "weather", "forecast", "web_search"],
        },
    }


# ---------------------------------------------------------------------------
# Degraded startup: GenieX unavailable → demoted, MLC works
# ---------------------------------------------------------------------------


class TestDegradedStartup:
    """When GenieX constructor fails (server unreachable), Engine degrades."""

    @pytest.mark.anyio
    async def test_real_geniex_warmup_failure_becomes_typed_unhealthy(self):
        """Actual GenieX connection failure is demoted before request routing."""
        import httpx

        from tether.context.memory_store import MemoryStore
        from tether.protocol.parsers.sliding import SlidingParser
        from tether.runtime.hw_watchdog import HardwareWatchdog

        def _unavailable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_unavailable),
            base_url="http://test",
        )
        provider = GenieXProvider(
            base_url="http://test",
            model_id="qwen3-npu",
            http_client=http_client,
        )
        provider._client.aclose = AsyncMock()
        providers = {"geniex-npu": provider}
        engine = Engine(
            providers=providers,
            default_provider_id="geniex-npu",
            parser=SlidingParser(),
            session_store=MemoryStore(),
            tools={},
            system_prompt="test",
            hw_watchdog=HardwareWatchdog(list(providers.values())),
        )

        await engine._warm_up_providers_degraded()

        assert "geniex-npu" not in engine.providers
        assert "geniex-npu" in engine._provider_start_failures
        provider._client.aclose.assert_awaited_once()
        with pytest.raises(ProviderUnhealthyError):
            async for _ in engine.chat(
                session_id="s1",
                prompt="hi",
                model_name="qwen3-npu",
                provider_id="geniex-npu",
            ):
                pass
        await http_client.aclose()

    def test_geniex_failure_captured_mlc_still_healthy(self, tmp_path):
        """Engine starts with MLC healthy + GenieX in _provider_start_failures."""
        db = str(tmp_path / "engine.db")
        registry = {
            "mlc-local": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "geniex-npu": {
                "impl": "tests.integration.test_geniex_engine_wiring._FailingGenieX",
                "args": {},
            },
        }
        settings = Settings.model_validate(
            _settings_dict(db, registry, default="mlc-local")
        )
        engine = Engine.from_settings(settings)

        # GenieX failed to start
        assert "geniex-npu" in engine._provider_start_failures
        assert "geniex-npu" not in engine.providers
        # MLC (dummy) is healthy
        assert "mlc-local" in engine.providers
        assert engine.default_provider_id == "mlc-local"

    def test_health_map_shows_geniex_unhealthy(self, tmp_path):
        """list_provider_health shows geniex as unhealthy with error."""
        db = str(tmp_path / "engine.db")
        registry = {
            "mlc-local": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "geniex-npu": {
                "impl": "tests.integration.test_geniex_engine_wiring._FailingGenieX",
                "args": {},
            },
        }
        settings = Settings.model_validate(
            _settings_dict(db, registry, default="mlc-local")
        )
        engine = Engine.from_settings(settings)
        health = engine.list_provider_health()

        assert health["mlc-local"]["healthy"] is True
        assert health["geniex-npu"]["healthy"] is False
        assert health["geniex-npu"]["error"] is not None
        assert "unavailable" in health["geniex-npu"]["error"].lower() or \
               "connect" in health["geniex-npu"]["error"].lower() or \
               "refused" in health["geniex-npu"]["error"].lower()

    def test_default_fallback_when_geniex_is_default(self, tmp_path, caplog):
        """If geniex is default but fails, Engine falls back to MLC."""
        db = str(tmp_path / "engine.db")
        registry = {
            "geniex-npu": {
                "impl": "tests.integration.test_geniex_engine_wiring._FailingGenieX",
                "args": {},
            },
            "mlc-local": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
        }
        settings = Settings.model_validate(
            _settings_dict(db, registry, default="geniex-npu")
        )
        with caplog.at_level(logging.WARNING, logger="tether.engine"):
            engine = Engine.from_settings(settings)

        assert engine.default_provider_id == "mlc-local"
        assert any(
            "default_unhealthy_fallback" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Provider ID routing
# ---------------------------------------------------------------------------


class TestProviderRouting:
    """chat() routes to the correct provider by provider_id."""

    @pytest.mark.anyio
    async def test_unknown_provider_id_raises(self, tmp_path):
        """Requesting a non-existent provider_id → UnknownProviderError."""
        db = str(tmp_path / "engine.db")
        registry = {
            "mlc-local": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
        }
        settings = Settings.model_validate(
            _settings_dict(db, registry, default="mlc-local")
        )
        engine = Engine.from_settings(settings)

        with pytest.raises(UnknownProviderError):
            async for _ in engine.chat(
                session_id="s1",
                prompt="hi",
                model_name="dummy-model-1",
                provider_id="geniex-npu",
            ):
                pass  # pragma: no cover

    @pytest.mark.anyio
    async def test_unhealthy_provider_id_raises(self, tmp_path):
        """Requesting a failed provider_id → ProviderUnhealthyError."""
        db = str(tmp_path / "engine.db")
        registry = {
            "mlc-local": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "geniex-npu": {
                "impl": "tests.integration.test_geniex_engine_wiring._FailingGenieX",
                "args": {},
            },
        }
        settings = Settings.model_validate(
            _settings_dict(db, registry, default="mlc-local")
        )
        engine = Engine.from_settings(settings)

        with pytest.raises(ProviderUnhealthyError):
            async for _ in engine.chat(
                session_id="s1",
                prompt="hi",
                model_name="some-model",
                provider_id="geniex-npu",
            ):
                pass  # pragma: no cover


# ---------------------------------------------------------------------------
# Model detail shape
# ---------------------------------------------------------------------------


class TestModelDetails:
    """Engine model info includes GenieX entries with correct shape."""

    def test_model_info_contains_geniex_provider_id(self, tmp_path):
        """When GenieX is healthy, list_model_info has provider_id set."""
        from tether.context.memory_store import MemoryStore
        from tether.protocol.parsers.sliding import SlidingParser
        from tether.providers.dummy.provider import DummyProvider
        from tether.runtime.hw_watchdog import HardwareWatchdog

        # Use a _FakeGenieXProvider that doesn't require network
        fake_geniex = _FakeGenieXProvider()

        providers = {
            "mlc-local": DummyProvider(),
            "geniex-npu": fake_geniex,
        }
        watchdog = HardwareWatchdog(list(providers.values()))
        engine = Engine(
            providers=providers,
            default_provider_id="mlc-local",
            parser=SlidingParser(),
            session_store=MemoryStore(),
            tools={},
            system_prompt="test",
            hw_watchdog=watchdog,
        )

        infos = engine.list_model_info()
        geniex_infos = [i for i in infos if i.provider_id == "geniex-npu"]
        assert len(geniex_infos) >= 1
        for info in geniex_infos:
            assert info.provider_kind == "geniex"
            assert info.source == "remote"
            assert info.supports_thinking is False


# ---------------------------------------------------------------------------
# Readyz shape
# ---------------------------------------------------------------------------


class TestReadyz:
    """readyz endpoint exposes geniex status."""

    def test_readyz_providers_map_includes_geniex(self, tmp_path):
        """readyz provider map includes GenieX with healthy/error status."""
        from fastapi import APIRouter, FastAPI
        from fastapi.testclient import TestClient

        from tether.app.http.routers.health import router as health_router
        from tether.context.memory_store import MemoryStore
        from tether.protocol.parsers.sliding import SlidingParser
        from tether.providers.dummy.provider import DummyProvider
        from tether.runtime.hw_watchdog import HardwareWatchdog

        providers = {"mlc-local": DummyProvider()}
        watchdog = HardwareWatchdog(list(providers.values()))
        engine = Engine(
            providers=providers,
            default_provider_id="mlc-local",
            provider_start_failures={"geniex-npu": "ConnectionError: refused"},
            parser=SlidingParser(),
            session_store=MemoryStore(),
            tools={},
            system_prompt="test",
            hw_watchdog=watchdog,
        )

        app = FastAPI()
        v1 = APIRouter(prefix="/api/v1")
        v1.include_router(health_router)
        app.include_router(v1)
        app.state.gen_svc = engine

        client = TestClient(app)
        body = client.get("/api/v1/readyz").json()

        assert body["ready"] is True  # MLC still healthy
        pmap = body["providers"]
        assert "geniex-npu" in pmap
        assert pmap["geniex-npu"]["healthy"] is False
        assert "refused" in pmap["geniex-npu"]["error"].lower()
        assert pmap["mlc-local"]["healthy"] is True


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FailingGenieX:
    """GenieX stub whose __init__ simulates connection refused."""

    def __init__(self, **kwargs: Any) -> None:
        raise ConnectionError("GenieX server unavailable: Connection refused")


class _FakeGenieXProvider(ModelProvider):
    """Minimal GenieX-shaped provider for engine integration tests."""

    @property
    def kind(self) -> str:
        return "geniex"

    @property
    def source(self) -> str:
        return "remote"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools_native=False,
            tools_marker=True,
            warm_up_required=True,
            multi_model=False,
        )

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "fake-geniex-response"

    def list_models(self) -> List[str]:
        return ["qwen3-npu"]

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 4096

    def list_model_info(self) -> List[ModelDetails]:
        return [
            ModelDetails(
                id="qwen3-npu",
                provider_id="_unwrapped_",
                provider_kind="geniex",
                source="remote",
                context_window=4096,
                supports_thinking=False,
                supports_reasoning_effort=False,
            )
        ]
