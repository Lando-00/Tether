"""ADR-0021 Phase 2.A — Engine multi-provider degraded-mode startup tests.

Covers the contract clauses in `docs/adr/0021-contract-stubs.md` §2:

  - `from_settings` degraded construction (try/except per registry entry).
  - All-failing entries raises ConfigError.
  - Default-provider failure remains visible instead of silently switching.
  - HardwareWatchdog receives the full provider list (post-filter to HW only).
  - `chat(...)` rejects unknown provider_ids with UnknownProviderError.
  - `chat(...)` rejects unhealthy provider_ids with ProviderUnhealthyError.

Uses DummyProvider for healthy entries and a `_FailingProvider` class whose
constructor raises for failing entries. Settings are routed through
`Settings.model_validate` (no YAML file needed).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tether import Engine
from tether.config.settings import Settings
from tether.core.errors import (
    AmbiguousModelError,
    ConfigError,
    ProviderUnhealthyError,
    UnknownModelError,
    UnknownProviderError,
)
from tether.core.interfaces import ModelProvider
from tether.protocol.parsers.sliding import SlidingParser


# Module-level marker so factory.load can instantiate it via dotted path.
class _FailingProvider:
    """Provider stub whose ``__init__`` raises — exercises factory.load
    failure capture in Engine.from_settings."""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("synthetic boot failure")


class _StaticProvider(ModelProvider):
    """Small in-memory provider for resolver-only contract tests."""

    def __init__(self, models: list[str]) -> None:
        self._models = models

    async def stream(self, model_name, messages, tools=None, *, request_id=None):
        if False:
            yield ""

    def list_models(self) -> list[str]:
        return list(self._models)

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 4096


def _direct_engine(
    providers: dict[str, ModelProvider],
    *,
    default: str,
    failures: dict[str, str] | None = None,
) -> Engine:
    return Engine(
        providers=providers,
        default_provider_id=default,
        provider_start_failures=failures,
        parser=SlidingParser(),
        session_store=MagicMock(),
        tools={},
        system_prompt="test",
    )


def _settings_dict(tmp_db: str, registry: dict, default: str) -> dict:
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


@pytest.fixture
def healthy_provider_impl() -> str:
    return "tether.providers.dummy.provider.DummyProvider"


@pytest.fixture
def failing_provider_impl() -> str:
    return (
        "tests.unit.engine.test_multi_provider_degraded_startup._FailingProvider"
    )


def test_one_healthy_one_failing_provider_starts(
    tmp_path, healthy_provider_impl, failing_provider_impl
):
    """A registry with one good + one bad provider must construct an
    Engine. The good id sits in ``providers``; the bad id sits in
    ``_provider_start_failures`` with the error message captured."""
    db = tmp_path / "engine.db"
    s_dict = _settings_dict(
        str(db),
        registry={
            "good": {"impl": healthy_provider_impl, "args": {}},
            "bad": {"impl": failing_provider_impl, "args": {}},
        },
        default="good",
    )
    settings = Settings.model_validate(s_dict)
    engine = Engine.from_settings(settings)

    assert "good" in engine.providers
    assert "bad" not in engine.providers
    assert "bad" in engine._provider_start_failures
    assert "synthetic boot failure" in engine._provider_start_failures["bad"]
    assert engine.default_provider_id == "good"
    # list_provider_health surfaces both.
    health = engine.list_provider_health()
    assert health["good"]["healthy"] is True
    assert health["bad"]["healthy"] is False
    assert health["bad"]["error"] is not None


def test_all_failing_raises_config_error(
    tmp_path, failing_provider_impl
):
    """Every registry entry failing → ConfigError (no engine emitted)."""
    db = tmp_path / "engine.db"
    s_dict = _settings_dict(
        str(db),
        registry={
            "bad1": {"impl": failing_provider_impl, "args": {}},
            "bad2": {"impl": failing_provider_impl, "args": {}},
        },
        default="bad1",
    )
    settings = Settings.model_validate(s_dict)
    with pytest.raises(ConfigError) as exc_info:
        Engine.from_settings(settings)
    assert "failed to construct" in str(exc_info.value)


def test_default_provider_unhealthy_remains_unavailable(
    tmp_path, healthy_provider_impl, failing_provider_impl
):
    """A failed configured default never silently becomes another backend."""
    db = tmp_path / "engine.db"
    s_dict = _settings_dict(
        str(db),
        registry={
            # Declaration order: bad first (the default), then good.
            "bad": {"impl": failing_provider_impl, "args": {}},
            "good": {"impl": healthy_provider_impl, "args": {}},
        },
        default="bad",
    )
    settings = Settings.model_validate(s_dict)
    engine = Engine.from_settings(settings)
    assert engine.default_provider_id == "bad"
    # Legacy singular-provider readers still receive a usable object, but
    # it is not the configured routing default.
    assert engine.provider is engine.providers["good"]
    with pytest.raises(ProviderUnhealthyError):
        engine.resolve_provider_id("dummy-model-1", provider_id="bad")


def test_watchdog_receives_all_providers_pre_filter(
    tmp_path, healthy_provider_impl
):
    """HardwareWatchdog gets the full provider list; its internal
    isinstance filter removes non-HW providers. With two DummyProvider
    entries (no HardwareLifecycle), the watchdog's _hw_providers is []."""
    db = tmp_path / "engine.db"
    s_dict = _settings_dict(
        str(db),
        registry={
            "a": {"impl": healthy_provider_impl, "args": {}},
            "b": {"impl": healthy_provider_impl, "args": {}},
        },
        default="a",
    )
    settings = Settings.model_validate(s_dict)
    engine = Engine.from_settings(settings)
    assert engine.hw_watchdog is not None
    # Filter ran: both DummyProviders dropped.
    assert engine.hw_watchdog.hw_provider_count == 0


async def _drain(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


@pytest.mark.anyio
async def test_unknown_provider_id_raises_in_chat(
    tmp_path, healthy_provider_impl
):
    """``engine.chat(..., provider_id="bogus")`` must raise
    UnknownProviderError before any orchestration starts."""
    db = tmp_path / "engine.db"
    s_dict = _settings_dict(
        str(db),
        registry={"good": {"impl": healthy_provider_impl, "args": {}}},
        default="good",
    )
    settings = Settings.model_validate(s_dict)
    engine = Engine.from_settings(settings)
    with pytest.raises(UnknownProviderError) as exc_info:
        await _drain(
            engine.chat(
                session_id="s1",
                prompt="hi",
                model_name="dummy-model-1",
                provider_id="bogus",
            )
        )
    assert exc_info.value.provider_id == "bogus"


@pytest.mark.anyio
async def test_unhealthy_provider_id_raises_in_chat(
    tmp_path, healthy_provider_impl, failing_provider_impl
):
    """``engine.chat(..., provider_id="<failed_id>")`` must raise
    ProviderUnhealthyError (not Unknown — the id is known, just dead)."""
    db = tmp_path / "engine.db"
    s_dict = _settings_dict(
        str(db),
        registry={
            "good": {"impl": healthy_provider_impl, "args": {}},
            "bad": {"impl": failing_provider_impl, "args": {}},
        },
        default="good",
    )
    settings = Settings.model_validate(s_dict)
    engine = Engine.from_settings(settings)
    with pytest.raises(ProviderUnhealthyError) as exc_info:
        await _drain(
            engine.chat(
                session_id="s1",
                prompt="hi",
                model_name="dummy-model-1",
                provider_id="bad",
            )
        )
    assert exc_info.value.provider_id == "bad"
    assert "synthetic boot failure" in exc_info.value.message


def test_automatic_routing_uses_the_unique_model_owner():
    engine = _direct_engine(
        {
            "mlc": _StaticProvider(["mlc-model"]),
            "geniex": _StaticProvider(["unsloth/Qwen3-4B-GGUF:Q4_0"]),
        },
        default="mlc",
    )

    assert (
        engine.resolve_provider_id("unsloth/Qwen3-4B-GGUF:Q4_0")
        == "geniex"
    )


def test_automatic_routing_rejects_duplicate_model_names():
    engine = _direct_engine(
        {
            "mlc": _StaticProvider(["shared-model"]),
            "geniex": _StaticProvider(["shared-model"]),
        },
        default="mlc",
    )

    with pytest.raises(AmbiguousModelError) as exc_info:
        engine.resolve_provider_id("shared-model")
    assert set(exc_info.value.provider_ids) == {"mlc", "geniex"}


def test_explicit_provider_rejects_another_provider_model():
    engine = _direct_engine(
        {
            "mlc": _StaticProvider(["mlc-model"]),
            "geniex": _StaticProvider(["geniex-model"]),
        },
        default="mlc",
    )

    with pytest.raises(UnknownModelError) as exc_info:
        engine.resolve_provider_id("geniex-model", provider_id="mlc")
    assert exc_info.value.provider_id == "mlc"


@pytest.fixture
def anyio_backend():
    return "asyncio"
