"""ADR-0021 Phase 2.A — Engine multi-provider degraded-mode startup tests.

Covers the contract clauses in `docs/adr/0021-contract-stubs.md` §2:

  - `from_settings` degraded construction (try/except per registry entry).
  - All-failing entries raises ConfigError.
  - Default-provider failure falls back to first healthy id (warning logged).
  - HardwareWatchdog receives the full provider list (post-filter to HW only).
  - `chat(...)` rejects unknown provider_ids with UnknownProviderError.
  - `chat(...)` rejects unhealthy provider_ids with ProviderUnhealthyError.

Uses DummyProvider for healthy entries and a `_FailingProvider` class whose
constructor raises for failing entries. Settings are routed through
`Settings.model_validate` (no YAML file needed).
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from tether import Engine
from tether.config.settings import Settings
from tether.core.errors import (
    ConfigError,
    ProviderUnhealthyError,
    UnknownProviderError,
)
from tether.providers.dummy.provider import DummyProvider


# Module-level marker so factory.load can instantiate it via dotted path.
class _FailingProvider:
    """Provider stub whose ``__init__`` raises — exercises factory.load
    failure capture in Engine.from_settings."""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("synthetic boot failure")


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


def test_default_provider_unhealthy_falls_back_to_first_healthy(
    tmp_path, healthy_provider_impl, failing_provider_impl, caplog
):
    """default_model_provider points at a failing entry → fall back to
    first healthy in declaration order (with a logger.warning)."""
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
    with caplog.at_level(logging.WARNING, logger="tether.engine"):
        engine = Engine.from_settings(settings)
    assert engine.default_provider_id == "good"
    # Loud warning emitted.
    assert any(
        "default_unhealthy_fallback" in rec.message
        for rec in caplog.records
    )


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


@pytest.fixture
def anyio_backend():
    return "asyncio"
