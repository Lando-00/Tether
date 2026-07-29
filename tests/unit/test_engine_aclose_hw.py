"""Tests for Phase 3 step 35 — Engine.aclose routing through HardwareWatchdog.

Verifies F1 of p3-lifespan-slim: ``Engine.from_settings`` builds a
:class:`HardwareWatchdog`, ``Engine.aclose`` routes through it when present,
and the legacy direct-constructor path still falls back to
``provider.shutdown_all()`` for tests / the deprecated alias.

Synthesis §4 Phase 3 step 35; §11.3 R22 (placeholder superseded).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tether import Engine
from tether.config.settings import Settings
from tether.runtime.hw_watchdog import HardwareWatchdog
from tether.runtime.watchdog_mode import WatchdogMode


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _settings_dict(tmp_db: str) -> dict:
    return {
        "system": {"prompt": "test-prompt"},
        "providers": {
            "model": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
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
        "tools": {"registry": [], "enabled": []},
        "orchestrator": {
            "registry": {
                "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
            },
        },
    }


@pytest.fixture
def settings(tmp_path) -> Settings:
    db = tmp_path / "engine_aclose_hw.db"
    return Settings.model_validate(_settings_dict(str(db)))


def test_engine_from_settings_builds_hw_watchdog(settings):
    """Engine.from_settings always populates engine.hw_watchdog with a
    HardwareWatchdog instance — even when the provider isn't a
    HardwareLifecycle (DummyProvider here is filtered out so the watchdog
    has zero HW providers, but it still exists)."""
    engine = Engine.from_settings(settings)
    assert engine.hw_watchdog is not None
    assert isinstance(engine.hw_watchdog, HardwareWatchdog)
    # DummyProvider is NOT HardwareLifecycle → filtered out → 0 HW providers.
    assert engine.hw_watchdog.hw_provider_count == 0


@pytest.mark.anyio
async def test_engine_aclose_calls_hw_watchdog_shutdown_all():
    """When hw_watchdog is present, aclose routes through it (does NOT
    call provider.shutdown_all directly)."""
    fake_provider = MagicMock()
    fake_provider.shutdown_all = MagicMock()
    fake_watchdog = MagicMock(spec=HardwareWatchdog)
    eng = Engine(
        provider=fake_provider,
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        hw_watchdog=fake_watchdog,
    )
    await eng.aclose()
    fake_watchdog.shutdown_all.assert_called_once()
    fake_provider.shutdown_all.assert_not_called()


@pytest.mark.anyio
async def test_engine_aclose_idempotent():
    """Calling aclose twice does not call hw_watchdog.shutdown_all twice."""
    fake_watchdog = MagicMock(spec=HardwareWatchdog)
    eng = Engine(
        provider=MagicMock(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        hw_watchdog=fake_watchdog,
    )
    await eng.aclose()
    await eng.aclose()
    fake_watchdog.shutdown_all.assert_called_once()


@pytest.mark.anyio
async def test_engine_aclose_fallback_no_watchdog():
    """Direct constructor with hw_watchdog=None and a provider that exposes
    shutdown_all() falls back to the legacy path."""
    fake_provider = MagicMock()
    fake_provider.shutdown_all = MagicMock()
    eng = Engine(
        provider=fake_provider,
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        hw_watchdog=None,
    )
    await eng.aclose()
    fake_provider.shutdown_all.assert_called_once()


@pytest.mark.anyio
async def test_engine_aclose_fallback_no_watchdog_no_shutdown_all():
    """Direct constructor with hw_watchdog=None and a provider that doesn't
    expose shutdown_all() must not raise."""
    bare_provider = MagicMock(spec=[])  # no shutdown_all
    eng = Engine(
        provider=bare_provider,
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        hw_watchdog=None,
    )
    # Must not raise.
    await eng.aclose()
    assert eng._closed is True


@pytest.mark.anyio
async def test_engine_aclose_direct_non_hw_provider_closes_once():
    """The provider-map fan-out owns async close for direct engines."""
    provider = MagicMock(spec=[])
    provider.aclose = AsyncMock()
    eng = Engine(
        provider=provider,
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        hw_watchdog=None,
    )

    await eng.aclose()

    provider.aclose.assert_awaited_once()


def test_engine_watchdog_mode_library_default(settings):
    """from_settings defaults watchdog_mode to LIBRARY."""
    engine = Engine.from_settings(settings)
    assert engine.watchdog_mode is WatchdogMode.LIBRARY
    # The watchdog instance carries the same mode.
    assert engine.hw_watchdog.mode is WatchdogMode.LIBRARY


def test_engine_watchdog_mode_server_via_create_app():
    """create_app() builds Engine with watchdog_mode=WatchdogMode.SERVER per
    Phase 3 step 35 (HTTP entry point is the canonical SERVER-mode caller)."""
    from tether.app.http.api import create_app

    app = create_app()
    engine = app.state.gen_svc
    assert engine.watchdog_mode is WatchdogMode.SERVER
    # The watchdog instance carries the same mode.
    assert engine.hw_watchdog is not None
    assert engine.hw_watchdog.mode is WatchdogMode.SERVER
