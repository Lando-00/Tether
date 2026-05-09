"""Tests for ``tether_service.engine.Engine``.

Cited in _synthesis.md §4 Phase 2 (steps 21, 22, 25).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tether_service import Engine
from tether_service.config.settings import Settings
from tether_service.runtime.watchdog_mode import WatchdogMode


def _settings_dict(tmp_db: str) -> dict:
    """Minimal Settings dict using DummyProvider + sqlite session store."""
    return {
        "system": {"prompt": "test-prompt"},
        "providers": {
            "model": {
                "impl": "tether_service.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "parser": {
                "impl": "tether_service.protocol.parsers.sliding.SlidingParser",
                "args": {},
            },
            "session_store": {
                "impl": "tether_service.context.sqlite_store.SqliteSessionStore",
                "args": {"dsn": f"sqlite:///{tmp_db}"},
            },
        },
        "tools": {"registry": [], "enabled": []},
    }


@pytest.fixture
def settings(tmp_path) -> Settings:
    db = tmp_path / "engine_test.db"
    return Settings.model_validate(_settings_dict(str(db)))


def test_engine_from_settings_constructs(settings):
    engine = Engine.from_settings(settings)
    assert isinstance(engine, Engine)
    assert engine.provider is not None
    assert engine.parser is not None
    assert engine.store is not None
    assert engine.tools == {}
    assert engine.system_prompt == "test-prompt"
    assert engine.watchdog_mode is WatchdogMode.LIBRARY


def test_engine_watchdog_mode_default(settings):
    eng_default = Engine.from_settings(settings)
    assert eng_default.watchdog_mode is WatchdogMode.LIBRARY

    eng_server = Engine.from_settings(settings, watchdog_mode=WatchdogMode.SERVER)
    assert eng_server.watchdog_mode is WatchdogMode.SERVER


@pytest.mark.anyio
async def test_engine_async_context_manager(settings):
    async with Engine.from_settings(settings) as eng:
        assert isinstance(eng, Engine)
        assert eng._closed is False
    assert eng._closed is True


@pytest.mark.anyio
async def test_engine_aclose_idempotent(settings):
    eng = Engine.from_settings(settings)
    await eng.aclose()
    assert eng._closed is True
    # Second call must not raise.
    await eng.aclose()
    assert eng._closed is True


@pytest.mark.anyio
async def test_engine_aclose_calls_provider_shutdown_all():
    fake_provider = MagicMock()
    fake_provider.shutdown_all = MagicMock()
    eng = Engine(
        provider=fake_provider,
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
    )
    await eng.aclose()
    fake_provider.shutdown_all.assert_called_once()


def test_engine_method_surface_matches_generationservice():
    """Engine must keep the same method surface as the legacy GenerationService
    so existing routers (chat / sessions / models / health) work unchanged.
    """
    expected = [
        "stream",
        "create_session",
        "list_sessions",
        "get_session_messages",
        "delete_session",
        "delete_all_sessions",
        "list_models",
        "unload_model",
    ]
    for name in expected:
        assert hasattr(Engine, name), f"Engine missing method: {name}"


@pytest.fixture
def anyio_backend():
    return "asyncio"
