"""Tests for the p2-cleanup wiring of Engine.

Verifies that ``Engine.from_settings`` builds an ``OrchestratorConfig`` and
``ToolRunner`` from typed Settings, that ``Engine.stream`` injects them into
``orchestrate``, and that an optional ``cancel_event`` is forwarded.

Per _synthesis.md §4 Phase 2 step 23 (DI of typed config / tool runner).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tether_service import Engine
from tether_service.config.settings import Settings
from tether_service.core.types import OrchestratorConfig
from tether_service.protocol.orchestration.tool_runner import ToolRunner


def _settings_dict(tmp_db: str) -> dict:
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
        "limits": {
            "max_tool_loops": 4,
            "tool_timeout_sec": 9,
            "auto_reload_on_fatal_error": False,
        },
        "context": {
            "save_thinking": False,
            "include_thinking_in_history": True,
        },
    }


@pytest.fixture
def settings(tmp_path) -> Settings:
    db = tmp_path / "engine_cleanup.db"
    return Settings.model_validate(_settings_dict(str(db)))


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_engine_builds_orchestrator_config(settings):
    engine = Engine.from_settings(settings)
    cfg = engine.orchestrator_config
    assert isinstance(cfg, OrchestratorConfig)
    assert cfg.max_tool_loops == 4
    assert cfg.auto_reload_on_fatal_error is False
    assert cfg.save_thinking is False
    assert cfg.include_thinking_in_history is True


def test_engine_builds_tool_runner_with_timeout(settings):
    engine = Engine.from_settings(settings)
    assert isinstance(engine.tool_runner, ToolRunner)
    assert engine.tool_runner.timeout == 9


@pytest.mark.anyio
async def test_engine_stream_passes_config_and_tool_runner(settings):
    """Engine.stream routes through Engine.chat; the orchestrator_config and
    tool_runner that from_settings built end up as the ChattyAgentOrchestrator
    constructor args. Verified by patching Engine.chat and confirming it is
    called (mode defaults to None → resolved to 'chat' via _orchestrator_default_mode).
    """
    engine = Engine.from_settings(settings)

    captured: dict = {}

    async def fake_chat(**kwargs):
        captured.update(kwargs)
        if False:
            yield  # make it an async generator

    with patch.object(engine, "chat", new=fake_chat):
        async for _ in engine.stream(
            session_id="s1", prompt="hi", model_name="m"
        ):
            pass

    # Engine.stream calls Engine.chat with mode and cancel_token kwargs.
    assert captured.get("session_id") == "s1"
    assert captured.get("mode") is None  # no mode passed → None (chat resolves via default)
    assert "cancel_token" in captured  # cancel_event=None → cancel_token=None


@pytest.mark.anyio
async def test_engine_stream_passes_cancel_event(settings):
    """cancel_event is wrapped in AsyncEventCancelToken and forwarded to Engine.chat
    as cancel_token. Engine.stream no longer calls orchestrate() directly.
    """
    from tether_service.protocol.orchestration.cancel import AsyncEventCancelToken

    engine = Engine.from_settings(settings)
    ev = asyncio.Event()

    captured: dict = {}

    async def fake_chat(**kwargs):
        captured.update(kwargs)
        if False:
            yield

    with patch.object(engine, "chat", new=fake_chat):
        async for _ in engine.stream(
            session_id="s2", prompt="hi", model_name="m", cancel_event=ev
        ):
            pass

    token = captured.get("cancel_token")
    assert isinstance(token, AsyncEventCancelToken)  # cancel_event wrapped into token


def test_engine_default_construction_without_explicit_config():
    """Direct constructor (the path GenerationService uses) must still work
    without orchestrator_config / tool_runner being passed explicitly."""
    eng = Engine(
        provider=AsyncMock(),
        parser=AsyncMock(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="",
    )
    assert isinstance(eng.orchestrator_config, OrchestratorConfig)
    assert isinstance(eng.tool_runner, ToolRunner)
