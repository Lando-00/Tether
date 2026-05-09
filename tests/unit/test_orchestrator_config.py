"""Tests for ``tether_service.core.types.OrchestratorConfig``.

Per _synthesis.md §4 Phase 2 step 23 (DI of typed config slice into
orchestrator + tool runner; replaces ``load_settings_legacy()`` calls in
business logic).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tether_service.config.settings import Settings
from tether_service.core.types import OrchestratorConfig


def _settings_dict() -> dict:
    return {
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
                "args": {"dsn": "sqlite:///:memory:"},
            },
        },
        "limits": {
            "max_tool_loops": 7,
            "auto_reload_on_fatal_error": False,
            "tool_timeout_sec": 11,
        },
        "context": {
            "save_thinking": False,
            "include_thinking_in_history": True,
        },
    }


def test_orchestrator_config_from_settings():
    settings = Settings.model_validate(_settings_dict())
    cfg = OrchestratorConfig.from_settings(settings)
    assert cfg.max_tool_loops == 7
    assert cfg.auto_reload_on_fatal_error is False
    assert cfg.save_thinking is False
    assert cfg.include_thinking_in_history is True


def test_orchestrator_config_frozen():
    cfg = OrchestratorConfig(
        max_tool_loops=3,
        auto_reload_on_fatal_error=True,
        save_thinking=True,
        include_thinking_in_history=False,
    )
    with pytest.raises(FrozenInstanceError):
        cfg.max_tool_loops = 99  # type: ignore[misc]


def test_orchestrator_config_explicit_fields():
    cfg = OrchestratorConfig(
        max_tool_loops=5,
        auto_reload_on_fatal_error=True,
        save_thinking=True,
        include_thinking_in_history=False,
    )
    assert isinstance(cfg.max_tool_loops, int)
    assert isinstance(cfg.auto_reload_on_fatal_error, bool)
    assert isinstance(cfg.save_thinking, bool)
    assert isinstance(cfg.include_thinking_in_history, bool)
