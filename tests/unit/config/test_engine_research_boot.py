from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from tether.config.settings import Settings
from tether.core.errors import ConfigError
from tether.engine import Engine


def _settings_dict(tmp_path) -> dict:
    db = tmp_path / "research_boot.db"
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
        "storage": {"sqlite": {"dsn": f"sqlite:///{db}"}},
        "tools": {
            "registry": [
                {
                    "name": "web_search",
                    "impl": "tether.tools.web_search_tool.WebSearchTool",
                }
            ],
            "enabled": ["web_search"],
            "disabled": [],
        },
        # ADR-0020 §D6: research mode is opt-in via orchestrator.registry.
        # These boot-validation tests must explicitly enable research so the
        # web_search-required + per-phase-model-override gates actually fire.
        "orchestrator": {
            "registry": {
                "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
                "research": "tether.protocol.orchestration.notebook.NotebookOrchestrator",
            }
        },
    }


def _settings(tmp_path, **updates) -> Settings:
    raw = _settings_dict(tmp_path)
    for key, value in updates.items():
        target = raw
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return Settings.model_validate(raw)


def _close(engine: Engine) -> None:
    asyncio.run(engine.aclose())


def test_research_mode_requires_web_search(tmp_path):
    raw = _settings_dict(tmp_path)
    raw["tools"]["enabled"] = []
    settings = Settings.model_validate(raw)

    with pytest.raises(ConfigError, match="web_search"):
        Engine.from_settings(settings)


def test_research_mode_with_web_search_enabled_passes(tmp_path):
    engine = Engine.from_settings(_settings(tmp_path))
    try:
        assert isinstance(engine, Engine)
    finally:
        _close(engine)


def test_per_phase_model_override_unknown_raises(tmp_path):
    settings = _settings(
        tmp_path,
        **{"orchestrator.research.planner_model": "nonexistent-model"},
    )

    with pytest.raises(ConfigError, match="planner_model.*nonexistent-model"):
        Engine.from_settings(settings)


def test_per_phase_model_override_none_passes(tmp_path):
    raw = deepcopy(_settings_dict(tmp_path))
    raw["orchestrator"]["research"] = {
        "planner_model": None,
        "extractor_model": None,
        "synthesizer_model": None,
    }
    engine = Engine.from_settings(Settings.model_validate(raw))
    try:
        assert isinstance(engine, Engine)
    finally:
        _close(engine)


def test_per_phase_model_override_valid_passes(tmp_path):
    settings = _settings(
        tmp_path,
        **{"orchestrator.research.planner_model": "dummy-model-1"},
    )

    engine = Engine.from_settings(settings)
    try:
        assert isinstance(engine, Engine)
    finally:
        _close(engine)
